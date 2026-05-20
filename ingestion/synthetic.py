"""
ingestion/synthetic.py
Generates realistic synthetic coffee futures + FX data for:
  - System testing when live feeds are unavailable
  - Backtesting framework validation
  - CI/CD pipelines

Simulation properties match empirical Arabica KC=F stylized facts:
  - Mean-reverting price (Ornstein-Uhlenbeck around a seasonal trend)
  - GARCH-like volatility clustering
  - Regime switching: 3 states (calm / trending / crisis)
  - Brazilian harvest seasonality (peak supply: May–Sep → price pressure)
  - USD/BRL co-integrated with coffee price (negative correlation)
  - Fat-tailed shocks (t-distributed innovations for crisis regimes)

Replace this module by swapping DataRegistry to YFinanceFuturesSource
once live network access is available.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from schemas.types import PriceFrame


class SyntheticCoffeeSource:
    """
    Produces 5 years of daily synthetic Arabica and USD/BRL data.

    Arabica price in ¢/lb, approximate historical range 100–350.
    USD/BRL in BRL per USD, approximate range 4.5–5.8.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def fetch_arabica(self, start: date, end: date) -> PriceFrame:
        dates, prices = self._simulate_arabica(start, end)
        df = self._to_ohlcv(dates, prices, base_vol=0.015)
        return PriceFrame(symbol="KC=F (synthetic)", data=df, source="synthetic", currency="USc/lb")

    def fetch_usd_brl(self, start: date, end: date) -> PriceFrame:
        dates, arabica_prices = self._simulate_arabica(start, end)
        # USD/BRL negatively correlated with coffee: cheaper BRL → cheaper coffee in USD
        brl_log = np.log(5.0) - 0.35 * (np.log(arabica_prices) - np.log(200))
        brl_log += self.rng.normal(0, 0.003, len(brl_log)).cumsum()
        brl_prices = np.exp(brl_log).clip(4.2, 6.5)
        df = self._to_ohlcv(dates, brl_prices, base_vol=0.005)
        return PriceFrame(symbol="BRL=X (synthetic)", data=df, source="synthetic", currency="BRL/USD")

    # ── Internal simulation ───────────────────────────────────────────────────

    def _simulate_arabica(self, start: date, end: date):
        dates = pd.bdate_range(start=str(start), end=str(end))
        n = len(dates)

        # Regime sequence: 3 states with transition matrix
        P = np.array([
            [0.97, 0.025, 0.005],   # calm → calm / trending / crisis
            [0.02, 0.96,  0.02 ],   # trending
            [0.05, 0.10,  0.85 ],   # crisis (persistent but recovers)
        ])
        regimes = self._markov_chain(P, n)

        # Per-regime drift and vol
        drift_map = {0: 0.0001, 1: 0.0004, 2: -0.0008}
        vol_map   = {0: 0.010,  1: 0.020,  2: 0.045 }

        # OU mean-reversion toward seasonal + long-run mean
        log_price = np.zeros(n)
        log_price[0] = np.log(200.0)  # start ~200¢/lb
        theta = 0.008  # mean-reversion speed

        for t in range(1, n):
            month = dates[t].month
            # Harvest seasonality: May–Sep = slight downward pressure on price
            seasonal_mean = np.log(200) + 0.08 * np.cos(2 * np.pi * (month - 2) / 12)
            reg = regimes[t]
            drift = drift_map[reg] + theta * (seasonal_mean - log_price[t - 1])
            # t-dist for crisis, normal otherwise
            if reg == 2:
                shock = self.rng.standard_t(df=4) * vol_map[reg]
            else:
                shock = self.rng.normal(0, vol_map[reg])
            log_price[t] = log_price[t - 1] + drift + shock

        prices = np.exp(log_price).clip(80, 450)
        return dates, prices

    def _markov_chain(self, P: np.ndarray, n: int) -> np.ndarray:
        states = np.zeros(n, dtype=int)
        states[0] = 0
        for t in range(1, n):
            states[t] = self.rng.choice(len(P), p=P[states[t - 1]])
        return states

    def _to_ohlcv(self, dates, closes: np.ndarray, base_vol: float) -> pd.DataFrame:
        n = len(dates)
        noise = self.rng.normal(0, base_vol, (n, 2))
        highs  = closes * (1 + np.abs(noise[:, 0]))
        lows   = closes * (1 - np.abs(noise[:, 1]))
        opens  = closes * (1 + self.rng.normal(0, base_vol / 2, n))
        volume = (self.rng.integers(5000, 50000, n)).astype(float)
        df = pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volume,
        }, index=dates)
        df.index.name = "Date"
        return df
