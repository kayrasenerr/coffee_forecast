"""
tests/evaluation/market_simulator.py
======================================
Realistic synthetic coffee market data calibrated to known
KC=F market characteristics (2022-2026).

Used when external APIs are unavailable (network restrictions).

Calibration parameters come from historical KC=F statistics:
  - Annualised vol:    ~35-50%
  - Mean daily return: ~0.0%
  - Price range 2022-2026: ~150-380 cents/lb
  - Volatility clustering (GARCH alpha~0.08, beta~0.87)
  - BRL/USD correlation with KC: ~-0.45
  - Seasonal pattern: Q1 pre-harvest strength, Q3 harvest weakness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


def simulate_garch_prices(
    n: int,
    start_price: float,
    mu: float = 0.0002,
    omega: float = 0.00001,
    alpha: float = 0.08,
    beta: float = 0.87,
    df_t: float = 5.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GARCH(1,1) price simulation with Student-t innovations.
    Returns (prices, log_returns).
    """
    rng = np.random.default_rng(seed)
    sigma2 = np.zeros(n)
    log_ret = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        # Student-t innovation
        z = rng.standard_t(df_t)
        eps = np.sqrt(sigma2[t-1]) * z
        log_ret[t] = mu + eps
        sigma2[t] = omega + alpha * eps**2 + beta * sigma2[t-1]

    prices = start_price * np.exp(np.cumsum(log_ret))
    return prices, log_ret


def add_seasonality(prices: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
    """
    Add seasonal component: coffee prices historically stronger
    in Q1 (pre-Brazil flowering) and Q4 (pre-harvest uncertainty),
    weaker in Q2-Q3 (Brazil harvest).
    """
    seasonal = np.zeros(len(index))
    for i, ts in enumerate(index):
        month = ts.month
        if month in [1, 2, 3]:
            seasonal[i] = 0.0008    # pre-harvest strength
        elif month in [4, 5, 6]:
            seasonal[i] = -0.0005   # harvest pressure
        elif month in [7, 8, 9]:
            seasonal[i] = -0.0003   # peak harvest
        else:
            seasonal[i] = 0.0004    # Q4 positioning
    return prices * np.exp(np.cumsum(seasonal))


def simulate_brl(arabica_ret: np.ndarray, seed: int = 7) -> np.ndarray:
    """
    USD/BRL with ~-0.45 correlation to KC returns.
    BRL weakens when KC rises (supply expectations embedded).
    """
    n = len(arabica_ret)
    rng = np.random.default_rng(seed)
    rho = -0.45
    independent = rng.normal(0, 0.008, n)
    brl_ret = rho * arabica_ret + np.sqrt(1 - rho**2) * independent
    brl_ret *= 0.3  # scale down
    brl = 5.3 * np.exp(np.cumsum(brl_ret))
    return np.clip(brl, 4.5, 7.5)


def simulate_enso(n_months: int, seed: int = 13) -> np.ndarray:
    """
    ONI with realistic autocorrelation (AR(1), rho=0.75).
    Includes a mild El Niño episode in 2023-24.
    """
    rng = np.random.default_rng(seed)
    oni = np.zeros(n_months)
    oni[0] = 0.1
    for t in range(1, n_months):
        oni[t] = 0.75 * oni[t-1] + rng.normal(0, 0.35)
    # Embed realistic El Niño 2023 (well-documented)
    el_nino_start = min(18, n_months - 12)
    el_nino_end   = min(30, n_months - 1)
    oni[el_nino_start:el_nino_end] += np.linspace(0, 1.8, el_nino_end - el_nino_start)
    return np.clip(oni, -2.5, 2.5)


def simulate_cot(
    n_weekly: int,
    arabica_ret_weekly: np.ndarray,
    seed: int = 99,
) -> pd.DataFrame:
    """
    COT positioning correlated with medium-term price trend.
    Speculators are trend-followers; commercials are hedgers.
    """
    rng = np.random.default_rng(seed)
    trend = pd.Series(arabica_ret_weekly).rolling(4, min_periods=1).mean().values

    base_nc_long  = 45_000
    base_nc_short = 35_000
    base_oi       = 160_000

    nc_long  = np.clip(base_nc_long  + trend * 500_000 + rng.normal(0, 3000, n_weekly), 15_000, 90_000)
    nc_short = np.clip(base_nc_short - trend * 300_000 + rng.normal(0, 2500, n_weekly), 10_000, 70_000)
    oi       = np.clip(base_oi + rng.normal(0, 8000, n_weekly), 80_000, 250_000)
    comm_short = np.clip(oi * 0.45 + rng.normal(0, 4000, n_weekly), 30_000, 130_000)
    comm_long  = np.clip(oi * 0.30 + rng.normal(0, 3000, n_weekly), 20_000, 100_000)
    nr_long    = rng.integers(4_000, 14_000, n_weekly).astype(float)
    nr_short   = rng.integers(4_000, 14_000, n_weekly).astype(float)

    return pd.DataFrame({
        "noncommercial_long":  nc_long,
        "noncommercial_short": nc_short,
        "commercial_long":     comm_long,
        "commercial_short":    comm_short,
        "nonreportable_long":  nr_long,
        "nonreportable_short": nr_short,
        "open_interest":       oi,
    })


def generate_full_dataset(
    start: str = "2022-05-13",
    end:   str = "2026-05-12",
    seed:  int = 42,
) -> dict:
    """
    Generate the full realistic synthetic dataset matching
    the 4-year evaluation window.

    Price calibration:
      - 2022: ~215 cents/lb (post-COVID recovery peak)
      - 2023: ~175-220 range (supply normalization)
      - 2024: ~220-290 range (weather concerns Brazil)
      - 2025: ~280-380 range (ongoing supply tightness, BRL weakness)
      - 2026 YTD: ~310-380 range (elevated volatility regime)
    """
    idx = pd.bdate_range(start, end)
    n   = len(idx)

    # Arabica: GARCH(1,1) with positive drift calibrated to known price path
    # 2022-05: ~215  →  2026-05: ~350 cts/lb  (+63% over 1043 bdays)
    # Required daily drift: log(350/215)/1043 ≈ 0.000467/day
    arabica_prices, arabica_ret = simulate_garch_prices(
        n, start_price=215.0, mu=0.000467,
        omega=0.000010, alpha=0.07, beta=0.88, df_t=5, seed=seed + 10
    )
    arabica_prices = add_seasonality(arabica_prices, idx)
    # Soft floor: price-mean-reversion rather than hard clip
    floor = 150.0
    for i in range(1, n):
        if arabica_prices[i] < floor:
            arabica_prices[i] = floor * (1 + 0.001)  # nudge up from floor

    rng = np.random.default_rng(seed)
    arabica_df = pd.DataFrame({
        "open":   arabica_prices * rng.uniform(0.997, 1.003, n),
        "high":   arabica_prices * rng.uniform(1.002, 1.015, n),
        "low":    arabica_prices * rng.uniform(0.985, 0.998, n),
        "close":  arabica_prices,
        "volume": rng.integers(8_000, 65_000, n).astype(float),
    }, index=idx)

    # Robusta
    rob_prices, rob_ret = simulate_garch_prices(
        n, start_price=2200.0, mu=0.00015, omega=0.000008,
        alpha=0.07, beta=0.88, df_t=5, seed=seed + 1
    )
    rob_prices = np.clip(rob_prices, 1200, 5500)
    robusta_df = pd.DataFrame({
        "open":  rob_prices * rng.uniform(0.998, 1.002, n),
        "high":  rob_prices * rng.uniform(1.001, 1.010, n),
        "low":   rob_prices * rng.uniform(0.990, 0.999, n),
        "close": rob_prices,
        "volume": rng.integers(3_000, 25_000, n).astype(float),
    }, index=idx)

    # USD/BRL
    brl_rates = simulate_brl(arabica_ret, seed=seed + 2)
    brl_df = pd.DataFrame({"rate": brl_rates}, index=idx)

    # EUR/USD
    eur_rates = 1.08 + np.cumsum(rng.normal(0, 0.003, n))
    eur_rates = np.clip(eur_rates, 0.92, 1.22)
    eur_df = pd.DataFrame({"rate": eur_rates}, index=idx)

    # ENSO (monthly)
    enso_idx    = pd.date_range(start, end, freq="MS")
    oni_vals    = simulate_enso(len(enso_idx), seed=seed + 3)
    enso_df     = pd.DataFrame({"oni": oni_vals}, index=enso_idx)

    # COT (weekly — every Tuesday)
    cot_idx     = pd.bdate_range(start, end, freq="W-TUE")
    # Compute weekly arabica returns for correlation
    weekly_ret  = pd.Series(arabica_ret, index=idx).resample("W-TUE").sum().reindex(cot_idx, method="ffill").values
    cot_data    = simulate_cot(len(cot_idx), weekly_ret, seed=seed + 4)
    cot_data.index = cot_idx

    return {
        "arabica_futures": arabica_df,
        "robusta_futures":  robusta_df,
        "fx_usdbrl":        brl_df,
        "fx_eurusd":        eur_df,
        "enso":             enso_df,
        "cot_arabica":      cot_data,
    }
