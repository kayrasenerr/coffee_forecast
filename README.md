# Coffee Quant v0.2

**Modular quantitative forecasting, regime-detection, and interactive research tool for global coffee markets.**

Arabica (KC=F) · Robusta (RC=F) · Any custom Yahoo Finance ticker

---

## Quick Start

```bash
git clone <repo> && cd coffee_quant
pip install -r requirements.txt
./run_app.sh              # opens http://localhost:8501
```

---

## Interactive Tool

The Streamlit app (`app.py`) gives full point-and-click access to every module.

### Sidebar Controls

| Control | What it does |
|---|---|
| **Variety** | Arabica (KC=F), Robusta (RC=F), or any custom ticker |
| **Train / Test dates** | Fully configurable — test any period |
| **Forecast horizon** | 1–21 days ahead |
| **Live data** | yfinance first; fallback to calibrated simulator |
| **Models** | Enable/disable HMM, GARCH, AR, Bayesian, SARIMAX, Ensemble |
| **Exogenous data** | 28 external variables; optional NASA POWER climate |
| **Significance tests** | Granger + LASSO + RF variable selection |
| **FRED API key** | Optional — improves macro data |
| **HMM regimes** | 2–5 states |
| **Backtest windows** | Train/test window, folds — all configurable |

### Eight Tabs

| Tab | Content |
|---|---|
| 📊 **Data** | OHLCV chart, price stats |
| 🌍 **Exogenous Variables** | Correlation bar chart; Granger/LASSO/RF significance table |
| 🎯 **Regime Detection** | HMM regime overlay; transition matrix |
| 📈 **Volatility** | GARCH conditional vol bands |
| 🔮 **Directional Models** | AR/Bayesian/SARIMAX; fan chart; calibration |
| 🔄 **Walk-Forward Backtest** | Per-fold DA / Brier / Sharpe |
| 📋 **Model Comparison** | Ranked metric table |
| 📡 **Live Forecast** | Current regime + directional probability |

---

## Exogenous Variables (28 series, 6 categories)

### FX — Producer Currency Competitiveness
`fx_usdbrl` `fx_usdvnd` `fx_usdcop` `fx_usdidr` `fx_usdugx` `fx_eurusd` `fx_gbpusd` `fx_jpyusd`

Higher producer-currency/USD = cheaper exports = more supply pressure = bearish coffee.
EUR/GBP/JPY strength = stronger demand side.

### Commodity Prices
`sugar_price` `corn_price` `crude_oil` `nat_gas` `cocoa_price`

Sugar competes for Brazilian land. Crude/nat-gas drive logistics and roasting costs.

### Macro / Financial
`vix` `dxy_proxy` `spx` `us_10y_yield` `us_cpi` `fed_rate` `brazil_rate`

VIX and DXY are the key risk/USD regime signals. Brazil Selic constrains supply-side investment.

### Climate / ENSO
`oni` `enso_el_nino` `enso_la_nina` `oni_lag3m` `oni_lag6m` `oni_lag9m`

El Niño → Brazil/Vietnam drought → supply risk → bullish. Lags 3–9 months for production response.

### Shipping / Logistics
`shipping_matx` `heating_oil`

Container and bunker cost proxies.

### NASA POWER Regional Climate (optional)
Daily rainfall and temperature for Brazil Sul de Minas, Cerrado, Vietnam Dak Lak, Colombia Huila, Ethiopia Sidama.

---

## Significance Testing Pipeline

```
For each candidate exogenous variable:
  1. Granger causality (lags 1–10)     → keep if min p-value < 0.10
  2. Spearman cross-correlation         → keep if |max corr| > 0.05 at any lag 0–21
  3. LASSO path (5-fold CV)             → keep if non-zero at best λ
  4. Random Forest importance           → keep if in top-15 features

Variable passes if it satisfies ≥ 1 test.
Final SARIMAX uses ONLY passing variables.
```

---

## Testing Different Periods — Recommended Configurations

| Use case | Train window | Test window | Folds |
|---|---|---|---|
| Standard | 756d (3yr) | 63d (3mo) | 5 |
| Crisis period | 252d (1yr) | 21d (1mo) | 8 |
| Long-run stability | 1260d (5yr) | 126d (6mo) | 4 |
| Single regime | 252d | 63d | 5 |

---

## Evaluation Results (90-day OOS, synthetic data calibrated to KC=F 2022-2026)

| Model | DA | Brier | BSS | Sharpe | p-value |
|---|---|---|---|---|---|
| **AR(1)** | **0.800** | **0.166** | **0.338** | **12.91** | 0.064 |
| AR(2) / AR(5) | 0.733 | 0.177 | 0.293 | 11.59 | ~0.18 |
| Ensemble (equal) | 0.733 | 0.188 | 0.249 | 11.20 | 0.116 |
| Ensemble (regime) | 0.667 | 0.206 | 0.178 | 10.65 | 0.272 |
| BayesianRidge | 0.267 | 0.330 | −0.321 | −6.90 | 1.000 |
| Random baseline | 0.500 | 0.250 | 0.000 | 0.000 | — |

AR(1) dominates in a trending market. Bayesian underperforms because regularisation
priors assume stationarity — correct fix is regime-conditional model selection via HMM.
No model reaches p<0.05 with n=30 OOS predictions; use 5-fold walk-forward for power.

---

## Project Structure

```
coffee_quant/
├── app.py                   ← Streamlit tool (entry point)
├── run_app.sh               ← One-command launcher
├── requirements.txt
├── contracts/               ← Schemas + interfaces
├── config/                  ← Settings + YAML
├── ingestion/               ← Data sources
│   └── exog_fetcher.py      ← 28-variable comprehensive fetcher
├── preprocessing/
├── features/
│   └── significance.py      ← Granger + LASSO + RF selection
├── models/                  ← HMM, GARCH, SARIMAX, Bayesian, Ensemble
├── backtesting/             ← Walk-forward engine + metrics
├── experiment/              ← Run tracking
├── visualization/           ← Plotting functions
├── orchestration/           ← Pipeline + CLI
└── tests/
    ├── unit/
    ├── integration/
    └── evaluation/          ← Standalone evaluation + simulator
```

## CLI

```bash
python -m orchestration.cli run --variety arabica --backtest
python tests/evaluation/model_evaluation.py
```
