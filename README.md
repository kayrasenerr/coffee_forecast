# Coffee Quant — Regime-Aware Probabilistic Forecasting for Coffee Futures

**A modular quantitative research pipeline that turns 28 raw exogenous data series into a statistically-screened, regime-conditioned, probabilistic forecast for Arabica (KC=F) and Robusta (RC=F) coffee futures — with a Streamlit front end for live, point-and-click experimentation.**

---

## 1. What this project actually does

Coffee is a commodity whose price is pushed around by dozens of weakly- and non-linearly-related forces: producer-currency depreciation, El Niño/La Niña rainfall shocks 3–9 months upstream of harvest, competing soft commodities, freight costs, and macro risk regimes (VIX, USD strength, rate cycles). Most public coffee-forecasting projects fit a single model (ARIMA, Prophet, an LSTM) to price history alone and report an in-sample R².

This project instead treats forecasting as a **pipeline problem**: ingest the raw drivers → statistically test which of them actually carry information → detect the prevailing market regime → fit several competing forecasting models *only* on variables that survived testing → and validate every claim through walk-forward, out-of-sample backtesting with a formal significance test, not just a single train/test split.

The deliverable is not "a model" but a research tool: an end-to-end framework for asking "does this variable matter, and how do I know I'm not fooling myself?"

---

## 2. Quick Start

```bash
git clone https://github.com/kayrasenerr/coffee_forecast.git
cd coffee_forecast
pip install -r requirements.txt
./run_app.sh              # or: streamlit run app.py
# opens http://localhost:8501
```

If yfinance can't reach live data (rate limits, no network), the app falls back automatically to a calibrated GARCH-based price simulator (`tests/evaluation/market_simulator.py`) so the pipeline always has data to run against.

---

## 3. Pipeline Architecture

```
ingestion/exog_fetcher.py          28 exogenous series across 6 categories (yfinance, FRED, NASA POWER)
        │
        ▼
features/significance.py           Granger + Spearman + LASSO + Random Forest screening
        │  (keep variable if it passes ≥ 1 of 4 tests)
        ▼
features/pipeline.py               Builds the modeling frame: target = 5-day forward log return,
                                    engineered features (realised vol, z-scores, returns), surviving exog
        │
        ▼
models/                            HMM regime detector · GARCH(1,1)-t volatility · AR(1)/(2)/(5)
                                    BayesianRidge · SARIMAX(1,0,1)+exog vs ARMA(1,0,1) control · Ensemble
        │
        ▼
backtesting/                       Rolling walk-forward folds → DA, Brier, Brier Skill Score,
                                    signal Sharpe, permutation p-value, calibration curve
        │
        ▼
app.py (Streamlit)                 Monte-Carlo fan-chart forecast + 8 interactive tabs
```

Every stage is independently importable — the Streamlit app is a thin orchestration layer over `ingestion/`, `features/`, `models/`, and `backtesting/`, so the same pipeline can also be driven headlessly via the CLI (`orchestration/cli.py`).

---

## 4. Exogenous Variable Universe — 28 series, 6 categories

Variables were chosen from commodity-market theory, not by throwing every available series at the model:

**FX — producer-currency competitiveness**
`fx_usdbrl` `fx_usdvnd` `fx_usdcop` `fx_usdidr` `fx_usdugx` `fx_eurusd` `fx_gbpusd` `fx_jpyusd`
A weaker BRL/VND/COP/IDR/UGX against USD makes exports cheaper for Brazil, Vietnam, Colombia, Indonesia, and Uganda — more supply pressure, bearish coffee. EUR/GBP/JPY strength instead proxies demand-side purchasing power in the largest importing blocs.

**Commodity prices**
`sugar_price` `corn_price` `crude_oil` `nat_gas` `cocoa_price`
Sugar and corn compete with coffee for Brazilian farmland (land-substitution effect); crude/nat-gas feed directly into shipping and roasting energy costs.

**Macro / financial**
`vix` `dxy_proxy` `spx` `us_10y_yield` `us_cpi` `fed_rate` `brazil_rate`
VIX and the dollar index are the key risk-regime and USD-strength signals; Brazil's Selic rate constrains producer-side investment and storage/carry decisions.

**Climate / ENSO**
`oni` `enso_el_nino` `enso_la_nina` `oni_lag3m` `oni_lag6m` `oni_lag9m`
El Niño conditions raise drought risk in Brazil and Vietnam, which only shows up in production 3–9 months later — hence the explicit lagged ONI features rather than only contemporaneous readings.

**Shipping / logistics**
`shipping_matx` `heating_oil`
Container and bunker-fuel cost proxies.

**NASA POWER regional climate (optional, slower fetch)**
Daily rainfall and temperature for the five growing regions that matter most: Brazil's Sul de Minas and Cerrado, Vietnam's Dak Lak, Colombia's Huila, and Ethiopia's Sidama.

---

## 5. Variable Significance Testing — how a variable earns its place in the model

Feeding 28 (often collinear, often lagged, often nonlinear) candidate variables straight into a regression invites overfitting and spurious coefficients. Every candidate is instead run through four independent tests, each catching a different kind of relationship a single test would miss:

| Test | What it catches | Pass threshold |
|---|---|---|
| **Granger causality** (lags 1–10) | Does the variable's *past* help predict price beyond price's own history? | min p-value across lags < 0.10 |
| **Spearman cross-correlation** (lags 0–21 days) | Monotonic, possibly nonlinear, possibly lagged association | \|max corr\| > 0.05 at any lag |
| **LASSO path** (5-fold CV) | Does the variable survive regularization against everything else simultaneously? | non-zero coefficient at best λ |
| **Random Forest importance** | Nonlinear / interaction effects linear tests miss | ranks in top 15 features |

```
A variable passes if it satisfies ≥ 1 of the 4 tests.
Only passing variables are exposed to the final SARIMAX(1,0,1)+exog model.
```

Requiring unanimous agreement across all four tests was deliberately rejected — many of the climate and FX series have weak, regime-dependent, or nonlinear effects that only one test family is sensitive to, so an AND-rule would have discarded real signal. The OR-rule, combined with downstream coefficient-significance checks inside SARIMAX itself, is the actual guard against false positives (see §7). Both the Granger p-value threshold and the minimum correlation threshold are exposed as sliders in the app, so the strictness of variable admission is a tunable research parameter, not a hard-coded assumption.

---

## 6. Models

| Model | Role | Key detail |
|---|---|---|
| **HMM Regime Detector** | Classifies the market into 2–5 latent regimes (bull / bear / volatile / neutral / etc.) | Gaussian HMM fit via Baum-Welch on 1-day return, 21-day realised vol, 5-day return, 63-day price z-score |
| **GARCH(1,1)** | Conditional volatility forecast | Student-t innovations to capture fat tails in daily coffee returns |
| **AR(1) / AR(2) / AR(5)** | Baseline directional forecasters on the 5-day forward log-return target | Closed-form normal-CDF conversion of the point forecast into P(price up) |
| **BayesianRidge** | Probabilistic regression over engineered + exogenous features | Returns predictive mean *and* standard deviation, so uncertainty is propagated, not bolted on |
| **SARIMAX(1,0,1) + exog vs. ARMA(1,0,1) control** | The actual "do the screened exogenous variables help" test | Always fits both, in parallel, on identical data, so directional-accuracy / Brier / Sharpe *lift* from adding exog is directly measurable rather than asserted |
| **Ensemble** | Equal-weight and regime-conditional blends of the above | — |

---

## 7. Backtesting & Statistical Validation

This is the part of the project built specifically to prevent the model from reporting accuracy that is really just noise:

- **Walk-forward folds**, not a single train/test split — configurable train window, test window, and number of rolling folds (defaults: 756-day train / 63-day test / 5 folds), so every metric below is an out-of-sample average across multiple, non-overlapping market periods rather than one lucky window.
- **Directional Accuracy (DA)** — fraction of correct up/down calls.
- **Brier score & Brier Skill Score** — proper scoring rule for the predicted probabilities themselves, benchmarked against the climatological p = 0.5 baseline (BSS = 1 − Brier / 0.25), so a model can't game DA with overconfident 50/50 calls.
- **Signal Sharpe ratio** — annualized Sharpe of a simple threshold rule (long if P(up) > 0.55, short if P(up) < 0.45, flat otherwise), so "accurate" forecasts are also checked for whether they'd translate into a usable trading signal net of noise.
- **Permutation test (500 draws)** — the predicted probabilities are randomly shuffled 500 times to build a null distribution of DA under "no real skill," and the empirical p-value reports how often that null beats the observed DA. This is the step that stops a model from being declared "good" just because it beat 50% on a small sample.
- **Calibration curve** — predicted probabilities are bucketed into five bins and checked against realized hit-rate in each bin, so "70% confident" calls are checked to actually resolve up ~70% of the time.

---

## 8. Probabilistic Forecast Engine

The live forecast is not a single number. A fitted AR(1) process is used as the data-generating process for a Monte Carlo simulation (100–2,000 configurable paths), producing day-by-day price percentile bands (p5 / p25 / median / p75 / p95) and a horizon-level P(price up), visualized as a fan chart alongside GARCH volatility regime and the currently-detected HMM market regime.

---

## 9. Sample Evaluation Results

90-day out-of-sample window, synthetic data calibrated to KC=F price/volatility behavior 2022–2026:

| Model | DA | Brier | BSS | Sharpe | p-value |
|---|---|---|---|---|---|
| **AR(1)** | **0.800** | **0.166** | **0.338** | **12.91** | 0.064 |
| AR(2) / AR(5) | 0.733 | 0.177 | 0.293 | 11.59 | ~0.18 |
| Ensemble (equal) | 0.733 | 0.188 | 0.249 | 11.20 | 0.116 |
| Ensemble (regime) | 0.667 | 0.206 | 0.178 | 10.65 | 0.272 |
| BayesianRidge | 0.267 | 0.330 | −0.321 | −6.90 | 1.000 |
| Random baseline | 0.500 | 0.250 | 0.000 | 0.000 | — |

Read honestly: AR(1) wins in a trending synthetic market, and BayesianRidge actively underperforms — its regularization prior implicitly assumes stationarity, which breaks down across regime shifts; the documented fix is regime-conditional model selection driven by the HMM output rather than a single global model. **None of the models reach p < 0.05 at n = 30 OOS predictions** — the permutation test is doing exactly what it's supposed to do here, flagging that this sample size lacks the statistical power to distinguish skill from luck, which is why the walk-forward backtest (§7) with multiple folds, not this single 90-day window, is the metric to trust.

---

## 10. Interactive Tool (Streamlit)

### Sidebar controls
| Control | What it does |
|---|---|
| Variety | Arabica (KC=F), Robusta (RC=F), or any custom Yahoo Finance ticker |
| Train / Test dates | Fully configurable date ranges |
| Forecast horizon | 1–30 days ahead |
| Live data | yfinance first, calibrated simulator fallback |
| Models | Toggle HMM / GARCH / AR / Bayesian / SARIMAX / Ensemble independently |
| Exogenous data | All 28 variables, optional NASA POWER climate (slower) |
| Significance tests | Granger p-value and min-correlation thresholds, live-adjustable |
| FRED API key | Optional, improves macro series quality |
| HMM regimes | 2–5 states |
| Backtest | Train/test window length and number of folds |

### Eight tabs
📊 Data · 🌍 Exogenous Variables · 🎯 Regime Detection · 📈 Volatility · 🔮 Directional Models · 🔄 Walk-Forward Backtest · 📋 Model Comparison · 📡 Live Forecast

---

## 11. CLI

```bash
python -m orchestration.cli run --variety arabica --backtest
python tests/evaluation/model_evaluation.py
```

---

## 12. Project Structure

```
coffee_forecast/
├── app.py                   ← Streamlit research tool (entry point, ~1,600 lines)
├── run_app.sh / launch.bat  ← One-command launchers
├── conftest.py               ← pytest fixtures
├── pyproject.toml / requirements.txt
├── .env.example               ← FRED_API_KEY and other secrets template
├── contracts/                ← Shared schemas (CoffeeVariety, DataFrequency, FeatureFrame)
├── config/                    ← Settings + YAML configuration
├── ingestion/
│   └── exog_fetcher.py       ← 28-variable comprehensive exogenous fetcher
├── preprocessing/             ← Cleaning, alignment, frequency conversion
├── features/
│   ├── significance.py       ← Granger + Spearman + LASSO + RF selection
│   └── pipeline.py            ← Builds the final modeling FeatureFrame
├── models/                    ← HMM, GARCH, AR, Bayesian, SARIMAX, Ensemble
├── backtesting/               ← Walk-forward engine + scoring (DA/Brier/BSS/Sharpe/permutation)
├── data/features/             ← Cached/generated feature artifacts
├── experiment/                 ← Run tracking
├── visualization/              ← Plotly chart builders
├── orchestration/              ← Pipeline orchestration + CLI
└── tests/
    ├── unit/
    ├── integration/
    └── evaluation/
        ├── model_evaluation.py    ← Standalone OOS evaluation harness
        └── market_simulator.py    ← Calibrated GARCH-based synthetic price generator (live-data fallback)
```

---

## 13. Recommended Backtest Configurations

| Use case | Train window | Test window | Folds |
|---|---|---|---|
| Standard | 756d (3yr) | 63d (3mo) | 5 |
| Crisis period | 252d (1yr) | 21d (1mo) | 8 |
| Long-run stability | 1260d (5yr) | 126d (6mo) | 4 |
| Single regime | 252d | 63d | 5 |

---

## 14. Tech Stack

Python · Streamlit · pandas / NumPy · statsmodels (AutoReg, SARIMAX, Granger causality) · scikit-learn (LassoCV, RandomForestRegressor, BayesianRidge) · hmmlearn (Gaussian HMM) · arch (GARCH) · SciPy (Spearman correlation, normal-CDF probability conversion, permutation testing) · Plotly (fan charts, regime overlays, calibration plots) · yfinance / FRED / NASA POWER for data ingestion.

---

## 15. Known Limitations / Roadmap

- BayesianRidge's stationarity-assuming prior underperforms across regime shifts — regime-conditional model switching (already partially scaffolded via the HMM output) is the planned fix rather than discarding the model.
- At n≈30 OOS predictions, no single model reaches conventional significance — the walk-forward, multi-fold backtest is the metric that should be trusted over any single evaluation window.
- The Robusta companion-ticker fallback and NASA POWER fetch are best-effort and degrade gracefully (sparse-data warnings shown in the Exogenous Variables tab) rather than failing the whole pipeline.
