# Coffee Market Forecasting System

Modular, probabilistic forecasting and regime-detection for Arabica/Robusta futures.

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py
# Charts saved to ./output/
```

---

## Architecture

```
coffee_forecast/
├── config/settings.py        ← ALL tunable parameters (Pydantic)
├── schemas/types.py          ← Canonical data contracts (dataclasses)
├── ingestion/
│   ├── base.py               ← Abstract DataSource
│   ├── futures.py            ← yfinance: KC=F (Arabica), RB=F (Robusta)
│   ├── fx.py                 ← yfinance: BRL=X (USD/BRL)
│   └── registry.py           ← Single entry point, returns dict[name→PriceFrame]
├── features/
│   ├── base.py               ← Abstract FeatureBuilder
│   ├── price_features.py     ← Momentum, z-scores, FX causal features
│   ├── volatility_features.py← Realised vol, Parkinson, vol-ratio
│   └── pipeline.py           ← Assembles FeatureMatrix + HMM input helper
├── models/
│   ├── base.py               ← Abstract BaseModel
│   ├── regime.py             ← GaussianHMM (3-state: calm/trending/crisis)
│   ├── volatility.py         ← GARCH(1,1) conditional vol
│   └── forecast.py           ← SARIMAX with exogenous variables
├── backtest/
│   └── walk_forward.py       ← Expanding-window OOS validation
├── viz/
│   └── plots.py              ← Standalone chart functions → matplotlib Figure
└── main.py                   ← Orchestration entry point
```

---

## Module Contracts

Each module is independently replaceable. The contracts:

| Module | Input | Output |
|---|---|---|
| `DataSource.fetch()` | symbol, start, end | `PriceFrame` |
| `FeatureBuilder.build()` | `dict[str, PriceFrame]` | `pd.DataFrame` |
| `FeaturePipeline.build()` | `dict[str, PriceFrame]` | `FeatureMatrix` |
| `HMMRegimeModel.fit_predict()` | `np.ndarray`, index | `RegimeResult` |
| `GARCHVolatilityModel.fit_predict()` | log-return `pd.Series` | `VolatilityResult` |
| `SARIMAXForecastModel.fit_predict()` | `FeatureMatrix` | `ForecastResult` |
| `WalkForwardBacktester.run()` | `FeatureMatrix`, `RegimeResult` | `BacktestReport` |

---

## Live Data

When `finance.yahoo.com` is reachable, all data is fetched live:

| Logical Name | Ticker | Exchange | Unit |
|---|---|---|---|
| `arabica` | `KC=F` | ICE New York | cents/lb |
| `robusta` | `RB=F` | LIFFE London | USD/MT |
| `usd_brl` | `BRL=X` | FX | BRL per USD |

To add new sources, subclass `DataSource` and register in `DataRegistry`.

---

## Adding Real Data (when you have it)

### Required PriceFrame schema

```python
# Any CSV with these columns works — just load and wrap:
import pandas as pd
from schemas.types import PriceFrame

df = pd.read_csv("my_data.csv", index_col="date", parse_dates=True)
df.columns = ["open", "high", "low", "close", "volume"]   # rename to match
pf = PriceFrame(symbol="KC=F", data=df, source="my_source")

# Then inject:
frames = registry.fetch_all()
frames["arabica"] = pf    # override live fetch with your data
```

### To add climate/ENSO/inventory data

1. Create `ingestion/climate.py` subclassing `DataSource`
2. Create `features/climate_features.py` subclassing `FeatureBuilder`
3. Register in `ingestion/registry.py` and `features/pipeline.py`

No other files need to change.

---

## Configuration

All parameters in `config/settings.py`. Key knobs:

```python
SETTINGS.data.history_years = 5        # training data window
SETTINGS.backtest.test_months = 3      # OOS backtest period
SETTINGS.models.hmm_n_states = 3       # regime count
SETTINGS.models.sarimax_order = (2,1,2)
SETTINGS.models.forecast_horizon = 5   # days ahead
```

---

## Outputs

| File | Description |
|---|---|
| `01_regime_overlay.png` | Price chart with HMM regime colour bands |
| `02_regime_probs.png` | Stacked regime probability (last 252 days) |
| `03_volatility.png` | Price + GARCH conditional volatility |
| `04_backtest.png` | OOS returns vs forecast + cumulative PnL |
| `05_forecast_oos.png` | 5-day forward forecast with 80% CI |

---

## Roadmap (Extension Points)

- [ ] NOAA ENSO index (`ingestion/climate.py`)
- [ ] USDA/ICO export stats (`ingestion/ico.py`)
- [ ] COT positioning data (`ingestion/cftc.py`)
- [ ] Robusta/Arabica spread features
- [ ] Frost probability index (Brazil June–August)
- [ ] Bayesian model averaging ensemble
- [ ] MLflow experiment tracking
