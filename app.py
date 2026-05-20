"""
app.py — Coffee Quant Interactive Research Tool
================================================
Run with:  streamlit run app.py
           streamlit run app.py --server.port 8501

Tabs:
  1. Configuration & Data    — instrument, dates, live data fetch
  2. Exogenous Variables     — fetch, preview, significance test
  3. Regime Detection (HMM)  — interactive regime chart
  4. Volatility (GARCH)      — conditional vol + forecast
  5. Directional Models      — AR / SARIMAX / Bayesian forecasts
  6. Walk-Forward Backtest   — per-fold metrics, calibration chart
  7. Model Comparison        — side-by-side ranking table
  8. Live Forecast           — current regime + probability output
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Coffee Quant",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS polish ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container{padding-top:1rem}
  [data-testid="stSidebar"] .stMarkdown h2{color:#6f4e37;font-size:1.1rem}
  .metric-card{background:#1e1e1e;border-radius:8px;padding:12px;margin:4px}
  .sig-pass{color:#4caf50;font-weight:bold}
  .sig-fail{color:#f44336}
  div[data-testid="stExpander"] summary{font-size:.9rem}
</style>
""", unsafe_allow_html=True)

from contracts.schemas import CoffeeVariety, DataFrequency


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR — all configuration
# ═══════════════════════════════════════════════════════════════════

def sidebar() -> dict:
    st.sidebar.title("☕ Coffee Quant")
    st.sidebar.markdown("---")

    # Instrument
    st.sidebar.header("Instrument")
    variety_name = st.sidebar.selectbox(
        "Variety", ["Arabica (KC=F)", "Robusta (RC=F)", "Custom ticker"],
        key="variety_select",
    )
    if variety_name == "Custom ticker":
        ticker = st.sidebar.text_input("Yahoo Finance ticker", value="KC=F")
        variety = CoffeeVariety.ARABICA
    elif "Arabica" in variety_name:
        ticker  = "KC=F"
        variety = CoffeeVariety.ARABICA
    else:
        ticker  = "RC=F"
        variety = CoffeeVariety.ROBUSTA

    # Dates
    st.sidebar.header("Date Ranges")
    from datetime import date, timedelta
    train_start = st.sidebar.date_input("Train start",  date(2019, 1, 1))
    train_end   = st.sidebar.date_input("Train end",    date(2024, 12, 31))
    test_start  = st.sidebar.date_input("Test start",   date(2025, 1, 1))
    test_end    = st.sidebar.date_input("Test end",     date.today())

    # Horizon
    st.sidebar.header("Forecast Horizon")
    horizon = st.sidebar.slider("Days ahead", 1, 21, 5)

    # Data source
    st.sidebar.header("Data Source")
    use_live  = st.sidebar.checkbox("Live market data (yfinance)", value=True)
    use_sim   = st.sidebar.checkbox("Fallback to simulator if live fails", value=True)

    # Models
    st.sidebar.header("Models to Run")
    run_hmm    = st.sidebar.checkbox("HMM Regime Detection", value=True)
    run_garch  = st.sidebar.checkbox("GARCH Volatility",     value=True)
    run_ar     = st.sidebar.checkbox("AR / ARMA Models",     value=True)
    run_bayes  = st.sidebar.checkbox("Bayesian Ridge",        value=True)
    run_sarimax= st.sidebar.checkbox("SARIMAX w/ Exog",      value=True)
    run_ens    = st.sidebar.checkbox("Ensemble",              value=True)

    # Exog settings
    st.sidebar.header("Exogenous Variables")
    fetch_exog     = st.sidebar.checkbox("Fetch exogenous data",   value=True)
    run_sig_test   = st.sidebar.checkbox("Run significance tests", value=True)
    include_nasa   = st.sidebar.checkbox("Include NASA POWER climate (slow)", value=False)
    fred_key       = st.sidebar.text_input("FRED API key (optional)", type="password")

    # HMM
    st.sidebar.header("HMM Settings")
    hmm_k = st.sidebar.slider("Number of regimes", 2, 5, 3)

    # Backtest
    st.sidebar.header("Backtest Settings")
    bt_train_window = st.sidebar.number_input("Train window (days)", 252, 1260, 756, step=63)
    bt_test_window  = st.sidebar.number_input("Test window (days)",   21,  252,  63, step=21)
    bt_folds        = st.sidebar.slider("Number of folds", 2, 10, 5)

    # Run button
    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button("🚀  Run Analysis", use_container_width=True, type="primary")

    return {
        "ticker": ticker, "variety": variety,
        "train_start": train_start, "train_end": train_end,
        "test_start": test_start,   "test_end": test_end,
        "horizon": horizon,
        "use_live": use_live, "use_sim": use_sim,
        "run_hmm": run_hmm, "run_garch": run_garch,
        "run_ar": run_ar, "run_bayes": run_bayes,
        "run_sarimax": run_sarimax, "run_ens": run_ens,
        "fetch_exog": fetch_exog, "run_sig_test": run_sig_test,
        "include_nasa": include_nasa, "fred_key": fred_key or None,
        "hmm_k": hmm_k,
        "bt_train": bt_train_window, "bt_test": bt_test_window, "bt_folds": bt_folds,
        "run_clicked": run_clicked,
    }


# ═══════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV from yfinance."""
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        raw = raw.rename(columns=str.lower)[["open","high","low","close","volume"]]
        raw.index = raw.index.tz_localize(None)
        return raw[raw["close"] > 0]
    except Exception:
        return pd.DataFrame()


def load_or_simulate(cfg: dict) -> pd.DataFrame:
    """Load live data or fall back to simulator."""
    start_str = str(cfg["train_start"])
    end_str   = str(cfg["test_end"])

    if cfg["use_live"]:
        with st.spinner(f"Downloading {cfg['ticker']} …"):
            df = load_price_data(cfg["ticker"], start_str, end_str)
        if not df.empty:
            st.success(f"✓ Live data: {len(df)} bars  "
                       f"({df['close'].iloc[0]:.1f} → {df['close'].iloc[-1]:.1f})")
            return df

    if cfg["use_sim"]:
        st.info("Live data unavailable — using calibrated GARCH simulator")
        from tests.evaluation.market_simulator import generate_full_dataset
        raw = generate_full_dataset(start_str, end_str, seed=42)
        return raw["arabica_futures"]

    st.error("No data available.")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════
# FEATURE BUILDING
# ═══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def build_full_features(
    price_df_json: str,
    ticker: str,
    variety_val: str,
    train_start: str,
    test_end: str,
    fetch_exog: bool,
    include_nasa: bool,
    fred_key: str | None,
) -> tuple[str, str]:
    """
    Cached feature building. Returns (feature_df_json, exog_df_json).
    All args must be JSON-serialisable for cache key.
    """
    from features.pipeline import FeaturePipeline
    from ingestion.exog_fetcher import ExogFetcher
    from datetime import date as dt

    price_df = pd.read_json(price_df_json)
    price_df.index = pd.to_datetime(price_df.index)

    # Companion ticker
    other_ticker = "RC=F" if "KC" in ticker else "KC=F"
    other_df = load_price_data(other_ticker, train_start, test_end)

    raw_inputs = {
        f"{'arabica' if 'KC' in ticker else 'robusta'}_futures": price_df,
        f"{'robusta' if 'KC' in ticker else 'arabica'}_futures": other_df,
    }

    # Exogenous
    exog_df = pd.DataFrame()
    if fetch_exog:
        fetcher = ExogFetcher(
            fred_api_key=fred_key,
            include_nasa=include_nasa,
            verbose=False,
        )
        start_d = dt.fromisoformat(train_start)
        end_d   = dt.fromisoformat(test_end)
        exog_df = fetcher.fetch_all(start_d, end_d, instrument=variety_val)
        if not exog_df.empty:
            for col in exog_df.columns:
                if col not in ["arabica_close", "robusta_close"]:
                    raw_inputs[col] = exog_df[[col]].rename(columns={col: "rate"})

    fp    = FeaturePipeline()
    frame = fp.run(raw_inputs, CoffeeVariety(variety_val), DataFrequency.DAILY)

    return frame.df.to_json(), exog_df.to_json() if not exog_df.empty else "{}"


# ═══════════════════════════════════════════════════════════════════
# PLOTTING HELPERS (plotly)
# ═══════════════════════════════════════════════════════════════════

def px():
    import plotly.express as px
    return px

def go():
    import plotly.graph_objects as go
    return go

def make_price_chart(price_df: pd.DataFrame, title: str = "Price"):
    import plotly.graph_objects as fig_go
    fig = fig_go.Figure()
    fig.add_trace(fig_go.Scatter(
        x=price_df.index, y=price_df["close"],
        mode="lines", name="Close", line=dict(color="#6f4e37", width=1.5)
    ))
    if all(c in price_df.columns for c in ["high","low"]):
        fig.add_trace(fig_go.Scatter(
            x=pd.concat([price_df.index.to_series(), price_df.index.to_series()[::-1]]),
            y=pd.concat([price_df["high"], price_df["low"][::-1]]),
            fill="toself", fillcolor="rgba(111,78,55,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Hi/Lo range",
        ))
    fig.update_layout(title=title, height=380, margin=dict(t=40,b=20,l=0,r=0),
                      xaxis_rangeslider_visible=False,
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"))
    return fig


def make_regime_chart(price_df: pd.DataFrame, regimes: pd.Series, title: str = "Regimes"):
    import plotly.graph_objects as fig_go

    COLORS = {"bull":"#2196F3","bear":"#F44336","volatile":"#FF9800",
               "neutral":"#9E9E9E","supply_stress":"#9C27B0","low_vol":"#4CAF50"}
    fig = fig_go.Figure()
    # Coloured background bands
    reg_aligned = regimes.reindex(price_df.index, method="ffill")
    changes = reg_aligned[reg_aligned != reg_aligned.shift()].index
    dates_list = list(changes) + [price_df.index[-1]]
    for i in range(len(dates_list)-1):
        regime = str(reg_aligned.loc[dates_list[i]])
        fig.add_vrect(
            x0=dates_list[i], x1=dates_list[i+1],
            fillcolor=COLORS.get(regime, "#555"), opacity=0.15,
            layer="below", line_width=0,
        )
    fig.add_trace(fig_go.Scatter(
        x=price_df.index, y=price_df["close"],
        mode="lines", line=dict(color="white", width=1.2), name="Close",
    ))
    fig.update_layout(title=title, height=420, margin=dict(t=40,b=20,l=0,r=0),
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"),
                      xaxis_rangeslider_visible=False)
    return fig


def make_vol_chart(price_df: pd.DataFrame, cond_vol: pd.Series):
    import plotly.graph_objects as fig_go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.05)
    fig.add_trace(fig_go.Scatter(x=price_df.index, y=price_df["close"],
                                  mode="lines", line=dict(color="#6f4e37", width=1.2),
                                  name="Close"), row=1, col=1)
    fig.add_trace(fig_go.Scatter(x=cond_vol.index, y=cond_vol.values * 100,
                                  mode="lines", line=dict(color="#FF9800", width=1.4),
                                  fill="tozeroy", fillcolor="rgba(255,152,0,0.15)",
                                  name="Cond. Vol %"), row=2, col=1)
    fig.update_layout(height=420, margin=dict(t=20,b=20,l=0,r=0),
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"), showlegend=False)
    return fig


def make_calibration_chart(cal_df: pd.DataFrame):
    import plotly.graph_objects as fig_go
    fig = fig_go.Figure()
    fig.add_trace(fig_go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                  line=dict(color="grey",dash="dash"), name="Perfect"))
    fig.add_trace(fig_go.Scatter(
        x=cal_df["mean_predicted"], y=cal_df["fraction_positive"],
        mode="markers+lines", marker=dict(size=cal_df["count"].clip(5,30),
                                          color="#2196F3", opacity=0.8),
        name="Model",
    ))
    fig.update_layout(title="Calibration (Reliability Diagram)", height=350,
                      xaxis_title="Predicted Prob", yaxis_title="Actual Freq",
                      xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1]),
                      margin=dict(t=40,b=30,l=0,r=0),
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"))
    return fig


def make_fan_chart(history: pd.Series, fc_dict: dict, title="Forecast Fan"):
    import plotly.graph_objects as fig_go
    fig = fig_go.Figure()
    hist_tail = history.iloc[-60:]
    fig.add_trace(fig_go.Scatter(x=hist_tail.index, y=hist_tail.values,
                                  line=dict(color="white",width=1.5), name="History"))
    if fc_dict:
        last = float(history.iloc[-1])
        horizon = fc_dict.get("horizon", 5)
        fcast_idx = pd.bdate_range(history.index[-1], periods=horizon+1)[1:]
        mean_r = fc_dict.get("mean_return", 0)
        q10    = fc_dict.get("q10", mean_r)
        q90    = fc_dict.get("q90", mean_r)
        fc_prices   = [last * np.exp(mean_r * (i+1)) for i in range(horizon)]
        fc_prices10 = [last * np.exp(q10   * (i+1)) for i in range(horizon)]
        fc_prices90 = [last * np.exp(q90   * (i+1)) for i in range(horizon)]
        fig.add_trace(fig_go.Scatter(
            x=list(fcast_idx)+list(fcast_idx[::-1]),
            y=fc_prices90 + fc_prices10[::-1],
            fill="toself", fillcolor="rgba(33,150,243,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="10-90% CI",
        ))
        fig.add_trace(fig_go.Scatter(x=fcast_idx, y=fc_prices,
                                      line=dict(color="#2196F3",dash="dash",width=2),
                                      name="Forecast"))
    fig.update_layout(title=title, height=380, margin=dict(t=40,b=20,l=0,r=0),
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"))
    return fig


# ═══════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def run_hmm(frame, cfg):
    from models.hmm_model import HMMRegimeDetector
    cutoff = pd.Timestamp(cfg["test_start"])
    train_df = frame.df[frame.df.index < cutoff]
    train_fr = _sub_frame(frame, train_df)
    FEATS = ["log_return_1d","realised_vol_21d","log_return_5d","price_z_63d"]
    det = HMMRegimeDetector(n_regimes=cfg["hmm_k"], variety=cfg["variety"],
                            feature_cols=FEATS, n_iter=300, random_state=0)
    det.fit(train_fr)
    snaps = det.predict(frame)
    regimes = pd.Series({pd.Timestamp(s.timestamp): s.regime.value for s in snaps})
    confs   = pd.Series({pd.Timestamp(s.timestamp): s.probability for s in snaps})
    stats   = det.regime_statistics(train_fr)
    trans   = det.transition_matrix()
    latest  = snaps[-1] if snaps else None
    return {"regimes": regimes, "confs": confs, "stats": stats,
            "trans": trans, "latest": latest, "detector": det}


def run_garch(frame, cfg):
    from models.garch_model import GARCHVolatilityModel
    ret_col = "log_return_1d"
    if ret_col not in frame.df.columns:
        return None
    cutoff   = pd.Timestamp(cfg["test_start"])
    all_rets = frame.df[ret_col].dropna()
    tr_rets  = all_rets[all_rets.index < cutoff]
    m = GARCHVolatilityModel(p=1, q=1, model_type="GARCH", dist="t",
                              variety=cfg["variety"])
    m.fit(tr_rets)
    cond_vol = m.conditional_volatility()
    fc = m.forecast(tr_rets.iloc[-252:], horizon_days=10)
    return {"model": m, "cond_vol": cond_vol, "forecast": fc,
            "train_rets": tr_rets, "test_rets": all_rets[all_rets.index >= cutoff]}


def run_ar_models(frame, cfg):
    from statsmodels.tsa.ar_model import AutoReg
    from scipy import stats as sp

    TARGET = "target_log_return_5d"
    cutoff = pd.Timestamp(cfg["test_start"])
    tr_df  = frame.df[frame.df.index < cutoff].dropna(subset=[TARGET])
    te_df  = frame.df[frame.df.index >= cutoff].dropna(subset=[TARGET])
    if len(te_df) == 0:
        return {}

    full_y   = pd.concat([tr_df[TARGET], te_df[TARGET]]).sort_index()
    full_arr = full_y.values
    full_map = {t: i for i, t in enumerate(full_y.index)}
    h        = cfg["horizon"]
    results  = {}

    for lags, label in [(1,"AR(1)"), (2,"AR(2)"), (5,"AR(5)")]:
        try:
            m      = AutoReg(tr_df[TARGET].values, lags=lags, old_names=False).fit()
            coef   = m.params
            c      = float(coef[0])
            phis   = [float(coef[i+1]) for i in range(lags)]
            h_std  = max(float(m.resid.std()) * np.sqrt(h), 1e-8)
            prob_ups, actuals = [], []
            for dt in te_df.index[::2]:
                pos = full_map.get(dt)
                if pos is None or pos < lags: continue
                fc   = c + sum(phis[i] * full_arr[pos-1-i] for i in range(lags))
                pu   = float(1 - sp.norm.cdf(0, loc=fc*h, scale=h_std))
                act  = float(te_df.loc[dt, TARGET])
                prob_ups.append(pu); actuals.append(act)
            results[label] = {
                "model": m, "prob_ups": prob_ups, "actuals": actuals,
                "aic": round(m.aic,2), "bic": round(m.bic,2),
                **_score(prob_ups, actuals, label),
            }
        except Exception as e:
            results[label] = {"error": str(e)}
    return results


def run_bayesian(frame, cfg):
    from models.bayesian_model import BayesianForecaster
    from scipy import stats as sp

    TARGET = "target_log_return_5d"
    cutoff = pd.Timestamp(cfg["test_start"])
    tr_df  = frame.df[frame.df.index < cutoff].dropna(subset=[TARGET])
    te_df  = frame.df[frame.df.index >= cutoff].dropna(subset=[TARGET])
    if len(te_df) == 0:
        return None

    from contracts.schemas import FeatureFrame
    tr_fr = _sub_frame(frame, tr_df)
    m = BayesianForecaster(variety=cfg["variety"])
    m.fit(tr_fr, TARGET)

    avail = [c for c in m._fitted_cols if c in te_df.columns]
    prob_ups, actuals = [], []
    for dt in te_df.index[::2]:
        try:
            row  = te_df[avail].loc[:dt].ffill().iloc[-1:].fillna(0)
            X    = m._scaler.transform(row.values)
            mu, sigma = m._model.predict(X, return_std=True)
            h_std = max(float(sigma[0]) * np.sqrt(cfg["horizon"]), 1e-8)
            pu   = float(1 - sp.norm.cdf(0, loc=float(mu[0]), scale=h_std))
            act  = float(te_df.loc[dt, TARGET])
            prob_ups.append(pu); actuals.append(act)
        except Exception:
            continue

    return {
        "model": m, "prob_ups": prob_ups, "actuals": actuals,
        "importance": m.feature_importance().head(15),
        **_score(prob_ups, actuals, "BayesianRidge"),
    }


def run_sarimax_exog(frame, cfg, sig_cols: list[str]):
    """SARIMAX with auto-selected exogenous variables."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from scipy import stats as sp

    TARGET = "target_log_return_5d"
    cutoff = pd.Timestamp(cfg["test_start"])
    tr_df  = frame.df[frame.df.index < cutoff].dropna(subset=[TARGET])
    te_df  = frame.df[frame.df.index >= cutoff].dropna(subset=[TARGET])
    if len(te_df) == 0:
        return None

    avail_exog = [c for c in sig_cols if c in tr_df.columns and c != TARGET]
    target_s   = tr_df[TARGET]
    exog_tr    = tr_df[avail_exog].ffill().fillna(0) if avail_exog else None

    try:
        res = SARIMAX(endog=target_s, exog=exog_tr, order=(1,0,1),
                      enforce_stationarity=False).fit(disp=False, maxiter=200)
    except Exception as e:
        return {"error": str(e)}

    h_std = max(float(res.resid.std()) * np.sqrt(cfg["horizon"]), 1e-8)
    full_y   = pd.concat([tr_df[TARGET], te_df[TARGET]]).sort_index()
    full_arr = full_y.values
    full_map = {t: i for i, t in enumerate(full_y.index)}

    ar_key = "ar.L1" if "ar.L1" in res.params.index else None
    ma_key = "ma.L1" if "ma.L1" in res.params.index else None
    c_val  = float(res.params.get("const", 0))
    ar_phi = float(res.params.get(ar_key, 0)) if ar_key else 0.0
    ma_phi = float(res.params.get(ma_key, 0)) if ma_key else 0.0

    prob_ups, actuals = [], []
    resid_arr = res.resid.values
    resid_map = {t: i for i, t in enumerate(tr_df.index)}

    for dt in te_df.index[::2]:
        pos = full_map.get(dt)
        if pos is None or pos < 2: continue
        fc   = c_val + ar_phi * full_arr[pos-1]
        pu   = float(1 - sp.norm.cdf(0, loc=fc * cfg["horizon"], scale=h_std))
        act  = float(te_df.loc[dt, TARGET])
        prob_ups.append(pu); actuals.append(act)

    return {
        "model": res, "prob_ups": prob_ups, "actuals": actuals,
        "exog_used": avail_exog,
        "aic": round(res.aic, 2), "bic": round(res.bic, 2),
        **_score(prob_ups, actuals, f"SARIMAX({len(avail_exog)} exog)"),
    }


def run_backtest(frame, cfg, model_type: str = "ar1"):
    """Walk-forward backtest for the chosen model type."""
    from statsmodels.tsa.ar_model import AutoReg
    from scipy import stats as sp

    TARGET  = "target_log_return_5d"
    h       = cfg["horizon"]
    df      = frame.df.dropna(subset=[TARGET]).copy()
    n       = len(df)
    tw      = min(cfg["bt_train"], n - 30)
    tsw     = cfg["bt_test"]
    n_folds = cfg["bt_folds"]

    folds_data = []
    start = tw
    for fold in range(n_folds):
        if start + tsw > n:
            break
        train_idx = range(max(0, start - tw), start)
        test_idx  = range(start, min(start + tsw, n))
        tr = df.iloc[train_idx]; te = df.iloc[test_idx]
        try:
            m     = AutoReg(tr[TARGET].values, lags=1, old_names=False).fit()
            c     = float(m.params[0]); phi = float(m.params[1])
            h_std = max(float(m.resid.std()) * np.sqrt(h), 1e-8)
            full_y = pd.concat([tr[TARGET], te[TARGET]])
            fa = full_y.values
            prob_ups, actuals = [], []
            for i, (dt, row) in enumerate(te.iterrows()):
                pos = len(tr) + i
                if pos < 1: continue
                fc   = c + phi * fa[pos-1]
                pu   = float(1 - sp.norm.cdf(0, loc=fc*h, scale=h_std))
                act  = float(row[TARGET])
                prob_ups.append(pu); actuals.append(act)
            sc = _score(prob_ups, actuals, f"fold_{fold}")
            folds_data.append({
                "fold": fold,
                "train_start": tr.index[0].date(),
                "train_end":   tr.index[-1].date(),
                "test_start":  te.index[0].date(),
                "test_end":    te.index[-1].date(),
                "n": len(prob_ups),
                "da": round(sc["directional_accuracy"], 4),
                "brier": round(sc["brier_score"], 4),
                "sharpe": round(sc.get("signal_sharpe_ann") or 0, 3),
            })
        except Exception:
            pass
        start += tsw

    return pd.DataFrame(folds_data)


# ── Small helpers ─────────────────────────────────────────────────

def _sub_frame(frame, sub_df):
    from contracts.schemas import FeatureFrame
    return FeatureFrame(variety=frame.variety, frequency=frame.frequency,
                        feature_names=frame.feature_names, df=sub_df)


def _score(prob_ups, actuals, label):
    if not prob_ups:
        return {"directional_accuracy": None, "brier_score": None,
                "brier_skill_score": None, "signal_sharpe_ann": None,
                "permutation_pvalue": None, "n_forecasts": 0}
    p, a = np.array(prob_ups), np.array(actuals)
    da   = float(np.mean((p>0.5)==(a>0)))
    bs   = float(np.mean((p-(a>0).astype(float))**2))
    bss  = float(1 - bs/0.25)
    sig  = np.where(p>0.55, 1., np.where(p<0.45, -1., 0.))
    pnl  = sig * a
    sh   = float(pnl.mean()/pnl.std()*np.sqrt(252)) if pnl.std()>0 else None
    rng  = np.random.default_rng(0)
    null = [float(np.mean((rng.permutation(p)>0.5)==(a>0))) for _ in range(500)]
    pval = float(np.mean(np.array(null) >= da))

    # Calibration
    bins = np.linspace(0,1,6)
    cal_rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m2 = (p>=lo)&(p<hi)
        if m2.sum()>0:
            cal_rows.append({"bin_centre": (lo+hi)/2,
                             "mean_predicted": float(p[m2].mean()),
                             "fraction_positive": float((a[m2]>0).mean()),
                             "count": int(m2.sum())})

    return {
        "directional_accuracy": round(da, 4),
        "brier_score":          round(bs, 4),
        "brier_skill_score":    round(bss, 4),
        "signal_sharpe_ann":    round(sh, 3) if sh else None,
        "permutation_pvalue":   round(pval, 4),
        "significant_5pct":     bool(pval < 0.05),
        "n_forecasts":          len(p),
        "n_long":               int((p>0.55).sum()),
        "n_short":              int((p<0.45).sum()),
        "calibration_df":       pd.DataFrame(cal_rows),
    }


# ═══════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ═══════════════════════════════════════════════════════════════════

def tab_data(price_df: pd.DataFrame, cfg: dict):
    st.subheader(f"Price Data — {cfg['ticker']}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Bars", f"{len(price_df):,}")
    col2.metric("Start Price", f"{price_df['close'].iloc[0]:.2f}")
    col3.metric("End Price",   f"{price_df['close'].iloc[-1]:.2f}")
    chg = (price_df["close"].iloc[-1] / price_df["close"].iloc[0] - 1) * 100
    col4.metric("Total Return", f"{chg:+.1f}%")
    st.plotly_chart(make_price_chart(price_df, f"{cfg['ticker']} Continuous Front-Month"),
                    use_container_width=True)
    with st.expander("Raw data preview"):
        st.dataframe(price_df.tail(20).style.format("{:.2f}"), use_container_width=True)


def tab_exog(exog_df: pd.DataFrame, frame_df: pd.DataFrame, cfg: dict):
    if exog_df.empty:
        st.info("No exogenous data fetched. Enable 'Fetch exogenous data' in sidebar.")
        return
    st.subheader(f"Exogenous Variables  ({exog_df.shape[1]} series, {len(exog_df)} bars)")

    # Preview heatmap
    target = "target_log_return_5d"
    if target in frame_df.columns:
        tgt = frame_df[target].dropna()
        exog_aligned = exog_df.reindex(tgt.index).ffill().dropna(how="all")
        corrs = exog_aligned.corrwith(tgt).dropna().sort_values()
        import plotly.express as pxp
        fig = pxp.bar(corrs, orientation="h", color=corrs,
                      color_continuous_scale="RdBu", range_color=(-0.3,0.3),
                      title="Correlation with 5-day target return",
                      labels={"value":"Corr", "index":"Variable"})
        fig.update_layout(height=max(300, len(corrs)*18),
                          margin=dict(t=40,b=20,l=0,r=0),
                          plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                          font=dict(color="#fafafa"))
        st.plotly_chart(fig, use_container_width=True)

    if cfg["run_sig_test"] and target in frame_df.columns:
        st.subheader("Significance Testing")
        with st.spinner("Running Granger causality + LASSO + RF …"):
            from features.significance import SignificanceTester
            tester = SignificanceTester(max_lag=10, granger_alpha=0.10)
            exog_clean = exog_df.reindex(tgt.index).ffill().dropna(how="all")
            report = tester.full_run(exog_clean, tgt)

        passed = report[report["passes"]]
        failed = report[~report["passes"]]
        st.success(f"✓ {len(passed)} variables pass significance  |  "
                   f"✗ {len(failed)} fail")

        st.markdown("**Selected variables (passed ≥ 1 test):**")
        disp = passed[["granger_pval","max_corr","best_lag","lasso_selected",
                        "rf_importance","composite_score","reason"]].round(4)
        st.dataframe(disp, use_container_width=True)

        with st.expander("All variables"):
            st.dataframe(report[["granger_pval","max_corr","best_lag",
                                  "lasso_selected","rf_importance","passes",
                                  "reason"]].round(4), use_container_width=True)

        st.session_state["sig_cols"] = report[report["passes"]].index.tolist()
    else:
        with st.expander("Raw exogenous data (last 10 rows)"):
            st.dataframe(exog_df.tail(10).style.format("{:.4f}"), use_container_width=True)


def tab_hmm(price_df: pd.DataFrame, hmm_result: dict | None):
    if hmm_result is None:
        st.info("HMM not run. Enable it in the sidebar.")
        return
    st.subheader("HMM Regime Detection")
    regimes = hmm_result["regimes"]
    st.plotly_chart(make_regime_chart(price_df, regimes), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Regime Statistics (train)**")
        st.dataframe(hmm_result["stats"].round(4), use_container_width=True)
    with col2:
        st.markdown("**Transition Matrix**")
        st.dataframe(hmm_result["trans"].round(4), use_container_width=True)

    latest = hmm_result.get("latest")
    if latest:
        st.markdown("---")
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Current Regime", latest.regime.value.upper())
        rc2.metric("Confidence",     f"{latest.probability:.2%}")
        rc3.metric("Transition Prob",f"{latest.transition_probability:.2%}")


def tab_garch(price_df: pd.DataFrame, garch_result: dict | None):
    if garch_result is None:
        st.info("GARCH not run.")
        return
    st.subheader("GARCH Volatility")
    cv = garch_result["cond_vol"]
    st.plotly_chart(make_vol_chart(price_df.loc[cv.index[0]:], cv),
                    use_container_width=True)
    fc = garch_result["forecast"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Vol (ann)", f"{fc.current_annualized_vol:.1%}")
    col2.metric("10-day Forecast",   f"{fc.forecast_annualized_vol:.1%}")
    col3.metric("Vol Regime",         fc.vol_regime.upper())

    m = garch_result["model"]
    with st.expander("Model diagnostics"):
        st.code(m._result.summary().as_text())


def tab_models(price_df: pd.DataFrame, ar_results: dict, bayes_result,
               sarimax_result, cfg: dict):
    st.subheader("Directional Forecast Models")
    history = price_df["close"]

    # Pick best AR for fan chart
    best_ar = None
    if ar_results:
        best_key = max(ar_results, key=lambda k: ar_results[k].get("directional_accuracy") or 0)
        best_ar = ar_results[best_key]

    col_ar, col_bay = st.columns(2)
    with col_ar:
        st.markdown("**AR/ARMA Models**")
        if ar_results:
            for label, r in ar_results.items():
                if "error" in r:
                    st.error(f"{label}: {r['error']}")
                    continue
                _model_metric_row(label, r)
        else:
            st.info("No AR results.")

    with col_bay:
        st.markdown("**Bayesian Ridge**")
        if bayes_result and "error" not in bayes_result:
            _model_metric_row("BayesianRidge", bayes_result)
            with st.expander("Top feature importances"):
                imp = bayes_result.get("importance")
                if imp is not None:
                    st.bar_chart(imp)
        else:
            st.info("No Bayesian results.")

    if sarimax_result and "error" not in sarimax_result:
        st.markdown("**SARIMAX with Exogenous Variables**")
        _model_metric_row(f"SARIMAX ({len(sarimax_result.get('exog_used',[]))} vars)", sarimax_result)
        exog_used = sarimax_result.get("exog_used", [])
        if exog_used:
            st.caption(f"Exog used: {', '.join(exog_used)}")

    # Fan chart for best model
    if best_ar:
        fc_dict = {
            "mean_return": np.mean(best_ar.get("prob_ups", [0.5])) - 0.5,
            "q10": -0.02, "q90": 0.02, "horizon": cfg["horizon"],
        }
        st.plotly_chart(make_fan_chart(history, fc_dict, "Latest Directional Forecast"),
                        use_container_width=True)

    # Calibration for best model
    best = best_ar or bayes_result
    if best:
        cal_df = best.get("calibration_df")
        if cal_df is not None and not cal_df.empty:
            st.plotly_chart(make_calibration_chart(cal_df), use_container_width=True)


def tab_backtest(bt_df: pd.DataFrame):
    st.subheader("Walk-Forward Backtest Results")
    if bt_df.empty:
        st.info("No backtest results.")
        return
    st.dataframe(bt_df.style.format({
        "da": "{:.4f}", "brier": "{:.4f}", "sharpe": "{:.3f}",
    }).background_gradient(subset=["da"], cmap="RdYlGn"),
    use_container_width=True)

    import plotly.graph_objects as fig_go
    fig = fig_go.Figure()
    fig.add_trace(fig_go.Bar(x=bt_df["fold"].astype(str), y=bt_df["da"],
                              marker_color=["#4caf50" if v>0.5 else "#f44336"
                                            for v in bt_df["da"]], name="DA"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey")
    fig.update_layout(title="Directional Accuracy by Fold", height=300,
                      yaxis=dict(range=[0.3,0.8]),
                      plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                      font=dict(color="#fafafa"), margin=dict(t=40,b=20,l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)


def tab_comparison(all_model_results: dict):
    st.subheader("Model Comparison")
    rows = []
    for label, r in all_model_results.items():
        if not r or "error" in r or not r.get("n_forecasts"):
            continue
        rows.append({
            "model":     label,
            "N":         r.get("n_forecasts", 0),
            "DA":        r.get("directional_accuracy"),
            "Brier":     r.get("brier_score"),
            "BSS":       r.get("brier_skill_score"),
            "Sharpe":    r.get("signal_sharpe_ann"),
            "p-val":     r.get("permutation_pvalue"),
            "Sig 5%":    "✓" if r.get("significant_5pct") else "",
        })
    if not rows:
        st.info("Run the analysis first.")
        return
    df = pd.DataFrame(rows).sort_values("DA", ascending=False)
    st.dataframe(df.style.format({
        "DA": "{:.4f}", "Brier": "{:.4f}", "BSS": "{:.4f}",
        "Sharpe": "{:.3f}", "p-val": "{:.4f}",
    }).background_gradient(subset=["DA"], cmap="RdYlGn"),
    use_container_width=True)
    st.markdown("> **Random baseline**: DA=0.5, Brier=0.25, BSS=0, Sharpe=0")


def tab_live_forecast(hmm_result, ar_results, bayes_result, garch_result, cfg):
    st.subheader("📡 Live Forecast Summary")
    if not any([hmm_result, ar_results, bayes_result, garch_result]):
        st.info("Run the analysis first.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Current Market State")
        if hmm_result and hmm_result.get("latest"):
            lt = hmm_result["latest"]
            reg_color = {"bull":"🔵","bear":"🔴","volatile":"🟠","neutral":"⚪"}.get(
                lt.regime.value, "⚪")
            st.metric("Regime", f"{reg_color} {lt.regime.value.upper()}")
            st.metric("Regime Confidence", f"{lt.probability:.1%}")
            st.metric("Transition Risk",   f"{lt.transition_probability:.1%}")
        if garch_result:
            fc = garch_result["forecast"]
            vol_emoji = {"low":"🟢","medium":"🟡","high":"🟠","extreme":"🔴"}.get(
                fc.vol_regime, "⚪")
            st.metric("Current Vol",     f"{vol_emoji} {fc.current_annualized_vol:.1%}")
            st.metric("10d Forecast Vol", f"{fc.forecast_annualized_vol:.1%}")

    with col2:
        st.markdown("### Directional Probability")
        all_pup = {}
        if ar_results:
            for label, r in ar_results.items():
                if r.get("prob_ups"):
                    all_pup[label] = float(np.mean(r["prob_ups"][-5:]))
        if bayes_result and bayes_result.get("prob_ups"):
            all_pup["Bayesian"] = float(np.mean(bayes_result["prob_ups"][-5:]))

        for model, pup in all_pup.items():
            direction = "LONG  📈" if pup > 0.55 else ("SHORT 📉" if pup < 0.45 else "FLAT  ➡️")
            col_a, col_b, col_c = st.columns([2, 1, 2])
            col_a.write(f"**{model}**")
            col_b.progress(int(pup * 100), text=f"{pup:.0%}")
            col_c.write(direction)

    st.markdown("---")
    st.caption(
        f"Forecast horizon: {cfg['horizon']} trading days  |  "
        f"Test period: {cfg['test_start']} → {cfg['test_end']}  |  "
        "⚠️ Not financial advice."
    )


def _model_metric_row(label, r):
    da  = r.get("directional_accuracy")
    bs  = r.get("brier_score")
    sh  = r.get("signal_sharpe_ann")
    pv  = r.get("permutation_pvalue")
    n   = r.get("n_forecasts", 0)
    sig = "✓" if r.get("significant_5pct") else ""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{label} DA",    f"{da:.4f}" if da else "n/a")
    c2.metric("Brier",           f"{bs:.4f}" if bs else "n/a")
    c3.metric("Sharpe",          f"{sh:.2f}" if sh else "n/a")
    c4.metric(f"p={pv:.3f} {sig}" if pv else "p=n/a", f"n={n}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    cfg = sidebar()

    st.title("☕ Coffee Quant — Interactive Research Tool")
    st.caption(
        "Modular regime-aware probabilistic forecasting for "
        "Arabica (KC=F) and Robusta (RC=F) coffee futures."
    )

    tabs = st.tabs([
        "📊 Data",
        "🌍 Exogenous Variables",
        "🎯 Regime Detection",
        "📈 Volatility",
        "🔮 Directional Models",
        "🔄 Walk-Forward Backtest",
        "📋 Model Comparison",
        "📡 Live Forecast",
    ])

    # ── Load data ────────────────────────────────────────────────────────
    if "price_df" not in st.session_state:
        st.session_state["price_df"] = pd.DataFrame()
    if "frame_df" not in st.session_state:
        st.session_state["frame_df"] = None
    if "exog_df" not in st.session_state:
        st.session_state["exog_df"] = pd.DataFrame()

    if cfg["run_clicked"]:
        # ── Price data ──────────────────────────────────────────────────
        price_df = load_or_simulate(cfg)
        st.session_state["price_df"] = price_df
        st.session_state["cfg"] = cfg

        if price_df.empty:
            st.error("Could not load price data.")
            return

        # ── Features ────────────────────────────────────────────────────
        with st.spinner("Building feature matrix …"):
            from features.pipeline import FeaturePipeline
            from ingestion.exog_fetcher import ExogFetcher
            from datetime import date as dt_cls

            fetcher = ExogFetcher(
                fred_api_key=cfg.get("fred_key"),
                include_nasa=cfg["include_nasa"],
                verbose=False,
            )
            start_d = cfg["train_start"]
            end_d   = cfg["test_end"]
            exog_df = pd.DataFrame()
            raw_inputs = {
                "arabica_futures" if cfg["variety"] == CoffeeVariety.ARABICA
                else "robusta_futures": price_df,
            }

            if cfg["fetch_exog"]:
                with st.spinner("Fetching exogenous data …"):
                    exog_df = fetcher.fetch_all(start_d, end_d,
                                                instrument=cfg["variety"].value)
                st.session_state["exog_df"] = exog_df

            other = load_price_data(
                "RC=F" if "KC" in cfg["ticker"] else "KC=F",
                str(start_d), str(end_d),
            )
            other_key = ("robusta_futures" if cfg["variety"] == CoffeeVariety.ARABICA
                         else "arabica_futures")
            if not other.empty:
                raw_inputs[other_key] = other

            fp = FeaturePipeline()
            frame = fp.run(raw_inputs, cfg["variety"], DataFrequency.DAILY)
            st.session_state["frame"] = frame

        # ── Run models ──────────────────────────────────────────────────
        model_results = {}

        if cfg["run_hmm"]:
            with st.spinner("Running HMM …"):
                try:
                    st.session_state["hmm_result"] = run_hmm(frame, cfg)
                except Exception as e:
                    st.warning(f"HMM failed: {e}")
                    st.session_state["hmm_result"] = None

        if cfg["run_garch"]:
            with st.spinner("Running GARCH …"):
                try:
                    st.session_state["garch_result"] = run_garch(frame, cfg)
                except Exception as e:
                    st.warning(f"GARCH failed: {e}")
                    st.session_state["garch_result"] = None

        if cfg["run_ar"]:
            with st.spinner("Running AR models …"):
                try:
                    ar_r = run_ar_models(frame, cfg)
                    st.session_state["ar_results"] = ar_r
                    model_results.update(ar_r)
                except Exception as e:
                    st.warning(f"AR failed: {e}")
                    st.session_state["ar_results"] = {}

        if cfg["run_bayes"]:
            with st.spinner("Running Bayesian Ridge …"):
                try:
                    br = run_bayesian(frame, cfg)
                    st.session_state["bayes_result"] = br
                    if br:
                        model_results["BayesianRidge"] = br
                except Exception as e:
                    st.warning(f"Bayesian failed: {e}")
                    st.session_state["bayes_result"] = None

        if cfg["run_sarimax"]:
            sig_cols = st.session_state.get("sig_cols", [])
            if sig_cols:
                with st.spinner(f"Running SARIMAX ({len(sig_cols)} exog vars) …"):
                    try:
                        sr = run_sarimax_exog(frame, cfg, sig_cols)
                        st.session_state["sarimax_result"] = sr
                        if sr:
                            model_results[f"SARIMAX_{len(sig_cols)}exog"] = sr
                    except Exception as e:
                        st.warning(f"SARIMAX failed: {e}")
                        st.session_state["sarimax_result"] = None
            else:
                st.info("SARIMAX: run Exogenous Variables tab first to select variables")

        if cfg["run_ar"]:
            with st.spinner("Running walk-forward backtest …"):
                try:
                    bt_df = run_backtest(frame, cfg)
                    st.session_state["bt_df"] = bt_df
                except Exception as e:
                    st.warning(f"Backtest failed: {e}")
                    st.session_state["bt_df"] = pd.DataFrame()

        st.session_state["model_results"] = model_results
        st.success("✅  Analysis complete!")

    # ── Render tabs ─────────────────────────────────────────────────────
    price_df = st.session_state.get("price_df", pd.DataFrame())
    frame    = st.session_state.get("frame")
    exog_df  = st.session_state.get("exog_df", pd.DataFrame())
    cfg_run  = st.session_state.get("cfg", cfg)

    with tabs[0]:
        if not price_df.empty:
            tab_data(price_df, cfg_run)
        else:
            st.info("Click **Run Analysis** to load data.")

    with tabs[1]:
        if frame is not None:
            tab_exog(exog_df, frame.df, cfg_run)
        else:
            st.info("Run analysis first.")

    with tabs[2]:
        tab_hmm(price_df if not price_df.empty else pd.DataFrame({"close":[0]}),
                st.session_state.get("hmm_result"))

    with tabs[3]:
        tab_garch(price_df if not price_df.empty else pd.DataFrame({"close":[0]}),
                  st.session_state.get("garch_result"))

    with tabs[4]:
        tab_models(
            price_df if not price_df.empty else pd.DataFrame({"close":[0]}),
            st.session_state.get("ar_results", {}),
            st.session_state.get("bayes_result"),
            st.session_state.get("sarimax_result"),
            cfg_run,
        )

    with tabs[5]:
        tab_backtest(st.session_state.get("bt_df", pd.DataFrame()))

    with tabs[6]:
        tab_comparison(st.session_state.get("model_results", {}))

    with tabs[7]:
        tab_live_forecast(
            st.session_state.get("hmm_result"),
            st.session_state.get("ar_results", {}),
            st.session_state.get("bayes_result"),
            st.session_state.get("garch_result"),
            cfg_run,
        )


if __name__ == "__main__":
    main()
