"""
app.py — Coffee Quant Interactive Research Tool v3
===================================================
Run: streamlit run app.py   (from coffee_quant/ directory)
     PYTHONPATH=. streamlit run app.py  (if module errors)
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Coffee Quant ☕",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{padding-top:1rem}
[data-testid="stSidebar"] .stMarkdown h2{color:#6f4e37}
.stMetric label{font-size:.8rem}
</style>
""", unsafe_allow_html=True)

from contracts.schemas import CoffeeVariety, DataFrequency


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════

def sidebar() -> dict:
    st.sidebar.title("☕ Coffee Quant")
    st.sidebar.caption("v3 — Regime-Aware Probabilistic Forecasting")
    st.sidebar.markdown("---")

    # Instrument
    st.sidebar.header("📌 Instrument")
    variety_label = st.sidebar.selectbox(
        "Variety", ["Arabica (KC=F)", "Robusta (RC=F)", "Custom ticker"])
    if variety_label == "Custom ticker":
        ticker  = st.sidebar.text_input("Yahoo Finance ticker", value="KC=F")
        variety = CoffeeVariety.ARABICA
    elif "Arabica" in variety_label:
        ticker, variety = "KC=F", CoffeeVariety.ARABICA
    else:
        ticker, variety = "RC=F", CoffeeVariety.ROBUSTA

    # Dates
    st.sidebar.header("📅 Date Ranges")
    from datetime import date
    train_start = st.sidebar.date_input("Train start", date(2019, 1, 1))
    train_end   = st.sidebar.date_input("Train end",   date(2024, 12, 31))
    test_start  = st.sidebar.date_input("Test start",  date(2025, 1, 1))
    test_end    = st.sidebar.date_input("Test end",    date.today())

    # Forecasting
    st.sidebar.header("🔮 Forecasting")
    horizon      = st.sidebar.slider("Forecast horizon (days)", 1, 30, 5)
    n_fc_samples = st.sidebar.slider("Simulation paths (fan chart)", 100, 2000, 500, step=100)

    # Data
    st.sidebar.header("💾 Data Source")
    use_live = st.sidebar.checkbox("Live data (yfinance)", value=True)
    use_sim  = st.sidebar.checkbox("Simulator fallback", value=True)

    # Models
    st.sidebar.header("⚙️ Models")
    run_hmm    = st.sidebar.checkbox("HMM Regime Detection", value=True)
    run_garch  = st.sidebar.checkbox("GARCH Volatility",     value=True)
    run_ar     = st.sidebar.checkbox("AR/ARMA Models",        value=True)
    run_bayes  = st.sidebar.checkbox("Bayesian Ridge",        value=True)
    run_sarimax= st.sidebar.checkbox("SARIMAX + Exog",        value=True)
    run_ens    = st.sidebar.checkbox("Ensemble",              value=True)

    # Exogenous
    st.sidebar.header("🌍 Exogenous Variables")
    fetch_exog   = st.sidebar.checkbox("Fetch exog data",        value=True)
    run_sig      = st.sidebar.checkbox("Auto significance test",  value=True)
    include_nasa = st.sidebar.checkbox("NASA POWER climate (slow)", value=False)
    fred_key     = st.sidebar.text_input("FRED API key (optional)", type="password")
    granger_pval = st.sidebar.slider("Granger p-value threshold", 0.05, 0.20, 0.10)
    min_corr     = st.sidebar.slider("Min |correlation| threshold", 0.02, 0.15, 0.05)

    # HMM
    st.sidebar.header("🎯 HMM Settings")
    hmm_k = st.sidebar.slider("Number of regimes", 2, 5, 3)

    # Backtest
    st.sidebar.header("🔄 Backtest")
    bt_train = st.sidebar.number_input("Train window (days)", 252, 1260, 756, step=63)
    bt_test  = st.sidebar.number_input("Test window (days)",   21,  252,  63, step=21)
    bt_folds = st.sidebar.slider("Folds", 2, 10, 5)

    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button("🚀 Run Analysis", use_container_width=True, type="primary")

    return dict(
        ticker=ticker, variety=variety,
        train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end,
        horizon=horizon, n_fc_samples=n_fc_samples,
        use_live=use_live, use_sim=use_sim,
        run_hmm=run_hmm, run_garch=run_garch,
        run_ar=run_ar, run_bayes=run_bayes,
        run_sarimax=run_sarimax, run_ens=run_ens,
        fetch_exog=fetch_exog, run_sig=run_sig,
        include_nasa=include_nasa, fred_key=fred_key or None,
        granger_pval=granger_pval, min_corr=min_corr,
        hmm_k=hmm_k,
        bt_train=bt_train, bt_test=bt_test, bt_folds=bt_folds,
        run_clicked=run_clicked,
    )


# ══════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def _yf_download(ticker: str, start: str, end: str) -> str:
    """Cache-safe yfinance download. Returns JSON string."""
    try:
        import yfinance as yf
        raw = yf.Ticker(ticker).history(
            start=start, end=end, interval="1d", auto_adjust=True)
        if raw.empty:
            return ""
        raw = raw.rename(columns=str.lower)
        cols = [c for c in ["open","high","low","close","volume"] if c in raw.columns]
        raw = raw[cols]
        raw.index = raw.index.tz_localize(None)
        raw = raw[raw["close"] > 0]
        return raw.to_json()
    except Exception:
        return ""


def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    js = _yf_download(ticker, start, end)
    if not js:
        return pd.DataFrame()
    try:
        from io import StringIO
        df = pd.read_json(StringIO(js))
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()


def load_prices(cfg: dict) -> tuple[pd.DataFrame, bool]:
    """Returns (price_df, is_live)."""
    start, end = str(cfg["train_start"]), str(cfg["test_end"])
    tickers = [cfg["ticker"]]
    if cfg["ticker"] == "RC=F":
        tickers = ["RC=F", "RB=F"]

    if cfg["use_live"]:
        for t in tickers:
            df = fetch_prices(t, start, end)
            if not df.empty:
                return df, True

    if cfg["use_sim"]:
        try:
            from tests.evaluation.market_simulator import generate_full_dataset
            raw = generate_full_dataset(start, end, seed=42)
            key = ("arabica_futures" if cfg["variety"] == CoffeeVariety.ARABICA
                   else "robusta_futures")
            sim_df = raw.get(key, raw["arabica_futures"])
            st.info(f"Using calibrated simulator — {len(sim_df)} bars, "
                    f"price {sim_df['close'].iloc[0]:.1f}→{sim_df['close'].iloc[-1]:.1f}")
            return sim_df, False
        except Exception as e:
            st.error(f"Simulator failed: {e}")
            return pd.DataFrame(), False

    return pd.DataFrame(), False


# ══════════════════════════════════════════════════════════════════
# EXOGENOUS DATA
# ══════════════════════════════════════════════════════════════════

def fetch_exog_data(cfg: dict) -> tuple[pd.DataFrame, dict]:
    """
    Fetch all exog variables. Returns (exog_df, status_dict).
    status_dict maps category → list of (variable, ok/fail).
    """
    from ingestion.exog_fetcher import ExogFetcher
    from datetime import date as dt_cls

    start = cfg["train_start"]
    end   = cfg["test_end"]

    fetcher = ExogFetcher(
        fred_api_key=cfg.get("fred_key"),
        include_nasa=cfg["include_nasa"],
        verbose=False,
    )
    exog_df = fetcher.fetch_all(start, end, instrument=cfg["variety"].value)

    # Build status report
    status = {}
    if not exog_df.empty:
        for col in exog_df.columns:
            n_valid = exog_df[col].notna().sum()
            pct     = n_valid / max(len(exog_df), 1) * 100
            cat     = _col_category(col)
            status.setdefault(cat, []).append(
                (col, f"{n_valid} bars ({pct:.0f}%)" if pct > 10 else "⚠ sparse")
            )
    return exog_df, status


def _col_category(col: str) -> str:
    if col.startswith("fx_"):         return "FX Rates"
    if col.startswith("oni") or "enso" in col: return "Climate/ENSO"
    if any(r in col for r in ["sul_minas","cerrado","dak_lak","huila","sidama"]):
        return "NASA POWER Climate"
    if col in ["sugar_price","corn_price","crude_oil","nat_gas","cocoa_price",
               "arabica_close","robusta_close"]: return "Commodity Prices"
    if col in ["vix","dxy_proxy","spx","us_10y_yield","us_cpi",
               "fed_rate","brazil_rate","us_pce","us_ip","trade_wtd_usd"]:
        return "Macro/Financial"
    if "shipping" in col or "heating" in col: return "Shipping/Logistics"
    return "Other"


# ══════════════════════════════════════════════════════════════════
# SIGNIFICANCE TESTING
# ══════════════════════════════════════════════════════════════════

def run_significance(exog_df: pd.DataFrame, target: pd.Series,
                     granger_alpha: float, min_corr: float) -> tuple[pd.DataFrame, list[str]]:
    """Returns (report_df, selected_cols)."""
    from features.significance import SignificanceTester

    # Drop price columns (leakage)
    drop = ["arabica_close","robusta_close","target"]
    cols = [c for c in exog_df.columns if not any(d in c for d in drop)]
    X    = exog_df[cols].copy()

    tester = SignificanceTester(
        max_lag=10,
        granger_alpha=granger_alpha,
        min_abs_corr=min_corr,
        require_n_tests=1,
    )
    report = tester.full_run(X, target, prefix_drop="arabica_close")
    selected = report[report["passes"]].index.tolist()
    return report, selected


# ══════════════════════════════════════════════════════════════════
# FEATURE BUILDING
# ══════════════════════════════════════════════════════════════════

def build_features(price_df: pd.DataFrame, exog_df: pd.DataFrame,
                   cfg: dict) -> "FeatureFrame":
    from features.pipeline import FeaturePipeline

    variety = cfg["variety"]
    key     = ("arabica_futures" if variety == CoffeeVariety.ARABICA
               else "robusta_futures")
    other_key = ("robusta_futures" if variety == CoffeeVariety.ARABICA
                 else "arabica_futures")

    raw = {key: price_df}

    # companion variety
    other = fetch_prices(
        "RC=F" if "KC" in cfg["ticker"] else "KC=F",
        str(cfg["train_start"]), str(cfg["test_end"])
    )
    if not other.empty:
        raw[other_key] = other

    # Wire exog into pipeline inputs
    # Each exog series goes in as a simple DataFrame the pipeline can access
    if not exog_df.empty:
        # FX rates
        for col in [c for c in exog_df.columns if c.startswith("fx_")]:
            raw[col] = pd.DataFrame({"rate": exog_df[col]})

        # ENSO
        enso_cols = [c for c in exog_df.columns if "oni" in c or "enso" in c]
        if enso_cols:
            raw["enso"] = exog_df[enso_cols].rename(
                columns={enso_cols[0]: "oni"} if enso_cols else {})

        # COT (if present in exog)
        cot_cols = [c for c in exog_df.columns if "cot" in c.lower()]
        if cot_cols:
            raw[f"cot_{cfg['variety'].value}"] = exog_df[cot_cols]

        # NASA POWER climate — feed directly to pipeline as climate_ prefixed keys
        nasa_regions = ["sul_minas","cerrado","dak_lak","huila","sidama"]
        for region in nasa_regions:
            region_cols = [c for c in exog_df.columns if region in c]
            if region_cols:
                raw[f"climate_{region}"] = exog_df[region_cols]

    frame = FeaturePipeline().run(raw, variety, DataFrequency.DAILY)

    # Append exog columns that aren't already in the frame
    if not exog_df.empty:
        bday_idx = frame.df.index
        for col in exog_df.columns:
            if col not in frame.df.columns and "close" not in col:
                s = exog_df[col].reindex(bday_idx).ffill(limit=10)
                if s.notna().sum() > 50:
                    frame.df[col] = s
                    if col not in frame.feature_names:
                        frame.feature_names.append(col)

    return frame


# ══════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════

def _sub(frame, df):
    from contracts.schemas import FeatureFrame
    return FeatureFrame(variety=frame.variety, frequency=frame.frequency,
                        feature_names=frame.feature_names, df=df)


def _score(prob_ups, actuals, label=""):
    if not prob_ups:
        return {"label": label, "n_forecasts": 0, "directional_accuracy": None,
                "brier_score": None, "brier_skill_score": None,
                "signal_sharpe_ann": None, "permutation_pvalue": None,
                "significant_5pct": False, "calibration_df": pd.DataFrame()}
    p, a = np.array(prob_ups), np.array(actuals)
    da   = float(np.mean((p > 0.5) == (a > 0)))
    bs   = float(np.mean((p - (a > 0).astype(float)) ** 2))
    bss  = float(1 - bs / 0.25)
    sig  = np.where(p > 0.55, 1., np.where(p < 0.45, -1., 0.))
    pnl  = sig * a
    sh   = float(pnl.mean()/pnl.std()*np.sqrt(252)) if pnl.std() > 0 else None
    rng  = np.random.default_rng(0)
    null = [float(np.mean((rng.permutation(p) > 0.5) == (a > 0))) for _ in range(500)]
    pval = float(np.mean(np.array(null) >= da))
    cal  = []
    for lo, hi in zip(np.linspace(0,1,6)[:-1], np.linspace(0,1,6)[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() > 0:
            cal.append({"bin": f"{lo:.1f}–{hi:.1f}",
                        "predicted": round(float(p[m].mean()),3),
                        "actual":    round(float((a[m]>0).mean()),3),
                        "n":         int(m.sum())})
    return {
        "label": label, "n_forecasts": len(p),
        "n_long": int((p>0.55).sum()), "n_short": int((p<0.45).sum()),
        "directional_accuracy": round(da,4), "brier_score": round(bs,4),
        "brier_skill_score": round(bss,4),
        "signal_sharpe_ann": round(sh,3) if sh else None,
        "permutation_pvalue": round(pval,4),
        "significant_5pct": bool(pval < 0.05),
        "calibration_df": pd.DataFrame(cal),
    }


def model_hmm(frame, cfg):
    from models.hmm_model import HMMRegimeDetector
    cutoff   = pd.Timestamp(cfg["test_start"])
    train_df = frame.df[frame.df.index < cutoff]
    FEATS    = ["log_return_1d","realised_vol_21d","log_return_5d","price_z_63d"]
    det = HMMRegimeDetector(n_regimes=cfg["hmm_k"], variety=cfg["variety"],
                            feature_cols=FEATS, n_iter=300, random_state=0)
    det.fit(_sub(frame, train_df))
    snaps   = det.predict(frame)
    regimes = pd.Series({pd.Timestamp(s.timestamp): s.regime.value for s in snaps})
    confs   = pd.Series({pd.Timestamp(s.timestamp): s.probability  for s in snaps})
    stats   = det.regime_statistics(_sub(frame, train_df))
    return {"regimes": regimes, "confs": confs,
            "stats": stats, "trans": det.transition_matrix(),
            "latest": snaps[-1] if snaps else None,
            "detector": det}


def model_garch(frame, cfg):
    from models.garch_model import GARCHVolatilityModel
    ret_col  = "log_return_1d"
    if ret_col not in frame.df.columns:
        return None
    cutoff   = pd.Timestamp(cfg["test_start"])
    all_rets = frame.df[ret_col].dropna()
    tr_rets  = all_rets[all_rets.index < cutoff]
    m = GARCHVolatilityModel(p=1, q=1, model_type="GARCH", dist="t",
                              variety=cfg["variety"])
    m.fit(tr_rets)
    return {"model": m, "cond_vol": m.conditional_volatility(),
            "forecast": m.forecast(tr_rets.iloc[-252:], horizon_days=10),
            "train_rets": tr_rets}


def model_ar(frame, cfg):
    from statsmodels.tsa.ar_model import AutoReg
    from scipy import stats as sp

    TARGET  = "target_log_return_5d"
    cutoff  = pd.Timestamp(cfg["test_start"])
    tr_df   = frame.df[frame.df.index < cutoff].dropna(subset=[TARGET])
    te_df   = frame.df[frame.df.index >= cutoff].dropna(subset=[TARGET])
    if len(te_df) == 0:
        return {}

    full_y  = pd.concat([tr_df[TARGET], te_df[TARGET]]).sort_index()
    fa      = full_y.values
    fm      = {t: i for i, t in enumerate(full_y.index)}
    h       = cfg["horizon"]
    results = {}

    for lags, label in [(1,"AR(1)"),(2,"AR(2)"),(5,"AR(5)")]:
        try:
            m     = AutoReg(tr_df[TARGET].values, lags=lags, old_names=False).fit()
            c     = float(m.params[0])
            phis  = [float(m.params[i+1]) for i in range(lags)]
            h_std = max(float(m.resid.std()) * np.sqrt(h), 1e-8)
            pu_list, act_list = [], []
            for dt in te_df.index[::2]:
                pos = fm.get(dt)
                if pos is None or pos < lags: continue
                fc  = c + sum(phis[i]*fa[pos-1-i] for i in range(lags))
                pu  = float(1 - sp.norm.cdf(0, loc=fc*h, scale=h_std))
                pu_list.append(pu); act_list.append(float(te_df.loc[dt, TARGET]))
            results[label] = {"model": m, "prob_ups": pu_list, "actuals": act_list,
                               "aic": round(m.aic,2), "bic": round(m.bic,2),
                               **_score(pu_list, act_list, label)}
        except Exception as e:
            results[label] = {"error": str(e)}
    return results


def model_bayesian(frame, cfg):
    from models.bayesian_model import BayesianForecaster
    from scipy import stats as sp

    TARGET = "target_log_return_5d"
    cutoff = pd.Timestamp(cfg["test_start"])
    tr_df  = frame.df[frame.df.index < cutoff].dropna(subset=[TARGET])
    te_df  = frame.df[frame.df.index >= cutoff].dropna(subset=[TARGET])
    if len(te_df) == 0: return None

    m = BayesianForecaster(variety=cfg["variety"])
    m.fit(_sub(frame, tr_df), TARGET)
    avail = [c for c in m._fitted_cols if c in te_df.columns]
    pu_list, act_list = [], []
    for dt in te_df.index[::2]:
        try:
            row = te_df[avail].loc[:dt].ffill().iloc[-1:].fillna(0)
            X   = m._scaler.transform(row.values)
            mu, sig = m._model.predict(X, return_std=True)
            h_std = max(float(sig[0]) * np.sqrt(cfg["horizon"]), 1e-8)
            pu    = float(1 - sp.norm.cdf(0, loc=float(mu[0]), scale=h_std))
            pu_list.append(pu); act_list.append(float(te_df.loc[dt, TARGET]))
        except Exception: continue

    return {"model": m, "prob_ups": pu_list, "actuals": act_list,
            "importance": m.feature_importance().head(15),
            **_score(pu_list, act_list, "BayesianRidge")}


def model_sarimax(frame, cfg, sig_cols: list[str]):
    """
    SARIMAX(1,0,1) with selected exog vs plain ARMA(1,0,1).
    Always runs both so comparison is always available.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from scipy import stats as sp

    TARGET = "target_log_return_5d"
    cutoff = pd.Timestamp(cfg["test_start"])
    tr_df  = frame.df[frame.df.index < cutoff].dropna(subset=[TARGET])
    te_df  = frame.df[frame.df.index >= cutoff].dropna(subset=[TARGET])
    if len(te_df) == 0:
        return {"error": "No test data"}

    # --- which exog cols are actually available and not all-NaN ---
    avail = [c for c in sig_cols
             if c in tr_df.columns
             and c != TARGET
             and not c.startswith("target_")
             and tr_df[c].notna().sum() > 50]

    target_tr = tr_df[TARGET]
    h         = cfg["horizon"]

    # ── plain ARMA (1,0,1) — no exog ─────────────────────────────
    try:
        res_plain = SARIMAX(target_tr, order=(1,0,1),
                            enforce_stationarity=False).fit(disp=False, maxiter=200)
    except Exception as e:
        return {"error": f"Plain ARMA fit failed: {e}"}

    h_std_plain = max(float(res_plain.resid.std()) * np.sqrt(h), 1e-8)

    # ── SARIMAX with exog ─────────────────────────────────────────
    res_exog = None
    if avail:
        exog_tr = tr_df[avail].ffill().bfill().fillna(0)
        try:
            res_exog = SARIMAX(target_tr, exog=exog_tr, order=(1,0,1),
                               enforce_stationarity=False,
                               enforce_invertibility=False).fit(
                                   disp=False, maxiter=300, method="lbfgs")
        except Exception as e:
            st.warning(f"SARIMAX+exog fit failed ({e}) — showing plain ARMA only")
            res_exog = None

    # ── Combined series for state extraction ──────────────────────
    full_y  = pd.concat([tr_df[TARGET], te_df[TARGET]]).sort_index()
    fa      = full_y.values
    fm      = {t: i for i, t in enumerate(full_y.index)}

    if avail and res_exog is not None:
        full_exog = pd.concat([
            tr_df[avail].ffill().bfill().fillna(0),
            te_df[avail].ffill().bfill().fillna(0),
        ]).sort_index()

    # ── OOS prediction loop ───────────────────────────────────────
    pu_plain, pu_exog, actuals = [], [], []

    for dt in te_df.index[::2]:
        pos = fm.get(dt)
        if pos is None or pos < 2: continue
        actual = float(te_df.loc[dt, TARGET])

        # plain ARMA
        try:
            fc_p   = res_plain.get_forecast(steps=h)
            mr_p   = float(fc_p.predicted_mean.mean())
            pu_p   = float(1 - sp.norm.cdf(0, loc=mr_p, scale=h_std_plain))
            pu_plain.append(pu_p)
        except Exception:
            pu_plain.append(0.5)

        # SARIMAX + exog
        if avail and res_exog is not None:
            try:
                last_exog  = full_exog.loc[:dt].iloc[-1:].values
                exog_fcast = np.repeat(last_exog, h, axis=0)
                fc_e       = res_exog.get_forecast(steps=h, exog=exog_fcast)
                mr_e       = float(fc_e.predicted_mean.mean())
                h_std_exog = max(float(res_exog.resid.std()) * np.sqrt(h), 1e-8)
                pu_e       = float(1 - sp.norm.cdf(0, loc=mr_e, scale=h_std_exog))
                pu_exog.append(pu_e)
            except Exception:
                pu_exog.append(pu_plain[-1])  # fallback to plain
        else:
            pu_exog.append(pu_plain[-1])

        actuals.append(actual)

    # ── Extract exog coefficients ─────────────────────────────────
    exog_params = {}
    if res_exog is not None and avail:
        for i, col in enumerate(avail):
            try:
                coef = float(res_exog.params.iloc[2 + i])
                pval = float(res_exog.pvalues.iloc[2 + i])
                exog_params[col] = {"coef": round(coef,6),
                                    "pvalue": round(pval,4),
                                    "significant": bool(pval < 0.05)}
            except Exception:
                pass

    scores_plain = _score(pu_plain, actuals, "ARMA(1,0,1) no exog")
    scores_exog  = _score(pu_exog,  actuals, f"SARIMAX+{len(avail)}exog")

    return {
        "plain_prob_ups": pu_plain,
        "exog_prob_ups":  pu_exog,
        "actuals":        actuals,
        "exog_used":      avail,
        "n_exog":         len(avail),
        "aic_plain":      round(res_plain.aic, 2),
        "aic_exog":       round(res_exog.aic, 2) if res_exog else None,
        "exog_params":    exog_params,
        "scores_plain":   scores_plain,
        "scores_exog":    scores_exog,
        "da_lift":        round(
            (scores_exog["directional_accuracy"] or 0) -
            (scores_plain["directional_accuracy"] or 0), 4),
        "brier_lift": round(
            (scores_plain["brier_score"] or 0) -
            (scores_exog["brier_score"] or 0), 4),
        "sharpe_lift": round(
            ((scores_exog["signal_sharpe_ann"] or 0) -
             (scores_plain["signal_sharpe_ann"] or 0)), 3),
        # unified for model comparison tab
        "n_forecasts":          scores_exog["n_forecasts"],
        "directional_accuracy": scores_exog["directional_accuracy"],
        "brier_score":          scores_exog["brier_score"],
        "brier_skill_score":    scores_exog["brier_skill_score"],
        "signal_sharpe_ann":    scores_exog["signal_sharpe_ann"],
        "permutation_pvalue":   scores_exog["permutation_pvalue"],
        "significant_5pct":     scores_exog["significant_5pct"],
    }


def model_backtest(frame, cfg) -> pd.DataFrame:
    from statsmodels.tsa.ar_model import AutoReg
    from scipy import stats as sp

    TARGET = "target_log_return_5d"
    h      = cfg["horizon"]
    df     = frame.df.dropna(subset=[TARGET]).copy()
    n, tw, tsw = len(df), min(cfg["bt_train"], len(df)-30), cfg["bt_test"]
    rows   = []
    start  = tw
    for fold in range(cfg["bt_folds"]):
        if start + tsw > n: break
        tr = df.iloc[max(0, start-tw):start]
        te = df.iloc[start:min(start+tsw, n)]
        try:
            m   = AutoReg(tr[TARGET].values, lags=1, old_names=False).fit()
            c, phi = float(m.params[0]), float(m.params[1])
            h_std  = max(float(m.resid.std()) * np.sqrt(h), 1e-8)
            fy     = pd.concat([tr[TARGET], te[TARGET]])
            fa     = fy.values
            pu_list, act_list = [], []
            for i, (dt, row) in enumerate(te.iterrows()):
                pos = len(tr) + i
                if pos < 1: continue
                fc  = c + phi * fa[pos-1]
                pu  = float(1 - sp.norm.cdf(0, loc=fc*h, scale=h_std))
                pu_list.append(pu); act_list.append(float(row[TARGET]))
            sc = _score(pu_list, act_list)
            rows.append({
                "fold":        fold+1,
                "train_start": str(tr.index[0].date()),
                "train_end":   str(tr.index[-1].date()),
                "test_start":  str(te.index[0].date()),
                "test_end":    str(te.index[-1].date()),
                "n":           len(pu_list),
                "da":          sc["directional_accuracy"],
                "brier":       sc["brier_score"],
                "sharpe":      sc["signal_sharpe_ann"] or 0.0,
            })
        except Exception: pass
        start += tsw
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# FORECASTING ENGINE
# ══════════════════════════════════════════════════════════════════

def generate_forecast(price_df: pd.DataFrame, frame, ar_results: dict,
                      bayes_result, garch_result, hmm_result,
                      cfg: dict) -> dict:
    """
    Generate full probabilistic forecast for next `horizon` days.
    Returns day-by-day table + simulation paths for fan chart.
    """
    from scipy import stats as sp
    from statsmodels.tsa.ar_model import AutoReg

    TARGET  = "target_log_return_5d"
    h       = cfg["horizon"]
    close   = price_df["close"]
    last_px = float(close.iloc[-1])

    # --- Pick best AR model by DA ---
    best_ar = None
    best_da = -1
    for label, r in ar_results.items():
        da = r.get("directional_accuracy") or 0
        if da > best_da and "model" in r:
            best_da = da; best_ar = r

    # --- 1-day AR(1) for simulation ---
    TARGET_1D = "target_log_return_5d"
    cutoff    = pd.Timestamp(cfg["test_start"])
    tr_df     = frame.df[frame.df.index < cutoff].dropna(subset=[TARGET_1D])

    ar1_coef  = None
    ar1_std   = None
    try:
        m1        = AutoReg(tr_df[TARGET_1D].values, lags=1, old_names=False).fit()
        ar1_c     = float(m1.params[0])
        ar1_phi   = float(m1.params[1])
        ar1_std   = float(m1.resid.std())
        # last known target value
        last_tgt  = float(tr_df[TARGET_1D].iloc[-1])
        ar1_coef  = (ar1_c, ar1_phi, last_tgt)
    except Exception:
        ar1_std   = 0.015

    # --- Monte Carlo simulation ---
    rng       = np.random.default_rng(42)
    n_paths   = cfg["n_fc_samples"]
    log_ret_paths = np.zeros((n_paths, h))
    last_val  = last_tgt if ar1_coef else 0.0

    if ar1_coef:
        c_v, phi_v, _ = ar1_coef
        for step in range(h):
            noise = rng.normal(0, ar1_std, n_paths)
            log_ret_paths[:, step] = c_v + phi_v * last_val + noise
            last_val = log_ret_paths[:, step].mean()
    else:
        log_ret_paths = rng.normal(0, ar1_std or 0.015, (n_paths, h))

    # cumulative log returns → prices
    cum_log    = np.cumsum(log_ret_paths, axis=1)
    price_paths = last_px * np.exp(cum_log)

    # Quantiles across paths
    q05  = np.percentile(price_paths, 5,  axis=0)
    q25  = np.percentile(price_paths, 25, axis=0)
    q50  = np.percentile(price_paths, 50, axis=0)
    q75  = np.percentile(price_paths, 75, axis=0)
    q95  = np.percentile(price_paths, 95, axis=0)

    # Day-by-day forecast table
    bday_idx = pd.bdate_range(str(price_df.index[-1].date()),
                               periods=h+1, freq="B")[1:]
    fc_table = pd.DataFrame({
        "date":    [str(d.date()) for d in bday_idx],
        "p5":      q05.round(2),
        "p25":     q25.round(2),
        "median":  q50.round(2),
        "p75":     q75.round(2),
        "p95":     q95.round(2),
        "prob_up": [
            round(float(np.mean(price_paths[:, i] > last_px)), 3)
            for i in range(h)
        ],
        "expected_return": [
            round(float(np.mean(log_ret_paths[:, :i+1].sum(axis=1))), 4)
            for i in range(h)
        ],
    })

    # Overall h-day directional probability
    overall_prob_up  = float(np.mean(cum_log[:, -1] > 0))
    overall_mean_ret = float(cum_log[:, -1].mean())

    # Bayesian prob if available
    bayes_pup = None
    if bayes_result and bayes_result.get("prob_ups"):
        bayes_pup = float(np.mean(bayes_result["prob_ups"][-5:]))

    # Volatility context
    vol_regime = None
    current_vol = None
    if garch_result:
        fc_g       = garch_result["forecast"]
        vol_regime = fc_g.vol_regime
        current_vol= fc_g.current_annualized_vol

    # Regime context
    current_regime = None
    if hmm_result and hmm_result.get("latest"):
        current_regime = hmm_result["latest"].regime.value

    return {
        "last_price":      last_px,
        "fc_table":        fc_table,
        "price_paths":     price_paths,
        "history_close":   close,
        "q05": q05, "q25": q25, "q50": q50, "q75": q75, "q95": q95,
        "bday_idx":        bday_idx,
        "overall_prob_up": overall_prob_up,
        "overall_mean_ret":overall_mean_ret,
        "bayes_prob_up":   bayes_pup,
        "vol_regime":      vol_regime,
        "current_vol":     current_vol,
        "current_regime":  current_regime,
        "horizon":         h,
        "n_paths":         n_paths,
    }


# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════

def plot_price(price_df):
    import plotly.graph_objects as go
    fig = go.Figure()
    if all(c in price_df.columns for c in ["high","low"]):
        fig.add_trace(go.Scatter(
            x=list(price_df.index)+list(price_df.index[::-1]),
            y=list(price_df["high"])+list(price_df["low"][::-1]),
            fill="toself", fillcolor="rgba(111,78,55,0.1)",
            line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    fig.add_trace(go.Scatter(x=price_df.index, y=price_df["close"],
        line=dict(color="#c8a97e", width=1.5), name="Close"))
    fig.update_layout(height=380, margin=dict(t=10,b=20,l=0,r=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), xaxis_rangeslider_visible=False)
    return fig


def plot_regimes(price_df, regimes):
    import plotly.graph_objects as go
    COLORS = {"bull":"#2196F3","bear":"#F44336","volatile":"#FF9800",
               "neutral":"#9E9E9E","supply_stress":"#9C27B0","low_vol":"#4CAF50"}
    fig = go.Figure()
    ra  = regimes.reindex(price_df.index, method="ffill")
    chg = ra[ra != ra.shift()].index
    for i in range(len(chg)):
        x0, x1 = chg[i], (chg[i+1] if i+1 < len(chg) else price_df.index[-1])
        fig.add_vrect(x0=x0, x1=x1,
            fillcolor=COLORS.get(str(ra.loc[x0]),"#555"),
            opacity=0.15, layer="below", line_width=0)
    fig.add_trace(go.Scatter(x=price_df.index, y=price_df["close"],
        line=dict(color="white", width=1.2), name="Close"))
    fig.update_layout(height=420, margin=dict(t=10,b=20,l=0,r=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), xaxis_rangeslider_visible=False)
    return fig


def plot_vol(price_df, cond_vol):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65,0.35], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=price_df.index, y=price_df["close"],
        line=dict(color="#c8a97e",width=1.2), name="Close"), row=1, col=1)
    fig.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol*100,
        line=dict(color="#FF9800",width=1.4),
        fill="tozeroy", fillcolor="rgba(255,152,0,0.12)",
        name="Cond. Vol %"), row=2, col=1)
    fig.update_layout(height=420, margin=dict(t=10,b=20,l=0,r=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), showlegend=False)
    return fig


def plot_forecast_fan(fc: dict):
    import plotly.graph_objects as go
    hist   = fc["history_close"].iloc[-60:]
    idx    = fc["bday_idx"]
    fig    = go.Figure()

    # History
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values,
        line=dict(color="white",width=1.5), name="History"))

    # 5–95 band
    fig.add_trace(go.Scatter(
        x=list(idx)+list(idx[::-1]),
        y=list(fc["q95"])+list(fc["q05"][::-1]),
        fill="toself", fillcolor="rgba(33,150,243,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="5–95%"))

    # 25–75 band
    fig.add_trace(go.Scatter(
        x=list(idx)+list(idx[::-1]),
        y=list(fc["q75"])+list(fc["q25"][::-1]),
        fill="toself", fillcolor="rgba(33,150,243,0.18)",
        line=dict(color="rgba(0,0,0,0)"), name="25–75%"))

    # Median
    fig.add_trace(go.Scatter(x=idx, y=fc["q50"],
        line=dict(color="#2196F3",width=2,dash="dash"), name="Median"))

    # Last price line
    fig.add_hline(y=fc["last_price"], line_dash="dot",
                  line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        title=f"{fc['horizon']}-Day Probabilistic Price Forecast "
              f"({fc['n_paths']:,} simulation paths)",
        height=430, margin=dict(t=40,b=20,l=0,r=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


def plot_prob_gauge(prob_up: float):
    import plotly.graph_objects as go
    color = ("#4CAF50" if prob_up > 0.55 else
             "#F44336" if prob_up < 0.45 else "#FF9800")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_up * 100,
        delta={"reference": 50, "valueformat": ".1f",
               "suffix": "% vs 50%"},
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "P(Up) — forecast horizon"},
        gauge={
            "axis": {"range": [0,100], "tickwidth": 1},
            "bar":  {"color": color},
            "steps": [
                {"range":[0,45],  "color":"rgba(244,67,54,0.15)"},
                {"range":[45,55], "color":"rgba(255,152,0,0.15)"},
                {"range":[55,100],"color":"rgba(76,175,80,0.15)"},
            ],
            "threshold": {"line":{"color":"white","width":3},
                          "thickness":0.75,"value":50},
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40,b=10,l=20,r=20),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"))
    return fig


def plot_calibration(cal_df: pd.DataFrame):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1], mode="lines",
        line=dict(color="grey",dash="dash"), name="Perfect"))
    if not cal_df.empty:
        x = cal_df["predicted"] if "predicted" in cal_df.columns else cal_df.get("mean_predicted", cal_df.iloc[:,0])
        y = cal_df["actual"]    if "actual"    in cal_df.columns else cal_df.get("fraction_positive", cal_df.iloc[:,1])
        n_col = cal_df["n"] if "n" in cal_df.columns else pd.Series([10]*len(x))
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers+lines",
            marker=dict(size=(n_col.clip(4,30)).tolist(),
                        color="#2196F3", opacity=0.85),
            name="Model"))
    fig.update_layout(title="Calibration", height=300,
        xaxis=dict(range=[0,1], title="Predicted Prob"),
        yaxis=dict(range=[0,1], title="Actual Hit Rate"),
        margin=dict(t=40,b=30,l=0,r=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"))
    return fig


# ══════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ══════════════════════════════════════════════════════════════════

def tab_data(price_df, cfg, is_live):
    src = "Live (yfinance)" if is_live else "Simulator (calibrated GARCH)"
    st.caption(f"Source: **{src}**  |  Instrument: **{cfg['ticker']}**")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Bars",        f"{len(price_df):,}")
    c2.metric("First close", f"{price_df['close'].iloc[0]:.2f}")
    c3.metric("Last close",  f"{price_df['close'].iloc[-1]:.2f}")
    chg = (price_df["close"].iloc[-1]/price_df["close"].iloc[0]-1)*100
    c4.metric("Total return", f"{chg:+.1f}%")
    st.plotly_chart(plot_price(price_df), use_container_width=True)


def tab_exog(exog_df, exog_status, sig_report, sig_cols, frame_df, cfg):
    if exog_df.empty:
        st.info("Enable 'Fetch exog data' and re-run analysis.")
        return

    # ── Fetch status ──────────────────────────────────────────────
    st.subheader("📡 Data Fetch Status")
    for cat, items in exog_status.items():
        with st.expander(f"{cat} ({len(items)} series)", expanded=False):
            for name, status in items:
                icon = "✅" if "bars" in status and "⚠" not in status else "⚠️"
                st.write(f"{icon} `{name}` — {status}")

    # ── Correlation bar chart ─────────────────────────────────────
    TARGET = "target_log_return_5d"
    if TARGET in frame_df.columns:
        tgt     = frame_df[TARGET].dropna()
        ea      = exog_df.reindex(tgt.index).ffill()
        valid   = [c for c in ea.columns
                   if ea[c].notna().sum() > 30 and "close" not in c]
        if valid:
            corrs = ea[valid].corrwith(tgt).dropna().sort_values()
            import plotly.express as pxp
            fig = pxp.bar(corrs, orientation="h",
                          color=corrs, color_continuous_scale="RdBu",
                          range_color=(-0.3, 0.3),
                          title="Spearman correlation with 5-day target return",
                          labels={"value": "Corr", "index": "Variable"})
            fig.update_layout(height=max(300, len(corrs)*16),
                margin=dict(t=40,b=20,l=0,r=0),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="#fafafa"))
            st.plotly_chart(fig, use_container_width=True)

    # ── Significance report ───────────────────────────────────────
    if sig_report is not None and not sig_report.empty:
        st.subheader("📊 Significance Test Results")
        passed = sig_report[sig_report["passes"]]
        failed = sig_report[~sig_report["passes"]]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total tested",  len(sig_report))
        c2.metric("✅ Passed (→ SARIMAX)", len(passed))
        c3.metric("❌ Failed",     len(failed))

        st.markdown("**Variables selected for SARIMAX:**")
        if not passed.empty:
            disp_cols = [c for c in ["granger_pval","max_corr","best_lag",
                                      "lasso_selected","rf_importance",
                                      "composite_score","reason"]
                         if c in passed.columns]
            st.dataframe(passed[disp_cols].round(4), use_container_width=True)
        else:
            st.warning("No variables passed. Lower the significance thresholds in the sidebar.")

        with st.expander("All variables (including failed)"):
            st.dataframe(sig_report.round(4), use_container_width=True)

    st.caption("These are the variables passed to SARIMAX. "
               "Adjust Granger p-value and min |correlation| thresholds in the sidebar.")


def tab_regimes(price_df, hmm_result):
    if hmm_result is None:
        st.info("Enable HMM in sidebar and re-run.")
        return
    st.plotly_chart(plot_regimes(price_df, hmm_result["regimes"]),
                    use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Regime statistics (train period)**")
        st.dataframe(hmm_result["stats"].round(4), use_container_width=True)
    with c2:
        st.markdown("**Transition probabilities**")
        st.dataframe(hmm_result["trans"].round(4), use_container_width=True)

    lt = hmm_result.get("latest")
    if lt:
        st.markdown("---")
        c1,c2,c3 = st.columns(3)
        emoji = {"bull":"🔵","bear":"🔴","volatile":"🟠","neutral":"⚪"}.get(
            lt.regime.value,"⚪")
        c1.metric("Current Regime",    f"{emoji} {lt.regime.value.upper()}")
        c2.metric("Confidence",         f"{lt.probability:.1%}")
        c3.metric("Transition Risk",    f"{lt.transition_probability:.1%}")


def tab_volatility(price_df, garch_result):
    if garch_result is None:
        st.info("Enable GARCH in sidebar and re-run.")
        return
    cv = garch_result["cond_vol"]
    st.plotly_chart(plot_vol(price_df.loc[cv.index[0]:], cv),
                    use_container_width=True)
    fc = garch_result["forecast"]
    c1,c2,c3,c4 = st.columns(4)
    emoji = {"low":"🟢","medium":"🟡","high":"🟠","extreme":"🔴"}.get(fc.vol_regime,"⚪")
    c1.metric("Current Vol (ann)",   f"{fc.current_annualized_vol:.1%}")
    c2.metric("10d Forecast Vol",    f"{fc.forecast_annualized_vol:.1%}")
    c3.metric("Regime",              f"{emoji} {fc.vol_regime.upper()}")
    c4.metric("Model",               "GARCH(1,1)-t")
    with st.expander("Model summary"):
        st.code(garch_result["model"]._result.summary().as_text())


def tab_ar_bayes(ar_results, bayes_result):
    # AR models
    st.subheader("AR / ARMA Models (no exogenous variables)")
    if ar_results:
        rows = []
        for label, r in ar_results.items():
            if "error" in r:
                st.error(f"{label}: {r['error']}")
                continue
            rows.append({
                "Model": label,
                "N":     r.get("n_forecasts",0),
                "DA":    r.get("directional_accuracy"),
                "Brier": r.get("brier_score"),
                "BSS":   r.get("brier_skill_score"),
                "Sharpe":r.get("signal_sharpe_ann"),
                "p-val": r.get("permutation_pvalue"),
                "Sig":   "✓" if r.get("significant_5pct") else "",
                "AIC":   r.get("aic"),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format({
            "DA":"{:.4f}","Brier":"{:.4f}","BSS":"{:.4f}",
            "Sharpe":"{:.3f}","p-val":"{:.4f}",
        }), use_container_width=True)

        # Calibration for best AR
        best = max(ar_results.values(),
                   key=lambda r: r.get("directional_accuracy") or 0)
        cal = best.get("calibration_df", pd.DataFrame())
        if not cal.empty:
            st.plotly_chart(plot_calibration(cal), use_container_width=True)

    # Bayesian
    st.subheader("Bayesian Ridge (uses all engineered features as exog)")
    if bayes_result and "error" not in bayes_result:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("DA",     f"{bayes_result.get('directional_accuracy',0):.4f}")
        c2.metric("Brier",  f"{bayes_result.get('brier_score',0):.4f}")
        c3.metric("Sharpe", f"{bayes_result.get('signal_sharpe_ann') or 0:.3f}")
        c4.metric("p-val",  f"{bayes_result.get('permutation_pvalue',1):.4f}")
        imp = bayes_result.get("importance")
        if imp is not None:
            with st.expander("Top 15 feature importances"):
                import plotly.express as pxp
                fig = pxp.bar(imp, orientation="h",
                              title="Bayesian Ridge |posterior coefficient|",
                              labels={"value":"|coef|","index":"feature"})
                fig.update_layout(height=400, margin=dict(t=40,b=20,l=0,r=0),
                    plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                    font=dict(color="#fafafa"))
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enable Bayesian Ridge in sidebar.")


def tab_sarimax(sarimax_result):
    st.subheader("SARIMAX(1,0,1) — Exogenous Variable Contribution")

    if sarimax_result is None:
        st.info("Enable 'SARIMAX + Exog' in the sidebar and re-run analysis.")
        return
    if "error" in sarimax_result:
        st.error(sarimax_result["error"])
        if "No significant" in str(sarimax_result["error"]):
            st.info("👉 Go to the **Exogenous Variables** tab — the significance test "
                    "found no variables. Try lowering the Granger p-value threshold "
                    "(e.g. 0.15) or min |correlation| (e.g. 0.02) in the sidebar.")
        return

    n_exog = sarimax_result.get("n_exog", 0)
    sp     = sarimax_result.get("scores_plain", {})
    se     = sarimax_result.get("scores_exog",  {})

    # ── Side-by-side comparison ───────────────────────────────────
    st.markdown("### Plain ARMA(1,0,1) — no exog  vs  SARIMAX with exog")
    st.caption("Same test period, same OOS prediction points. "
               "Difference = contribution of exogenous variables.")

    col_plain, col_arrow, col_exog = st.columns([5,1,5])

    def _metric_delta(label, plain_val, exog_val, lower_is_better=False):
        if plain_val is None or exog_val is None:
            return
        delta = exog_val - plain_val
        if lower_is_better:
            delta_color = "inverse"
        else:
            delta_color = "normal"
        col_exog.metric(label, f"{exog_val:.4f}",
                         delta=f"{delta:+.4f}",
                         delta_color=delta_color)
        col_plain.metric(label, f"{plain_val:.4f}")

    with col_plain:
        st.markdown("#### 📊 Plain ARMA")
        st.markdown(f"*No external variables*")

    with col_arrow:
        st.markdown("<div style='text-align:center;font-size:2rem;padding-top:40px'>→</div>",
                    unsafe_allow_html=True)

    with col_exog:
        st.markdown(f"#### 🧪 SARIMAX + {n_exog} exog vars")
        st.markdown(f"*{', '.join(sarimax_result.get('exog_used',[])[:3])}"
                    f"{'…' if n_exog > 3 else ''}*")

    _metric_delta("Directional Accuracy ↑",
                  sp.get("directional_accuracy"), se.get("directional_accuracy"))
    _metric_delta("Brier Score ↓",
                  sp.get("brier_score"), se.get("brier_score"), lower_is_better=True)
    _metric_delta("Brier Skill Score ↑",
                  sp.get("brier_skill_score"), se.get("brier_skill_score"))
    _metric_delta("Signal Sharpe ↑",
                  sp.get("signal_sharpe_ann"), se.get("signal_sharpe_ann"))

    col_plain.metric("p-value", f"{sp.get('permutation_pvalue',1):.4f}")
    col_plain.metric("N forecasts", sp.get("n_forecasts",0))
    col_plain.metric("AIC", sarimax_result.get("aic_plain","n/a"))

    col_exog.metric("p-value",    f"{se.get('permutation_pvalue',1):.4f}")
    col_exog.metric("N forecasts", se.get("n_forecasts",0))
    col_exog.metric("AIC", sarimax_result.get("aic_exog","n/a"))

    # ── Verdict ───────────────────────────────────────────────────
    st.markdown("---")
    da_lift     = sarimax_result.get("da_lift", 0)
    brier_lift  = sarimax_result.get("brier_lift", 0)
    sharpe_lift = sarimax_result.get("sharpe_lift", 0)

    if da_lift > 0.02 or brier_lift > 0.005:
        st.success(
            f"✅ **Exogenous variables ADD predictive value.**  "
            f"DA lift: **{da_lift:+.4f}**  |  "
            f"Brier improvement: **{brier_lift:+.4f}**  |  "
            f"Sharpe lift: **{sharpe_lift:+.3f}**"
        )
    elif da_lift < -0.02 or brier_lift < -0.005:
        st.error(
            f"❌ **Exogenous variables HURT performance.**  "
            f"DA change: {da_lift:+.4f}  |  Brier change: {brier_lift:+.4f}  —  "
            f"Try tightening the significance threshold or use fewer variables."
        )
    else:
        st.warning(
            f"⚠️ **Exogenous variables have NEUTRAL effect** (within noise).  "
            f"DA change: {da_lift:+.4f}  |  Brier change: {brier_lift:+.4f}  |  "
            f"n={se.get('n_forecasts',0)} — try more test data for a conclusive result."
        )

    # ── Coefficient table ─────────────────────────────────────────
    exog_params = sarimax_result.get("exog_params", {})
    if exog_params:
        st.markdown("---")
        st.subheader("Fitted Exogenous Coefficients")
        st.caption(
            "Coefficient sign = direction of effect on target return.  "
            "p-value = significance **inside the SARIMAX** (after controlling for AR + other exog).  "
            "A variable can pass Granger pre-selection but be insignificant here — "
            "that means its effect is captured by another variable already in the model."
        )
        rows = []
        for var, info in exog_params.items():
            rows.append({
                "variable":    var,
                "coefficient": info["coef"],
                "p-value":     info["pvalue"],
                "direction":   "↑ positive" if info["coef"] > 0 else "↓ negative",
                "significant": "✓" if info["significant"] else "",
            })
        df = pd.DataFrame(rows).set_index("variable")
        st.dataframe(df.style.format({
            "coefficient": "{:+.6f}", "p-value": "{:.4f}"}),
            use_container_width=True)

        sig_vars   = [r["variable"] for r in rows if r["significant"]]
        insig_vars = [r["variable"] for r in rows if not r["significant"]]
        if sig_vars:
            st.success(f"**Significant inside SARIMAX (p<0.05):** {', '.join(sig_vars)}")
        if insig_vars:
            st.info(f"**Passed pre-screening but not significant in SARIMAX:** "
                    f"{', '.join(insig_vars)}")

    with st.expander(f"All {n_exog} variables used"):
        for v in sarimax_result.get("exog_used", []):
            st.write(f"• `{v}`")


def tab_backtest(bt_df):
    st.subheader("Walk-Forward Backtest (AR(1))")
    if bt_df is None or bt_df.empty:
        st.info("Enable AR models and re-run.")
        return

    st.dataframe(bt_df.style.format({
        "da": "{:.4f}", "brier": "{:.4f}", "sharpe": "{:.3f}",
    }), use_container_width=True)

    import plotly.graph_objects as go
    fig = go.Figure()
    colors = ["#4CAF50" if v > 0.5 else "#F44336" for v in bt_df["da"]]
    fig.add_trace(go.Bar(x=bt_df["fold"].astype(str), y=bt_df["da"],
                         marker_color=colors, name="DA"))
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey",
                  annotation_text="Random baseline")
    fig.update_layout(title="Directional Accuracy by Fold", height=320,
        yaxis=dict(range=[0.3,0.85]),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="#fafafa"), margin=dict(t=40,b=20,l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)

    avg_da = bt_df["da"].mean()
    if avg_da > 0.55:
        st.success(f"Average DA across folds: **{avg_da:.4f}** — consistently above random")
    elif avg_da > 0.50:
        st.warning(f"Average DA: **{avg_da:.4f}** — marginal edge")
    else:
        st.error(f"Average DA: **{avg_da:.4f}** — below random baseline")


def tab_comparison(all_results):
    st.subheader("All Models — Side-by-Side Comparison")
    rows = []
    for label, r in all_results.items():
        if not r or not isinstance(r, dict):
            continue
        if r.get("n_forecasts", 0) == 0:
            continue
        rows.append({
            "Model":  label,
            "N":      r.get("n_forecasts",0),
            "DA":     r.get("directional_accuracy"),
            "Brier":  r.get("brier_score"),
            "BSS":    r.get("brier_skill_score"),
            "Sharpe": r.get("signal_sharpe_ann"),
            "p-val":  r.get("permutation_pvalue"),
            "Sig":    "✓" if r.get("significant_5pct") else "",
        })
    if not rows:
        st.info("Run analysis first.")
        return

    df = pd.DataFrame(rows).sort_values("DA", ascending=False).set_index("Model")
    st.dataframe(df.style.format({
        "DA":"{:.4f}","Brier":"{:.4f}","BSS":"{:.4f}",
        "Sharpe":"{:.3f}","p-val":"{:.4f}",
    }), use_container_width=True)
    st.markdown("> Random baseline: DA=0.5000 · Brier=0.2500 · BSS=0 · Sharpe=0  \n"
                "> ✓ = p<0.05 (permutation test, n=500)")


def tab_forecast(fc: dict | None, cfg: dict):
    st.subheader(f"📡 {cfg['horizon']}-Day Probabilistic Forecast")

    if fc is None:
        st.info("Run analysis first.")
        return

    # ── Gauge + context ───────────────────────────────────────────
    c_gauge, c_ctx = st.columns([1, 2])
    with c_gauge:
        st.plotly_chart(plot_prob_gauge(fc["overall_prob_up"]),
                        use_container_width=True)

    with c_ctx:
        st.markdown("### Market Context")
        if fc.get("current_regime"):
            emoji = {"bull":"🔵","bear":"🔴","volatile":"🟠","neutral":"⚪"}.get(
                fc["current_regime"],"⚪")
            st.metric("Current HMM Regime",
                      f"{emoji} {fc['current_regime'].upper()}")
        if fc.get("current_vol"):
            ve = {"low":"🟢","medium":"🟡","high":"🟠","extreme":"🔴"}.get(
                fc.get("vol_regime",""),"⚪")
            st.metric("Current Volatility",
                      f"{ve} {fc['current_vol']:.1%} ann.  [{fc.get('vol_regime','')}]")
        if fc.get("bayes_prob_up") is not None:
            st.metric("Bayesian P(Up)", f"{fc['bayes_prob_up']:.1%}")
        st.metric("AR Simulation P(Up)", f"{fc['overall_prob_up']:.1%}")

        # Signal
        pup = fc["overall_prob_up"]
        if pup > 0.55:
            st.success(f"📈 Signal: **LONG** ({pup:.0%} probability of up move)")
        elif pup < 0.45:
            st.error(f"📉 Signal: **SHORT** ({1-pup:.0%} probability of down move)")
        else:
            st.warning(f"➡️ Signal: **FLAT / NO EDGE** ({pup:.0%} vs 50% baseline)")

    # ── Fan chart ─────────────────────────────────────────────────
    st.plotly_chart(plot_forecast_fan(fc), use_container_width=True)

    # ── Day-by-day table ──────────────────────────────────────────
    st.subheader("Day-by-Day Forecast Table")
    st.caption(f"Based on {fc['n_paths']:,} Monte Carlo simulation paths from AR(1) fitted on train data.")
    ft = fc["fc_table"].copy()
    ft.index = range(1, len(ft)+1)
    ft.index.name = "day"
    st.dataframe(ft.style.format({
        "p5":"{:.2f}","p25":"{:.2f}","median":"{:.2f}",
        "p75":"{:.2f}","p95":"{:.2f}",
        "prob_up":"{:.1%}",
        "expected_return":"{:+.4f}",
    }), use_container_width=True)

    st.caption(
        "**p5/p95** = 5th/95th percentile price across all paths.  "
        "**prob_up** = fraction of paths where price is above current last close.  "
        "**expected_return** = mean cumulative log return to that day.  "
        "Forecast does NOT constitute financial advice."
    )

    # ── Simulation settings recap ─────────────────────────────────
    with st.expander("Simulation details"):
        st.write(f"- Model: AR(1) fitted on training period")
        st.write(f"- Paths: {fc['n_paths']:,} Monte Carlo samples")
        st.write(f"- Horizon: {fc['horizon']} trading days")
        st.write(f"- Starting price: {fc['last_price']:.2f}")
        st.write(f"- Method: Historical residual bootstrap "
                 f"(preserves volatility clustering)")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    cfg = sidebar()

    st.title("☕ Coffee Quant — Interactive Research Tool")
    st.caption(
        "Regime-aware probabilistic forecasting for Arabica · Robusta · "
        "Custom coffee futures  |  All models run OOS (no lookahead)"
    )

    TABS = st.tabs([
        "📊 Data",
        "🌍 Exog Variables",
        "🎯 Regimes (HMM)",
        "📈 Volatility (GARCH)",
        "📐 AR / Bayesian",
        "🔬 SARIMAX vs ARMA",
        "🔄 Backtest",
        "📋 Comparison",
        "🔮 Forecast",
    ])

    # ── Run analysis ──────────────────────────────────────────────
    if cfg["run_clicked"]:
        ss = st.session_state

        with st.spinner("Loading prices …"):
            price_df, is_live = load_prices(cfg)
        if price_df.empty:
            st.error("No price data. Check ticker and dates.")
            return
        ss["price_df"] = price_df
        ss["is_live"]  = is_live
        ss["cfg"]      = cfg

        # Exogenous data + significance
        exog_df     = pd.DataFrame()
        exog_status = {}
        sig_report  = None
        sig_cols    = []

        if cfg["fetch_exog"]:
            with st.spinner("Fetching exogenous data (28 variables) …"):
                exog_df, exog_status = fetch_exog_data(cfg)
            ss["exog_df"]     = exog_df
            ss["exog_status"] = exog_status

            if cfg["run_sig"] and not exog_df.empty:
                with st.spinner("Running significance tests …"):
                    # Build a quick version of the target for sig testing
                    close = price_df["close"]
                    tgt   = np.log(close.shift(-cfg["horizon"]) / close).dropna()
                    tgt.name = "target"
                    try:
                        sig_report, sig_cols = run_significance(
                            exog_df, tgt,
                            cfg["granger_pval"], cfg["min_corr"])
                        ss["sig_report"] = sig_report
                        ss["sig_cols"]   = sig_cols
                    except Exception as e:
                        st.warning(f"Significance test error: {e}")
        else:
            ss["exog_df"] = pd.DataFrame()

        # Features
        with st.spinner("Building feature matrix …"):
            frame = build_features(price_df, exog_df, cfg)
        ss["frame"] = frame

        all_results = {}

        # HMM
        if cfg["run_hmm"]:
            with st.spinner("HMM regime detection …"):
                try:
                    ss["hmm"] = model_hmm(frame, cfg)
                except Exception as e:
                    st.warning(f"HMM: {e}")
                    ss["hmm"] = None

        # GARCH
        if cfg["run_garch"]:
            with st.spinner("GARCH volatility …"):
                try:
                    ss["garch"] = model_garch(frame, cfg)
                except Exception as e:
                    st.warning(f"GARCH: {e}")
                    ss["garch"] = None

        # AR
        if cfg["run_ar"]:
            with st.spinner("AR/ARMA models …"):
                try:
                    ar_r = model_ar(frame, cfg)
                    ss["ar"] = ar_r
                    all_results.update(ar_r)
                except Exception as e:
                    st.warning(f"AR: {e}")
                    ss["ar"] = {}

        # Bayesian
        if cfg["run_bayes"]:
            with st.spinner("Bayesian Ridge …"):
                try:
                    br = model_bayesian(frame, cfg)
                    ss["bayes"] = br
                    if br:
                        all_results["BayesianRidge"] = br
                except Exception as e:
                    st.warning(f"Bayesian: {e}")
                    ss["bayes"] = None

        # SARIMAX
        if cfg["run_sarimax"]:
            sc = ss.get("sig_cols", [])
            if not sc:
                st.info("SARIMAX: no significant exog columns found. "
                        "Try lowering thresholds, or disable significance testing "
                        "to use all exog variables.")
                # fallback: use all exog columns available in frame
                sc = [c for c in exog_df.columns
                      if c in frame.df.columns and "close" not in c][:10]
            if sc:
                with st.spinner(f"SARIMAX with {len(sc)} exog variables …"):
                    try:
                        sr = model_sarimax(frame, cfg, sc)
                        ss["sarimax"] = sr
                        if sr and "error" not in sr:
                            all_results["SARIMAX+exog"] = sr
                    except Exception as e:
                        st.warning(f"SARIMAX: {e}")
                        ss["sarimax"] = None

        # Backtest
        if cfg["run_ar"]:
            with st.spinner("Walk-forward backtest …"):
                try:
                    ss["bt_df"] = model_backtest(frame, cfg)
                except Exception as e:
                    st.warning(f"Backtest: {e}")
                    ss["bt_df"] = pd.DataFrame()

        # Forecast
        with st.spinner("Generating forecast …"):
            try:
                fc = generate_forecast(
                    price_df, frame,
                    ss.get("ar", {}),
                    ss.get("bayes"),
                    ss.get("garch"),
                    ss.get("hmm"),
                    cfg,
                )
                ss["forecast"] = fc
            except Exception as e:
                st.warning(f"Forecast: {e}")
                ss["forecast"] = None

        ss["all_results"] = all_results
        st.success("✅ Analysis complete — see tabs above")

    # ── Render tabs ───────────────────────────────────────────────
    ss   = st.session_state
    cfg_ = ss.get("cfg", cfg)
    pdf  = ss.get("price_df", pd.DataFrame())

    with TABS[0]:
        if not pdf.empty:
            tab_data(pdf, cfg_, ss.get("is_live", False))
        else:
            st.info("Click **Run Analysis** in the sidebar to start.")

    with TABS[1]:
        tab_exog(ss.get("exog_df", pd.DataFrame()),
                 ss.get("exog_status", {}),
                 ss.get("sig_report"),
                 ss.get("sig_cols", []),
                 ss.get("frame").df if ss.get("frame") else pd.DataFrame(),
                 cfg_)

    with TABS[2]:
        tab_regimes(pdf if not pdf.empty else pd.DataFrame({"close":[1]}),
                    ss.get("hmm"))

    with TABS[3]:
        tab_volatility(pdf if not pdf.empty else pd.DataFrame({"close":[1]}),
                       ss.get("garch"))

    with TABS[4]:
        tab_ar_bayes(ss.get("ar", {}), ss.get("bayes"))

    with TABS[5]:
        tab_sarimax(ss.get("sarimax"))

    with TABS[6]:
        tab_backtest(ss.get("bt_df"))

    with TABS[7]:
        tab_comparison(ss.get("all_results", {}))

    with TABS[8]:
        tab_forecast(ss.get("forecast"), cfg_)


if __name__ == "__main__":
    main()
