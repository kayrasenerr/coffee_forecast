"""
tests/evaluation/model_evaluation.py
======================================
Independent evaluation of every model on simulated KC=F Arabica futures data.
Calibrated GARCH simulator matching known 2022-2026 KC=F market characteristics.

Test window  : 2026-02-11 → 2026-05-12  (~90 calendar days out-of-sample)
Train window : 2022-05-13 → 2026-02-11  (~4 years)
Horizon      : 5-day log return
"""

from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
from datetime import date
from typing import Dict, List, Tuple
from scipy import stats as sp
from contracts.schemas import CoffeeVariety, DataFrequency, FeatureFrame

TRAIN_START  = date(2022, 5, 13)
TEST_START   = date(2026, 2, 11)
TEST_END     = date(2026, 5, 12)
HORIZON      = 5          # days
LONG_TH      = 0.55
SHORT_TH     = 0.45
RESULTS_DIR  = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# 1. SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────

def _garch_path(n, start, mu, omega, alpha, beta, df_t, seed):
    rng = np.random.default_rng(seed)
    s2  = np.zeros(n);  s2[0] = omega / max(1 - alpha - beta, 1e-8)
    r   = np.zeros(n)
    for t in range(1, n):
        eps  = rng.standard_t(df_t) * np.sqrt(s2[t-1])
        r[t] = mu + eps
        s2[t] = omega + alpha * eps**2 + beta * s2[t-1]
    return start * np.exp(np.cumsum(r)), r


def build_raw(seed: int = 42) -> Dict[str, pd.DataFrame]:
    idx = pd.bdate_range(str(TRAIN_START), str(TEST_END))
    n   = len(idx)
    rng = np.random.default_rng(seed)

    # Arabica – calibrated drift = log(350/215)/n ≈ 0.000467 /day
    drift = np.log(350.0 / 215.0) / n
    arab_p, arab_r = _garch_path(n, 215.0, drift, 1e-5, 0.07, 0.88, 5, seed + 10)
    arab_p = np.maximum(arab_p, 150.0)

    rob_p, _ = _garch_path(n, 2200.0, 0.0002, 8e-6, 0.06, 0.88, 5, seed + 20)

    brl_r = -0.45 * arab_r + np.sqrt(1 - 0.45**2) * rng.normal(0, 0.006, n)
    brl   = np.clip(5.3 * np.exp(np.cumsum(brl_r * 0.3)), 4.2, 8.0)
    eur   = np.clip(1.08 + np.cumsum(rng.normal(0, 0.003, n)), 0.90, 1.22)

    enso_idx = pd.date_range(str(TRAIN_START), str(TEST_END), freq="MS")
    oni = np.zeros(len(enso_idx)); oni[0] = 0.1
    for t in range(1, len(oni)):
        oni[t] = 0.75 * oni[t-1] + rng.normal(0, 0.35)
    oni[18:30] += np.linspace(0, 1.8, 12)
    oni = np.clip(oni, -2.5, 2.5)

    cot_idx = pd.bdate_range(str(TRAIN_START), str(TEST_END), freq="W-TUE")
    nc = len(cot_idx)
    cot = pd.DataFrame({
        "noncommercial_long":  np.clip(45000 + rng.normal(0, 3000, nc), 10000, 90000),
        "noncommercial_short": np.clip(35000 + rng.normal(0, 2500, nc), 10000, 70000),
        "commercial_long":     np.clip(48000 + rng.normal(0, 3000, nc), 15000, 100000),
        "commercial_short":    np.clip(72000 + rng.normal(0, 4000, nc), 20000, 130000),
        "nonreportable_long":  rng.integers(4000, 14000, nc).astype(float),
        "nonreportable_short": rng.integers(4000, 14000, nc).astype(float),
        "open_interest":       np.clip(160000 + rng.normal(0, 8000, nc), 80000, 250000),
    }, index=cot_idx)

    def ohlcv(p, seed_offset):
        r2 = np.random.default_rng(seed + seed_offset)
        return pd.DataFrame({
            "open":  p * r2.uniform(0.997, 1.003, n),
            "high":  p * r2.uniform(1.002, 1.012, n),
            "low":   p * r2.uniform(0.988, 0.998, n),
            "close": p,
            "volume": r2.integers(8000, 60000, n).astype(float),
        }, index=idx)

    return {
        "arabica_futures": ohlcv(arab_p, 1),
        "robusta_futures": ohlcv(rob_p, 2),
        "fx_usdbrl": pd.DataFrame({"rate": brl}, index=idx),
        "fx_eurusd": pd.DataFrame({"rate": eur}, index=idx),
        "enso": pd.DataFrame({"oni": oni}, index=enso_idx),
        "cot_arabica": cot,
    }


# ─────────────────────────────────────────────────────────────
# 2. FEATURES
# ─────────────────────────────────────────────────────────────

def build_features(raw: Dict) -> FeatureFrame:
    from features.pipeline import FeaturePipeline
    return FeaturePipeline().run(raw, CoffeeVariety.ARABICA, DataFrequency.DAILY)


def split(frame: FeatureFrame):
    cutoff   = pd.Timestamp(TEST_START)
    train_df = frame.df[frame.df.index <  cutoff]
    test_df  = frame.df[frame.df.index >= cutoff]
    def sub(df):
        return FeatureFrame(variety=frame.variety, frequency=frame.frequency,
                            feature_names=frame.feature_names, df=df)
    return sub(train_df), sub(test_df)


# ─────────────────────────────────────────────────────────────
# 3. SHARED METRICS
# ─────────────────────────────────────────────────────────────

def score(label, prob_up, actuals, extras=None):
    if not prob_up:
        return {"label": label, "n": 0, "error": "no predictions"}
    p, a = np.array(prob_up), np.array(actuals)
    # directional accuracy
    da  = float(np.mean((p > 0.5) == (a > 0)))
    # brier
    bs  = float(np.mean((p - (a > 0).astype(float))**2))
    bss = float(1 - bs / 0.25)
    # IC
    ic  = float(np.corrcoef(p, a)[0, 1]) if len(p) > 3 else np.nan
    # signal returns
    sig = np.where(p > LONG_TH, 1.0, np.where(p < SHORT_TH, -1.0, 0.0))
    pnl = sig * a
    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else np.nan
    # permutation test
    rng   = np.random.default_rng(99)
    null  = [float(np.mean((rng.permutation(p) > 0.5) == (a > 0))) for _ in range(500)]
    pval  = float(np.mean(np.array(null) >= da))
    # calibration (5 bins)
    cal_rows = []
    for lo, hi in zip(np.linspace(0,1,6)[:-1], np.linspace(0,1,6)[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() > 0:
            cal_rows.append({"bin": f"{lo:.1f}-{hi:.1f}",
                             "pred": round(float(p[m].mean()),3),
                             "actual": round(float((a[m]>0).mean()),3),
                             "n": int(m.sum())})
    result = {
        "label": label, "n": len(p),
        "n_long": int((p > LONG_TH).sum()),
        "n_short": int((p < SHORT_TH).sum()),
        "directional_accuracy": round(da, 4),
        "da_vs_random": round(da - 0.5, 4),
        "brier_score": round(bs, 4),
        "brier_skill_score": round(bss, 4),
        "information_coefficient": round(ic, 4) if not np.isnan(ic) else None,
        "signal_sharpe_ann": round(sharpe, 3) if not np.isnan(sharpe) else None,
        "permutation_pvalue": round(pval, 4),
        "significant_5pct": bool(pval < 0.05),
        "calibration": cal_rows,
    }
    if extras:
        result.update(extras)
    return result


# ─────────────────────────────────────────────────────────────
# 4. HMM
# ─────────────────────────────────────────────────────────────

def eval_hmm(train_frame, test_frame):
    from models.hmm_model import HMMRegimeDetector
    results = {}
    FEATS = ["log_return_1d", "realised_vol_21d", "log_return_5d", "price_z_63d"]
    for k in [2, 3]:
        det = HMMRegimeDetector(n_regimes=k, variety=CoffeeVariety.ARABICA,
                                feature_cols=FEATS, n_iter=300, random_state=0)
        det.fit(train_frame)
        snaps_test = det.predict(test_frame)
        snaps_all  = det.predict(train_frame)

        # Regime-conditional annualised return (in-sample)
        ret_col = "log_return_1d"
        regime_rets = {}
        if ret_col in train_frame.df.columns:
            rets = train_frame.df[ret_col].dropna()
            avail = [f for f in FEATS if f in train_frame.df.columns]
            clean = train_frame.df[avail].dropna()
            states = det._model.predict(clean.values)
            for s in range(k):
                mask = states == s
                label = det._regime_map.get(s, "?").value if hasattr(det._regime_map.get(s,"?"), "value") else str(det._regime_map.get(s,"?"))
                r_sub = rets.reindex(clean.index[mask]).dropna()
                if len(r_sub):
                    regime_rets[label] = round(float(r_sub.mean() * 252), 4)

        runs = []
        if snaps_test:
            cur, cnt = snaps_test[0].regime_id, 1
            for s in snaps_test[1:]:
                if s.regime_id == cur: cnt += 1
                else: runs.append(cnt); cur, cnt = s.regime_id, 1
            runs.append(cnt)

        results[f"hmm_k{k}"] = {
            "n_regimes": k,
            "converged": bool(det._model.monitor_.converged),
            "log_likelihood": round(float(det._model.score(
                train_frame.df[[f for f in FEATS if f in train_frame.df.columns]].dropna().values)), 2),
            "train_snapshots": len(snaps_all),
            "test_snapshots": len(snaps_test),
            "mean_confidence": round(float(np.mean([s.probability for s in snaps_test])), 4) if snaps_test else None,
            "avg_run_length_test": round(float(np.mean(runs)), 2) if runs else None,
            "regime_ann_returns_train": regime_rets,
            "current_regime": snaps_test[-1].regime.value if snaps_test else "n/a",
            "current_confidence": round(snaps_test[-1].probability, 4) if snaps_test else None,
            "transition_matrix": det.transition_matrix().round(4).to_dict(),
        }
    return results


# ─────────────────────────────────────────────────────────────
# 5. GARCH
# ─────────────────────────────────────────────────────────────

def eval_garch(train_frame, test_frame):
    from models.garch_model import GARCHVolatilityModel
    results = {}

    # Build aligned return series
    all_df   = pd.concat([train_frame.df, test_frame.df])
    ret_col  = "log_return_1d"
    all_rets = all_df[ret_col].dropna() if ret_col in all_df.columns else pd.Series(dtype=float)
    cut      = pd.Timestamp(TEST_START)
    tr_rets  = all_rets[all_rets.index <  cut]
    te_rets  = all_rets[all_rets.index >= cut]

    for spec, mtype in [("garch11", "GARCH"), ("egarch11", "EGARCH")]:
        m = GARCHVolatilityModel(p=1, q=1, model_type=mtype, dist="t",
                                 variety=CoffeeVariety.ARABICA)
        m.fit(tr_rets)
        cv  = m.conditional_volatility()
        diag = m.residual_diagnostics()

        # OOS: rolling 21-day realised vol vs 5-day-ahead forecast
        forecasts_v, realised_v, regimes = [], [], []
        dates = te_rets.index[5::3]
        for dt in dates:
            loc = all_rets.index.get_loc(dt) if dt in all_rets.index else None
            if loc is None or loc < 100:
                continue
            window = all_rets.iloc[max(0, loc-252):loc]
            try:
                fc = m.forecast(window, horizon_days=5)
                rv = float(te_rets.loc[dt:].iloc[:5].std() * np.sqrt(252)) if len(te_rets.loc[dt:]) >= 5 else np.nan
                forecasts_v.append(fc.forecast_annualized_vol)
                realised_v.append(rv)
                regimes.append(fc.vol_regime)
            except Exception:
                continue

        fv = np.array(forecasts_v); rv = np.array(realised_v)
        valid = ~(np.isnan(fv) | np.isnan(rv))
        rmse = float(np.sqrt(np.mean((fv[valid]-rv[valid])**2))) if valid.sum() > 0 else np.nan
        mae  = float(np.mean(np.abs(fv[valid]-rv[valid])))       if valid.sum() > 0 else np.nan
        corr = float(np.corrcoef(fv[valid], rv[valid])[0,1])     if valid.sum() > 2 else np.nan

        latest_fc = m.forecast(tr_rets.iloc[-252:], horizon_days=10)
        results[spec] = {
            "model": f"{mtype}(1,1)-t",
            "aic": round(m._result.aic, 2),
            "bic": round(m._result.bic, 2),
            "n_forecasts": int(valid.sum()),
            "forecast_rmse_ann": round(rmse, 4),
            "forecast_mae_ann":  round(mae,  4),
            "forecast_corr":     round(corr, 4),
            "current_vol_ann":   round(latest_fc.current_annualized_vol, 4),
            "forecast_vol_10d":  round(latest_fc.forecast_annualized_vol, 4),
            "current_vol_regime": latest_fc.vol_regime,
            "no_remaining_arch": diag.get("no_remaining_arch"),
            "vol_regime_dist": pd.Series(regimes).value_counts().to_dict() if regimes else {},
        }
    return results


# ─────────────────────────────────────────────────────────────
# 6. AR / ARMA (statsmodels)
# ─────────────────────────────────────────────────────────────

def eval_ar(train_frame, test_frame):
    from statsmodels.tsa.ar_model import AutoReg

    TARGET = "target_log_return_5d"
    tr_df  = train_frame.df.dropna(subset=[TARGET])
    te_df  = test_frame.df.dropna(subset=[TARGET])
    if len(te_df) == 0:
        return {}

    # Combined series for AR state at each test bar
    full_y = pd.concat([tr_df[TARGET], te_df[TARGET]]).sort_index()
    full_arr = full_y.values
    full_idx = {t: i for i, t in enumerate(full_y.index)}

    results = {}
    for lags, label in [(1, "AR(1)"), (2, "AR(2)"), (5, "AR(5)")]:
        try:
            m    = AutoReg(tr_df[TARGET].values, lags=lags, old_names=False).fit()
            coef = m.params   # [const, phi_1, ..., phi_p]
            c    = float(coef[0])
            phis = [float(coef[i+1]) for i in range(lags)]
            res_std  = float(m.resid.std())
            h_std    = max(res_std * np.sqrt(HORIZON), 1e-8)
        except Exception as e:
            results[label] = {"label": label, "n": 0, "error": str(e)}
            continue

        prob_ups, actuals = [], []
        for dt in te_df.index[::2]:
            pos = full_idx.get(dt)
            if pos is None or pos < lags:
                continue
            forecast = c + sum(phis[i] * full_arr[pos - 1 - i] for i in range(lags))
            prob_up  = float(1 - sp.norm.cdf(0, loc=forecast * HORIZON, scale=h_std))
            actual   = float(te_df.loc[dt, TARGET])
            prob_ups.append(prob_up)
            actuals.append(actual)

        results[label] = score(label, prob_ups, actuals, {
            "aic": round(m.aic, 2),
            "bic": round(m.bic, 2),
            "resid_std": round(res_std, 6),
        })
    return results


# ─────────────────────────────────────────────────────────────
# 7. BAYESIAN RIDGE
# ─────────────────────────────────────────────────────────────

def eval_bayesian(train_frame, test_frame):
    from models.bayesian_model import BayesianForecaster

    TARGET = "target_log_return_5d"
    tr_df  = train_frame.df.dropna(subset=[TARGET])
    te_df  = test_frame.df.dropna(subset=[TARGET])
    if len(te_df) == 0:
        return {"BayesianRidge": {"label": "BayesianRidge", "n": 0, "error": "no test data"}}

    m = BayesianForecaster(variety=CoffeeVariety.ARABICA)
    m.fit(FeatureFrame(variety=train_frame.variety, frequency=train_frame.frequency,
                       feature_names=train_frame.feature_names, df=tr_df), TARGET)

    avail = [c for c in m._fitted_cols if c in te_df.columns]
    prob_ups, actuals = [], []
    for dt in te_df.index[::2]:
        try:
            row  = te_df[avail].loc[:dt].ffill().iloc[-1:].fillna(0)
            X    = m._scaler.transform(row.values)
            mu, sigma = m._model.predict(X, return_std=True)
            h_std = max(float(sigma[0]) * np.sqrt(HORIZON), 1e-8)
            prob_up = float(1 - sp.norm.cdf(0, loc=float(mu[0]), scale=h_std))
            prob_ups.append(prob_up)
            actuals.append(float(te_df.loc[dt, TARGET]))
        except Exception:
            continue

    top5 = m.feature_importance().head(5).round(5).to_dict()
    return {"BayesianRidge": score("BayesianRidge", prob_ups, actuals, {"top5_features": top5})}


# ─────────────────────────────────────────────────────────────
# 8. ENSEMBLE
# ─────────────────────────────────────────────────────────────

def eval_ensemble(train_frame, test_frame):
    from statsmodels.tsa.ar_model import AutoReg
    from models.bayesian_model import BayesianForecaster
    from models.hmm_model import HMMRegimeDetector

    TARGET = "target_log_return_5d"
    tr_df  = train_frame.df.dropna(subset=[TARGET])
    te_df  = test_frame.df.dropna(subset=[TARGET])
    if len(te_df) == 0:
        return {}

    # Pre-fit component models
    ar1   = AutoReg(tr_df[TARGET].values, lags=1, old_names=False).fit()
    ar1_c, ar1_phi = float(ar1.params[0]), float(ar1.params[1])
    ar1_hstd = max(float(ar1.resid.std()) * np.sqrt(HORIZON), 1e-8)

    bayes = BayesianForecaster(variety=CoffeeVariety.ARABICA)
    bayes.fit(FeatureFrame(variety=train_frame.variety, frequency=train_frame.frequency,
                           feature_names=train_frame.feature_names, df=tr_df), TARGET)
    b_avail = [c for c in bayes._fitted_cols if c in te_df.columns]

    FEATS = ["log_return_1d", "realised_vol_21d", "log_return_5d", "price_z_63d"]
    hmm   = HMMRegimeDetector(n_regimes=3, variety=CoffeeVariety.ARABICA,
                               feature_cols=FEATS, n_iter=300, random_state=0)
    hmm.fit(train_frame)

    full_y   = pd.concat([tr_df[TARGET], te_df[TARGET]]).sort_index()
    full_arr = full_y.values
    full_map = {t: i for i, t in enumerate(full_y.index)}

    # Collect per-bar predictions from all three models
    pred_equal, pred_regime = [], []
    actuals = []

    for dt in te_df.index[::2]:
        pos = full_map.get(dt)
        if pos is None or pos < 1:
            continue

        # AR(1) component
        ar_fc  = ar1_c + ar1_phi * full_arr[pos - 1]
        p_ar   = float(1 - sp.norm.cdf(0, loc=ar_fc * HORIZON, scale=ar1_hstd))

        # Bayesian component
        try:
            row   = te_df[b_avail].loc[:dt].ffill().iloc[-1:].fillna(0)
            X     = bayes._scaler.transform(row.values)
            mu_b, sig_b = bayes._model.predict(X, return_std=True)
            h_b   = max(float(sig_b[0]) * np.sqrt(HORIZON), 1e-8)
            p_bay = float(1 - sp.norm.cdf(0, loc=float(mu_b[0]), scale=h_b))
        except Exception:
            p_bay = 0.5

        # HMM regime for regime-switching ensemble
        try:
            ctx_df = pd.concat([train_frame.df, test_frame.df.loc[:dt]])
            ctx_fr = FeatureFrame(variety=train_frame.variety, frequency=train_frame.frequency,
                                  feature_names=train_frame.feature_names, df=ctx_df)
            snap = hmm.predict_latest(ctx_fr)
            from contracts.schemas import MarketRegime
            if snap.regime == MarketRegime.BULL:
                p_regime = 0.6 * p_bay + 0.4 * p_ar
            elif snap.regime == MarketRegime.BEAR:
                p_regime = 0.4 * p_bay + 0.6 * p_ar
            else:
                p_regime = 0.5 * p_bay + 0.5 * p_ar
        except Exception:
            p_regime = 0.5 * p_bay + 0.5 * p_ar

        p_equal = 0.5 * p_ar + 0.5 * p_bay
        pred_equal.append(p_equal)
        pred_regime.append(p_regime)
        actuals.append(float(te_df.loc[dt, TARGET]))

    return {
        "ensemble_equal_weight": score("ensemble_equal_weight", pred_equal, actuals),
        "ensemble_regime_switch": score("ensemble_regime_switch", pred_regime, actuals),
    }


# ─────────────────────────────────────────────────────────────
# 9. FEATURE DIAGNOSTICS
# ─────────────────────────────────────────────────────────────

def eval_features(train_frame):
    from statsmodels.tsa.stattools import adfuller
    TARGET = "target_log_return_5d"
    df  = train_frame.df.dropna(subset=[TARGET])
    tgt = df[TARGET]
    rows = []
    for col in df.columns:
        if col.startswith("target"):
            continue
        s = df[col].dropna()
        if len(s) < 50:
            continue
        corr = float(s.corr(tgt.reindex(s.index)))
        try:
            _, pval, *_ = adfuller(s, autolag="AIC", maxlag=10)
            stationary = bool(pval < 0.05)
        except Exception:
            stationary = None
        rows.append({"feature": col, "corr": round(corr, 4),
                     "abs_corr": round(abs(corr), 4), "stationary": stationary})
    rows.sort(key=lambda x: x["abs_corr"], reverse=True)
    n_stat = sum(1 for r in rows if r["stationary"])
    return {"n_features": len(rows), "n_stationary": n_stat, "ranked": rows}


# ─────────────────────────────────────────────────────────────
# 10. REPORT
# ─────────────────────────────────────────────────────────────

def report(all_r: Dict) -> str:
    lines = []
    W = 72

    def hdr(t):
        return [f"\n{'='*W}", f"  {t}", f"{'='*W}"]
    def kv(k, v, w=42):
        return f"    {k:<{w}} {v}"
    def metric_line(k, v, good="high"):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return kv(k, "n/a")
        if isinstance(v, float):
            mark = ""
            if "accuracy" in k and v > 0.52: mark = "  ✓"
            elif "sharpe"  in k and v > 0:   mark = "  ✓"
            elif "bss"     in k and v > 0:   mark = "  ✓"
            return kv(k, f"{v:>10.4f}{mark}")
        return kv(k, str(v))

    lines += [
        "╔" + "═"*W + "╗",
        f"║{'COFFEE QUANT — MODEL EVALUATION REPORT':^{W}}║",
        f"║{'Test : 2026-02-11 → 2026-05-12  (~90 days OOS)':^{W}}║",
        f"║{'Train: 2022-05-13 → 2026-02-11  (~4 years)':^{W}}║",
        f"║{'Data : Calibrated GARCH Simulator (KC=F params)':^{W}}║",
        f"║{'Horizon: 5-day log return  |  Permutations: 500':^{W}}║",
        "╚" + "═"*W + "╝",
    ]

    # Feature diagnostics
    if "features" in all_r:
        fr = all_r["features"]
        lines += hdr("FEATURE DIAGNOSTICS")
        lines.append(kv("Total features computed", fr["n_features"]))
        lines.append(kv("Stationary (ADF p<0.05)", fr["n_stationary"]))
        lines.append(f"\n    {'Feature':<38} {'Corr':>8}  {'Stationary'}")
        lines.append("    " + "─"*56)
        for r in fr["ranked"][:12]:
            stat = "Yes" if r["stationary"] else "No " if r["stationary"] is False else "n/a"
            lines.append(f"    {r['feature']:<38} {r['corr']:>8.4f}  {stat}")

    # HMM
    if "hmm" in all_r:
        lines += hdr("1. HMM REGIME DETECTOR")
        for key, r in all_r["hmm"].items():
            lines.append(f"\n  ── {key.upper()} (n_regimes={r['n_regimes']}) ──")
            lines.append(kv("Converged", r["converged"]))
            lines.append(kv("Log-likelihood (train)", r["log_likelihood"]))
            lines.append(kv("Test snapshots", r["test_snapshots"]))
            if r.get("mean_confidence"):
                lines.append(kv("Mean confidence (test)", f"{r['mean_confidence']:.4f}"))
            if r.get("avg_run_length_test"):
                lines.append(kv("Avg regime run length (test)", f"{r['avg_run_length_test']:.1f} bars"))
            lines.append(kv("Current regime (last test bar)", r["current_regime"]))
            if r.get("current_confidence"):
                lines.append(kv("Current confidence", f"{r['current_confidence']:.4f}"))
            if r.get("regime_ann_returns_train"):
                lines.append("  Annualised return by regime (train):")
                for rn, rv in r["regime_ann_returns_train"].items():
                    lines.append(f"    {rn:<25}  {rv:+.4f}")
            tm = r.get("transition_matrix", {})
            if tm:
                lines.append("  Transition matrix (rows=from, cols=to):")
                ks = list(tm.keys())
                lines.append("    " + " ".join(f"{k:>14}" for k in [""]+ks))
                for rk, rv2 in tm.items():
                    row_str = " ".join(f"{rv2[c]:>14.4f}" for c in ks)
                    lines.append(f"    {rk:<14}" + row_str)

    # GARCH
    if "garch" in all_r:
        lines += hdr("2. GARCH VOLATILITY MODELS")
        for key, r in all_r["garch"].items():
            lines.append(f"\n  ── {r['model']} ──")
            lines.append(kv("AIC", r["aic"]))
            lines.append(kv("BIC", r["bic"]))
            lines.append(kv("N OOS forecasts", r["n_forecasts"]))
            lines.append(kv("Vol Forecast RMSE (ann)", f"{r['forecast_rmse_ann']:.4f}"))
            lines.append(kv("Vol Forecast MAE  (ann)", f"{r['forecast_mae_ann']:.4f}"))
            lines.append(kv("Vol Forecast Corr",       f"{r['forecast_corr']:.4f}"))
            lines.append(kv("Current cond. vol (ann)", f"{r['current_vol_ann']:.2%}"))
            lines.append(kv("10-day forecast vol",     f"{r['forecast_vol_10d']:.2%}"))
            lines.append(kv("Current vol regime",      r["current_vol_regime"]))
            lines.append(kv("No remaining ARCH",       r["no_remaining_arch"]))
            if r.get("vol_regime_dist"):
                lines.append("  Vol regime distribution over test period:")
                for vr, cnt in sorted(r["vol_regime_dist"].items()):
                    lines.append(f"    {vr:<12}  {cnt} bars")

    # AR / ARMA
    if "ar" in all_r:
        lines += hdr("3. AR / ARMA MOMENTUM MODELS")
        for label, r in all_r["ar"].items():
            lines.append(f"\n  ── {label} ──")
            _dir_block(lines, r)

    # Bayesian
    if "bayesian" in all_r:
        lines += hdr("4. BAYESIAN RIDGE FORECASTER")
        for label, r in all_r["bayesian"].items():
            lines.append(f"\n  ── {label} ──")
            _dir_block(lines, r)
            if "top5_features" in r:
                lines.append("  Top-5 features (|posterior coef|):")
                for fn, fv in r["top5_features"].items():
                    lines.append(f"    {fn:<38}  {fv:.6f}")

    # Ensemble
    if "ensemble" in all_r:
        lines += hdr("5. ENSEMBLE FORECASTERS")
        for label, r in all_r["ensemble"].items():
            lines.append(f"\n  ── {label} ──")
            _dir_block(lines, r)

    # Summary table
    lines += hdr("SUMMARY — ALL DIRECTIONAL MODELS")
    lines.append(f"  {'Model':<35} {'DA':>7} {'Brier':>8} {'BSS':>7} {'Sharpe':>8} {'p-val':>7} {'Sig':>4} {'N':>5}")
    lines.append("  " + "─"*77)
    rows = []
    for sec in ["ar", "bayesian", "ensemble"]:
        if sec in all_r:
            for label, r in all_r[sec].items():
                if "n" not in r or r["n"] == 0:
                    continue
                rows.append(r)
    rows.sort(key=lambda x: x.get("directional_accuracy", 0), reverse=True)
    for r in rows:
        da  = r.get("directional_accuracy", float("nan"))
        bs  = r.get("brier_score",          float("nan"))
        bss = r.get("brier_skill_score",     float("nan"))
        sh  = r.get("signal_sharpe_ann") or float("nan")
        pv  = r.get("permutation_pvalue",   float("nan"))
        n   = r.get("n", 0)
        sig = "  *" if pv < 0.05 else "   "
        lines.append(
            f"  {r['label']:<35} {da:>7.4f} {bs:>8.4f} {bss:>7.4f}"
            f" {sh:>8.3f} {pv:>7.4f}{sig} {n:>5}"
        )
    lines.append("")
    lines.append("  * = statistically significant at 5% level (permutation test, n=500)")
    lines.append("  Baseline: DA=0.5000  Brier=0.2500  BSS=0.0000  Sharpe=0.000")
    lines.append("")
    return "\n".join(lines)


def _dir_block(lines, r):
    if r.get("error"):
        lines.append(f"    ERROR: {r['error']}")
        return
    lines.append(kv("N forecasts",              r.get("n"), 42))
    lines.append(kv("N long / N short",         f"{r.get('n_long')} / {r.get('n_short')}", 42))
    lines.append(kv("Directional accuracy",     f"{r.get('directional_accuracy', 0):.4f}", 42))
    lines.append(kv("DA vs random (±0.5000)",   f"{r.get('da_vs_random', 0):+.4f}", 42))
    lines.append(kv("Brier score (↓ better)",   f"{r.get('brier_score', 0):.4f}", 42))
    lines.append(kv("Brier skill score (↑ >0)", f"{r.get('brier_skill_score', 0):.4f}", 42))
    ic = r.get("information_coefficient")
    lines.append(kv("Information coefficient",  f"{ic:.4f}" if ic else "n/a", 42))
    sh = r.get("signal_sharpe_ann")
    lines.append(kv("Signal Sharpe (ann)",       f"{sh:.3f}" if sh else "n/a", 42))
    lines.append(kv("Permutation p-value",       f"{r.get('permutation_pvalue', 1):.4f}", 42))
    lines.append(kv("Significant at 5%?",        r.get("significant_5pct", False), 42))
    cal = r.get("calibration", [])
    if cal:
        lines.append("  Calibration (predicted prob → actual hit rate):")
        for b in cal:
            bar = "█" * int(b["actual"] * 20)
            lines.append(f"    {b['bin']:>9}  pred={b['pred']:.2f}  actual={b['actual']:.2f}  n={b['n']:3d}  {bar}")


def kv(k, v, w=42):
    return f"    {k:<{w}} {v}"


def build_csv(all_r: Dict) -> pd.DataFrame:
    rows = []
    for sec in ["ar", "bayesian", "ensemble"]:
        if sec not in all_r:
            continue
        for label, r in all_r[sec].items():
            rows.append({
                "section": sec, "model": label,
                "n_forecasts": r.get("n"),
                "directional_accuracy": r.get("directional_accuracy"),
                "brier_score": r.get("brier_score"),
                "brier_skill_score": r.get("brier_skill_score"),
                "information_coefficient": r.get("information_coefficient"),
                "signal_sharpe_ann": r.get("signal_sharpe_ann"),
                "permutation_pvalue": r.get("permutation_pvalue"),
                "significant_5pct": r.get("significant_5pct"),
                "n_long": r.get("n_long"),
                "n_short": r.get("n_short"),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run():
    print("\n" + "="*72)
    print("  COFFEE QUANT — FULL MODEL EVALUATION")
    print(f"  Test : {TEST_START} → {TEST_END}")
    print(f"  Train: {TRAIN_START} → {TEST_START}")
    print("="*72)

    print("\n[1/8] Generating calibrated market data …")
    raw = build_raw(seed=42)
    close = raw["arabica_futures"]["close"]
    print(f"      Bars={len(close)}  price={close.iloc[0]:.1f}→{close.iloc[-1]:.1f} cts/lb  "
          f"range={close.min():.1f}–{close.max():.1f}")

    print("\n[2/8] Building feature matrix …")
    frame = build_features(raw)
    train_frame, test_frame = split(frame)
    print(f"      Train={len(train_frame.df)} bars  Test={len(test_frame.df)} bars  "
          f"Features={len(frame.feature_names)}")

    all_r = {}

    print("\n[3/8] Feature diagnostics …")
    all_r["features"] = eval_features(train_frame)
    print(f"      {all_r['features']['n_features']} features  "
          f"{all_r['features']['n_stationary']} stationary")

    print("\n[4/8] HMM regime detection …")
    all_r["hmm"] = eval_hmm(train_frame, test_frame)
    for k, v in all_r["hmm"].items():
        print(f"      {k}: regime={v['current_regime']}  "
              f"conf={v.get('current_confidence','?')}  "
              f"converged={v['converged']}")

    print("\n[5/8] GARCH volatility …")
    all_r["garch"] = eval_garch(train_frame, test_frame)
    for k, v in all_r["garch"].items():
        print(f"      {v['model']}: vol={v['current_vol_ann']:.1%}  "
              f"regime={v['current_vol_regime']}  RMSE={v['forecast_rmse_ann']:.4f}")

    print("\n[6/8] AR/ARMA models …")
    all_r["ar"] = eval_ar(train_frame, test_frame)
    for label, r in all_r["ar"].items():
        da = r.get("directional_accuracy", float("nan"))
        pv = r.get("permutation_pvalue",   float("nan"))
        n  = r.get("n", 0)
        print(f"      {label:<12} DA={da:.4f}  p={pv:.4f}  n={n}")

    print("\n[7/8] Bayesian Ridge …")
    all_r["bayesian"] = eval_bayesian(train_frame, test_frame)
    for label, r in all_r["bayesian"].items():
        da = r.get("directional_accuracy", float("nan"))
        pv = r.get("permutation_pvalue",   float("nan"))
        sh = r.get("signal_sharpe_ann") or float("nan")
        n  = r.get("n", 0)
        print(f"      {label:<20} DA={da:.4f}  Sharpe={sh:.3f}  p={pv:.4f}  n={n}")

    print("\n[8/8] Ensemble strategies …")
    all_r["ensemble"] = eval_ensemble(train_frame, test_frame)
    for label, r in all_r["ensemble"].items():
        da = r.get("directional_accuracy", float("nan"))
        pv = r.get("permutation_pvalue",   float("nan"))
        n  = r.get("n", 0)
        print(f"      {label:<30} DA={da:.4f}  p={pv:.4f}  n={n}")

    # Write outputs
    rpt_txt = RESULTS_DIR / "evaluation_report.txt"
    csv_out = RESULTS_DIR / "metrics_table.csv"
    rpt = report(all_r)
    with open(rpt_txt, "w") as f:
        f.write(rpt)
    build_csv(all_r).to_csv(csv_out, index=False)

    print(f"\n{'='*72}")
    print(f"  Results written to:")
    print(f"    {rpt_txt}")
    print(f"    {csv_out}")
    print(f"{'='*72}\n")
    print(rpt)
    return all_r


if __name__ == "__main__":
    run()
