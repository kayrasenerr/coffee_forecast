"""
backtesting/metrics.py
======================
Scoring metrics and model comparison utilities.

Metrics catalogue:
  Directional:  accuracy, precision, recall, F1
  Calibration:  Brier score, reliability diagram data, ECE
  Continuous:   RMSE, MAE, MAPE, IC (information coefficient)
  Risk-adjusted: Sharpe, Sortino, max drawdown of signal
  Regime:       per-regime accuracy breakdown

Overfitting detection:
  - Train vs test metric divergence
  - Permutation test for directional accuracy significance
  - Temporal autocorrelation of forecast errors
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def directional_accuracy(
    prob_up: np.ndarray,
    actuals: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Fraction of forecasts where direction is correct."""
    pred_up = prob_up > threshold
    actual_up = actuals > 0
    return float(np.mean(pred_up == actual_up))


def brier_score(prob_up: np.ndarray, actuals: np.ndarray) -> float:
    """Mean squared error of probability forecast. Range [0, 1]. Lower=better."""
    outcomes = (actuals > 0).astype(float)
    return float(np.mean((prob_up - outcomes) ** 2))


def brier_skill_score(
    prob_up: np.ndarray,
    actuals: np.ndarray,
    climatology: float = 0.5,
) -> float:
    """BSS relative to climatological forecast. Positive = better than climatology."""
    bs = brier_score(prob_up, actuals)
    bs_clim = brier_score(np.full_like(prob_up, climatology), actuals)
    return float(1 - bs / bs_clim) if bs_clim != 0 else float("nan")


def information_coefficient(
    forecasts: np.ndarray,
    actuals: np.ndarray,
) -> float:
    """Spearman rank correlation between forecast mean returns and actuals."""
    if len(forecasts) < 5:
        return float("nan")
    corr, _ = stats.spearmanr(forecasts, actuals, nan_policy="omit")
    return float(corr)


def calibration_curve(
    prob_up: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Reliability diagram data.
    Returns DataFrame with columns: bin_centre, mean_predicted, fraction_positive, count.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    outcomes = (actuals > 0).astype(float)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_up >= lo) & (prob_up < hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_centre": (lo + hi) / 2,
            "mean_predicted": float(prob_up[mask].mean()),
            "fraction_positive": float(outcomes[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def expected_calibration_error(
    prob_up: np.ndarray,
    actuals: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE: weighted average deviation from perfect calibration."""
    cal = calibration_curve(prob_up, actuals, n_bins)
    if cal.empty:
        return float("nan")
    n = cal["count"].sum()
    ece = (cal["count"] / n * (cal["mean_predicted"] - cal["fraction_positive"]).abs()).sum()
    return float(ece)


def signal_returns(
    prob_up: np.ndarray,
    actuals: np.ndarray,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
) -> pd.Series:
    """Compute PnL of threshold-based long/short signal."""
    signals = np.where(
        prob_up > long_threshold, 1.0,
        np.where(prob_up < short_threshold, -1.0, 0.0)
    )
    return pd.Series(signals * actuals)


def sharpe_ratio(returns: np.ndarray, annualise: int = 252) -> float:
    """Annualised Sharpe ratio."""
    r = np.array(returns)
    r = r[~np.isnan(r)]
    if len(r) < 3 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(annualise))


def sortino_ratio(returns: np.ndarray, annualise: int = 252) -> float:
    """Annualised Sortino ratio (penalises downside only)."""
    r = np.array(returns)
    r = r[~np.isnan(r)]
    downside = r[r < 0]
    if len(downside) < 3 or downside.std() == 0:
        return float("nan")
    return float(r.mean() / downside.std() * np.sqrt(annualise))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of equity curve."""
    cumulative = np.cumprod(1 + np.array(equity_curve))
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(drawdowns.min())


# ---------------------------------------------------------------------------
# Overfitting diagnostics
# ---------------------------------------------------------------------------

def permutation_test_da(
    prob_up: np.ndarray,
    actuals: np.ndarray,
    n_permutations: int = 1000,
    threshold: float = 0.5,
) -> Tuple[float, float]:
    """
    Permutation test for directional accuracy significance.

    Null hypothesis: forecasts have no predictive power (random).
    Returns: (observed_DA, p_value)
    """
    observed_da = directional_accuracy(prob_up, actuals, threshold)
    null_das = []
    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        shuffled = rng.permutation(prob_up)
        null_das.append(directional_accuracy(shuffled, actuals, threshold))
    p_value = float(np.mean(np.array(null_das) >= observed_da))
    return observed_da, p_value


def train_test_gap(
    train_metric: float,
    test_metric: float,
    higher_is_better: bool = True,
) -> Dict[str, float]:
    """
    Detect overfitting: compare train vs test metric.
    Returns gap and degradation ratio.
    """
    gap = train_metric - test_metric
    if train_metric != 0:
        ratio = test_metric / train_metric
    else:
        ratio = float("nan")
    sign = 1 if higher_is_better else -1
    overfitting_flag = bool(sign * gap > 0.1)  # >10% drop
    return {
        "train_metric": train_metric,
        "test_metric": test_metric,
        "gap": gap,
        "ratio": ratio,
        "overfitting_likely": overfitting_flag,
    }


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def compare_models(results: List[Dict]) -> pd.DataFrame:
    """
    Build a comparison table from a list of BacktestResult dicts or objects.

    Each item should have: model_name, directional_accuracy, mean_brier_score,
    rmse, signal_sharpe.
    """
    rows = []
    for r in results:
        if hasattr(r, "__dict__"):
            r = r.__dict__
        rows.append({
            "model": r.get("model_name", "unknown"),
            "variety": r.get("variety", ""),
            "DA": round(r.get("directional_accuracy", float("nan")), 4),
            "Brier": round(r.get("mean_brier_score", float("nan")) or float("nan"), 4),
            "RMSE": round(r.get("rmse", float("nan")) or float("nan"), 6),
            "MAE":  round(r.get("mae",  float("nan")) or float("nan"), 6),
            "Sharpe": round(r.get("signal_sharpe", float("nan")) or float("nan"), 3),
            "n_folds": r.get("n_folds", 0),
        })
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("DA", ascending=False)
