"""
visualization/plots.py
======================
Modular plotting functions for the Coffee Quant system.

All functions are pure: they accept data, return matplotlib figures.
No side effects. No global state. No hardcoded paths.

Plot catalogue:
  plot_regime_overlay      — price series with HMM regime colouring
  plot_forecast_fan        — probabilistic fan chart
  plot_volatility_bands    — GARCH conditional vol over price
  plot_calibration         — reliability diagram
  plot_feature_importance  — bar chart of Bayesian coefficients
  plot_cot_positioning     — COT net position with extremes flagged
  plot_enso_coffee_lag     — ENSO ONI vs coffee returns (lagged)
  plot_backtest_folds      — walk-forward fold performance
  plot_model_comparison    — metric comparison bar chart

Style: minimal, publication-quality, colour-blind safe.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Lazy import matplotlib to avoid import errors in headless environments
def _get_plt():
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    return plt, mpatches


# Colour-blind safe regime palette
_REGIME_COLORS = {
    "bull":           "#2196F3",   # blue
    "bear":           "#F44336",   # red
    "volatile":       "#FF9800",   # orange
    "supply_stress":  "#9C27B0",   # purple
    "neutral":        "#9E9E9E",   # grey
    "low_vol":        "#4CAF50",   # green
    "contango":       "#00BCD4",
    "backwardation":  "#FF5722",
}


def plot_regime_overlay(
    prices: pd.Series,
    regimes: pd.Series,                # DatetimeIndex → regime string
    title: str = "Coffee Futures — Regime Detection",
    figsize: Tuple[int, int] = (14, 6),
) -> "matplotlib.figure.Figure":
    """
    Price series with regime background colouring.

    Parameters
    ----------
    prices  : close price series
    regimes : Series of regime strings aligned to prices index
    """
    plt, mpatches = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)

    # Background regime shading
    aligned = regimes.reindex(prices.index, method="ffill")
    prev_regime, start_i = None, 0
    for i, (ts, regime) in enumerate(aligned.items()):
        if regime != prev_regime or i == len(aligned) - 1:
            if prev_regime is not None:
                color = _REGIME_COLORS.get(str(prev_regime), "#EEEEEE")
                ax.axvspan(
                    prices.index[start_i], prices.index[min(i, len(prices) - 1)],
                    alpha=0.15, color=color, linewidth=0,
                )
            start_i = i
            prev_regime = regime

    ax.plot(prices.index, prices.values, color="black", linewidth=1.2, label="Close Price")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)

    # Legend
    patches = [
        mpatches.Patch(color=c, alpha=0.4, label=r.replace("_", " ").title())
        for r, c in _REGIME_COLORS.items()
        if r in aligned.values
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_forecast_fan(
    history: pd.Series,
    forecast_date: pd.Timestamp,
    q10: List[float],
    q25: List[float],
    mean: List[float],
    q75: List[float],
    q90: List[float],
    horizon_days: int = 21,
    title: str = "Probabilistic Price Forecast",
    figsize: Tuple[int, int] = (12, 5),
) -> "matplotlib.figure.Figure":
    """
    Fan chart showing probabilistic forecast distribution over horizon.
    """
    plt, _ = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)

    # Historical prices (last 60 bars)
    hist_plot = history.iloc[-60:]
    ax.plot(hist_plot.index, hist_plot.values, color="black", linewidth=1.5, label="History")

    # Build forecast index
    last_price = float(history.iloc[-1])
    fcast_idx = pd.bdate_range(forecast_date, periods=horizon_days + 1)[1:]

    def _cumulative_price(log_returns: List[float]) -> np.ndarray:
        return last_price * np.exp(np.cumsum(log_returns))

    p10 = _cumulative_price(q10[:horizon_days])
    p25 = _cumulative_price(q25[:horizon_days])
    pmid = _cumulative_price(mean[:horizon_days])
    p75 = _cumulative_price(q75[:horizon_days])
    p90 = _cumulative_price(q90[:horizon_days])

    ax.fill_between(fcast_idx, p10, p90, alpha=0.15, color="#2196F3", label="10–90% CI")
    ax.fill_between(fcast_idx, p25, p75, alpha=0.25, color="#2196F3", label="25–75% CI")
    ax.plot(fcast_idx, pmid, "--", color="#2196F3", linewidth=2, label="Mean Forecast")
    ax.axvline(forecast_date, color="grey", linestyle=":", linewidth=1)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Price")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_volatility_bands(
    prices: pd.Series,
    cond_vol: pd.Series,
    title: str = "Price with GARCH Conditional Volatility",
    figsize: Tuple[int, int] = (14, 7),
) -> "matplotlib.figure.Figure":
    """
    Top panel: price with 1-sigma GARCH bands.
    Bottom panel: conditional annualised volatility.
    """
    plt, _ = _get_plt()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, height_ratios=[2, 1])

    # Price panel
    aligned_vol = cond_vol.reindex(prices.index, method="ffill")
    upper = prices * np.exp(aligned_vol / np.sqrt(252))
    lower = prices * np.exp(-aligned_vol / np.sqrt(252))
    ax1.fill_between(prices.index, lower, upper, alpha=0.2, color="#FF9800", label="±1σ band")
    ax1.plot(prices.index, prices.values, color="black", linewidth=1.0)
    ax1.set_ylabel("Price")
    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Vol panel
    ax2.plot(cond_vol.index, cond_vol.values * 100, color="#FF9800", linewidth=1.2)
    ax2.axhline(cond_vol.mean() * 100, color="grey", linestyle="--", linewidth=0.8, label="Mean vol")
    ax2.set_ylabel("Cond. Vol (%)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_calibration(
    calibration_df: pd.DataFrame,
    title: str = "Probability Calibration (Reliability Diagram)",
    figsize: Tuple[int, int] = (7, 7),
) -> "matplotlib.figure.Figure":
    """
    Reliability diagram from backtesting.metrics.calibration_curve output.
    """
    plt, _ = _get_plt()
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1.2, label="Perfect calibration")
    ax.scatter(
        calibration_df["mean_predicted"],
        calibration_df["fraction_positive"],
        s=calibration_df["count"].clip(10, 200),
        color="#2196F3", alpha=0.8, zorder=5,
    )
    ax.plot(
        calibration_df["mean_predicted"],
        calibration_df["fraction_positive"],
        color="#2196F3", linewidth=1.5, label="Model calibration",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction Positive (Actual)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_importance(
    importance: pd.Series,
    title: str = "Feature Importance (Bayesian Posterior |Coef|)",
    top_n: int = 20,
    figsize: Tuple[int, int] = (9, 7),
) -> "matplotlib.figure.Figure":
    """Horizontal bar chart of feature importances."""
    plt, _ = _get_plt()
    top = importance.nlargest(top_n).sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#F44336" if "brl" in f or "oni" in f else "#2196F3" for f in top.index]
    top.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("|Posterior Coefficient|")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def plot_enso_coffee_lag(
    oni: pd.Series,
    coffee_returns: pd.Series,
    lags: List[int] = [0, 3, 6, 9, 12],
    title: str = "ENSO (ONI) vs Coffee Returns — Lagged Correlations",
    figsize: Tuple[int, int] = (10, 5),
) -> "matplotlib.figure.Figure":
    """Bar chart of Pearson correlation between ONI and coffee returns at various lags."""
    plt, _ = _get_plt()
    from scipy.stats import pearsonr

    monthly_ret = coffee_returns.resample("ME").sum()
    monthly_oni = oni.resample("ME").last()
    aligned = pd.concat([monthly_oni.rename("oni"), monthly_ret.rename("ret")], axis=1).dropna()

    corrs, pvals = [], []
    for lag in lags:
        if lag == 0:
            r, p = pearsonr(aligned["oni"], aligned["ret"])
        else:
            shifted = aligned["oni"].shift(lag)
            sub = pd.concat([shifted, aligned["ret"]], axis=1).dropna()
            r, p = pearsonr(sub.iloc[:, 0], sub.iloc[:, 1])
        corrs.append(r)
        pvals.append(p)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2196F3" if p < 0.05 else "#BDBDBD" for p in pvals]
    ax.bar([str(l) for l in lags], corrs, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ONI Lag (months)")
    ax.set_ylabel("Pearson r")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.annotate("Blue = p < 0.05", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", fontsize=9, color="#2196F3")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_backtest_folds(
    fold_summary: pd.DataFrame,
    title: str = "Walk-Forward Backtest — Per-Fold Metrics",
    figsize: Tuple[int, int] = (13, 5),
) -> "matplotlib.figure.Figure":
    """Bar chart of directional accuracy per walk-forward fold."""
    plt, _ = _get_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    da = fold_summary["directional_accuracy"].dropna()
    colors = ["#4CAF50" if v > 0.5 else "#F44336" for v in da.values]
    ax1.bar(da.index.astype(str), da.values, color=colors)
    ax1.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Random baseline")
    ax1.set_title("Directional Accuracy by Fold")
    ax1.set_ylabel("DA")
    ax1.set_ylim(0.3, 0.8)
    ax1.legend()

    brier = fold_summary["brier_score"].dropna()
    ax2.bar(brier.index.astype(str), brier.values, color="#FF9800")
    ax2.axhline(0.25, color="grey", linestyle="--", linewidth=1, label="Random baseline")
    ax2.set_title("Brier Score by Fold (lower=better)")
    ax2.set_ylabel("Brier")
    ax2.legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def save_figure(fig: "matplotlib.figure.Figure", path: str, dpi: int = 150) -> None:
    """Save figure to disk and close it."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
