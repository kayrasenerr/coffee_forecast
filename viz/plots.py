"""
viz/plots.py
All visualisation in one place. Each function is standalone — pass in result
objects from schemas.types and get a matplotlib Figure back.
No side effects. Callers decide whether to show() or savefig().
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from schemas.types import PriceFrame, RegimeResult, ForecastResult, VolatilityResult, BacktestReport

REGIME_COLORS = {"calm": "#4caf50", "trending": "#ff9800", "crisis": "#f44336", "unknown": "#9e9e9e"}

def _dark(ax):
    ax.set_facecolor("#161b22")
    ax.grid(True, linestyle="--", alpha=0.4, color="#21262d")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#30363d")
    ax.tick_params(colors="#8b949e")
    ax.xaxis.label.set_color("#c9d1d9")
    ax.yaxis.label.set_color("#c9d1d9")
    ax.title.set_color("#e6edf3")

def _fmt_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    plt.xticks(rotation=30)


def plot_regime_overlay(pf: PriceFrame, regime: RegimeResult) -> plt.Figure:
    close = pf.close()
    states = regime.states.reindex(close.index).ffill()
    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")
    prev, start = states.iloc[0], states.index[0]
    for dt, state in states.items():
        if state != prev:
            ax.axvspan(start, dt, alpha=0.18, color=REGIME_COLORS.get(prev, "#9e9e9e"), lw=0)
            start, prev = dt, state
    ax.axvspan(start, states.index[-1], alpha=0.18, color=REGIME_COLORS.get(prev, "#9e9e9e"), lw=0)
    ax.plot(close.index, close.values, color="#58a6ff", lw=1.2, zorder=3)
    ax.set_title("Arabica Coffee (KC=F) — HMM Regime Overlay", fontsize=13)
    ax.set_ylabel("Price (cents/lb)")
    _fmt_xaxis(ax)
    legend = [Patch(facecolor=c, label=r, alpha=0.7) for r, c in REGIME_COLORS.items() if r != "unknown"]
    ax.legend(handles=legend, loc="upper left", framealpha=0.3, labelcolor="#c9d1d9", facecolor="#161b22")
    _dark(ax)
    fig.tight_layout()
    return fig


def plot_regime_probs(regime: RegimeResult, last_n: int = 252) -> plt.Figure:
    probs = regime.state_probs.iloc[-last_n:]
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")
    bottom = np.zeros(len(probs))
    for col in probs.columns:
        vals = probs[col].values
        ax.fill_between(probs.index, bottom, bottom + vals, alpha=0.75,
                        color=REGIME_COLORS.get(col, "#9e9e9e"), label=col)
        bottom += vals
    ax.set_ylim(0, 1)
    ax.set_ylabel("Regime Probability")
    ax.set_title("HMM Regime Probabilities (last 252 trading days)")
    _fmt_xaxis(ax)
    ax.legend(loc="upper left", framealpha=0.3, labelcolor="#c9d1d9", facecolor="#161b22")
    _dark(ax)
    fig.tight_layout()
    return fig


def plot_volatility(pf: PriceFrame, vol: VolatilityResult) -> plt.Figure:
    close = pf.close()
    cond_vol = vol.conditional_vol.reindex(close.index).ffill()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor="#0d1117", sharex=True)
    fig.patch.set_facecolor("#0d1117")
    ax1.plot(close.index, close.values, color="#58a6ff", lw=1.2)
    ax1.set_ylabel("Price (cents/lb)")
    ax1.set_title("Arabica Price + GARCH(1,1) Conditional Volatility", fontsize=13)
    _dark(ax1)
    ax2.fill_between(cond_vol.index, cond_vol.values, alpha=0.6, color="#ff9800", label="Cond. Vol (ann.)")
    ax2.set_ylabel("Annualised Vol")
    _fmt_xaxis(ax2)
    ax2.legend(framealpha=0.3, labelcolor="#c9d1d9", facecolor="#161b22")
    _dark(ax2)
    fig.tight_layout()
    return fig


def plot_backtest(report: BacktestReport, pf: PriceFrame) -> plt.Figure:
    preds = report.predictions.copy()
    close = pf.close().reindex(preds.index).ffill()
    preds["ret_act"]   = np.log(close / close.shift(1))
    preds["signal"]    = np.sign(preds["forecast"])
    preds["strat"]     = preds["signal"] * preds["ret_act"]
    preds["cum_strat"] = preds["strat"].cumsum()
    preds["cum_bh"]    = preds["ret_act"].cumsum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), facecolor="#0d1117", sharex=True)
    fig.patch.set_facecolor("#0d1117")

    colors = ["#4caf50" if r > 0 else "#f44336" for r in preds["ret_act"]]
    ax1.bar(preds.index, preds["ret_act"], color=colors, alpha=0.55, width=0.8, label="Actual return")
    ax1.plot(preds.index, preds["forecast"], color="#ff9800", lw=0.9, alpha=0.85, label="SARIMAX forecast")
    ax1.set_ylabel("Log Return")
    ax1.set_title(
        f"Walk-Forward OOS  |  Dir. Accuracy: {report.directional_accuracy:.1%}  |  "
        f"RMSE: {report.rmse:.5f}  |  Folds: {report.n_folds}", fontsize=12)
    ax1.legend(framealpha=0.3, labelcolor="#c9d1d9", facecolor="#161b22")
    _dark(ax1)

    ax2.plot(preds.index, preds["cum_strat"], color="#58a6ff", lw=1.4, label="Signal strategy")
    ax2.plot(preds.index, preds["cum_bh"], color="#8b949e", lw=1.0, alpha=0.7, linestyle="--", label="Buy & Hold")
    ax2.axhline(0, color="#30363d", lw=0.8)
    ax2.set_ylabel("Cumulative Log Return")
    _fmt_xaxis(ax2)
    ax2.legend(framealpha=0.3, labelcolor="#c9d1d9", facecolor="#161b22")
    _dark(ax2)
    fig.tight_layout()
    return fig


def plot_forecast_oos(fc: ForecastResult, actual: pd.Series = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")
    ax.fill_between(fc.forecast_dates, fc.lower, fc.upper, alpha=0.25, color="#58a6ff", label="80% CI")
    ax.plot(fc.forecast_dates, fc.mean, color="#58a6ff", lw=1.6, label="Forecast mean")
    ax.axhline(0, color="#8b949e", lw=0.8, linestyle="--")
    if actual is not None:
        ax.scatter(actual.index, actual.values, color="#f44336", s=18, zorder=5, label="Actual", alpha=0.85)
    ax.set_ylabel("Log Return Forecast")
    ax.set_title(f"SARIMAX OOS Forecast — {fc.symbol.upper()} ({fc.horizon}d)")
    ax.legend(framealpha=0.3, labelcolor="#c9d1d9", facecolor="#161b22")
    _dark(ax)
    fig.tight_layout()
    return fig
