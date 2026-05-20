"""
main.py
Orchestration entry point. Runs the full MVP pipeline:

  1. Fetch live data (Arabica futures + USD/BRL FX)
  2. Build feature matrix
  3. HMM regime detection (full history)
  4. GARCH volatility model
  5. Walk-forward backtest on last 3 months (SARIMAX)
  6. 5-day OOS forecast
  7. Save all charts to ./output/

Run:
    cd coffee_forecast
    python main.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt

from config.settings import SETTINGS
from ingestion.registry import DataRegistry
from features.pipeline import FeaturePipeline
from models.regime import HMMRegimeModel
from models.volatility import GARCHVolatilityModel
from models.forecast import SARIMAXForecastModel
from backtest.walk_forward import WalkForwardBacktester
from schemas.types import FeatureMatrix
from viz.plots import (
    plot_regime_overlay, plot_regime_probs,
    plot_volatility, plot_backtest, plot_forecast_oos,
)

OUTPUT_DIR = Path(__file__).parent / "output"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    cfg = SETTINGS

    # ── 1. Data ───────────────────────────────────────────────────────
    print("\n[1/6] Fetching live data …")
    registry = DataRegistry(cfg.data)
    frames = registry.fetch_all()

    arabica_pf = frames.get("arabica")
    if arabica_pf is None or arabica_pf.data.empty:
        print("ERROR: Arabica data unavailable. Exiting.")
        return

    # ── 2. Features ───────────────────────────────────────────────────
    print("\n[2/6] Building feature matrix …")
    pipeline = FeaturePipeline(cfg.features)
    fm = pipeline.build(frames, symbol="arabica")
    print(f"  Feature matrix: {fm.features.shape[0]} rows × {fm.features.shape[1]} cols")
    print(f"  Date range:     {fm.features.index[0].date()} → {fm.features.index[-1].date()}")

    # ── 3. Regime Detection ───────────────────────────────────────────
    print("\n[3/6] Fitting HMM regime model …")
    hmm_inputs = pipeline.get_hmm_inputs(fm)
    hmm_index  = fm.features.dropna(
        subset=["price_log_ret", "vol_realised_vol"]
    ).index[:len(hmm_inputs)]

    regime_model = HMMRegimeModel(cfg.models)
    regime = regime_model.fit_predict(hmm_inputs, hmm_index)

    current_regime = regime.states.iloc[-1]
    current_probs  = regime.state_probs.iloc[-1].to_dict()
    print(f"  Current regime: {current_regime.upper()}")
    print(f"  State probs:    { {k: f'{v:.1%}' for k, v in current_probs.items()} }")

    fig1 = plot_regime_overlay(arabica_pf, regime)
    fig1.savefig(OUTPUT_DIR / "01_regime_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    fig2 = plot_regime_probs(regime)
    fig2.savefig(OUTPUT_DIR / "02_regime_probs.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── 4. Volatility ─────────────────────────────────────────────────
    print("\n[4/6] Fitting GARCH(1,1) …")
    log_ret = arabica_pf.log_returns().dropna()
    garch   = GARCHVolatilityModel(cfg.models)
    vol_result = garch.fit_predict(log_ret)
    current_vol = vol_result.conditional_vol.iloc[-1]
    forecast_vol = float(vol_result.forecast_vol.iloc[0])
    print(f"  Current conditional vol (ann.): {current_vol:.1%}")
    print(f"  Next-step forecast vol  (ann.): {forecast_vol:.1%}")

    fig3 = plot_volatility(arabica_pf, vol_result)
    fig3.savefig(OUTPUT_DIR / "03_volatility.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # ── 5. Walk-Forward Backtest ──────────────────────────────────────
    print(f"\n[5/6] Walk-forward backtest (last {cfg.backtest.test_months} months OOS) …")
    backtester = WalkForwardBacktester(cfg.backtest, cfg.models)
    report = backtester.run(fm, regime)

    fig4 = plot_backtest(report, arabica_pf)
    fig4.savefig(OUTPUT_DIR / "04_backtest.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # ── 6. Forward Forecast ───────────────────────────────────────────
    print(f"\n[6/6] Generating {cfg.models.forecast_horizon}-day forward forecast …")
    final_model = SARIMAXForecastModel(cfg.models)
    final_model.fit(fm)
    fc_oos = final_model.forecast_oos(steps=cfg.models.forecast_horizon)
    direction = "UP ↑" if fc_oos.mean.mean() > 0 else "DOWN ↓"
    print(f"  Forecast direction (next {cfg.models.forecast_horizon}d): {direction}")
    print(f"  Mean log return:  {fc_oos.mean.mean():.5f}")
    print(f"  80% CI:           [{fc_oos.lower.mean():.5f}, {fc_oos.upper.mean():.5f}]")

    fig5 = plot_forecast_oos(fc_oos)
    fig5.savefig(OUTPUT_DIR / "05_forecast_oos.png", dpi=150, bbox_inches="tight")
    plt.close(fig5)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  OUTPUT FILES  →  {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"    {f.name}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
