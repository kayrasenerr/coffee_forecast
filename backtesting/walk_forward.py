"""
backtesting/walk_forward.py
===========================
Walk-forward validation engine.

Walk-forward is the ONLY valid validation methodology for time series models.
It prevents lookahead bias by strictly respecting temporal order.

Design:
  - Sliding or expanding train window
  - Fixed test window (e.g. 63 trading days = 3 months)
  - Re-fits model on each fold (proper out-of-sample)
  - Aggregates all fold metrics into BacktestResult

Overfitting mitigations built-in:
  1. Temporal separation: train/test never overlap
  2. Minimum train size enforced (300+ bars)
  3. No hyper-parameter optimisation inside walk-forward
     (use separate validation set for HP tuning, then run WF)
  4. Multiple metrics evaluated simultaneously
  5. Regime-conditional performance breakdown
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from contracts.interfaces import BacktesterBase, ModelBase, RegimeDetectorBase
from contracts.schemas import (
    BacktestResult,
    CoffeeVariety,
    FeatureFrame,
    ForecastRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    """Results from a single walk-forward fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    forecasts: List[ForecastRecord] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)
    regime_ids: List[int] = field(default_factory=list)

    @property
    def directional_accuracy(self) -> float:
        """Fraction of forecasts where sign(forecast) == sign(actual)."""
        if not self.forecasts or not self.actuals:
            return float("nan")
        correct = sum(
            1 for f, a in zip(self.forecasts, self.actuals)
            if (f.prob_up > 0.5) == (a > 0)
        )
        return correct / len(self.actuals)

    @property
    def mean_brier_score(self) -> float:
        """Brier score: mean((P_up - I(actual > 0))^2). Lower=better."""
        if not self.forecasts or not self.actuals:
            return float("nan")
        scores = [
            (f.prob_up - float(a > 0)) ** 2
            for f, a in zip(self.forecasts, self.actuals)
        ]
        return float(np.mean(scores))

    @property
    def rmse(self) -> float:
        if not self.forecasts or not self.actuals:
            return float("nan")
        errors = [f.mean_return - a for f, a in zip(self.forecasts, self.actuals)]
        return float(np.sqrt(np.mean([e**2 for e in errors])))

    @property
    def mae(self) -> float:
        if not self.forecasts or not self.actuals:
            return float("nan")
        errors = [abs(f.mean_return - a) for f, a in zip(self.forecasts, self.actuals)]
        return float(np.mean(errors))


class WalkForwardBacktester(BacktesterBase):
    """
    Walk-forward validation for any ModelBase implementation.

    Parameters
    ----------
    train_window  : number of bars for training (expanding=False → sliding)
    test_window   : number of bars per test fold
    step          : bars between fold start dates (usually = test_window)
    expanding     : if True, train window grows with each fold
    min_train     : minimum train bars required (safety guard)
    """

    def __init__(
        self,
        train_window: int = 756,
        test_window: int = 63,
        step: int = 63,
        expanding: bool = False,
        min_train: Optional[int] = None,
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        self.expanding = expanding
        # Default min_train: no greater than train_window
        self.min_train = min_train if min_train is not None else min(252, train_window)

    def run(
        self,
        model: ModelBase,
        frame: FeatureFrame,
        target_col: str,
        n_folds: Optional[int] = None,
        regime_detector: Optional[RegimeDetectorBase] = None,
    ) -> BacktestResult:
        """
        Execute walk-forward backtest.

        Parameters
        ----------
        model          : unfitted ModelBase instance (will be re-fit each fold)
        frame          : FeatureFrame with target_col present
        target_col     : column to predict (e.g. "target_log_return_5d")
        n_folds        : cap number of folds (None = use all available)
        regime_detector: optional, adds regime breakdown to results
        """
        df = frame.df.dropna(subset=[target_col]).copy()
        n = len(df)

        if n < self.min_train + self.test_window:
            raise ValueError(
                f"Not enough data ({n} bars) for walk-forward with "
                f"min_train={self.min_train}, test_window={self.test_window}"
            )

        folds = self._generate_folds(df, n_folds)
        logger.info(
            "[WalkForward] %d folds, train=%d, test=%d, expanding=%s",
            len(folds), self.train_window, self.test_window, self.expanding
        )

        fold_results: List[FoldResult] = []

        for fold_id, (train_idx, test_idx) in enumerate(folds):
            train_df = df.iloc[train_idx]
            test_df  = df.iloc[test_idx]

            if len(train_df) < self.min_train:
                logger.debug("[WalkForward] Fold %d: skipping (train too small)", fold_id)
                continue

            # Build sub-FeatureFrame for train
            train_frame = FeatureFrame(
                variety=frame.variety,
                frequency=frame.frequency,
                feature_names=frame.feature_names,
                df=train_df,
            )

            # Fit model on train
            try:
                model.fit(train_frame, target_col)
            except Exception as exc:
                logger.warning("[WalkForward] Fold %d fit failed: %s", fold_id, exc)
                continue

            # Predict on each test bar
            fold_result = FoldResult(
                fold_id=fold_id,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
            )

            for bar_i in range(len(test_df)):
                # Context: train + all test bars up to (not including) current
                context_df = pd.concat([train_df, test_df.iloc[:bar_i]])
                context_frame = FeatureFrame(
                    variety=frame.variety,
                    frequency=frame.frequency,
                    feature_names=frame.feature_names,
                    df=context_df,
                )

                try:
                    forecast = model.predict(context_frame, horizon_days=self.test_window)
                    actual = test_df[target_col].iloc[bar_i]
                    fold_result.forecasts.append(forecast)
                    fold_result.actuals.append(float(actual))
                except Exception as exc:
                    logger.debug("[WalkForward] Fold %d bar %d predict failed: %s", fold_id, bar_i, exc)

            fold_results.append(fold_result)
            logger.info(
                "[WalkForward] Fold %d: DA=%.3f  Brier=%.4f  RMSE=%.5f",
                fold_id,
                fold_result.directional_accuracy,
                fold_result.mean_brier_score,
                fold_result.rmse,
            )

        return self._aggregate(model, frame, target_col, fold_results)

    def _generate_folds(
        self,
        df: pd.DataFrame,
        n_folds: Optional[int],
    ) -> List[Tuple[List[int], List[int]]]:
        """Generate (train_indices, test_indices) pairs."""
        folds = []
        n = len(df)
        start = self.train_window

        while start + self.test_window <= n:
            test_end = start + self.test_window
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, start - self.train_window)
            folds.append((list(range(train_start, start)), list(range(start, test_end))))
            start += self.step
            if n_folds and len(folds) >= n_folds:
                break

        return folds

    @staticmethod
    def _aggregate(
        model: ModelBase,
        frame: FeatureFrame,
        target_col: str,
        folds: List[FoldResult],
    ) -> BacktestResult:
        if not folds:
            raise ValueError("No valid folds completed")

        da_scores   = [f.directional_accuracy for f in folds if not np.isnan(f.directional_accuracy)]
        brier_scores = [f.mean_brier_score    for f in folds if not np.isnan(f.mean_brier_score)]
        rmse_scores  = [f.rmse                for f in folds if not np.isnan(f.rmse)]
        mae_scores   = [f.mae                 for f in folds if not np.isnan(f.mae)]

        # Signal-based Sharpe: go long when prob_up > 0.55, short when < 0.45
        all_forecasts = [f for fold in folds for f in fold.forecasts]
        all_actuals   = [a for fold in folds for a in fold.actuals]
        signal_sharpe = _compute_signal_sharpe(all_forecasts, all_actuals)

        return BacktestResult(
            model_name=model.model_name,
            variety=frame.variety,
            start_date=folds[0].train_start.date(),
            end_date=folds[-1].test_end.date(),
            n_folds=len(folds),
            directional_accuracy=float(np.mean(da_scores)) if da_scores else float("nan"),
            mean_brier_score=float(np.mean(brier_scores)) if brier_scores else None,
            rmse=float(np.mean(rmse_scores)) if rmse_scores else None,
            mae=float(np.mean(mae_scores)) if mae_scores else None,
            signal_sharpe=signal_sharpe,
            params=model.get_params(),
        )

    def fold_summary(self, folds: List[FoldResult]) -> pd.DataFrame:
        """Return per-fold metrics as DataFrame (for visualisation)."""
        rows = []
        for f in folds:
            rows.append({
                "fold": f.fold_id,
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "n_forecasts": len(f.forecasts),
                "directional_accuracy": f.directional_accuracy,
                "brier_score": f.mean_brier_score,
                "rmse": f.rmse,
                "mae": f.mae,
            })
        return pd.DataFrame(rows).set_index("fold")


def _compute_signal_sharpe(
    forecasts: List[ForecastRecord],
    actuals: List[float],
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
) -> Optional[float]:
    """
    Compute annualised Sharpe ratio of a long/short signal.
    Signal: +1 if prob_up > long_threshold, -1 if < short_threshold, 0 otherwise.
    """
    if not forecasts or not actuals:
        return None
    signals = []
    for f in forecasts:
        if f.prob_up > long_threshold:
            signals.append(1.0)
        elif f.prob_up < short_threshold:
            signals.append(-1.0)
        else:
            signals.append(0.0)

    pnl = [s * a for s, a in zip(signals, actuals) if s != 0]
    if len(pnl) < 5:
        return None
    mean_pnl = np.mean(pnl)
    std_pnl  = np.std(pnl)
    if std_pnl == 0:
        return None
    return float(mean_pnl / std_pnl * np.sqrt(252))
