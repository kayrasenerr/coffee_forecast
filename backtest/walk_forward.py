"""
backtest/walk_forward.py
Walk-forward (expanding window) validation on last N months.

Logic:
  - Train on [start .. fold_start)
  - Predict fold_start .. fold_start+step
  - Roll forward by step_days until end of data
  - Last bt_cfg.test_months of data is always OOS

Metrics per fold + aggregated:
  - directional_accuracy (primary)
  - RMSE on log returns
  - hit_rate_by_regime
"""
import numpy as np
import pandas as pd
from config.settings import BacktestConfig, ModelConfig
from schemas.types import FeatureMatrix, RegimeResult, BacktestReport
from models.forecast import SARIMAXForecastModel


class WalkForwardBacktester:
    def __init__(self, bt_cfg: BacktestConfig, model_cfg: ModelConfig):
        self.bt_cfg = bt_cfg
        self.model_cfg = model_cfg

    def run(self, fm_full: FeatureMatrix, regime_full: RegimeResult) -> BacktestReport:
        idx = fm_full.features.index
        test_start_ts = idx[-1] - pd.DateOffset(months=self.bt_cfg.test_months)
        test_start = idx[idx >= test_start_ts][0]
        min_train  = self.bt_cfg.min_train_days

        if (idx < test_start).sum() < min_train:
            fallback = idx[min_train]
            print(f"[WARN] Insufficient training data before {test_start.date()}, "
                  f"shifting to {fallback.date()}")
            test_start = fallback

        step = self.bt_cfg.step_days
        records, fold_id = [], 0
        cursor = test_start

        while cursor <= idx[-1]:
            test_end  = min(cursor + pd.Timedelta(days=step - 1), idx[-1])
            train_mask = idx < cursor
            test_mask  = (idx >= cursor) & (idx <= test_end)

            if train_mask.sum() < min_train or test_mask.sum() == 0:
                cursor = test_end + pd.Timedelta(days=1)
                continue

            fm_tr = FeatureMatrix(fm_full.features[train_mask],
                                  fm_full.target[train_mask], fm_full.symbol)
            fm_te = FeatureMatrix(fm_full.features[test_mask],
                                  fm_full.target[test_mask], fm_full.symbol)
            try:
                model = SARIMAXForecastModel(self.model_cfg)
                model.fit(fm_tr)
                fc = model.predict(fm_te)
                actual   = fm_te.target
                forecast = fc.mean.reindex(actual.index).fillna(0)
                regime   = regime_full.states.reindex(actual.index).fillna("unknown")

                for dt in actual.index:
                    records.append({
                        "date":     dt,
                        "actual":   actual.loc[dt],
                        "forecast": forecast.loc[dt],
                        "regime":   regime.loc[dt],
                        "fold":     fold_id,
                    })
            except Exception as e:
                print(f"[WARN] Fold {fold_id} ({cursor.date()}): {e}")

            cursor = test_end + pd.Timedelta(days=1)
            fold_id += 1

        if not records:
            raise RuntimeError("No folds completed.")

        preds = pd.DataFrame(records).set_index("date").sort_index()
        preds["correct"] = np.sign(preds["actual"]) == np.sign(preds["forecast"])

        da   = float(preds["correct"].mean())
        rmse = float(np.sqrt(((preds["actual"] - preds["forecast"]) ** 2).mean()))
        hit_by_regime = preds.groupby("regime")["correct"].mean().to_dict()

        print(f"\n{'='*55}")
        print(f"  Walk-Forward Backtest — {fm_full.symbol.upper()}")
        print(f"  Folds:            {fold_id}")
        print(f"  OOS period:       {preds.index[0].date()} → {preds.index[-1].date()}")
        print(f"  Directional Acc:  {da:.1%}")
        print(f"  RMSE:             {rmse:.6f}")
        print(f"  Hit rate / regime:")
        for r, h in sorted(hit_by_regime.items()):
            print(f"    {r:12s}: {h:.1%}")
        print(f"{'='*55}\n")

        return BacktestReport(
            symbol=fm_full.symbol,
            model_name="SARIMAX",
            predictions=preds,
            directional_accuracy=da,
            rmse=rmse,
            hit_rate_by_regime=hit_by_regime,
            n_folds=fold_id,
        )
