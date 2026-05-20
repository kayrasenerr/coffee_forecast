"""
features/significance.py
=========================
Statistical significance testing for exogenous variables.

Tests run (in order of priority):
  1. Granger Causality — does variable X Granger-cause coffee returns?
     H0: X does not help predict Y.  Reject at p < 0.05.
  2. Rolling Spearman Cross-Correlation — at lags 0..30 days
     Flag if |max_corr| > 0.05 at any lag.
  3. LASSO Regularisation Path — variable enters path before λ_min?
     Signals multivariate explanatory power even under shrinkage.
  4. Random Forest Importance — top-K features by mean decrease impurity.

Variable passes selection if it satisfies ≥ 1 of the above.

Output:
  SignificanceReport — dataframe with one row per variable,
  columns: granger_pval, max_lag_corr, best_lag_days,
           lasso_selected, rf_importance, passes, reason.
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignificanceTester:
    """
    Run significance tests and return a ranked selection of exogenous
    variables suitable for inclusion in a SARIMAX / Bayesian model.

    Usage:
        tester = SignificanceTester(max_lag=21)
        report = tester.run(exog_df, target_series)
        selected = tester.get_selected(report)  # list of column names
    """

    def __init__(
        self,
        max_lag:          int   = 21,     # max Granger lag (trading days)
        granger_alpha:    float = 0.10,   # significance threshold
        min_abs_corr:     float = 0.05,   # minimum |Spearman corr| at any lag
        lasso_frac:       float = 0.25,   # LASSO: variable must enter top 25% of path
        rf_top_k:         int   = 15,     # Random Forest: keep top-K features
        require_n_tests:  int   = 1,      # variable must pass ≥ N tests
    ):
        self.max_lag         = max_lag
        self.granger_alpha   = granger_alpha
        self.min_abs_corr    = min_abs_corr
        self.lasso_frac      = lasso_frac
        self.rf_top_k        = rf_top_k
        self.require_n_tests = require_n_tests

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        exog_df: pd.DataFrame,
        target:  pd.Series,
        prefix_drop: str = "arabica_close",   # exclude target leakage
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        exog_df : DataFrame of candidate exogenous variables
        target  : target return series (aligned to exog_df index)

        Returns
        -------
        pd.DataFrame with significance results, sorted by composite score.
        """
        # Drop the price series itself (leakage)
        cols = [c for c in exog_df.columns if prefix_drop not in c
                and "robusta_close" not in c]
        X = exog_df[cols].copy()

        # Align to target
        aligned = pd.concat([X, target.rename("__target__")], axis=1).dropna()
        X_clean  = aligned[cols]
        y_clean  = aligned["__target__"]

        results = []
        print(f"  Testing {len(cols)} variables …", flush=True)

        for col in cols:
            row = self._test_variable(col, X_clean[col], y_clean, X_clean)
            results.append(row)
            if row["passes"]:
                print(f"    ✓ {col:<40} {row['reason']}", flush=True)

        report = pd.DataFrame(results).set_index("variable")
        report["composite_score"] = (
            report["granger_passes"].astype(float) * 3.0
            + report["corr_passes"].astype(float)  * 1.5
            + report["lasso_selected"].astype(float)* 1.5
            + report["rf_passes"].astype(float)    * 1.0
        )
        report = report.sort_values("composite_score", ascending=False)
        n_pass = report["passes"].sum()
        print(f"  → {n_pass}/{len(cols)} variables pass significance gate", flush=True)
        return report

    def get_selected(self, report: pd.DataFrame) -> List[str]:
        """Return list of variable names that passed selection."""
        return report[report["passes"]].index.tolist()

    def get_top_n(self, report: pd.DataFrame, n: int = 10) -> List[str]:
        """Return top-N variables by composite score regardless of pass/fail."""
        return report.head(n).index.tolist()

    # ── Per-variable tests ────────────────────────────────────────────────────

    def _test_variable(
        self,
        col:    str,
        series: pd.Series,
        target: pd.Series,
        X_all:  pd.DataFrame,
    ) -> Dict:
        row: Dict = {
            "variable":      col,
            "granger_pval":  np.nan,
            "granger_passes":False,
            "max_corr":      np.nan,
            "best_lag":      np.nan,
            "corr_passes":   False,
            "lasso_selected":False,
            "rf_importance": np.nan,
            "rf_passes":     False,
            "passes":        False,
            "reason":        "",
        }

        reasons = []

        # 1. Granger causality
        try:
            pval = self._granger_pval(series, target)
            row["granger_pval"] = round(pval, 4)
            if pval < self.granger_alpha:
                row["granger_passes"] = True
                reasons.append(f"Granger p={pval:.3f}")
        except Exception as exc:
            logger.debug("Granger failed for %s: %s", col, exc)

        # 2. Rolling cross-correlation
        try:
            max_corr, best_lag = self._max_lag_corr(series, target)
            row["max_corr"]   = round(max_corr, 4)
            row["best_lag"]   = int(best_lag)
            if abs(max_corr) >= self.min_abs_corr:
                row["corr_passes"] = True
                reasons.append(f"|corr|={abs(max_corr):.3f}@lag{best_lag}d")
        except Exception as exc:
            logger.debug("Corr failed for %s: %s", col, exc)

        # 3. LASSO (run once for all, but check per-variable)
        # Deferred — computed in batch by run()

        # 4. Mark passes
        n_pass = sum([row["granger_passes"], row["corr_passes"],
                      row["lasso_selected"], row["rf_passes"]])
        row["passes"] = n_pass >= self.require_n_tests
        row["reason"] = " | ".join(reasons) if reasons else "–"
        return row

    def _granger_pval(self, x: pd.Series, y: pd.Series) -> float:
        """Return minimum p-value across lags 1..max_lag."""
        from statsmodels.tsa.stattools import grangercausalitytests
        aligned = pd.concat([y, x], axis=1).dropna()
        if len(aligned) < self.max_lag * 3:
            return 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = grangercausalitytests(aligned.values, maxlag=self.max_lag, verbose=False)
        min_p = min(
            result[lag][0]["ssr_ftest"][1]
            for lag in range(1, self.max_lag + 1)
        )
        return float(min_p)

    def _max_lag_corr(
        self,
        x:      pd.Series,
        y:      pd.Series,
        lags:   Optional[List[int]] = None,
    ) -> Tuple[float, int]:
        """Return (max_abs_spearman_corr, best_lag) over specified lags."""
        from scipy.stats import spearmanr
        if lags is None:
            lags = [0, 1, 3, 5, 10, 15, 21]
        best_corr, best_lag = 0.0, 0
        for lag in lags:
            xlag = x.shift(lag) if lag >= 0 else x.shift(lag)
            combined = pd.concat([xlag, y], axis=1).dropna()
            if len(combined) < 30:
                continue
            corr, _ = spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
            if abs(corr) > abs(best_corr):
                best_corr, best_lag = float(corr), lag
        return best_corr, best_lag

    def run_lasso(
        self,
        exog_df: pd.DataFrame,
        target:  pd.Series,
    ) -> Dict[str, bool]:
        """
        Run LASSO regularisation path and return {col: enters_early} dict.
        Variables entering the path in the first `lasso_frac` of alphas
        are marked as lasso_selected=True.
        """
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler

        cols = list(exog_df.columns)
        aligned = pd.concat([exog_df, target.rename("__t__")], axis=1).dropna()
        X = aligned[cols].values
        y = aligned["__t__"].values

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso = LassoCV(cv=5, max_iter=5000, n_alphas=50).fit(X_s, y)

        # Non-zero coefficients at best alpha
        selected = set(np.where(lasso.coef_ != 0)[0])
        return {col: (i in selected) for i, col in enumerate(cols)}

    def run_rf_importance(
        self,
        exog_df: pd.DataFrame,
        target:  pd.Series,
    ) -> pd.Series:
        """Random Forest feature importance."""
        from sklearn.ensemble import RandomForestRegressor

        cols = list(exog_df.columns)
        aligned = pd.concat([exog_df, target.rename("__t__")], axis=1).dropna()
        X = aligned[cols].values
        y = aligned["__t__"].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                                       random_state=42, n_jobs=-1)
            rf.fit(X, y)

        return pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)

    def full_run(
        self,
        exog_df: pd.DataFrame,
        target:  pd.Series,
        prefix_drop: str = "arabica_close",
    ) -> pd.DataFrame:
        """
        Full run including LASSO and RF (slower).
        Adds lasso_selected and rf_importance columns to report.
        """
        report = self.run(exog_df, target, prefix_drop=prefix_drop)

        cols = [c for c in exog_df.columns
                if prefix_drop not in c and "robusta_close" not in c]
        X_sub = exog_df[cols].copy()

        # LASSO
        print("  Running LASSO path …", flush=True)
        try:
            lasso_sel = self.run_lasso(X_sub, target)
            for col, sel in lasso_sel.items():
                if col in report.index:
                    report.at[col, "lasso_selected"] = sel
                    if sel and not report.at[col, "passes"]:
                        report.at[col, "passes"] = True
                        report.at[col, "reason"] += " | LASSO"
        except Exception as exc:
            logger.warning("LASSO failed: %s", exc)

        # RF
        print("  Running Random Forest …", flush=True)
        try:
            imp = self.run_rf_importance(X_sub, target)
            top_k = set(imp.head(self.rf_top_k).index)
            for col in report.index:
                if col in imp.index:
                    report.at[col, "rf_importance"] = round(float(imp[col]), 6)
                    if col in top_k:
                        report.at[col, "rf_passes"] = True
                        if not report.at[col, "passes"]:
                            report.at[col, "passes"] = True
                            report.at[col, "reason"] += " | RF_top"
        except Exception as exc:
            logger.warning("RF importance failed: %s", exc)

        # Recompute composite score
        report["composite_score"] = (
            report["granger_passes"].astype(float) * 3.0
            + report["corr_passes"].astype(float)  * 1.5
            + report["lasso_selected"].astype(float)* 1.5
            + report["rf_passes"].astype(float)    * 1.0
        )
        return report.sort_values("composite_score", ascending=False)
