"""
models/hmm_model.py
===================
Hidden Markov Model for market regime detection.

Regimes capture the unobserved market state driving coffee price dynamics.
Typical regimes found in coffee markets:

  State 0: Low volatility / trend-following (bull or bear)
  State 1: High volatility / mean-reverting (range-bound)
  State 2: Stress / supply shock (explosive moves)

Model: Gaussian HMM on selected features
Library: hmmlearn (scikit-learn compatible)

Key outputs:
  - Regime assignment per time step
  - Transition probability matrix
  - Regime-conditioned return statistics
  - "Regime shift" probability (P(state change next period))

Methodology:
  - Fit on log returns + volatility (minimum required)
  - Optionally add COT, curve structure, ENSO as additional obs.
  - Use BIC to select number of states (2–5)
  - Validate: regime-conditioned Sharpe must be distinct
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from contracts.interfaces import RegimeDetectorBase
from contracts.schemas import (
    CoffeeVariety,
    DataFrequency,
    FeatureFrame,
    MarketRegime,
    RegimeSnapshot,
)
from config.settings import settings

logger = logging.getLogger(__name__)

# Feature columns used for regime inference (ordered by importance)
_DEFAULT_REGIME_FEATURES = [
    "log_return_1d",
    "realised_vol_21d",
    "log_return_5d",
    "log_return_21d",
    "price_z_63d",
]

# Human-interpretable regime labels (assigned post-hoc by return/vol stats)
_REGIME_LABELS = {
    "high_ret_low_vol": MarketRegime.BULL,
    "low_ret_low_vol":  MarketRegime.NEUTRAL,
    "high_vol":         MarketRegime.VOLATILE,
    "low_ret_high_vol": MarketRegime.BEAR,
    "stress":           MarketRegime.SUPPLY_STRESS,
}


class HMMRegimeDetector(RegimeDetectorBase):
    """
    Gaussian HMM-based regime detector for coffee futures.

    Usage:
        detector = HMMRegimeDetector(n_regimes=3, variety=CoffeeVariety.ARABICA)
        detector.fit(frame)
        snapshots = detector.predict(frame)
        latest = detector.predict_latest(frame)
    """

    model_name = "gaussian_hmm"

    def __init__(
        self,
        n_regimes: int = 3,
        variety: CoffeeVariety = CoffeeVariety.ARABICA,
        feature_cols: Optional[List[str]] = None,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.variety = variety
        self.feature_cols = feature_cols or _DEFAULT_REGIME_FEATURES
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state

        self._model = None
        self._regime_map: Dict[int, MarketRegime] = {}
        self._fitted = False

    def fit(self, frame: FeatureFrame) -> "HMMRegimeDetector":
        """Fit HMM on feature matrix."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise ImportError("pip install hmmlearn") from e

        X = self._prepare_observations(frame)
        logger.info(
            "[HMM] Fitting %d-regime model on %d obs, %d features",
            self.n_regimes, len(X), X.shape[1]
        )

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            verbose=False,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(X)
        converged = self._model.monitor_.converged
        logger.info("[HMM] Converged: %s  Log-likelihood: %.2f",
                    converged, self._model.score(X))
        if not converged:
            logger.warning("[HMM] Model did not converge — results may be unreliable. "
                           "Try increasing n_iter or reducing n_regimes.")

        # Assign semantic labels based on regime statistics
        states = self._model.predict(X)
        self._regime_map = self._label_regimes(X, states, frame)
        self._fitted = True
        return self

    def predict(self, frame: FeatureFrame) -> List[RegimeSnapshot]:
        """Return regime snapshot for every row in frame."""
        self._check_fitted()
        import warnings
        available = [col for col in self.feature_cols if col in frame.df.columns]
        clean_df  = frame.df[available].dropna()
        if len(clean_df) == 0:
            return []
        X     = clean_df.values.astype(np.float64)
        index = clean_df.index

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            states     = self._model.predict(X)
            posteriors = self._model.predict_proba(X)

        snapshots = []
        for i, (ts, state) in enumerate(zip(index, states)):
            regime = self._regime_map.get(int(state), MarketRegime.NEUTRAL)
            prob   = float(posteriors[i, state])
            trans_row  = self._model.transmat_[state]
            trans_prob = float(1.0 - trans_row[state])
            snapshots.append(RegimeSnapshot(
                timestamp=pd.Timestamp(ts).to_pydatetime(),
                variety=self.variety,
                regime=regime,
                regime_id=int(state),
                probability=prob,
                transition_probability=trans_prob,
            ))
        return snapshots

    def predict_latest(self, frame: FeatureFrame) -> RegimeSnapshot:
        """Regime for the most recent row."""
        snapshots = self.predict(frame)
        if not snapshots:
            raise ValueError("No valid observations for regime prediction")
        return snapshots[-1]

    def regime_statistics(self, frame: FeatureFrame) -> pd.DataFrame:
        """
        Return per-regime descriptive statistics.
        Useful for validating regime interpretability.
        """
        self._check_fitted()
        X = self._prepare_observations(frame)
        states = self._model.predict(X)
        index = frame.df.dropna(subset=[c for c in self.feature_cols if c in frame.df.columns]).index

        df = frame.df.loc[index].copy()
        df["__regime__"] = states

        rows = []
        for s in range(self.n_regimes):
            mask = df["__regime__"] == s
            sub = df[mask]
            ret_col = "log_return_1d" if "log_return_1d" in sub.columns else sub.columns[0]
            vol_col = "realised_vol_21d" if "realised_vol_21d" in sub.columns else None
            rows.append({
                "regime_id": s,
                "regime_label": self._regime_map.get(s, MarketRegime.NEUTRAL).value,
                "n_obs": len(sub),
                "pct_time": len(sub) / len(df) * 100,
                "mean_return": sub[ret_col].mean() * 252 if ret_col else None,
                "mean_vol": sub[vol_col].mean() if vol_col else None,
                "sharpe": (sub[ret_col].mean() / sub[ret_col].std() * np.sqrt(252))
                          if ret_col else None,
            })
        return pd.DataFrame(rows).set_index("regime_id")

    def transition_matrix(self) -> pd.DataFrame:
        """Return the learned transition probability matrix."""
        self._check_fitted()
        labels = [self._regime_map.get(i, MarketRegime.NEUTRAL).value for i in range(self.n_regimes)]
        return pd.DataFrame(
            self._model.transmat_,
            index=labels,
            columns=labels,
        )

    def select_n_regimes(
        self,
        frame: FeatureFrame,
        min_k: int = 2,
        max_k: int = 5,
    ) -> Tuple[int, pd.DataFrame]:
        """
        Use BIC to select optimal number of regimes.
        Returns (best_k, results_df).
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError as e:
            raise ImportError("pip install hmmlearn") from e

        X = self._prepare_observations(frame)
        results = []
        for k in range(min_k, max_k + 1):
            model = GaussianHMM(
                n_components=k,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            model.fit(X)
            log_lik = model.score(X)
            n_params = k * k + k * X.shape[1] + k * X.shape[1] * (X.shape[1] + 1) / 2
            bic = -2 * log_lik + n_params * np.log(len(X))
            results.append({"k": k, "log_likelihood": log_lik, "bic": bic})

        df = pd.DataFrame(results).set_index("k")
        best_k = int(df["bic"].idxmin())
        logger.info("[HMM] BIC-optimal regimes: %d\n%s", best_k, df.to_string())
        return best_k, df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "regime_map": self._regime_map,
                "feature_cols": self.feature_cols,
                "n_regimes": self.n_regimes,
                "variety": self.variety.value,
            }, f)
        logger.info("[HMM] Saved to %s", path)

    def load(self, path: str) -> "HMMRegimeDetector":
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._model = state["model"]
        self._regime_map = state["regime_map"]
        self.feature_cols = state["feature_cols"]
        self.n_regimes = state["n_regimes"]
        self.variety = CoffeeVariety(state["variety"])
        self._fitted = True
        logger.info("[HMM] Loaded from %s", path)
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prepare_observations(self, frame: FeatureFrame) -> np.ndarray:
        """Extract and clean observation matrix for HMM."""
        available = [c for c in self.feature_cols if c in frame.df.columns]
        if not available:
            raise ValueError(
                f"None of the required features {self.feature_cols} found in frame. "
                f"Available: {list(frame.df.columns)}"
            )
        sub = frame.df[available].dropna()
        return sub.values.astype(np.float64)

    def _label_regimes(
        self,
        X: np.ndarray,
        states: np.ndarray,
        frame: FeatureFrame,
    ) -> Dict[int, MarketRegime]:
        """
        Assign semantic MarketRegime to each HMM state based on statistics.
        Strategy:
          - Sort states by mean return (ascending)
          - Lowest mean return → BEAR
          - Highest mean return → BULL
          - Highest volatility (among remaining) → VOLATILE
        """
        available = [c for c in self.feature_cols if c in frame.df.columns]
        clean_idx = frame.df[available].dropna().index

        ret_col_idx = None
        vol_col_idx = None
        if "log_return_1d" in available:
            ret_col_idx = available.index("log_return_1d")
        if "realised_vol_21d" in available:
            vol_col_idx = available.index("realised_vol_21d")

        stats = {}
        for s in range(self.n_regimes):
            mask = states == s
            stats[s] = {
                "mean_ret": X[mask, ret_col_idx].mean() if ret_col_idx is not None else 0,
                "mean_vol": X[mask, vol_col_idx].mean() if vol_col_idx is not None else 1,
                "n": mask.sum(),
            }

        # Sort by mean return
        sorted_by_ret = sorted(stats.items(), key=lambda x: x[1]["mean_ret"])
        regime_map: Dict[int, MarketRegime] = {}

        if self.n_regimes == 2:
            regime_map[sorted_by_ret[0][0]] = MarketRegime.BEAR
            regime_map[sorted_by_ret[1][0]] = MarketRegime.BULL
        elif self.n_regimes == 3:
            # Among the middle regime, classify by vol
            regime_map[sorted_by_ret[0][0]] = MarketRegime.BEAR
            regime_map[sorted_by_ret[-1][0]] = MarketRegime.BULL
            mid_state = sorted_by_ret[1][0]
            regime_map[mid_state] = MarketRegime.VOLATILE
        else:
            # Generic fallback for 4+ regimes
            for rank, (state_id, _) in enumerate(sorted_by_ret):
                frac = rank / (self.n_regimes - 1)
                if frac < 0.25:
                    regime_map[state_id] = MarketRegime.BEAR
                elif frac > 0.75:
                    regime_map[state_id] = MarketRegime.BULL
                else:
                    regime_map[state_id] = MarketRegime.NEUTRAL

        logger.info("[HMM] Regime assignments: %s", {k: v.value for k, v in regime_map.items()})
        return regime_map

    def _check_fitted(self) -> None:
        if not self._fitted or self._model is None:
            raise RuntimeError("Call fit() before predict()")
