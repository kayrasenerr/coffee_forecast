"""
models/regime.py
Hidden Markov Model for coffee-market regime detection.

States are labelled post-hoc by sorting on mean volatility:
  0 = calm        (low vol, quiet drift)
  1 = trending    (moderate vol, directional pressure)
  2 = crisis      (high vol, supply-shock / macro disruption)

Intentionally uses raw numpy arrays instead of FeatureMatrix
because HMM requires a contiguous sequence — not a tabular frame.
"""
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from config.settings import ModelConfig
from schemas.types import RegimeResult


class HMMRegimeModel:
    model_name = "HMM_Regime"

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.n_states = cfg.hmm_n_states
        self._model = None
        self._scaler = StandardScaler()
        self._state_map: dict = {}

    def fit(self, X: np.ndarray, index: pd.DatetimeIndex) -> "HMMRegimeModel":
        """X: (T, D) array — [log_ret, realised_vol, (fx_zscore)]"""
        Xs = self._scaler.fit_transform(X)
        model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.cfg.hmm_covariance_type,
            n_iter=self.cfg.hmm_n_iter,
            random_state=42,
        )
        model.fit(Xs)
        self._model = model
        # Label states by ascending volatility mean (col 1)
        vol_means = model.means_[:, min(1, X.shape[1] - 1)]
        order = np.argsort(vol_means)
        labels = ["calm", "trending", "crisis"]
        self._state_map = {int(order[i]): labels[i] for i in range(self.n_states)}
        return self

    def predict(self, X: np.ndarray, index: pd.DatetimeIndex) -> RegimeResult:
        if self._model is None:
            raise RuntimeError("Call fit() first")
        Xs = self._scaler.transform(X)
        raw = self._model.predict(Xs)
        probs = self._model.predict_proba(Xs)
        states = pd.Series(
            [self._state_map.get(int(s), str(s)) for s in raw],
            index=index, name="regime",
        )
        state_probs = pd.DataFrame(
            probs, index=index,
            columns=[self._state_map.get(i, str(i)) for i in range(self.n_states)],
        )
        return RegimeResult(
            symbol="arabica",
            states=states,
            state_probs=state_probs,
            state_labels=self._state_map,
            model_name=self.model_name,
        )

    def fit_predict(self, X: np.ndarray, index: pd.DatetimeIndex) -> RegimeResult:
        return self.fit(X, index).predict(X, index)
