"""
models/base.py
Abstract base for all models. Each model receives a FeatureMatrix,
returns its canonical result type from schemas.types.
"""
from abc import ABC, abstractmethod
from schemas.types import FeatureMatrix


class BaseModel(ABC):
    model_name: str = ""

    @abstractmethod
    def fit(self, fm: FeatureMatrix) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, fm: FeatureMatrix):
        """Returns the appropriate result type for the model."""
        ...

    def fit_predict(self, fm: FeatureMatrix):
        return self.fit(fm).predict(fm)
