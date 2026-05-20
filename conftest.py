"""
conftest.py
===========
Shared pytest fixtures available across all test modules.
Place at project root so all test directories can use them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from contracts.schemas import CoffeeVariety, DataFrequency, FeatureFrame


# ---------------------------------------------------------------------------
# Synthetic data generators (deterministic for reproducibility)
# ---------------------------------------------------------------------------

def make_price_series(
    n: int = 500,
    start: str = "2022-01-01",
    start_price: float = 180.0,
    drift: float = 0.0003,
    vol: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    log_ret = rng.normal(drift, vol, n)
    close = start_price * np.exp(np.cumsum(log_ret))
    return pd.DataFrame({
        "open":   close * rng.uniform(0.998, 1.002, n),
        "high":   close * rng.uniform(1.001, 1.010, n),
        "low":    close * rng.uniform(0.990, 0.999, n),
        "close":  close,
        "volume": rng.integers(10_000, 60_000, n).astype(float),
    }, index=idx)


def make_raw_inputs(n: int = 600, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    prices  = make_price_series(n, seed=seed)
    robusta = make_price_series(n, start_price=2200.0, vol=0.012, seed=seed + 1)

    idx = prices.index
    raw = {
        "arabica_futures": prices,
        "robusta_futures": robusta,
        "fx_usdbrl": pd.DataFrame({"rate": 5.2 + np.cumsum(rng.normal(0, 0.02, n))}, index=idx),
        "fx_eurusd": pd.DataFrame({"rate": 1.08 + np.cumsum(rng.normal(0, 0.003, n))}, index=idx),
    }

    # ENSO (monthly)
    enso_idx = pd.date_range(prices.index[0], periods=n // 22, freq="MS")
    raw["enso"] = pd.DataFrame({"oni": rng.normal(0.1, 0.6, len(enso_idx))}, index=enso_idx)

    # COT (weekly)
    cot_idx = pd.bdate_range(prices.index[0], periods=n // 5, freq="W-TUE")
    raw["cot_arabica"] = pd.DataFrame({
        "noncommercial_long":  rng.integers(30_000, 70_000, len(cot_idx)).astype(float),
        "noncommercial_short": rng.integers(20_000, 50_000, len(cot_idx)).astype(float),
        "commercial_long":     rng.integers(40_000, 80_000, len(cot_idx)).astype(float),
        "commercial_short":    rng.integers(50_000, 90_000, len(cot_idx)).astype(float),
        "nonreportable_long":  rng.integers(5_000, 15_000, len(cot_idx)).astype(float),
        "nonreportable_short": rng.integers(5_000, 15_000, len(cot_idx)).astype(float),
        "open_interest":       rng.integers(100_000, 200_000, len(cot_idx)).astype(float),
    }, index=cot_idx)

    return raw


@pytest.fixture(scope="session")
def synthetic_raw():
    return make_raw_inputs(n=600, seed=42)


@pytest.fixture(scope="session")
def synthetic_feature_frame(synthetic_raw):
    from features.pipeline import FeaturePipeline
    fp = FeaturePipeline()
    return fp.run(
        raw_inputs=synthetic_raw,
        variety=CoffeeVariety.ARABICA,
        frequency=DataFrequency.DAILY,
    )
