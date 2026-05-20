"""
config/settings.py
Central configuration. All tuneable parameters live here.
Modules import SETTINGS — never hardcode values anywhere else.
"""
from pydantic import BaseModel, Field
from typing import Optional


class DataConfig(BaseModel):
    arabica_ticker:  str = "KC=F"
    robusta_ticker:  str = "RB=F"
    usd_brl_ticker:  str = "BRL=X"
    history_years:   int = 5
    interval:        str = "1d"
    # 'live' = yfinance | 'csv' = local files | 'synthetic' = generated
    data_mode:       str = "synthetic"
    csv_dir:         str = "./data"
    synthetic_seed:  int = 42


class FeatureConfig(BaseModel):
    momentum_windows: list[int] = [5, 10, 21, 63]
    vol_window:        int = 21
    zscore_window:     int = 63
    spread_window:     int = 21


class ModelConfig(BaseModel):
    hmm_n_states:          int   = 3
    hmm_n_iter:            int   = 200
    hmm_covariance_type:   str   = "full"
    garch_p:               int   = 1
    garch_q:               int   = 1
    sarimax_order:         tuple = (2, 1, 2)
    sarimax_seasonal_order: tuple = (0, 0, 0, 0)
    forecast_horizon:      int   = 5


class BacktestConfig(BaseModel):
    test_months:    int = 3
    min_train_days: int = 504
    step_days:      int = 5
    metric:         str = "directional_accuracy"


class Settings(BaseModel):
    data:     DataConfig    = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models:   ModelConfig   = Field(default_factory=ModelConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


SETTINGS = Settings()
