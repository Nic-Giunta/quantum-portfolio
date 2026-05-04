from __future__ import annotations
from collections.abc import Iterator
import numpy as np
import pandas as pd
from .validators import validate_prices_dataframe, validate_returns_dataframe

def handle_missing_data(data: pd.DataFrame, method: str = "drop", *, limit: int | None=None) -> pd.DataFrame:
    if method == "drop": return data.dropna()
    if method == "ffill": return data.ffill(limit=limit).dropna()
    if method == "bfill": return data.bfill(limit=limit).dropna()
    if method == "zero": return data.fillna(0.0)
    if method == "mean": return data.fillna(data.mean(numeric_only=True))
    raise ValueError("method must be drop, ffill, bfill, zero, or mean")

def simple_returns(prices: pd.DataFrame, *, periods: int=1, missing: str="drop") -> pd.DataFrame:
    validate_prices_dataframe(prices, allow_missing=True)
    return handle_missing_data(prices.pct_change(periods), missing)

def log_returns(prices: pd.DataFrame, *, periods: int=1, missing: str="drop") -> pd.DataFrame:
    validate_prices_dataframe(prices, allow_missing=True)
    return handle_missing_data(np.log(prices / prices.shift(periods)), missing)

def prices_to_returns(prices: pd.DataFrame, *, method: str="simple", periods: int=1, missing: str="drop") -> pd.DataFrame:
    if method == "simple": return simple_returns(prices, periods=periods, missing=missing)
    if method == "log": return log_returns(prices, periods=periods, missing=missing)
    raise ValueError("method must be simple or log")

def winsorize_returns(returns: pd.DataFrame, lower: float=0.01, upper: float=0.99) -> pd.DataFrame:
    validate_returns_dataframe(returns, allow_missing=True)
    return returns.clip(lower=returns.quantile(lower), upper=returns.quantile(upper), axis=1)

def zscore_outlier_flags(returns: pd.DataFrame, threshold: float=4.0) -> pd.DataFrame:
    z = (returns - returns.mean()) / returns.std(ddof=0).replace(0, np.nan)
    return z.abs() > threshold

def temporal_train_test_split(data: pd.DataFrame, test_size: float | int=0.2):
    n = int(round(len(data)*test_size)) if isinstance(test_size, float) else int(test_size)
    if n <= 0 or n >= len(data): raise ValueError("invalid test_size")
    return data.iloc[:-n].copy(), data.iloc[-n:].copy()

def rolling_windows(data: pd.DataFrame, window: int, *, step: int=1) -> Iterator[pd.DataFrame]:
    for end in range(window, len(data)+1, step): yield data.iloc[end-window:end].copy()
