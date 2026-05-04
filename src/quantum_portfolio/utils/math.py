from __future__ import annotations

import numpy as np
import pandas as pd


def annualization_factor(index: pd.Index | None, default: int = 252) -> int:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return default
    days = float(np.median(np.diff(index.view("i8")) / 86_400_000_000_000))
    if days <= 1.5: return 252
    if days <= 8: return 52
    if days <= 32: return 12
    return 1

def portfolio_variance(weights, covariance) -> float:
    w = np.asarray(weights, dtype=float); c = np.asarray(covariance, dtype=float)
    return float(w @ c @ w)

def portfolio_volatility(weights, covariance) -> float:
    return float(np.sqrt(max(portfolio_variance(weights, covariance), 0.0)))

def normalize_weights(weights: pd.Series, total: float = 1.0) -> pd.Series:
    s = float(weights.sum())
    return pd.Series(total/len(weights), index=weights.index) if abs(s) < 1e-12 else weights * total / s
