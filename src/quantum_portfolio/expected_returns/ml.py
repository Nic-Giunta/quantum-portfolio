import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor

from .base import ExpectedReturnModel


@dataclass
class FeatureBuilder:
    windows: tuple[int, ...] = (5, 21, 63)
    def transform(self, returns: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for w in self.windows:
            parts += [returns.rolling(w).mean().add_suffix(f"_mom_{w}"), returns.rolling(w).std().add_suffix(f"_vol_{w}")]
        wealth = (1+returns).cumprod()
        parts.append((wealth/wealth.cummax()-1).add_suffix("_drawdown"))
        return pd.concat(parts, axis=1).replace([np.inf,-np.inf], np.nan).dropna()

@dataclass
class MLExpectedReturns(ExpectedReturnModel):
    model: Any | None = None
    horizon: int = 1
    feature_builder: FeatureBuilder | None = None
    annualize: bool = True
    periods_per_year: int = 252
    name: str = "ml_expected_returns"
    def __post_init__(self):
        if self.model is None: self.model = RandomForestRegressor(n_estimators=30, random_state=0, min_samples_leaf=5)
        if self.feature_builder is None: self.feature_builder = FeatureBuilder()
        warnings.warn("MLExpectedReturns can leak and overfit; use walk-forward validation.", UserWarning, stacklevel=2)
    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        builder = self.feature_builder or FeatureBuilder()
        X = builder.transform(returns); latest = X.iloc[[-1]]
        out: dict[str, float] = {}
        for asset in returns.columns:
            y = returns[asset].shift(-self.horizon).reindex(X.index)
            mask = y.notna()
            if mask.sum() < 20:
                out[asset] = float(returns[asset].mean()) * (self.periods_per_year if self.annualize else 1)
            else:
                m = clone(self.model).fit(X.loc[mask], y.loc[mask])
                out[asset] = float(m.predict(latest)[0]) * (self.periods_per_year if self.annualize else 1)
        return pd.Series(out)

@dataclass
class WalkForwardForecaster:
    model: Any
    feature_builder: FeatureBuilder
    train_window: int = 252
    expanding: bool = False
    horizon: int = 1
    def forecast(self, returns: pd.DataFrame, asset: str) -> pd.Series:
        X = self.feature_builder.transform(returns); y = returns[asset].shift(-self.horizon).reindex(X.index)
        preds = {}
        for i in range(self.train_window, len(X)-self.horizon):
            start = 0 if self.expanding else i-self.train_window
            target = y.iloc[start:i].dropna()
            if len(target) < 20: continue
            m = clone(self.model).fit(X.loc[target.index], target)
            preds[X.index[i]] = float(m.predict(X.iloc[[i]])[0])
        return pd.Series(preds, name=asset)

@dataclass
class ModelSignalAllocator:
    long_only: bool = True
    signal_power: float = 1.0
    def signals_to_weights(self, signals: pd.Series) -> pd.Series:
        s = signals.astype(float)
        if self.long_only: s = s.clip(lower=0)
        s = np.sign(s) * np.abs(s)**self.signal_power
        denom = float(s.sum() if self.long_only else s.abs().sum())
        return pd.Series(1/len(s), index=s.index) if abs(denom) < 1e-12 else s/denom
