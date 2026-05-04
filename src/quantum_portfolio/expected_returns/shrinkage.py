from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import ExpectedReturnModel
from .historical import HistoricalMean


@dataclass
class JamesSteinShrinkage(ExpectedReturnModel):
    shrinkage: float | None = None
    annualize: bool = True
    name: str = "james_stein_shrinkage"
    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        sample = HistoricalMean(annualize=self.annualize).estimate(returns)
        target = pd.Series(float(sample.mean()), index=sample.index)
        if self.shrinkage is None:
            noise = float(returns.var().mean()) / max(len(returns), 1)
            dispersion = float(sample.var(ddof=1))
            a = float(np.clip(noise/(noise+dispersion+1e-12), 0, 1))
        else:
            a = float(np.clip(self.shrinkage, 0, 1))
        return (1-a)*sample + a*target

@dataclass
class BayesianShrinkage(ExpectedReturnModel):
    prior_mean: float | pd.Series = 0.0
    prior_strength: float = 20.0
    annualize: bool = True
    name: str = "bayesian_shrinkage"
    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        sample = HistoricalMean(annualize=self.annualize).estimate(returns)
        prior = self.prior_mean if isinstance(self.prior_mean, pd.Series) else pd.Series(self.prior_mean, index=sample.index)
        prior = prior.reindex(sample.index).astype(float)
        n = len(returns)
        return (n*sample + self.prior_strength*prior)/(n+self.prior_strength)
