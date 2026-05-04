from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantum_portfolio.utils.math import annualization_factor

from .base import RiskModel
from .psd import nearest_psd


@dataclass
class SemiCovariance(RiskModel):
    threshold: float = 0.0
    annualize: bool = True
    name: str = "semi_covariance"
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        d = returns.clip(upper=self.threshold)-self.threshold
        cov = d.cov()
        if self.annualize: cov *= annualization_factor(returns.index)
        return nearest_psd(cov)

@dataclass
class DownsideDeviation(RiskModel):
    threshold: float = 0.0
    annualize: bool = True
    name: str = "downside_deviation"
    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        dd = np.sqrt((np.minimum(returns-self.threshold, 0.0)**2).mean())
        return dd*np.sqrt(annualization_factor(returns.index)) if self.annualize else dd

@dataclass
class MAD(RiskModel):
    annualize: bool = True
    name: str = "mad"
    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        x = (returns-returns.mean()).abs().mean()
        return x*np.sqrt(annualization_factor(returns.index)) if self.annualize else x
