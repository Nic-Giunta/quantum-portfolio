from dataclasses import dataclass

import pandas as pd
from sklearn.covariance import OAS, LedoitWolf

from quantum_portfolio.data.validators import validate_returns_dataframe
from quantum_portfolio.utils.math import annualization_factor

from .base import RiskModel
from .psd import nearest_psd


@dataclass
class SampleCovariance(RiskModel):
    annualize: bool = True
    periods_per_year: int | None = None
    repair_psd: bool = True
    name: str = "sample_covariance"
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        validate_returns_dataframe(returns)
        cov = returns.cov()
        if self.annualize: cov *= self.periods_per_year or annualization_factor(returns.index)
        return nearest_psd(cov) if self.repair_psd else cov

@dataclass
class EWMACovariance(RiskModel):
    span: int = 60
    annualize: bool = True
    name: str = "ewma_covariance"
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        cov = returns.ewm(span=self.span, adjust=False).cov().dropna()
        mat = cov.loc[cov.index.get_level_values(0)[-1]]
        if self.annualize: mat *= annualization_factor(returns.index)
        return nearest_psd(mat)

@dataclass
class LedoitWolfCovariance(RiskModel):
    annualize: bool = True
    name: str = "ledoit_wolf"
    shrinkage_: float | None = None
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        m = LedoitWolf().fit(returns.to_numpy(dtype=float)); self.shrinkage_ = float(m.shrinkage_)
        cov = pd.DataFrame(m.covariance_, index=returns.columns, columns=returns.columns)
        if self.annualize: cov *= annualization_factor(returns.index)
        return nearest_psd(cov)

@dataclass
class OASCovariance(RiskModel):
    annualize: bool = True
    name: str = "oas"
    shrinkage_: float | None = None
    def estimate(self, returns: pd.DataFrame) -> pd.DataFrame:
        m = OAS().fit(returns.to_numpy(dtype=float)); self.shrinkage_ = float(m.shrinkage_)
        cov = pd.DataFrame(m.covariance_, index=returns.columns, columns=returns.columns)
        if self.annualize: cov *= annualization_factor(returns.index)
        return nearest_psd(cov)
