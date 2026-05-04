from dataclasses import dataclass
import pandas as pd
from .base import ExpectedReturnModel
from quantum_portfolio.data.validators import validate_returns_dataframe
from quantum_portfolio.utils.math import annualization_factor

@dataclass
class HistoricalMean(ExpectedReturnModel):
    annualize: bool = True
    periods_per_year: int | None = None
    name: str = "historical_mean"
    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        validate_returns_dataframe(returns)
        mu = returns.mean()
        return mu * (self.periods_per_year or annualization_factor(returns.index)) if self.annualize else mu

@dataclass
class ExponentialWeightedMean(ExpectedReturnModel):
    span: int = 60
    annualize: bool = True
    periods_per_year: int | None = None
    name: str = "ewma_mean"
    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        validate_returns_dataframe(returns)
        mu = returns.ewm(span=self.span, adjust=False).mean().iloc[-1]
        return mu * (self.periods_per_year or annualization_factor(returns.index)) if self.annualize else mu
