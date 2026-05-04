from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantum_portfolio.expected_returns import ExpectedReturnModel, HistoricalMean
from quantum_portfolio.optimization.optimizer import PortfolioOptimizer


@dataclass
class _FixedExpectedReturns(ExpectedReturnModel):
    values: pd.Series
    name: str = "fixed_expected_returns"

    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        return self.values


def sensitivity_to_expected_returns(optimizer: PortfolioOptimizer, shock: float=0.001) -> pd.DataFrame:
    base = optimizer.solve().weights
    expected_return_model = optimizer.expected_return_model or HistoricalMean()
    mu = expected_return_model.estimate(optimizer.returns)
    rows = {}
    for asset in optimizer.returns.columns:
        shocked = mu.copy(); shocked.loc[asset] += shock
        rows[asset] = PortfolioOptimizer(optimizer.returns, _FixedExpectedReturns(shocked), optimizer.risk_model, optimizer.objective, optimizer.constraints).solve().weights - base
    return pd.DataFrame(rows).T

def bootstrap_weight_stability(optimizer_factory, returns, n_bootstraps: int=25, seed: int=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows=[]
    for _ in range(n_bootstraps):
        rows.append(optimizer_factory(returns.iloc[rng.integers(0,len(returns),len(returns))]).solve().weights)
    return pd.DataFrame(rows)
