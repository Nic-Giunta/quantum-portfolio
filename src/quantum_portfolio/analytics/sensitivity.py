import numpy as np
import pandas as pd
from quantum_portfolio.optimization.optimizer import PortfolioOptimizer

def sensitivity_to_expected_returns(optimizer: PortfolioOptimizer, shock: float=0.001):
    base = optimizer.solve().weights
    mu = optimizer.expected_return_model.estimate(optimizer.returns)
    rows = {}
    for asset in optimizer.returns.columns:
        shocked = mu.copy(); shocked.loc[asset] += shock
        class FixedMu:
            def estimate(self, returns): return shocked
        rows[asset] = PortfolioOptimizer(optimizer.returns, FixedMu(), optimizer.risk_model, optimizer.objective, optimizer.constraints).solve().weights - base
    return pd.DataFrame(rows).T

def bootstrap_weight_stability(optimizer_factory, returns, n_bootstraps: int=25, seed: int=42):
    rng = np.random.default_rng(seed); rows=[]
    for _ in range(n_bootstraps):
        rows.append(optimizer_factory(returns.iloc[rng.integers(0,len(returns),len(returns))]).solve().weights)
    return pd.DataFrame(rows)
