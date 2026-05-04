from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np
import pandas as pd

from quantum_portfolio.constraints import Constraint, LongOnly
from quantum_portfolio.optimization.solvers import choose_solver
from quantum_portfolio.utils.exceptions import OptimizationError


@dataclass
class MultiPeriodResult:
    weights_by_period: pd.DataFrame
    solver_status: str
    solver_name: str
    objective_value: float | None

@dataclass
class MultiPeriodOptimizer:
    expected_returns_by_period: pd.DataFrame
    covariance_by_period: list[pd.DataFrame] | dict[object, pd.DataFrame]
    initial_weights: pd.Series
    constraints: list[Constraint] = field(default_factory=lambda: [LongOnly()])
    turnover_penalty: float = 0.001
    risk_aversion: float = 5.0
    max_turnover: float | None = None
    no_trade_zone: float = 0.0
    terminal_utility: pd.Series | None = None
    def _cov_at(self, i, label):
        return self.covariance_by_period[label] if isinstance(self.covariance_by_period, dict) else self.covariance_by_period[i]
    def solve(self, solver: str | None=None) -> MultiPeriodResult:
        mu = self.expected_returns_by_period.astype(float); assets = mu.columns; T, n = mu.shape
        w = cp.Variable((T,n)); terms=[]; cons=[]; prev = self.initial_weights.reindex(assets).fillna(0).to_numpy(float)
        for t, label in enumerate(mu.index):
            wt = w[t,:]; cov = self._cov_at(t,label).reindex(index=assets, columns=assets).to_numpy(float)
            delta = wt - (prev if t == 0 else w[t-1,:])
            trade = self.turnover_penalty * (cp.sum(cp.pos(cp.abs(delta)-self.no_trade_zone)) if self.no_trade_zone > 0 else cp.norm1(delta))
            terms.append(mu.iloc[t].to_numpy(float) @ wt - self.risk_aversion*cp.quad_form(wt, cov) - trade)
            cons.append(cp.sum(wt) == 1)
            if any(isinstance(c, LongOnly) for c in self.constraints): cons.append(wt >= 0)
            if self.max_turnover is not None: cons.append(cp.norm1(delta) <= self.max_turnover)
        if self.terminal_utility is not None: terms.append(self.terminal_utility.reindex(assets).to_numpy(float) @ w[-1,:])
        problem = cp.Problem(cp.Maximize(cp.sum(terms)), cons); chosen = choose_solver(solver); value = problem.solve(solver=chosen)
        if w.value is None: raise OptimizationError(f"multi-period failed: {problem.status}")
        return MultiPeriodResult(pd.DataFrame(np.asarray(w.value), index=mu.index, columns=assets), str(problem.status), chosen, None if value is None else float(value))
