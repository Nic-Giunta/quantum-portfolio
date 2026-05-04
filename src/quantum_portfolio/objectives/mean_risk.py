from dataclasses import dataclass

import cvxpy as cp

from quantum_portfolio.constraints.base import Constraint

from .base import Objective


@dataclass
class MinVariance(Objective):
    name: str = "min_variance"
    def build_cvxpy_objective(self, context, w): return cp.Minimize(cp.quad_form(w, context.covariance.to_numpy(float)))

@dataclass
class MeanVarianceUtility(Objective):
    risk_aversion: float = 3.0
    name: str = "mean_variance_utility"
    def build_cvxpy_objective(self, context, w):
        return cp.Maximize(context.expected_returns.to_numpy(float) @ w - self.risk_aversion*cp.quad_form(w, context.covariance.to_numpy(float)))

@dataclass
class MaxSharpe(Objective):
    risk_free_rate: float = 0.0
    risk_aversion: float = 3.0
    name: str = "max_sharpe_quadratic_surrogate"
    def build_cvxpy_objective(self, context, w):
        return cp.Maximize((context.expected_returns.to_numpy(float)-self.risk_free_rate) @ w - self.risk_aversion*cp.quad_form(w, context.covariance.to_numpy(float)))

@dataclass
class TargetReturn(Constraint):
    target_return: float
    description: str = "minimum expected return"
    def build_cvxpy_constraints(self, context, w): return [context.expected_returns.to_numpy(float) @ w >= self.target_return]

@dataclass
class TargetRisk(Constraint):
    target_risk: float
    description: str = "maximum risk"
    def build_cvxpy_constraints(self, context, w): return [cp.quad_form(w, context.covariance.to_numpy(float)) <= self.target_risk**2]

@dataclass
class TargetReturnObjective(Objective):
    target_return: float
    name: str = "target_return_objective"
    def build_cvxpy_objective(self, context, w): return cp.Minimize(cp.quad_form(w, context.covariance.to_numpy(float)))
    def extra_constraints(self, context, w): return [context.expected_returns.to_numpy(float) @ w >= self.target_return]

@dataclass
class TargetRiskObjective(Objective):
    target_risk: float
    name: str = "target_risk_objective"
    def build_cvxpy_objective(self, context, w): return cp.Maximize(context.expected_returns.to_numpy(float) @ w)
    def extra_constraints(self, context, w): return [cp.quad_form(w, context.covariance.to_numpy(float)) <= self.target_risk**2]

@dataclass
class MinCVaR(Objective):
    alpha: float = 0.95
    name: str = "min_cvar"
    def build_cvxpy_objective(self, context, w):
        self._z = cp.Variable()
        self._u = cp.Variable(context.returns.shape[0])
        return cp.Minimize(self._z + cp.sum(self._u)/((1-self.alpha)*context.returns.shape[0]))
    def extra_constraints(self, context, w):
        losses = -context.returns.to_numpy(float) @ w
        return [self._u >= 0, self._u >= losses - self._z]
