from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time
import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from quantum_portfolio.constraints import Constraint, WeightSum
from quantum_portfolio.data.validators import validate_returns_dataframe
from quantum_portfolio.expected_returns import ExpectedReturnModel, HistoricalMean
from quantum_portfolio.risk import RiskModel, SampleCovariance
from quantum_portfolio.objectives import Objective, MinVariance, RiskParityObjective, MaxDiversification
from quantum_portfolio.optimization.problem import OptimizationContext
from quantum_portfolio.optimization.result import OptimizationResult
from quantum_portfolio.optimization.solvers import choose_solver
from quantum_portfolio.utils.exceptions import OptimizationError

@dataclass
class PortfolioOptimizer:
    returns: pd.DataFrame
    expected_return_model: ExpectedReturnModel | None = None
    risk_model: RiskModel | None = None
    objective: Objective | None = None
    constraints: list[Constraint] = field(default_factory=list)
    previous_weights: pd.Series | None = None
    benchmark_weights: pd.Series | None = None
    factor_exposures: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    add_weight_sum: bool = True

    def validate(self): return validate_returns_dataframe(self.returns)

    def _context(self) -> OptimizationContext:
        self.validate()
        erm = self.expected_return_model or HistoricalMean()
        rm = self.risk_model or SampleCovariance()
        mu = erm.estimate(self.returns).reindex(self.returns.columns).astype(float)
        cov = rm.estimate(self.returns)
        if not isinstance(cov, pd.DataFrame): raise OptimizationError("risk_model must return covariance DataFrame")
        cov = cov.reindex(index=self.returns.columns, columns=self.returns.columns).astype(float)
        return OptimizationContext(self.returns, mu, cov, self.returns.columns, self.previous_weights, self.benchmark_weights, self.factor_exposures, metadata=self.metadata.copy())

    def diagnostics(self) -> dict[str, Any]:
        return {"data_validation": self.validate().to_dict(), "n_constraints": len(self.constraints)}

    def _all_constraints(self):
        c = list(self.constraints)
        if self.add_weight_sum and not any(isinstance(x, WeightSum) for x in c): c.append(WeightSum(1.0))
        return c

    def solve(self, solver: str | None=None, **solver_options: Any) -> OptimizationResult:
        ctx = self._context(); obj = self.objective or MinVariance()
        if isinstance(obj, RiskParityObjective): return self._solve_risk_parity(ctx, obj)
        if isinstance(obj, MaxDiversification): return self._solve_max_diversification(ctx, obj)
        w = cp.Variable(ctx.n_assets)
        cons = []; reports = []
        for c in self._all_constraints():
            c.validate(ctx)
            built = c.build_cvxpy_constraints(ctx, w)
            cons.extend(built)
            reports.append({"type": c.__class__.__name__, "description": getattr(c, "description", ""), "n_cvxpy": len(built)})
        obj.validate(ctx)
        cp_obj = obj.build_cvxpy_objective(ctx, w)
        cons.extend(obj.extra_constraints(ctx, w))
        problem = cp.Problem(cp_obj, cons)
        chosen = choose_solver(solver)
        start = time.perf_counter()
        try:
            value = problem.solve(solver=chosen, **solver_options)
        except Exception as exc:
            raise OptimizationError(f"CVXPY solve failed with {chosen}: {exc}") from exc
        solve_time = time.perf_counter() - start
        if w.value is None: raise OptimizationError(f"optimization failed: {problem.status}")
        weights = pd.Series(np.asarray(w.value).reshape(-1), index=ctx.assets, name="weight").where(lambda s: s.abs()>1e-10, 0.0)
        return OptimizationResult.from_weights(weights, ctx.expected_returns, ctx.covariance, objective_value=None if value is None else float(value), solver_status=str(problem.status), solver_name=chosen, solve_time=solve_time, diagnostics={**self.diagnostics(), "objective": obj.__class__.__name__, "input_shape": self.returns.shape}, constraints_report=reports)

    def _scipy_constraints(self, ctx):
        cons = [{"type": "eq", "fun": lambda x: np.sum(x)-1.0}]
        bounds = [(None, None)] * ctx.n_assets
        for c in self.constraints:
            if c.__class__.__name__ == "LongOnly": bounds = [(0.0, hi) for _, hi in bounds]
            elif c.__class__.__name__ == "MaxWeight": bounds = [(lo, c.maximum if hi is None else min(hi, c.maximum)) for lo, hi in bounds]
            elif c.__class__.__name__ == "MinWeight": bounds = [(c.minimum if lo is None else max(lo, c.minimum), hi) for lo, hi in bounds]
        return cons, bounds

    def _solve_risk_parity(self, ctx, obj):
        cov = ctx.covariance.to_numpy(float); n = ctx.n_assets
        budgets = np.repeat(1/n, n) if obj.budgets is None else np.asarray(obj.budgets, dtype=float); budgets = budgets/budgets.sum()
        def loss(x):
            vol = np.sqrt(max(float(x @ cov @ x), 1e-12))
            rc = x * (cov @ x) / vol
            pct = rc / max(rc.sum(), 1e-12)
            return float(((pct-budgets)**2).sum())
        res = minimize(loss, np.repeat(1/n, n), method="SLSQP", bounds=self._scipy_constraints(ctx)[1], constraints=self._scipy_constraints(ctx)[0], options={"maxiter": 1000})
        if not res.success: raise OptimizationError(f"risk parity failed: {res.message}")
        weights = pd.Series(res.x, index=ctx.assets, name="weight")
        return OptimizationResult.from_weights(weights, ctx.expected_returns, ctx.covariance, objective_value=float(res.fun), solver_status="optimal", solver_name="scipy-slsqp", solve_time=None, diagnostics={**self.diagnostics(), "objective": obj.__class__.__name__})

    def _solve_max_diversification(self, ctx, obj):
        cov = ctx.covariance.to_numpy(float); n = ctx.n_assets; vols = np.sqrt(np.diag(cov))
        def loss(x): return -float(vols @ x / np.sqrt(max(float(x @ cov @ x), 1e-12)))
        res = minimize(loss, np.repeat(1/n, n), method="SLSQP", bounds=self._scipy_constraints(ctx)[1], constraints=self._scipy_constraints(ctx)[0], options={"maxiter": 1000})
        if not res.success: raise OptimizationError(f"max diversification failed: {res.message}")
        weights = pd.Series(res.x, index=ctx.assets, name="weight")
        return OptimizationResult.from_weights(weights, ctx.expected_returns, ctx.covariance, objective_value=float(-res.fun), solver_status="optimal", solver_name="scipy-slsqp", solve_time=None, diagnostics={**self.diagnostics(), "objective": obj.__class__.__name__})
