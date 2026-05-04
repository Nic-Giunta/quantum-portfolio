from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantum_portfolio.constraints import Constraint, TargetReturn
from quantum_portfolio.expected_returns import ExpectedReturnModel
from quantum_portfolio.objectives import MinVariance
from quantum_portfolio.optimization.optimizer import PortfolioOptimizer
from quantum_portfolio.risk import RiskModel


@dataclass
class EfficientFrontier:
    returns: pd.DataFrame
    expected_return_model: ExpectedReturnModel
    risk_model: RiskModel
    constraints: list[Constraint]
    def compute(self, n_points: int=25, solver: str | None=None) -> pd.DataFrame:
        mu = self.expected_return_model.estimate(self.returns)
        rows = []
        for target in np.linspace(float(mu.min()), float(mu.max()), n_points):
            try:
                res = PortfolioOptimizer(self.returns, self.expected_return_model, self.risk_model, MinVariance(), [*self.constraints, TargetReturn(float(target))]).solve(solver=solver)
                rows.append({"target_return": target, "return": res.expected_return, "risk": res.risk, "sharpe": res.expected_return/res.risk if res.risk else np.nan, "status": res.solver_status})
            except Exception as exc:
                rows.append({"target_return": target, "return": np.nan, "risk": np.nan, "sharpe": np.nan, "status": f"infeasible/error: {exc}"})
        return pd.DataFrame(rows)
    @staticmethod
    def plot(frontier: pd.DataFrame):
        ax = frontier.plot(x="risk", y="return", kind="line", marker="o", title="Efficient frontier")
        ax.set_xlabel("Risk"); ax.set_ylabel("Expected return"); return ax
    @staticmethod
    def export_csv(frontier: pd.DataFrame, path: str | Path) -> None: frontier.to_csv(path, index=False)
