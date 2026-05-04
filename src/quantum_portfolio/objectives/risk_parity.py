from dataclasses import dataclass

from .base import Objective


@dataclass
class RiskParityObjective(Objective):
    budgets: object | None = None
    name: str = "risk_parity"
    def build_cvxpy_objective(self, context, w): raise NotImplementedError("Risk parity uses the scipy path.")
@dataclass
class EqualRiskContribution(RiskParityObjective):
    name: str = "equal_risk_contribution"
