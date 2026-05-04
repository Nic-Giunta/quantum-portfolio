from abc import ABC, abstractmethod

import cvxpy as cp

from quantum_portfolio.optimization.problem import OptimizationContext


class Constraint(ABC):
    description: str = "constraint"
    def validate(self, context: OptimizationContext) -> None: return None
    @abstractmethod
    def build_cvxpy_constraints(self, context: OptimizationContext, w: cp.Variable) -> list[cp.Constraint]: ...
