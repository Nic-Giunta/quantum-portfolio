from abc import ABC, abstractmethod
from dataclasses import dataclass
import cvxpy as cp

@dataclass
class Objective(ABC):
    name: str = "objective"
    def validate(self, context) -> None: return None
    @abstractmethod
    def build_cvxpy_objective(self, context, w): ...
    def extra_constraints(self, context, w) -> list[cp.Constraint]: return []
