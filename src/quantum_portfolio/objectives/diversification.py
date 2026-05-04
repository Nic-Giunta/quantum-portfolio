from dataclasses import dataclass

from .base import Objective


@dataclass
class MaxDiversification(Objective):
    name: str = "max_diversification"
    def build_cvxpy_objective(self, context, w): raise NotImplementedError("Max diversification uses the scipy path.")
