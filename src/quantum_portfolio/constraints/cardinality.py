from dataclasses import dataclass

import cvxpy as cp

from .base import Constraint


@dataclass
class CardinalityConstraint(Constraint):
    max_positions: int
    min_position_size: float = 0.0
    allow_mixed_integer: bool = False
    description: str = "cardinality constraint"
    def build_cvxpy_constraints(self, context, w):
        if not self.allow_mixed_integer:
            raise NotImplementedError("CardinalityConstraint requires allow_mixed_integer=True and a MIP solver.")
        z = cp.Variable(context.n_assets, boolean=True)
        return [w <= z, w >= self.min_position_size*z, cp.sum(z) <= self.max_positions]
