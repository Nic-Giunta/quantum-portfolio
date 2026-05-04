from dataclasses import dataclass

import cvxpy as cp

from .base import Constraint


@dataclass
class TaxBudgetConstraint(Constraint):
    max_realized_gain: float
    gain_per_weight_reduction: dict[str, float] | None = None
    description: str = "simplified tax budget"
    def build_cvxpy_constraints(self, context, w):
        if context.previous_weights is None or self.gain_per_weight_reduction is None: return []
        prev = context.previous_weights.reindex(context.assets).fillna(0).to_numpy(float)
        gains = [self.gain_per_weight_reduction.get(str(a), 0.0) for a in context.assets]
        return [cp.sum(cp.multiply(gains, cp.pos(prev - w))) <= self.max_realized_gain]
