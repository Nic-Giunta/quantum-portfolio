from dataclasses import dataclass

import cvxpy as cp
import pandas as pd

from .base import Objective


@dataclass
class MinTransactionCosts(Objective):
    cost_per_unit: float | pd.Series = 0.001
    risk_weight: float = 1.0
    name: str = "min_transaction_costs"
    def build_cvxpy_objective(self, context, w):
        prev = 0.0 if context.previous_weights is None else context.previous_weights.reindex(context.assets).fillna(0).to_numpy(float)
        cost = self.cost_per_unit if isinstance(self.cost_per_unit, float) else self.cost_per_unit.reindex(context.assets).to_numpy(float)
        return cp.Minimize(self.risk_weight*cp.quad_form(w, context.covariance.to_numpy(float)) + cp.sum(cp.multiply(cost, cp.abs(w-prev))))
