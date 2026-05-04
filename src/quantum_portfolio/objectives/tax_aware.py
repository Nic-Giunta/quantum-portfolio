from dataclasses import dataclass
import cvxpy as cp
from .mean_risk import MeanVarianceUtility

@dataclass
class TaxAwareUtility(MeanVarianceUtility):
    tax_penalty: float = 1.0
    gain_per_weight_reduction: dict[str, float] | None = None
    name: str = "tax_aware_utility"
    def build_cvxpy_objective(self, context, w):
        base = super().build_cvxpy_objective(context, w).args[0]
        if context.previous_weights is None or not self.gain_per_weight_reduction: return cp.Maximize(base)
        prev = context.previous_weights.reindex(context.assets).fillna(0).to_numpy(float)
        gains = [self.gain_per_weight_reduction.get(str(a), 0.0) for a in context.assets]
        return cp.Maximize(base - self.tax_penalty*cp.sum(cp.multiply(gains, cp.pos(prev-w))))
