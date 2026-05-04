from dataclasses import dataclass
import cvxpy as cp
import pandas as pd
from .base import Constraint

@dataclass
class TrackingErrorLimit(Constraint):
    benchmark_weights: pd.Series | None = None
    max_tracking_error: float = 0.05
    description: str = "tracking error limit"
    def build_cvxpy_constraints(self, context, w):
        b = self.benchmark_weights if self.benchmark_weights is not None else context.benchmark_weights
        if b is None: raise ValueError("TrackingErrorLimit requires benchmark_weights")
        arr = b.reindex(context.assets).fillna(0.0).to_numpy(float)
        return [cp.quad_form(w - arr, context.covariance.to_numpy(float)) <= self.max_tracking_error**2]
