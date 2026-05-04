from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .base import Objective


@dataclass
class RobustMeanVariance(Objective):
    risk_aversion: float = 3.0
    uncertainty_radius: float = 0.01
    uncertainty: str = "box"
    uncertainty_covariance: object | None = None
    name: str = "robust_mean_variance"
    def build_cvxpy_objective(self, context, w):
        cov = context.covariance.to_numpy(float)
        penalty = self.uncertainty_radius * cp.norm1(w)
        if self.uncertainty == "ellipsoid":
            vals, vecs = np.linalg.eigh((cov+cov.T)/2)
            root = vecs @ np.diag(np.sqrt(np.clip(vals, 0, None))) @ vecs.T
            penalty = self.uncertainty_radius * cp.norm(root @ w, 2)
        return cp.Maximize(context.expected_returns.to_numpy(float) @ w - penalty - self.risk_aversion*cp.quad_form(w, cov))
