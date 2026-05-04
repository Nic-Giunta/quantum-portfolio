from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from .base import Constraint


@dataclass
class LongOnly(Constraint):
    description: str = "weights must be non-negative"
    def build_cvxpy_constraints(self, context, w): return [w >= 0]

@dataclass
class WeightSum(Constraint):
    total: float = 1.0
    description: str = "weights sum target"
    def build_cvxpy_constraints(self, context, w): return [cp.sum(w) == self.total]

@dataclass
class MinWeight(Constraint):
    minimum: float
    description: str = "minimum asset weight"
    def build_cvxpy_constraints(self, context, w): return [w >= self.minimum]

@dataclass
class MaxWeight(Constraint):
    maximum: float
    description: str = "maximum asset weight"
    def build_cvxpy_constraints(self, context, w): return [w <= self.maximum]

@dataclass
class BoxBounds(Constraint):
    lower: float | pd.Series
    upper: float | pd.Series
    description: str = "box bounds"
    def build_cvxpy_constraints(self, context, w):
        lo = self.lower if np.isscalar(self.lower) else self.lower.reindex(context.assets).to_numpy(float)
        hi = self.upper if np.isscalar(self.upper) else self.upper.reindex(context.assets).to_numpy(float)
        return [w >= lo, w <= hi]

@dataclass
class LeverageLimit(Constraint):
    max_leverage: float = 1.0
    description: str = "gross leverage limit"
    def build_cvxpy_constraints(self, context, w): return [cp.norm1(w) <= self.max_leverage]

@dataclass
class DollarNeutral(Constraint):
    description: str = "dollar neutral"
    def build_cvxpy_constraints(self, context, w): return [cp.sum(w) == 0]

@dataclass
class MarketNeutral(Constraint):
    beta: pd.Series | None = None
    description: str = "market beta neutral"
    def build_cvxpy_constraints(self, context, w):
        beta = self.beta if self.beta is not None else context.metadata.get("beta")
        if beta is None: raise ValueError("MarketNeutral requires beta exposures")
        return [beta.reindex(context.assets).to_numpy(float) @ w == 0]
