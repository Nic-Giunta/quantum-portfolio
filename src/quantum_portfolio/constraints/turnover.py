from dataclasses import dataclass

import cvxpy as cp
import pandas as pd

from .base import Constraint


@dataclass
class TurnoverLimit(Constraint):
    previous_weights: pd.Series | None = None
    max_turnover: float = 0.25
    description: str = "L1 turnover limit"
    def build_cvxpy_constraints(self, context, w):
        prev = self.previous_weights if self.previous_weights is not None else context.previous_weights
        if prev is None: raise ValueError("TurnoverLimit requires previous_weights")
        return [cp.norm1(w - prev.reindex(context.assets).fillna(0.0).to_numpy(float)) <= self.max_turnover]
