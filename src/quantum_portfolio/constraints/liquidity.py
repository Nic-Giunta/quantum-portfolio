from dataclasses import dataclass
import cvxpy as cp
import pandas as pd
from .base import Constraint

@dataclass
class LiquidityLimit(Constraint):
    adv: pd.Series
    portfolio_value: float
    max_days_to_trade: float = 5.0
    participation_rate: float = 0.1
    previous_weights: pd.Series | None = None
    description: str = "ADV liquidity limit"
    def build_cvxpy_constraints(self, context, w):
        prev = self.previous_weights if self.previous_weights is not None else context.previous_weights
        prev_arr = 0.0 if prev is None else prev.reindex(context.assets).fillna(0).to_numpy(float)
        limit = self.adv.reindex(context.assets).to_numpy(float)*self.participation_rate*self.max_days_to_trade/self.portfolio_value
        return [cp.abs(w - prev_arr) <= limit]

@dataclass
class MaxADVParticipation(LiquidityLimit):
    description: str = "max ADV participation"
