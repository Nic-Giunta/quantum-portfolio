from dataclasses import dataclass
import cvxpy as cp
import pandas as pd
from .base import Constraint

@dataclass
class GroupExposure(Constraint):
    groups: pd.Series
    lower: dict[str, float] | None = None
    upper: dict[str, float] | None = None
    description: str = "group exposure bounds"
    def build_cvxpy_constraints(self, context, w):
        g = self.groups.reindex(context.assets)
        cons = []
        for group in sorted(g.dropna().unique()):
            idx = [i for i,a in enumerate(context.assets) if g.loc[a] == group]
            if self.lower and group in self.lower: cons.append(cp.sum(w[idx]) >= self.lower[group])
            if self.upper and group in self.upper: cons.append(cp.sum(w[idx]) <= self.upper[group])
        return cons

@dataclass
class RiskBudgetByGroup(GroupExposure):
    description: str = "simplified group risk-budget proxy as exposure bands"
