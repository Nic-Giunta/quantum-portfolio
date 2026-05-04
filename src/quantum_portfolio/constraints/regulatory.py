from dataclasses import dataclass

import pandas as pd

from .base import Constraint


@dataclass
class ESGScoreConstraint(Constraint):
    scores: pd.Series
    minimum_score: float
    description: str = "minimum ESG score"
    def build_cvxpy_constraints(self, context, w):
        return [self.scores.reindex(context.assets).to_numpy(float) @ w >= self.minimum_score]

@dataclass
class MinimumESGScore(ESGScoreConstraint): pass

@dataclass
class CarbonIntensityConstraint(Constraint):
    intensity: pd.Series
    maximum_intensity: float
    description: str = "maximum carbon intensity"
    def build_cvxpy_constraints(self, context, w):
        return [self.intensity.reindex(context.assets).to_numpy(float) @ w <= self.maximum_intensity]

@dataclass
class MaximumCarbonIntensity(CarbonIntensityConstraint): pass
