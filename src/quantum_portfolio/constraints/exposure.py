from dataclasses import dataclass
import cvxpy as cp
import pandas as pd
from .base import Constraint

@dataclass
class FactorExposure(Constraint):
    exposures: pd.DataFrame | None = None
    lower: pd.Series | None = None
    upper: pd.Series | None = None
    target: pd.Series | None = None
    tolerance: float = 0.0
    description: str = "factor exposure constraints"
    def build_cvxpy_constraints(self, context, w):
        exp = self.exposures if self.exposures is not None else context.factor_exposures
        if exp is None: raise ValueError("FactorExposure requires exposures")
        x = exp.reindex(context.assets).to_numpy(float).T
        names = exp.columns
        cons = []
        if self.lower is not None: cons.append(x @ w >= self.lower.reindex(names).to_numpy(float))
        if self.upper is not None: cons.append(x @ w <= self.upper.reindex(names).to_numpy(float))
        if self.target is not None:
            t = self.target.reindex(names).to_numpy(float)
            cons.extend([x @ w >= t - self.tolerance, x @ w <= t + self.tolerance])
        return cons

@dataclass
class SectorNeutrality(FactorExposure):
    description: str = "sector neutrality"

@dataclass
class CountryExposure(FactorExposure):
    description: str = "country exposure"

@dataclass
class DurationExposure(Constraint):
    duration: pd.Series
    min_duration: float | None = None
    max_duration: float | None = None
    description: str = "duration exposure"
    def build_cvxpy_constraints(self, context, w):
        d = self.duration.reindex(context.assets).to_numpy(float)
        cons = []
        if self.min_duration is not None: cons.append(d @ w >= self.min_duration)
        if self.max_duration is not None: cons.append(d @ w <= self.max_duration)
        return cons

@dataclass
class ConvexityExposure(DurationExposure):
    description: str = "convexity exposure"

@dataclass
class GreeksExposureConstraint(FactorExposure):
    description: str = "greeks exposure"
