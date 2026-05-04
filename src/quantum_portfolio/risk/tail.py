from dataclasses import dataclass
import numpy as np
import pandas as pd
from .base import RiskModel

@dataclass
class CVaR(RiskModel):
    alpha: float = 0.95
    name: str = "cvar"
    def estimate(self, returns):
        losses = -returns; var = losses.quantile(self.alpha)
        return losses.where(losses.ge(var), np.nan).mean()

@dataclass
class EVaR(RiskModel):
    alpha: float = 0.95
    name: str = "evar"
    def estimate(self, returns):
        raise NotImplementedError("EVaR needs an audited exponential-cone formulation; intentionally not implemented in this alpha.")
