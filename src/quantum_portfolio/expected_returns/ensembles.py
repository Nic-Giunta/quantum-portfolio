from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import ExpectedReturnModel


@dataclass
class EnsembleExpectedReturns(ExpectedReturnModel):
    models: list[ExpectedReturnModel]
    weights: list[float] | None = None
    name: str = "ensemble"
    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        if not self.models: raise ValueError("at least one model is required")
        est = [m.estimate(returns) for m in self.models]
        w = np.repeat(1/len(est), len(est)) if self.weights is None else np.asarray(self.weights, dtype=float)
        w = w / w.sum()
        frame = pd.concat(est, axis=1).fillna(0.0)
        return pd.Series(frame.to_numpy() @ w, index=frame.index)
