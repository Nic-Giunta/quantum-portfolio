from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OptimizerEnsemble:
    optimizers: list
    method: str = "mean"
    validation_scores: pd.Series | None = None
    def solve(self) -> pd.Series:
        results = [o.solve() for o in self.optimizers]
        frame = pd.concat([r.weights.rename(i) for i,r in enumerate(results)], axis=1).fillna(0)
        if self.method == "mean": return frame.mean(axis=1)
        if self.method == "median": return frame.median(axis=1)
        if self.method == "validation_sharpe":
            scores = self.validation_scores if self.validation_scores is not None else pd.Series([r.expected_return/max(r.risk,1e-12) for r in results])
            w = scores.clip(lower=0).to_numpy(float); w = w/w.sum() if w.sum() > 0 else np.repeat(1/len(results), len(results))
            return pd.Series(frame.to_numpy() @ w, index=frame.index)
        raise ValueError("unknown ensemble method")

@dataclass
class ConsensusPortfolio:
    weights: pd.Series
    @classmethod
    def from_results(cls, results, method: str="mean"):
        frame = pd.concat([r.weights for r in results], axis=1).fillna(0)
        w = frame.mean(axis=1) if method == "mean" else frame.median(axis=1)
        return cls(w/w.sum())
