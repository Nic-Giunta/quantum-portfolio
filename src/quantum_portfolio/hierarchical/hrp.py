from dataclasses import dataclass
import numpy as np
import pandas as pd
from .clustering import quasi_diagonal_order

def _cluster_variance(cov: pd.DataFrame, assets: list[str]) -> float:
    sub = cov.loc[assets, assets]
    inv = 1/np.diag(sub); w = inv/inv.sum()
    return float(w @ sub.to_numpy() @ w)

@dataclass
class HRPAllocator:
    linkage_method: str = "single"
    def allocate(self, returns: pd.DataFrame) -> pd.Series:
        cov = returns.cov(); ordered = quasi_diagonal_order(returns, self.linkage_method)
        w = pd.Series(1.0, index=ordered); clusters = [ordered]
        while clusters:
            c = clusters.pop(0)
            if len(c) <= 1: continue
            split = len(c)//2; left, right = c[:split], c[split:]
            vl, vr = _cluster_variance(cov,left), _cluster_variance(cov,right)
            alpha = 1 - vl/(vl+vr)
            w[left] *= alpha; w[right] *= 1-alpha
            clusters += [left, right]
        return w.reindex(returns.columns).fillna(0)/w.sum()
