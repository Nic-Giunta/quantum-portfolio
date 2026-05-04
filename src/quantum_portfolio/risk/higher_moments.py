from dataclasses import dataclass
import pandas as pd
from .base import RiskModel

@dataclass
class HigherMomentRisk(RiskModel):
    name: str = "higher_moments"
    def estimate(self, returns: pd.DataFrame):
        z = (returns-returns.mean())/returns.std(ddof=0).replace(0, pd.NA)
        cols = returns.columns
        coskew = pd.DataFrame(index=cols, columns=cols, dtype=float)
        cokurt = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for i in cols:
            for j in cols:
                coskew.loc[i,j] = float((z[i]*z[j]**2).mean())
                cokurt.loc[i,j] = float((z[i]**2*z[j]**2).mean())
        return {"skewness": (z**3).mean(), "kurtosis": (z**4).mean(), "coskewness": coskew, "cokurtosis": cokurt}
