from dataclasses import dataclass
import numpy as np
import pandas as pd
@dataclass
class MonteCarloScenarioGenerator:
    n_scenarios: int = 1000
    horizon: int = 1
    seed: int = 42
    def multivariate_normal(self, returns):
        rng = np.random.default_rng(self.seed)
        arr = rng.multivariate_normal(returns.mean(), returns.cov(), size=self.n_scenarios*self.horizon)
        return pd.DataFrame(arr, columns=returns.columns)
    def bootstrap(self, returns):
        rng = np.random.default_rng(self.seed); idx = rng.integers(0, len(returns), self.n_scenarios*self.horizon)
        return returns.iloc[idx].reset_index(drop=True)
