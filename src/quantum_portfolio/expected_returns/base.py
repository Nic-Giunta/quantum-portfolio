from abc import ABC, abstractmethod

import pandas as pd


class ExpectedReturnModel(ABC):
    name = "expected_return_model"
    def fit(self, returns: pd.DataFrame, **kwargs): return self
    @abstractmethod
    def estimate(self, returns: pd.DataFrame) -> pd.Series: ...
    def fit_estimate(self, returns: pd.DataFrame, **kwargs) -> pd.Series:
        return self.fit(returns, **kwargs).estimate(returns)
