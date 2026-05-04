from abc import ABC, abstractmethod
import pandas as pd
class RiskModel(ABC):
    name = "risk_model"
    @abstractmethod
    def estimate(self, returns: pd.DataFrame): ...
