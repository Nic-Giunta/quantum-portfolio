from __future__ import annotations
from typing import Protocol, Any, TypeAlias
import numpy as np
import pandas as pd
ArrayLike: TypeAlias = np.ndarray | pd.Series | pd.DataFrame
Weights: TypeAlias = pd.Series
class Fittable(Protocol):
    def fit(self, returns: pd.DataFrame, **kwargs: Any) -> Any: ...
