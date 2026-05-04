from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class OptimizationContext:
    returns: pd.DataFrame
    expected_returns: pd.Series
    covariance: pd.DataFrame
    assets: pd.Index
    previous_weights: pd.Series | None = None
    benchmark_weights: pd.Series | None = None
    factor_exposures: pd.DataFrame | None = None
    groups: pd.Series | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    @property
    def n_assets(self) -> int: return len(self.assets)
