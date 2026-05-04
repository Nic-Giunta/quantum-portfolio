from dataclasses import dataclass

from .base import Scenario


@dataclass
class ShockScenario(Scenario):
    asset_shocks: object | None = None
    parallel_shock: float = 0.0
    group_shocks: dict[str, float] | None = None
    groups: object | None = None
    volatility_multiplier: float = 1.0
    correlation_shock: float = 0.0
    def apply(self, returns):
        out = returns.copy()*self.volatility_multiplier + self.parallel_shock
        if self.asset_shocks is not None: out = out + self.asset_shocks.reindex(out.columns).fillna(0)
        if self.group_shocks and self.groups is not None:
            for g, shock in self.group_shocks.items():
                assets = self.groups[self.groups == g].index.intersection(out.columns)
                out.loc[:, assets] = out.loc[:, assets] + shock
        if self.correlation_shock:
            avg = out.mean(axis=1); out = (1-self.correlation_shock)*out + self.correlation_shock*avg.to_numpy()[:,None]
        return out
