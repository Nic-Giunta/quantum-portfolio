from dataclasses import dataclass

from .base import RiskModel


def drawdown_series(returns):
    wealth = (1+returns).cumprod()
    return wealth/wealth.cummax()-1

@dataclass
class MaxDrawdown(RiskModel):
    name: str = "max_drawdown"
    def estimate(self, returns): return drawdown_series(returns).min()

@dataclass
class AverageDrawdown(RiskModel):
    name: str = "average_drawdown"
    def estimate(self, returns):
        dd = drawdown_series(returns); return dd[dd < 0].mean()

@dataclass
class CDaR(RiskModel):
    alpha: float = 0.95
    name: str = "cdar"
    def estimate(self, returns):
        losses = -drawdown_series(returns); var = losses.quantile(self.alpha)
        return losses.where(losses.ge(var)).mean()
