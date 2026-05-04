from .base import Scenario
from .historical import HistoricalScenario
from .stress import ShockScenario
from .monte_carlo import MonteCarloScenarioGenerator
from .bootstrap import bootstrap_returns
class ScenarioAnalyzer:
    def __init__(self, weights): self.weights = weights
    def pnl(self, scenario_returns): return scenario_returns.reindex(columns=self.weights.index).fillna(0) @ self.weights
    def worst_scenario(self, scenario_returns) -> float: return float(self.pnl(scenario_returns).min())
    def report(self, scenario_returns):
        pnl = self.pnl(scenario_returns); return {"mean_pnl": float(pnl.mean()), "worst_pnl": float(pnl.min()), "best_pnl": float(pnl.max())}
