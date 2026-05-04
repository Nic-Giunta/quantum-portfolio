from dataclasses import dataclass

from .base import Scenario


@dataclass
class HistoricalScenario(Scenario):
    start: str
    end: str
    def apply(self, returns): return returns.loc[self.start:self.end].copy()
