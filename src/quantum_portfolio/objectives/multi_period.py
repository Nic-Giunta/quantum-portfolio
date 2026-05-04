from dataclasses import dataclass


@dataclass
class MultiPeriodUtility:
    risk_aversion: float = 5.0
    turnover_penalty: float = 0.001
    terminal_weight: float = 0.0
