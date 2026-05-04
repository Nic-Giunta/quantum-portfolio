from dataclasses import dataclass

import pandas as pd

from .metrics import performance_metrics
from .rebalancing import make_rebalance_calendar


@dataclass
class BacktestResult:
    returns: pd.Series
    equity_curve: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    metrics: dict[str, float]

@dataclass
class BacktestEngine:
    returns: pd.DataFrame
    strategy: object
    rebalance_frequency: str = "ME"
    lookback: int = 252
    transaction_cost_rate: float = 0.001
    slippage_rate: float = 0.0005
    execution_mode: str = "close_to_close"
    cash: bool = False
    benchmark: pd.Series | None = None
    def run(self):
        dates = make_rebalance_calendar(self.returns.index, self.rebalance_frequency)
        cur = pd.Series(0.0, index=self.returns.columns); rows=[]; bt={}; turns={}; costs={}
        for date in dates:
            loc = self.returns.index.get_loc(date)
            if isinstance(loc, slice) or loc < self.lookback: continue
            target = self.strategy(self.returns.iloc[loc-self.lookback:loc], cur).reindex(self.returns.columns).fillna(0)
            trn = float((target-cur).abs().sum()); cost = trn*(self.transaction_cost_rate+self.slippage_rate)
            turns[date]=trn; costs[date]=cost; rows.append(target.rename(date)); cur = target
            next_loc = min(loc+1, len(self.returns)-1) if self.execution_mode == "next_open" else loc
            bt[self.returns.index[next_loc]] = float(self.returns.iloc[next_loc] @ cur - cost)
        full = pd.Series(0.0, index=self.returns.index); full.loc[pd.Series(bt).index] = pd.Series(bt)
        equity = (1+full).cumprod(); weights = pd.DataFrame(rows) if rows else pd.DataFrame(columns=self.returns.columns)
        turnover = pd.Series(turns); tx = pd.Series(costs)
        return BacktestResult(full, equity, weights, turnover, tx, performance_metrics(full, benchmark=self.benchmark, turnover=turnover, transaction_costs=tx))
