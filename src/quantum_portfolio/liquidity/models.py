from dataclasses import dataclass

import pandas as pd


@dataclass
class ADVLiquidityModel:
    adv: pd.Series
    def max_trade_notional(self, max_participation: float=0.1, days: float=1.0): return self.adv*max_participation*days
    def report(self, weights: pd.Series, portfolio_value: float):
        notional = weights.abs()*portfolio_value; adv = self.adv.reindex(weights.index)
        return pd.DataFrame({"weight": weights, "notional": notional, "adv": adv, "days_to_trade_10pct_adv": notional/(0.1*adv)})
