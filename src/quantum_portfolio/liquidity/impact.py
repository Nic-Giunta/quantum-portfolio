import numpy as np


def linear_transaction_cost_model(trade_notional, rate: float=0.001): return trade_notional.abs()*rate
def square_root_market_impact(trade_notional, adv, volatility, coefficient: float=0.1):
    return coefficient * volatility.reindex(trade_notional.index) * np.sqrt((trade_notional.abs()/adv.reindex(trade_notional.index)).clip(lower=0))
