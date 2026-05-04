import pandas as pd
from quantum_portfolio.liquidity import ADVLiquidityModel, estimate_capacity
w = pd.Series({"A": .4, "B": .6}); adv = pd.Series({"A": 1_000_000, "B": 5_000_000})
print(ADVLiquidityModel(adv).report(w, 10_000_000)); print(estimate_capacity(w, .2, adv))
