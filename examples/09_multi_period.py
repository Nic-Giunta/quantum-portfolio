import pandas as pd
from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.optimization import MultiPeriodOptimizer
r = make_synthetic_returns(seed=15); assets = r.columns
mu = pd.DataFrame([r.mean()*252]*3, index=pd.date_range("2026-01-31", periods=3, freq="M"), columns=assets)
print(MultiPeriodOptimizer(mu, [r.cov()*252]*3, pd.Series(1/len(assets), index=assets)).solve().weights_by_period)
