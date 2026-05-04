import pandas as pd

from quantum_portfolio.optimization import MultiPeriodOptimizer


def test_multi_period(returns):
    assets = returns.columns; mu = pd.DataFrame([returns.mean()*252]*3, columns=assets, index=pd.date_range("2026-01-31", periods=3, freq="ME"))
    r = MultiPeriodOptimizer(mu, [returns.cov()*252]*3, pd.Series(1/len(assets), index=assets), max_turnover=2).solve()
    assert r.weights_by_period.shape == (3, len(assets))
