from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.constraints import LongOnly, TargetReturn
from quantum_portfolio.objectives import MinVariance


def test_target_return(returns):
    base = PortfolioOptimizer(returns, objective=MinVariance(), constraints=[LongOnly()]).solve()
    r = PortfolioOptimizer(returns, objective=MinVariance(), constraints=[LongOnly(), TargetReturn(min(base.expected_return, .5))]).solve()
    assert r.risk >= 0
