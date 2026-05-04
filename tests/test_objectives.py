from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.constraints import LongOnly
from quantum_portfolio.objectives import (
    EqualRiskContribution,
    MaxDiversification,
    MeanVarianceUtility,
    MinCVaR,
    MinVariance,
    RobustMeanVariance,
)


def test_objectives(returns):
    for obj in [MinVariance(), MeanVarianceUtility(), MinCVaR(), RobustMeanVariance(), MaxDiversification(), EqualRiskContribution()]:
        r = PortfolioOptimizer(returns, objective=obj, constraints=[LongOnly()]).solve()
        assert abs(r.weights.sum()-1) < 1e-5
