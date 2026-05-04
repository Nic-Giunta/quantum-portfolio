from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.expected_returns import HistoricalMean
from quantum_portfolio.risk import LedoitWolfCovariance
from quantum_portfolio.objectives import MeanVarianceUtility
from quantum_portfolio.constraints import LongOnly, MaxWeight
returns = make_synthetic_returns(seed=7)
res = PortfolioOptimizer(returns, HistoricalMean(), LedoitWolfCovariance(), MeanVarianceUtility(3.0), [LongOnly(), MaxWeight(0.4)]).solve()
print(res.summary())
