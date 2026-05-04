from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.objectives import EqualRiskContribution
from quantum_portfolio.constraints import LongOnly
returns = make_synthetic_returns(seed=9)
print(PortfolioOptimizer(returns, objective=EqualRiskContribution(), constraints=[LongOnly()]).solve().weights)
