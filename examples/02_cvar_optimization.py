from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.objectives import MinCVaR
from quantum_portfolio.constraints import LongOnly
returns = make_synthetic_returns(seed=8)
print(PortfolioOptimizer(returns, objective=MinCVaR(), constraints=[LongOnly()]).solve().weights)
