from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.objectives import RobustMeanVariance
from quantum_portfolio.constraints import LongOnly
print(PortfolioOptimizer(make_synthetic_returns(seed=12), objective=RobustMeanVariance(), constraints=[LongOnly()]).solve().summary())
