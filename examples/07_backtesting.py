from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.objectives import MinVariance
from quantum_portfolio.constraints import LongOnly
from quantum_portfolio.backtest import BacktestEngine
returns = make_synthetic_returns(n_periods=500, seed=13)
def strategy(train, prev): return PortfolioOptimizer(train, objective=MinVariance(), constraints=[LongOnly()], previous_weights=prev).solve().weights
print(BacktestEngine(returns, strategy, lookback=126).run().metrics)
