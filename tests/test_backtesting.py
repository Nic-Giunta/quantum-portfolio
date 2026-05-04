from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.objectives import MinVariance
from quantum_portfolio.constraints import LongOnly
from quantum_portfolio.backtest import BacktestEngine
def test_backtest(returns):
    def strategy(train, prev): return PortfolioOptimizer(train, objective=MinVariance(), constraints=[LongOnly()], previous_weights=prev).solve().weights
    r = BacktestEngine(returns, strategy, lookback=60).run()
    assert "sharpe" in r.metrics and len(r.equity_curve) == len(returns)
