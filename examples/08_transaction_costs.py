from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.objectives import MinTransactionCosts
returns = make_synthetic_returns(seed=14); prev = returns.iloc[-1]*0 + 1/returns.shape[1]
print(PortfolioOptimizer(returns, objective=MinTransactionCosts(), previous_weights=prev).solve().weights)
