# QuantumPortfolio

QuantumPortfolio is a clean-room Python framework for portfolio optimization, portfolio construction, risk modelling, backtesting, reporting, and reproducible quantitative research.

The design is composable:

```python
from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.expected_returns import HistoricalMean
from quantum_portfolio.risk import LedoitWolfCovariance
from quantum_portfolio.constraints import LongOnly, MaxWeight
from quantum_portfolio.objectives import MeanVarianceUtility

optimizer = PortfolioOptimizer(
    returns=returns,
    expected_return_model=HistoricalMean(),
    risk_model=LedoitWolfCovariance(),
    objective=MeanVarianceUtility(risk_aversion=3.0),
    constraints=[LongOnly(), MaxWeight(0.10)],
)
result = optimizer.solve()
print(result.summary())
```

## Clean-room originality

This repository intentionally does not copy source code, internal names, implementation structure, examples, docs, or tests from existing portfolio optimization libraries. It implements standard public mathematical and financial concepts through an original API and architecture.

## Features

- Data validation, price-to-return conversion, missing data handling, outlier flags, temporal splits, rolling windows.
- Expected return estimators: historical, EWMA, James-Stein-style shrinkage, Bayesian shrinkage, CAPM, factor/PCA, Black-Litterman, ensembles, ML wrapper.
- Risk models: sample/EWMA/Ledoit-Wolf/OAS covariance, semivariance, downside deviation, MAD, CVaR, drawdown, higher moments, PSD repair.
- Optimization: minimum variance, mean-variance utility, target return/risk, CVaR, risk parity, maximum diversification, robust mean-variance, transaction cost/tax aware, multi-period.
- Constraints: long-only, weights, bounds, leverage, turnover, group/factor exposure, tracking error, liquidity/ADV, ESG/carbon, tax budget, neutrality, optional cardinality.
- Hierarchical allocation: HRP, simplified HERC and NCO.
- Backtesting with transaction costs, slippage, rebalancing, benchmark metrics, and walk-forward style strategy injection.
- Scenario lab, analytics, reports, experiment tracking, plugin registry, config-driven optimization and CLI.

## Installation

```bash
pip install -e ".[dev,cli,docs]"
```

## Disclaimer

QuantumPortfolio is for research, education, and quantitative analysis. It is not financial, investment, legal, tax, accounting, fiduciary, or regulatory advice. Tax, liquidity, execution, and derivative models are simplified.
