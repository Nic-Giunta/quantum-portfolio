import pandas as pd
from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.expected_returns import BlackLitterman
returns = make_synthetic_returns(seed=10)
print(BlackLitterman(pd.Series(1/returns.shape[1], index=returns.columns)).estimate(returns))
