import pandas as pd
from quantum_portfolio.data.synthetic import make_synthetic_returns
from quantum_portfolio.factors import estimate_factor_betas
r = make_synthetic_returns(seed=17); print(estimate_factor_betas(r, pd.DataFrame({"MKT": r.mean(axis=1)})))
