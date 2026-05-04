import pandas as pd
from sklearn.linear_model import LinearRegression


def estimate_factor_betas(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    aligned = returns.join(factors, how="inner", rsuffix="_factor"); x = aligned[factors.columns]
    return pd.DataFrame({a: LinearRegression().fit(x, aligned[a]).coef_ for a in returns.columns}, index=factors.columns).T
