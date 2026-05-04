import numpy as np
import pandas as pd

def make_synthetic_returns(n_periods: int=252, n_assets: int=5, seed: int=42, freq: str="B") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n_assets, n_assets))
    cov = a @ a.T
    cov = cov / max(np.diag(cov).max(), 1e-12) * 0.0004
    mu = rng.normal(0.0003, 0.0002, n_assets)
    arr = rng.multivariate_normal(mu, cov, size=n_periods)
    return pd.DataFrame(arr, index=pd.date_range("2020-01-01", periods=n_periods, freq=freq), columns=[f"Asset_{i+1}" for i in range(n_assets)])

def make_synthetic_prices(n_periods: int=252, n_assets: int=5, seed: int=42) -> pd.DataFrame:
    return 100 * (1 + make_synthetic_returns(n_periods, n_assets, seed)).cumprod()
