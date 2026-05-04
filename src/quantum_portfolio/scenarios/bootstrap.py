import numpy as np
def bootstrap_returns(returns, n: int, seed: int=42):
    rng = np.random.default_rng(seed); return returns.iloc[rng.integers(0, len(returns), n)].reset_index(drop=True)
