import numpy as np
import pandas as pd
from quantum_portfolio.utils.exceptions import DataValidationError

def ensure_series(x, index=None, name="series") -> pd.Series:
    s = x if isinstance(x, pd.Series) else pd.Series(x, dtype=float)
    if index is not None:
        missing = set(index) - set(s.index)
        if missing: raise DataValidationError(f"{name} missing assets: {sorted(missing)}")
        s = s.reindex(index)
    return s.astype(float)

def ensure_psd_matrix(cov, tol: float = 1e-8) -> bool:
    arr = np.asarray(cov, dtype=float)
    return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and np.linalg.eigvalsh((arr+arr.T)/2).min() >= -tol
