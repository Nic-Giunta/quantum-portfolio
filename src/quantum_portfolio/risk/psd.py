import numpy as np
import pandas as pd

def eigenvalue_clipping(covariance: pd.DataFrame, min_eigenvalue: float=1e-8) -> pd.DataFrame:
    vals, vecs = np.linalg.eigh((covariance.to_numpy()+covariance.to_numpy().T)/2)
    arr = vecs @ np.diag(np.clip(vals, min_eigenvalue, None)) @ vecs.T
    return pd.DataFrame((arr+arr.T)/2, index=covariance.index, columns=covariance.columns)

def diagonal_loading(covariance: pd.DataFrame, loading: float=1e-6) -> pd.DataFrame:
    arr = covariance.to_numpy().copy(); arr += np.eye(arr.shape[0])*loading
    return pd.DataFrame(arr, index=covariance.index, columns=covariance.columns)

def nearest_psd(covariance: pd.DataFrame, min_eigenvalue: float=1e-8) -> pd.DataFrame:
    return eigenvalue_clipping(covariance, min_eigenvalue)
