import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def correlation_distance(correlation: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(np.sqrt((1-correlation.clip(-1,1))/2), index=correlation.index, columns=correlation.columns)

def hierarchical_linkage(returns: pd.DataFrame, method: str="single"):
    dist = correlation_distance(returns.corr().fillna(0.0))
    return linkage(squareform(dist.to_numpy(), checks=False), method=method)

def quasi_diagonal_order(returns: pd.DataFrame, method: str="single") -> list[str]:
    return list(returns.columns[leaves_list(hierarchical_linkage(returns, method))])
