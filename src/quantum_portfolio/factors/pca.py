import pandas as pd
from sklearn.decomposition import PCA


def pca_factor_exposures(returns: pd.DataFrame, n_components: int=3) -> pd.DataFrame:
    p = PCA(n_components=min(n_components, returns.shape[1]), random_state=0).fit(returns.fillna(0))
    return pd.DataFrame(p.components_.T, index=returns.columns, columns=[f"PC{i+1}" for i in range(p.n_components_)])
