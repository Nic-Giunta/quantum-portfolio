from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from quantum_portfolio.utils.math import annualization_factor

from .base import ExpectedReturnModel


@dataclass
class CAPMExpectedReturns(ExpectedReturnModel):
    market_returns: pd.Series
    risk_free_rate: float = 0.0
    annualize: bool = True
    name: str = "capm"
    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        aligned = returns.join(self.market_returns.rename("market"), how="inner")
        m = aligned.pop("market")
        factor = annualization_factor(aligned.index)
        premium = m.mean() * factor - self.risk_free_rate if self.annualize else m.mean() - self.risk_free_rate
        var_m = float(np.var(m, ddof=1))
        betas = {c: 0.0 if var_m <= 1e-12 else float(np.cov(aligned[c], m, ddof=1)[0,1]/var_m) for c in aligned.columns}
        return pd.Series({k: self.risk_free_rate + b*premium for k,b in betas.items()})

@dataclass
class FactorExpectedReturns(ExpectedReturnModel):
    factor_returns: pd.DataFrame
    factor_premia: pd.Series | None = None
    include_alpha: bool = True
    annualize: bool = True
    name: str = "factor"
    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        aligned = returns.join(self.factor_returns, how="inner", rsuffix="_factor")
        x = aligned[self.factor_returns.columns]
        premia = self.factor_premia.reindex(x.columns) if self.factor_premia is not None else x.mean()
        if self.annualize: premia = premia * annualization_factor(aligned.index)
        out = {}
        for asset in returns.columns:
            lr = LinearRegression(fit_intercept=self.include_alpha).fit(x, aligned[asset])
            alpha = float(lr.intercept_) * (annualization_factor(aligned.index) if self.annualize else 1.0)
            out[asset] = alpha + float(np.asarray(lr.coef_) @ premia.to_numpy())
        return pd.Series(out)

@dataclass
class PCAFactorExpectedReturns(ExpectedReturnModel):
    n_components: int = 3
    annualize: bool = True
    name: str = "pca_factor"
    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        k = min(self.n_components, returns.shape[1], max(1, returns.shape[0]-1))
        f = PCA(n_components=k, random_state=0).fit_transform(returns.fillna(0.0))
        factor_df = pd.DataFrame(f, index=returns.index, columns=[f"PC{i+1}" for i in range(k)])
        return FactorExpectedReturns(factor_df, annualize=self.annualize).estimate(returns)
