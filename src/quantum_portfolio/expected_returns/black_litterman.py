from dataclasses import dataclass
import numpy as np
import pandas as pd
from .base import ExpectedReturnModel
from quantum_portfolio.risk.covariance import SampleCovariance

@dataclass
class BlackLitterman(ExpectedReturnModel):
    market_weights: pd.Series
    risk_aversion: float = 2.5
    tau: float = 0.05
    P: pd.DataFrame | np.ndarray | None = None
    Q: pd.Series | np.ndarray | None = None
    omega: pd.DataFrame | np.ndarray | None = None
    covariance: pd.DataFrame | None = None
    name: str = "black_litterman"

    @staticmethod
    def market_implied_prior(covariance: pd.DataFrame, market_weights: pd.Series, risk_aversion: float=2.5) -> pd.Series:
        w = market_weights.reindex(covariance.index).astype(float)
        return pd.Series(risk_aversion * covariance.to_numpy() @ w.to_numpy(), index=covariance.index)

    def estimate(self, returns: pd.DataFrame) -> pd.Series:
        cov = self.covariance if self.covariance is not None else SampleCovariance().estimate(returns)
        pi = self.market_implied_prior(cov, self.market_weights, self.risk_aversion)
        if self.P is None or self.Q is None: return pi
        P = np.asarray(self.P, dtype=float); Q = np.asarray(self.Q, dtype=float).reshape(-1)
        sigma = cov.to_numpy(dtype=float); tau_sigma = self.tau * sigma
        omega = np.asarray(self.omega, dtype=float) if self.omega is not None else np.diag(np.diag(P @ tau_sigma @ P.T))
        post_cov = np.linalg.pinv(np.linalg.pinv(tau_sigma) + P.T @ np.linalg.pinv(omega) @ P)
        post_mu = post_cov @ (np.linalg.pinv(tau_sigma) @ pi.to_numpy() + P.T @ np.linalg.pinv(omega) @ Q)
        return pd.Series(post_mu, index=cov.index)

    def posterior_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        cov = self.covariance if self.covariance is not None else SampleCovariance().estimate(returns)
        if self.P is None: return cov
        P = np.asarray(self.P, dtype=float); sigma = cov.to_numpy(dtype=float); tau_sigma = self.tau*sigma
        omega = np.asarray(self.omega, dtype=float) if self.omega is not None else np.diag(np.diag(P @ tau_sigma @ P.T))
        post = sigma + np.linalg.pinv(np.linalg.pinv(tau_sigma)+P.T@np.linalg.pinv(omega)@P)
        return pd.DataFrame(post, index=cov.index, columns=cov.columns)
