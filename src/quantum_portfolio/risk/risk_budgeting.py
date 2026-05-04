import numpy as np
import pandas as pd


def marginal_risk_contribution(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    cov = covariance.loc[weights.index, weights.index]; sigw = cov @ weights
    vol = float(np.sqrt(max(weights @ sigw, 0.0)))
    return sigw / max(vol, 1e-12)
def component_risk_contribution(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    crc = weights * marginal_risk_contribution(weights, covariance)
    return crc / crc.sum() if abs(float(crc.sum())) > 1e-12 else crc
