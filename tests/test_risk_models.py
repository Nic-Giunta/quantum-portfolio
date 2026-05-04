import numpy as np, pytest
from quantum_portfolio.risk import SampleCovariance, LedoitWolfCovariance, OASCovariance, CVaR, EVaR
def test_covs(returns):
    for m in [SampleCovariance(), LedoitWolfCovariance(), OASCovariance()]:
        c = m.estimate(returns); assert c.shape == (returns.shape[1], returns.shape[1]); assert np.linalg.eigvalsh(c).min() > -1e-7
def test_tail(returns):
    assert CVaR().estimate(returns).shape[0] == returns.shape[1]
    with pytest.raises(NotImplementedError): EVaR().estimate(returns)
