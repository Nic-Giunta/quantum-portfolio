from quantum_portfolio.constraints import LongOnly
from quantum_portfolio.expected_returns import HistoricalMean
from quantum_portfolio.optimization import EfficientFrontier
from quantum_portfolio.risk import SampleCovariance


def test_frontier(returns):
    df = EfficientFrontier(returns, HistoricalMean(), SampleCovariance(), [LongOnly()]).compute(n_points=4)
    assert len(df) == 4 and "status" in df
