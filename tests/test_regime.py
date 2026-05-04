from quantum_portfolio.regimes import (
    CorrelationRegimeDetector,
    TrendRegimeDetector,
    VolatilityRegimeDetector,
)


def test_regimes(returns):
    assert len(VolatilityRegimeDetector(window=20).fit_predict(returns)) == len(returns)
    assert len(TrendRegimeDetector(5,20).fit_predict((1+returns).cumprod())) == len(returns)
    assert len(CorrelationRegimeDetector(window=20).fit_predict(returns)) == len(returns)
