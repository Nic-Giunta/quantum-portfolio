import pandas as pd
from quantum_portfolio.expected_returns import HistoricalMean, ExponentialWeightedMean, JamesSteinShrinkage, BayesianShrinkage, EnsembleExpectedReturns, BlackLitterman
def test_expected_shape(returns):
    for m in [HistoricalMean(), ExponentialWeightedMean(), JamesSteinShrinkage(), BayesianShrinkage(), EnsembleExpectedReturns([HistoricalMean(), ExponentialWeightedMean()])]:
        assert list(m.estimate(returns).index) == list(returns.columns)
def test_bl(returns):
    w = pd.Series(1/returns.shape[1], index=returns.columns); assert len(BlackLitterman(w).estimate(returns)) == returns.shape[1]
