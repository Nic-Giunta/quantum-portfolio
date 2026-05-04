from quantum_portfolio.returns import compound_simple_returns, infer_periods_per_year


def test_returns(returns):
    assert compound_simple_returns(returns).shape == returns.shape
    assert infer_periods_per_year(returns.index) in {252,52,12,1}
