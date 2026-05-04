from quantum_portfolio.data import (
    prices_to_returns,
    temporal_train_test_split,
    validate_prices_dataframe,
    validate_returns_dataframe,
)


def test_validate_returns(returns):
    s = validate_returns_dataframe(returns); assert s.rows == len(returns)
def test_prices_to_returns(prices):
    validate_prices_dataframe(prices); assert prices_to_returns(prices).shape[0] == prices.shape[0]-1
def test_split(returns):
    tr, te = temporal_train_test_split(returns, .25); assert tr.index.max() < te.index.min()
