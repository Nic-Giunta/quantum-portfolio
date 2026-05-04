import pytest
from quantum_portfolio.data.synthetic import make_synthetic_returns, make_synthetic_prices

@pytest.fixture
def returns():
    return make_synthetic_returns(n_periods=160, n_assets=5, seed=123)

@pytest.fixture
def prices():
    return make_synthetic_prices(n_periods=160, n_assets=5, seed=123)
