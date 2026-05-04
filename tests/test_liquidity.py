import pandas as pd

from quantum_portfolio.liquidity import ADVLiquidityModel, estimate_capacity


def test_liq():
    w = pd.Series({"A":.5,"B":.5}); adv = pd.Series({"A":1_000_000,"B":2_000_000})
    assert estimate_capacity(w, .2, adv) > 0
    assert "adv" in ADVLiquidityModel(adv).report(w, 1_000_000).columns
