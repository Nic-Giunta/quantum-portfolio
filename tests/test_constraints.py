import pandas as pd

from quantum_portfolio import PortfolioOptimizer
from quantum_portfolio.constraints import (
    FactorExposure,
    GroupExposure,
    LongOnly,
    MaxWeight,
    TrackingErrorLimit,
    TurnoverLimit,
)
from quantum_portfolio.objectives import MinVariance


def test_long_max(returns):
    r = PortfolioOptimizer(returns, objective=MinVariance(), constraints=[LongOnly(), MaxWeight(.5)]).solve()
    assert (r.weights >= -1e-7).all(); assert (r.weights <= .5+1e-6).all()
def test_group_factor_turnover_tracking(returns):
    prev = pd.Series(1/returns.shape[1], index=returns.columns)
    groups = pd.Series(["A","A","B","B","B"], index=returns.columns)
    exp = pd.DataFrame({"beta":[1,1,1,1,1]}, index=returns.columns)
    r = PortfolioOptimizer(returns, objective=MinVariance(), constraints=[LongOnly(), GroupExposure(groups, upper={"A": .8}), TurnoverLimit(prev, 1.0), TrackingErrorLimit(prev, 1.0), FactorExposure(exp, target=pd.Series({"beta":1.0}), tolerance=1e-6)], previous_weights=prev, benchmark_weights=prev, factor_exposures=exp).solve()
    assert abs(r.weights.sum()-1) < 1e-5
