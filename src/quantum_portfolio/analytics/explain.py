import pandas as pd

from quantum_portfolio.risk.risk_budgeting import (
    component_risk_contribution as component_risk_contribution,
)
from quantum_portfolio.risk.risk_budgeting import (
    marginal_risk_contribution as marginal_risk_contribution,
)


def explain_weights(weights): return pd.DataFrame({"weight": weights, "abs_weight": weights.abs(), "rank": weights.abs().rank(ascending=False)})
