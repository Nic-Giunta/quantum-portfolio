from quantum_portfolio.utils.math import annualization_factor
def infer_periods_per_year(index, default: int=252) -> int: return annualization_factor(index, default)
