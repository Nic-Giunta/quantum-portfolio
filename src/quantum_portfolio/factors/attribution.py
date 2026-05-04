def factor_return_attribution(weights, exposures, factor_returns):
    e = exposures.reindex(weights.index).fillna(0).T @ weights
    return e * factor_returns.reindex(e.index)
