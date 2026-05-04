def portfolio_factor_exposure(weights, exposures): return exposures.reindex(weights.index).fillna(0).T @ weights
