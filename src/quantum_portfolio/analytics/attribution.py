def return_attribution(weights, asset_returns): return weights.reindex(asset_returns.index).fillna(0) * asset_returns
def active_share(weights, benchmark_weights) -> float: return float(0.5*(weights-benchmark_weights.reindex(weights.index).fillna(0)).abs().sum())
