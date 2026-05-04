import pandas as pd
def adjust_prices_for_splits(prices: pd.DataFrame, split_factors: pd.DataFrame | None=None) -> pd.DataFrame:
    return prices.copy() if split_factors is None else prices / split_factors.reindex_like(prices).fillna(1.0).cumprod()
