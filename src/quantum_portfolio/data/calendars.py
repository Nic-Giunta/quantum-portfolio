import pandas as pd


def rebalance_dates(index: pd.DatetimeIndex, frequency: str="ME") -> pd.DatetimeIndex:
    groups = pd.Series(index, index=index).groupby(pd.Grouper(freq=frequency))
    return pd.DatetimeIndex([g.iloc[-1] for _, g in groups if len(g)])
