import pandas as pd


def estimate_capacity(weights, turnover: float, adv, max_participation: float=0.05) -> float:
    active = weights.abs().replace(0, pd.NA).dropna()
    if active.empty or turnover <= 0: return float("inf")
    return float((adv.reindex(active.index)*max_participation/(active*turnover)).min())
