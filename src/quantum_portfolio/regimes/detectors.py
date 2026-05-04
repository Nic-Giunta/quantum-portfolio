from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VolatilityRegimeDetector:
    window: int = 63
    low_quantile: float = 0.33
    high_quantile: float = 0.67
    def fit_predict(self, returns: pd.DataFrame) -> pd.Series:
        vol = returns.mean(axis=1).rolling(self.window).std()
        lo, hi = vol.quantile([self.low_quantile, self.high_quantile])
        out = pd.Series("medium_vol", index=returns.index)
        out[vol <= lo] = "low_vol"; out[vol >= hi] = "high_vol"
        return out

@dataclass
class TrendRegimeDetector:
    fast_window: int = 21
    slow_window: int = 126
    def fit_predict(self, prices_or_wealth: pd.DataFrame) -> pd.Series:
        s = prices_or_wealth.mean(axis=1); f = s.rolling(self.fast_window).mean(); sl = s.rolling(self.slow_window).mean()
        out = pd.Series("sideways", index=prices_or_wealth.index); out[f>sl] = "uptrend"; out[f<sl] = "downtrend"; return out

@dataclass
class CorrelationRegimeDetector:
    window: int = 63
    high_quantile: float = 0.75
    def fit_predict(self, returns: pd.DataFrame) -> pd.Series:
        vals = {}
        for i in range(self.window, len(returns)+1):
            c = returns.iloc[i-self.window:i].corr().to_numpy()
            vals[returns.index[i-1]] = float(c[~np.eye(c.shape[0], dtype=bool)].mean())
        avg = pd.Series(vals).reindex(returns.index)
        threshold = avg.quantile(self.high_quantile)
        out = pd.Series("normal_corr", index=returns.index); out[avg >= threshold] = "high_corr"; return out
