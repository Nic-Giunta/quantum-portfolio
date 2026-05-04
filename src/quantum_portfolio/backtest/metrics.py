import numpy as np
import pandas as pd


def performance_metrics(returns, *, benchmark=None, periods_per_year: int=252, turnover=None, transaction_costs=None):
    r = returns.dropna(); wealth = (1+r).cumprod(); years = len(r)/periods_per_year
    cumulative = float(wealth.iloc[-1]-1) if len(wealth) else 0.0
    cagr = float(wealth.iloc[-1]**(1/years)-1) if years > 0 and len(wealth) else 0.0
    vol = float(r.std(ddof=1)*np.sqrt(periods_per_year)) if len(r)>1 else 0.0
    sharpe = float(r.mean()*periods_per_year/vol) if vol > 1e-12 else np.nan
    downside = r.clip(upper=0).std(ddof=1)*np.sqrt(periods_per_year)
    wealth = wealth if len(wealth) else pd.Series([1.0])
    dd = wealth/wealth.cummax()-1; maxdd = float(dd.min())
    var = float(r.quantile(0.05)) if len(r) else np.nan
    out = {"cumulative_return": cumulative, "cagr": cagr, "annualized_volatility": vol, "sharpe": sharpe, "sortino": float(r.mean()*periods_per_year/downside) if downside>1e-12 else np.nan, "max_drawdown": maxdd, "calmar": float(cagr/abs(maxdd)) if abs(maxdd)>1e-12 else np.nan, "hit_ratio": float((r>0).mean()) if len(r) else np.nan, "skew": float(r.skew()) if len(r) else np.nan, "kurtosis": float(r.kurtosis()) if len(r) else np.nan, "var_95": var, "cvar_95": float(r[r<=var].mean()) if len(r) else np.nan, "average_turnover": float(turnover.mean()) if turnover is not None and len(turnover) else 0.0, "average_transaction_cost": float(transaction_costs.mean()) if transaction_costs is not None and len(transaction_costs) else 0.0}
    if benchmark is not None:
        aligned = pd.concat([r, benchmark], axis=1).dropna(); active = aligned.iloc[:,0]-aligned.iloc[:,1]
        te = float(active.std(ddof=1)*np.sqrt(periods_per_year)) if len(active)>1 else 0.0
        out["tracking_error"] = te; out["information_ratio"] = float(active.mean()*periods_per_year/te) if te>1e-12 else np.nan
    return out
