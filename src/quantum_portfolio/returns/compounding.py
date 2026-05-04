import numpy as np
def compound_simple_returns(returns): return (1+returns).cumprod()-1
def cumulative_log_returns(log_returns): return np.exp(log_returns.cumsum())-1
