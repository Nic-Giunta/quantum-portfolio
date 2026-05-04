def plot_equity_curve(equity_curve): return equity_curve.plot(title="Equity curve")
def plot_drawdown(returns):
    wealth = (1+returns).cumprod(); return (wealth/wealth.cummax()-1).plot(title="Drawdown")
def plot_turnover(turnover): return turnover.plot(kind="bar", title="Turnover")
