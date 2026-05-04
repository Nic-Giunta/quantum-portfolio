from quantum_portfolio.data.calendars import rebalance_dates


def make_rebalance_calendar(index, frequency: str='ME'): return rebalance_dates(index, frequency)
