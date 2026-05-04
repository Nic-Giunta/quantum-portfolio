from quantum_portfolio.data.calendars import rebalance_dates
def make_rebalance_calendar(index, frequency: str='M'): return rebalance_dates(index, frequency)
