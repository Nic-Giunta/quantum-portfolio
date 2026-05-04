from datetime import date

import pandas as pd

from quantum_portfolio.tax import TaxLot, TaxLotBook, tax_loss_harvesting_candidates


def test_tax():
    b = TaxLotBook([TaxLot("A", 10, 100, date(2024,1,1))]); p = pd.Series({"A": 90})
    assert b.unrealized_gains(p).loc["A"] == -100
    assert len(tax_loss_harvesting_candidates(b,p)) == 1
