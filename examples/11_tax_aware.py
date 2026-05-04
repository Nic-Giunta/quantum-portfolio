from datetime import date
import pandas as pd
from quantum_portfolio.tax import TaxLot, TaxLotBook, tax_loss_harvesting_candidates
book = TaxLotBook([TaxLot("A", 10, 100, date(2024,1,1))])
print(tax_loss_harvesting_candidates(book, pd.Series({"A": 90})))
