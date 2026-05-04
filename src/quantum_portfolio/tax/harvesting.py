import pandas as pd

from .lots import TaxLotBook


def tax_loss_harvesting_candidates(book: TaxLotBook, prices: pd.Series, min_loss: float=0.0) -> pd.DataFrame:
    rows = []
    for lot in book.lots:
        gain = lot.unrealized_gain(float(prices.loc[lot.asset]))
        if gain < -abs(min_loss): rows.append({"asset": lot.asset, "quantity": lot.quantity, "unrealized_loss": gain, "acquisition_date": lot.acquisition_date})
    return pd.DataFrame(rows)
