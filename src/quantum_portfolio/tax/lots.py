from dataclasses import dataclass
from datetime import date
import pandas as pd

@dataclass(frozen=True)
class TaxLot:
    asset: str
    quantity: float
    cost_basis: float
    acquisition_date: date
    def unrealized_gain(self, price: float) -> float: return self.quantity * (price - self.cost_basis)

@dataclass
class TaxLotBook:
    lots: list[TaxLot]
    @classmethod
    def from_positions(cls, positions: pd.DataFrame):
        return cls([TaxLot(str(r.asset), float(r.quantity), float(r.cost_basis), pd.Timestamp(r.acquisition_date).date()) for r in positions.itertuples(index=False)])
    def unrealized_gains(self, prices: pd.Series) -> pd.Series:
        out = {}
        for lot in self.lots: out[lot.asset] = out.get(lot.asset, 0.0) + lot.unrealized_gain(float(prices.loc[lot.asset]))
        return pd.Series(out)
    def realized_gain_for_sale(self, asset: str, quantity: float, price: float) -> float:
        remaining = quantity; gain = 0.0
        for lot in sorted([x for x in self.lots if x.asset == asset], key=lambda x: x.acquisition_date):
            q = min(remaining, lot.quantity); gain += q*(price-lot.cost_basis); remaining -= q
            if remaining <= 1e-12: break
        return gain
