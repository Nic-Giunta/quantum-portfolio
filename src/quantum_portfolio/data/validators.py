from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from quantum_portfolio.utils.exceptions import DataValidationError

@dataclass(frozen=True)
class DataValidationSummary:
    rows: int
    columns: int
    missing_cells: int
    duplicate_columns: list[str] = field(default_factory=list)
    monotonic_index: bool = True
    estimated_frequency: str | None = None
    warnings: list[str] = field(default_factory=list)
    def to_dict(self) -> dict[str, object]: return self.__dict__.copy()

def estimate_frequency(index: pd.Index) -> str | None:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3: return None
    inferred = pd.infer_freq(index)
    if inferred: return inferred
    days = float(np.median(np.diff(index.view("i8")) / 86_400_000_000_000))
    return "D/B" if days <= 1.5 else "W" if days <= 8 else "M" if days <= 32 else "A"

def _validate_frame(df: pd.DataFrame, label: str, allow_missing: bool, min_rows: int) -> DataValidationSummary:
    if not isinstance(df, pd.DataFrame): raise DataValidationError(f"{label} must be a DataFrame")
    if df.shape[0] < min_rows or df.shape[1] < 1: raise DataValidationError(f"{label} has invalid shape {df.shape}")
    if not isinstance(df.index, pd.DatetimeIndex): raise DataValidationError(f"{label} index must be DatetimeIndex")
    dups = df.columns[df.columns.duplicated()].astype(str).tolist()
    if dups: raise DataValidationError(f"duplicate columns: {dups}")
    if not df.index.is_monotonic_increasing: raise DataValidationError(f"{label} index must be monotonic increasing")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    missing = int(numeric.isna().sum().sum())
    if missing and not allow_missing: raise DataValidationError(f"{label} contains {missing} missing/non-numeric cells")
    warnings = []
    if label == "returns" and (numeric.abs() > 5).any().any(): warnings.append("returns above 500%; check scale")
    return DataValidationSummary(len(df), df.shape[1], missing, dups, True, estimate_frequency(df.index), warnings)

def validate_returns_dataframe(returns: pd.DataFrame, *, allow_missing: bool=False, min_rows: int=2, require_datetime_index: bool=True) -> DataValidationSummary:
    return _validate_frame(returns, "returns", allow_missing, min_rows)

def validate_prices_dataframe(prices: pd.DataFrame, *, allow_missing: bool=False, min_rows: int=2, require_positive: bool=True) -> DataValidationSummary:
    summary = _validate_frame(prices, "prices", allow_missing, min_rows)
    if require_positive and (prices.apply(pd.to_numeric, errors="coerce").dropna() <= 0).any().any():
        raise DataValidationError("prices must be positive")
    return summary
