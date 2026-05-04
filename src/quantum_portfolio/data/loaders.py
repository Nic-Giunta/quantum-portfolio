from pathlib import Path
import pandas as pd
def load_csv_timeseries(path: str | Path, *, index_col: int | str=0) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=index_col, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index)
    return df.sort_index()
