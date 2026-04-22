from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise ImportError(
        "yfinance is required. Install with: pip install yfinance"
    ) from exc


def download_prices(
    ticker: str,
    start: str,
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Download OHLCV price data for a ticker and normalize columns.

    Returns a DataFrame with lowercase column names.
    Expected main column for the framework: 'close'
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker={ticker}")

    df = df.copy()
    
    # Handle multi-level column names from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [str(col).lower() for col in df.columns]
    df.index.name = "date"

    required = {"close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {sorted(missing)}")

    return df


def save_dataframe_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Save a DataFrame to CSV, creating parent folders if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def load_prices_csv(path: str | Path, index_col: str = "date") -> pd.DataFrame:
    """
    Load a CSV saved from save_dataframe_csv / download_prices output.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, parse_dates=[index_col], index_col=index_col)
    df.columns = [str(col).lower() for col in df.columns]
    return df