from __future__ import annotations

import pandas as pd


def add_return_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Add simple return-based features for mini research experiments.

    Expected input:
    - DataFrame indexed by date
    - one price column, default: 'close'
    """
    if price_col not in df.columns:
        raise ValueError(f"Missing required price column: {price_col}")

    out = df.copy()

    out["ret_1d"] = out[price_col].pct_change(1)
    out["ret_3d"] = out[price_col].pct_change(3)
    out["ret_5d"] = out[price_col].pct_change(5)
    out["ret_10d"] = out[price_col].pct_change(10)
    out["ret_20d"] = out[price_col].pct_change(20)

    out["abs_ret_1d"] = out["ret_1d"].abs()
    out["abs_ret_3d"] = out["ret_3d"].abs()
    out["abs_ret_5d"] = out["ret_5d"].abs()
    out["abs_ret_10d"] = out["ret_10d"].abs()
    out["abs_ret_20d"] = out["ret_20d"].abs()

    out["fwd_ret_1d"] = out[price_col].shift(-1) / out[price_col] - 1
    out["fwd_ret_3d"] = out[price_col].shift(-3) / out[price_col] - 1
    out["fwd_ret_5d"] = out[price_col].shift(-5) / out[price_col] - 1
    out["fwd_ret_10d"] = out[price_col].shift(-10) / out[price_col] - 1
    out["fwd_ret_20d"] = out[price_col].shift(-20) / out[price_col] - 1
    
    out["fwd_abs_ret_1d"] = out["fwd_ret_1d"].abs()
    out["fwd_abs_ret_3d"] = out["fwd_ret_3d"].abs()
    out["fwd_abs_ret_5d"] = out["fwd_ret_5d"].abs()
    out["fwd_abs_ret_10d"] = out["fwd_ret_10d"].abs()
    out["fwd_abs_ret_20d"] = out["fwd_ret_20d"].abs()

    # Add moving average distance features
    out["ma10"] = out[price_col].rolling(window=10).mean()
    out["ma20"] = out[price_col].rolling(window=20).mean()
    out["ma50"] = out[price_col].rolling(window=50).mean()
    
    # Calculate percentage distance from moving averages
    out["dist_from_ma10"] = (out[price_col] - out["ma10"]) / out["ma10"]
    out["dist_from_ma20"] = (out[price_col] - out["ma20"]) / out["ma20"]
    out["dist_from_ma50"] = (out[price_col] - out["ma50"]) / out["ma50"]

    return out