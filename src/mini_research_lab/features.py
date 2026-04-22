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

    out["abs_ret_1d"] = out["ret_1d"].abs()

    out["fwd_ret_1d"] = out[price_col].shift(-1) / out[price_col] - 1
    out["fwd_ret_3d"] = out[price_col].shift(-3) / out[price_col] - 1
    out["fwd_abs_ret_1d"] = out["fwd_ret_1d"].abs()

    out["ma_20"] = out[price_col].rolling(20).mean()
    out["dist_from_ma20"] = out[price_col] / out["ma_20"] - 1

    return out