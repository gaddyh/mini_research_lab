from __future__ import annotations

import pandas as pd
import numpy as np


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


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_rsi_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Add RSI-based features for mean reversion analysis.
    
    Expected input:
    - DataFrame indexed by date
    - one price column, default: 'close'
    """
    if price_col not in df.columns:
        raise ValueError(f"Missing required price column: {price_col}")
    
    out = df.copy()
    
    # Calculate RSI with different periods
    out["rsi_14"] = calculate_rsi(out[price_col], 14)
    out["rsi_7"] = calculate_rsi(out[price_col], 7)
    
    # RSI oversold recovery signals
    # Signal when RSI crosses above 30 after being oversold (RSI < 30)
    out["rsi_14_oversold"] = (out["rsi_14"] < 30).astype(int)
    out["rsi_14_oversold_recovery"] = ((out["rsi_14_oversold"].shift(1) == 1) & (out["rsi_14"] > 30)).astype(int)
    out["rsi_7_oversold"] = (out["rsi_7"] < 20).astype(int)  # More aggressive for shorter period
    out["rsi_7_oversold_recovery"] = ((out["rsi_7_oversold"].shift(1) == 1) & (out["rsi_7"] > 20)).astype(int)
    
    # RSI overbought signals (for completeness)
    out["rsi_14_overbought"] = (out["rsi_14"] > 70).astype(int)
    out["rsi_14_overbought_breakdown"] = ((out["rsi_14_overbought"].shift(1) == 1) & (out["rsi_14"] < 70)).astype(int)
    out["rsi_7_overbought"] = (out["rsi_7"] > 80).astype(int)
    out["rsi_7_overbought_breakdown"] = ((out["rsi_7_overbought"].shift(1) == 1) & (out["rsi_7"] < 80)).astype(int)
    
    # RSI bucket features for better analysis
    out["rsi_14_bucket"] = pd.cut(
        out["rsi_14"], 
        bins=[0, 30, 50, 70, 100], 
        labels=["oversold", "neutral_low", "neutral_high", "overbought"],
        include_lowest=True
    )
    
    # Create binary features for each bucket (for regression)
    out["rsi_14_oversold_bucket"] = (out["rsi_14_bucket"] == "oversold").astype(int)
    out["rsi_14_neutral_low_bucket"] = (out["rsi_14_bucket"] == "neutral_low").astype(int)
    out["rsi_14_neutral_high_bucket"] = (out["rsi_14_bucket"] == "neutral_high").astype(int)
    out["rsi_14_overbought_bucket"] = (out["rsi_14_bucket"] == "overbought").astype(int)
    
    return out


def add_donchian_features(df: pd.DataFrame, price_col: str = "close", period: int = 20) -> pd.DataFrame:
    """
    Add Donchian channel breakout features.
    
    Expected input:
    - DataFrame indexed by date
    - one price column, default: 'close'
    """
    if price_col not in df.columns:
        raise ValueError(f"Missing required price column: {price_col}")
    
    out = df.copy()
    
    # Calculate Donchian channels
    out[f"donchian_{period}d_high"] = out[price_col].rolling(window=period).max()
    out[f"donchian_{period}d_low"] = out[price_col].rolling(window=period).min()
    out[f"donchian_{period}d_mid"] = (out[f"donchian_{period}d_high"] + out[f"donchian_{period}d_low"]) / 2
    
    # Breakout signals
    out[f"donchian_{period}d_position"] = (out[price_col] - out[f"donchian_{period}d_low"]) / (out[f"donchian_{period}d_high"] - out[f"donchian_{period}d_low"])
    
    # Breakout events
    out[f"donchian_{period}d_upper_breakout"] = (out[price_col] > out[f"donchian_{period}d_high"].shift(1)).astype(int)
    out[f"donchian_{period}d_lower_breakout"] = (out[price_col] < out[f"donchian_{period}d_low"].shift(1)).astype(int)
    
    # Breakout after being in channel (stronger signal)
    in_channel_upper = out[price_col] <= out[f"donchian_{period}d_high"].shift(1)
    in_channel_lower = out[price_col] >= out[f"donchian_{period}d_low"].shift(1)
    in_channel = in_channel_upper & in_channel_lower
    
    out[f"donchian_{period}d_upper_breakout_strong"] = (in_channel.shift(1) & out[f"donchian_{period}d_upper_breakout"]).astype(int)
    out[f"donchian_{period}d_lower_breakout_strong"] = (in_channel.shift(1) & out[f"donchian_{period}d_lower_breakout"]).astype(int)
    
    return out


def add_event_based_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Add event-based signal features for crossover and threshold crossing analysis.
    
    This function generates binary event signals rather than level-based features.
    """
    out = df.copy()
    
    # Add base features needed for event detection
    out = add_return_features(out, price_col)
    out = add_rsi_features(out, price_col)
    out = add_donchian_features(out, price_col, 20)
    out = add_donchian_features(out, price_col, 10)
    
    # Moving average crossovers
    out["ma10"] = out[price_col].rolling(window=10).mean()
    out["ma20"] = out[price_col].rolling(window=20).mean()
    out["ma50"] = out[price_col].rolling(window=50).mean()
    
    # Golden cross events (short MA crosses above long MA)
    out["golden_cross_10_20"] = ((out["ma10"] > out["ma20"]) & (out["ma10"].shift(1) <= out["ma20"].shift(1))).astype(int)
    out["golden_cross_20_50"] = ((out["ma20"] > out["ma50"]) & (out["ma20"].shift(1) <= out["ma50"].shift(1))).astype(int)
    
    # Death cross events (short MA crosses below long MA)
    out["death_cross_10_20"] = ((out["ma10"] < out["ma20"]) & (out["ma10"].shift(1) >= out["ma20"].shift(1))).astype(int)
    out["death_cross_20_50"] = ((out["ma20"] < out["ma50"]) & (out["ma20"].shift(1) >= out["ma50"].shift(1))).astype(int)
    
    # Price crossovers with moving averages
    out["price_cross_above_ma20"] = ((out[price_col] > out["ma20"]) & (out[price_col].shift(1) <= out["ma20"].shift(1))).astype(int)
    out["price_cross_below_ma20"] = ((out[price_col] < out["ma20"]) & (out[price_col].shift(1) >= out["ma20"].shift(1))).astype(int)
    
    # Enhanced RSI events (already exist but ensure they're included)
    # These are already created in add_rsi_features:
    # - rsi_14_oversold_recovery
    # - rsi_14_overbought_breakdown
    # - rsi_7_oversold_recovery  
    # - rsi_7_overbought_breakdown
    
    # Donchian channel breakout events (already exist but ensure they're included)
    # These are already created in add_donchian_features:
    # - donchian_10d_upper_breakout_strong
    # - donchian_10d_lower_breakout_strong
    # - donchian_20d_upper_breakout_strong
    # - donchian_20d_lower_breakout_strong
    
    return out


def add_strategy_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Add all strategy-based features (RSI, Donchian, etc.).
    
    This function combines all strategy feature generators.
    """
    out = df.copy()
    out = add_return_features(out, price_col)
    out = add_rsi_features(out, price_col)
    out = add_donchian_features(out, price_col, 20)  # 20-day Donchian
    out = add_donchian_features(out, price_col, 10)  # 10-day Donchian
    return out