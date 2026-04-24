"""
Core Enums for Signal Interpretation

This is the MOST IMPORTANT file in the interpretation system.
All signal analysis flows from these deterministic states.
"""

from enum import Enum


class SignalExistence(Enum):
    """Signal existence classification - deterministic."""
    NONE = "NONE"
    WEAK = "WEAK"
    PARTIAL = "PARTIAL"
    STRONG = "STRONG"


class AssetBehavior(Enum):
    """Signal behavior across assets - deterministic."""
    SINGLE = "SINGLE"          # Only works on one asset
    SELECTIVE = "SELECTIVE"    # Works on subset of assets
    UNIVERSAL = "UNIVERSAL"    # Works across all assets


class TimeStability(Enum):
    """Signal stability across time - deterministic."""
    NOT_ESTABLISHED = "NOT_ESTABLISHED"  # No stability data
    FRAGILE = "FRAGILE"                   # Direction survives, significance doesn't
    STABLE = "STABLE"                     # Both direction and significance survive


class Strength(Enum):
    """Overall signal strength classification - deterministic."""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


class AssetSignalStrength(Enum):
    """Strength of signal for individual assets - deterministic."""
    NONE = "NONE"
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"


# Consistency rules for deterministic state combinations
VALID_COMBINATIONS = {
    # Format: (existence, asset_behavior, time_stability, strength)
    # These are the only allowed combinations to prevent contradictions
    
    # No signal cases
    (SignalExistence.NONE, AssetBehavior.SINGLE, TimeStability.NOT_ESTABLISHED, Strength.WEAK),
    (SignalExistence.NONE, AssetBehavior.SELECTIVE, TimeStability.NOT_ESTABLISHED, Strength.WEAK),
    (SignalExistence.NONE, AssetBehavior.UNIVERSAL, TimeStability.NOT_ESTABLISHED, Strength.WEAK),
    
    # Weak signal cases
    (SignalExistence.WEAK, AssetBehavior.SINGLE, TimeStability.NOT_ESTABLISHED, Strength.MODERATE),
    (SignalExistence.WEAK, AssetBehavior.SELECTIVE, TimeStability.NOT_ESTABLISHED, Strength.MODERATE),
    (SignalExistence.WEAK, AssetBehavior.SINGLE, TimeStability.FRAGILE, Strength.MODERATE),
    
    # Partial signal cases
    (SignalExistence.PARTIAL, AssetBehavior.SELECTIVE, TimeStability.NOT_ESTABLISHED, Strength.MODERATE),
    (SignalExistence.PARTIAL, AssetBehavior.SELECTIVE, TimeStability.FRAGILE, Strength.MODERATE),
    (SignalExistence.PARTIAL, AssetBehavior.SELECTIVE, TimeStability.STABLE, Strength.MODERATE),
    (SignalExistence.PARTIAL, AssetBehavior.UNIVERSAL, TimeStability.NOT_ESTABLISHED, Strength.MODERATE),
    (SignalExistence.PARTIAL, AssetBehavior.UNIVERSAL, TimeStability.FRAGILE, Strength.MODERATE),
    (SignalExistence.PARTIAL, AssetBehavior.UNIVERSAL, TimeStability.STABLE, Strength.STRONG),
    
    # Strong signal cases
    (SignalExistence.STRONG, AssetBehavior.UNIVERSAL, TimeStability.STABLE, Strength.STRONG),
    (SignalExistence.STRONG, AssetBehavior.SELECTIVE, TimeStability.STABLE, Strength.STRONG),
}


def is_valid_combination(existence, asset_behavior, time_stability, strength):
    """Check if a combination is valid (no contradictions)."""
    return (existence, asset_behavior, time_stability, strength) in VALID_COMBINATIONS


def get_valid_combinations():
    """Get all valid combinations for reference."""
    return VALID_COMBINATIONS.copy()
