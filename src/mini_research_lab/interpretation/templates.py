"""
Deterministic Templates for Signal Interpretation

This file contains ONLY text templates - no logic, no calculations.
Each enum state maps to exact human-readable text.
"""

from typing import Dict, List
from .enums import SignalExistence, AssetBehavior, TimeStability, Strength


class ExistenceTemplates:
    """Templates for signal existence descriptions."""
    
    TEMPLATES = {
        SignalExistence.NONE: "No reliable signal detected.",
        SignalExistence.WEAK: "Weak evidence of signal, not statistically reliable.",
        SignalExistence.PARTIAL: "Signal exists in some assets but not universally.",
        SignalExistence.STRONG: "Signal exists reliably across tested assets."
    }
    
    @classmethod
    def get_text(cls, existence: SignalExistence) -> str:
        """Get deterministic text for existence state."""
        return cls.TEMPLATES[existence]


class AssetBehaviorTemplates:
    """Templates for asset behavior descriptions."""
    
    UNIVERSAL_TEMPLATE = "Universal. Signal works consistently across all analyzed assets."
    SELECTIVE_TEMPLATE = "Selective. Signal works in specific assets but not universally."
    SINGLE_TEMPLATE = "Single. Signal works in only one tested asset."
    NONE_TEMPLATE = "No meaningful signal across assets."
    
    @classmethod
    def get_text(cls, behavior: AssetBehavior, asset_details: Dict[str, List[str]]) -> str:
        """Get deterministic text for asset behavior with asset details."""
        if behavior == AssetBehavior.UNIVERSAL:
            return cls.UNIVERSAL_TEMPLATE
        elif behavior == AssetBehavior.SELECTIVE:
            parts = []
            for strength, assets in asset_details.items():
                if assets:
                    parts.append(f"{strength.capitalize()}: {', '.join(assets)}")
            if parts:
                return f"Selective.\n- " + "\n- ".join(parts) + "."
            else:
                return cls.SELECTIVE_TEMPLATE
        elif behavior == AssetBehavior.SINGLE:
            return cls.SINGLE_TEMPLATE
        else:
            return cls.NONE_TEMPLATE


class TimeStabilityTemplates:
    """Templates for time stability descriptions."""
    
    TEMPLATES = {
        TimeStability.NOT_ESTABLISHED: "Not established. Stability analysis does not confirm robustness.",
        TimeStability.FRAGILE: "Fragile. Direction is consistent but significance does not survive train/test split.",
        TimeStability.STABLE: "Stable. Direction and significance both survive train/test splits."
    }
    
    @classmethod
    def get_text(cls, stability: TimeStability) -> str:
        """Get deterministic text for time stability state."""
        return cls.TEMPLATES[stability]


class StrengthTemplates:
    """Templates for signal strength descriptions."""
    
    TEMPLATES = {
        Strength.WEAK: "Low explanatory power. Limited practical value without significant improvements.",
        Strength.MODERATE: "Moderate explanatory power with consistent direction.",
        Strength.STRONG: "Strong explanatory power and consistent performance across assets."
    }
    
    @classmethod
    def get_text(cls, strength: Strength) -> str:
        """Get deterministic text for strength state."""
        return cls.TEMPLATES[strength]


class ConclusionTemplates:
    """Templates for conclusions based on combined state."""
    
    @classmethod
    def get_conclusion(cls, existence: SignalExistence, behavior: AssetBehavior, 
                      stability: TimeStability, strength: Strength) -> str:
        """Get deterministic conclusion based on combined state."""
        
        # No signal cases
        if existence == SignalExistence.NONE:
            return "No reliable predictive signal detected in this formulation."
        
        # Weak signal cases
        if existence == SignalExistence.WEAK:
            return "Signal shows weak evidence but lacks statistical reliability for practical use."
        
        # Strong universal cases
        if existence == SignalExistence.STRONG and behavior == AssetBehavior.UNIVERSAL:
            if stability == TimeStability.STABLE:
                return "Universal and stable signal representing a fundamental market mechanism."
            else:
                return "Universal signal with mixed temporal stability."
        
        # Selective cases
        if behavior == AssetBehavior.SELECTIVE:
            if stability == TimeStability.FRAGILE:
                return "Asset-selective and time-fragile signal requiring refinement for specific assets."
            elif stability == TimeStability.STABLE:
                return "Asset-selective but temporally stable signal suitable for targeted implementation."
            else:
                return "Asset-selective signal with unconfirmed temporal stability."
        
        # Single asset cases
        if behavior == AssetBehavior.SINGLE:
            if stability == TimeStability.STABLE:
                return "Single-asset signal with confirmed temporal stability."
            else:
                return "Single-asset signal with unconfirmed temporal stability."
        
        # Partial universal cases
        if existence == SignalExistence.PARTIAL and behavior == AssetBehavior.UNIVERSAL:
            if stability == TimeStability.STABLE:
                return "Partially universal signal with good temporal stability."
            else:
                return "Partially universal signal with mixed temporal stability."
        
        # Default case
        return "Signal shows selective behavior with mixed temporal stability."


class RecommendationTemplates:
    """Templates for actionable recommendations."""
    
    @classmethod
    def get_recommendations(cls, existence: SignalExistence, behavior: AssetBehavior,
                           stability: TimeStability, strength: Strength,
                           asset_details: Dict[str, List[str]]) -> List[str]:
        """Get deterministic recommendations based on combined state."""
        recommendations = []
        
        # Existence-based recommendations
        if existence == SignalExistence.NONE:
            recommendations.append("Consider rejecting signal or fundamental redesign of signal construction")
            return recommendations
        
        # Asset behavior recommendations
        if behavior == AssetBehavior.UNIVERSAL:
            recommendations.append("Proceed with implementation across all assets")
        elif behavior == AssetBehavior.SELECTIVE:
            strong_assets = asset_details.get("strong", [])
            moderate_assets = asset_details.get("moderate", [])
            if strong_assets:
                recommendations.append(f"Prioritize implementation on {', '.join(strong_assets)}")
            if moderate_assets:
                recommendations.append(f"Consider {', '.join(moderate_assets)} for further testing")
        elif behavior == AssetBehavior.SINGLE:
            recommendations.append("Focus on single-asset implementation strategy")
        
        # Time stability recommendations
        if stability == TimeStability.FRAGILE:
            recommendations.append("Add regime-filtering mechanisms or combine with market condition indicators")
        elif stability == TimeStability.STABLE:
            recommendations.append("Signal is stable over time, but strength may limit practical tradability")
        elif stability == TimeStability.NOT_ESTABLISHED:
            recommendations.append("Conduct additional stability testing before implementation")
        
        # Strength-based recommendations
        if strength == Strength.WEAK:
            recommendations.append("Consider signal combination approaches or feature engineering improvements")
        elif strength == Strength.MODERATE:
            recommendations.append("Signal shows promise but may need refinement for practical use")
        elif strength == Strength.STRONG:
            recommendations.append("Signal shows strong potential for standalone implementation")
        
        # Combined recommendations
        if behavior == AssetBehavior.SELECTIVE and stability == TimeStability.FRAGILE:
            recommendations.append("Focus on asset-specific parameter tuning and regime-dependent implementation")
        elif behavior == AssetBehavior.SELECTIVE and stability == TimeStability.STABLE:
            recommendations.append("Investigate what makes specific assets more responsive to the signal")
        
        return recommendations


class CombinedClassificationTemplate:
    """Template for combined classification label."""
    
    @classmethod
    def get_label(cls, existence: SignalExistence, behavior: AssetBehavior,
                  stability: TimeStability, strength: Strength) -> str:
        """Get combined classification label."""
        return f"{behavior.value} + {stability.value} + {strength.value}"
