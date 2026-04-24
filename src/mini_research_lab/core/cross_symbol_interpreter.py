"""
Cross-Symbol Interpretation Layer

Converts raw experimental results across multiple symbols into human-readable insights
about signal behavior, asset dependence, and universality.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np


# Layer 1: Clean deterministic enums
class SignalExistence(Enum):
    """Signal existence classification - deterministic."""
    YES = "YES"
    PARTIAL = "PARTIAL"
    WEAK = "WEAK"
    NONE = "NONE"

class AssetBehavior(Enum):
    """Signal behavior across assets - deterministic."""
    UNIVERSAL = "UNIVERSAL"
    SELECTIVE = "SELECTIVE"
    NONE = "NONE"

class TimeStability(Enum):
    """Signal stability across time - deterministic."""
    STABLE = "STABLE"
    FRAGILE = "FRAGILE"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"

class SignalStrength(Enum):
    """Overall signal strength classification - deterministic."""
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"

class AssetSignalStrength(Enum):
    """Strength of signal for individual assets - deterministic."""
    STRONG = "STRONG"
    MODERATE = "MODERATE" 
    WEAK = "WEAK"
    NONE = "NONE"


# Layer 1: Deterministic SignalSummary class
@dataclass
class SignalSummary:
    """Deterministic signal summary - no contradictions allowed."""
    existence: SignalExistence
    asset_behavior: AssetBehavior
    time_stability: TimeStability
    strength: SignalStrength
    
    def __post_init__(self):
        """Consistency assertions to prevent contradictions."""
        # Rule: WEAK existence should NEVER become NONE
        if self.existence == SignalExistence.WEAK:
            assert self.asset_behavior != AssetBehavior.NONE, "WEAK existence cannot have NONE asset behavior"
            assert self.strength != SignalStrength.WEAK, "WEAK existence cannot have WEAK strength (should be at least MODERATE)"
        
        # Rule: UNKNOWN time stability only when no stability test ran
        # (This will be handled in classification logic)
        
        # Rule: YES existence requires at least some signal strength
        if self.existence == SignalExistence.YES:
            assert self.strength in [SignalStrength.STRONG, SignalStrength.MODERATE], "YES existence requires STRONG or MODERATE strength"
        
        # Rule: UNIVERSAL asset behavior requires YES or PARTIAL existence
        if self.asset_behavior == AssetBehavior.UNIVERSAL:
            assert self.existence in [SignalExistence.YES, SignalExistence.PARTIAL], "UNIVERSAL asset behavior requires YES or PARTIAL existence"


@dataclass
class AssetSignalProfile:
    """Signal profile for individual asset."""
    symbol: str
    strength: AssetSignalStrength
    confidence: float
    decision: str
    avg_score: float
    consistency: float
    explanatory_power: float
    stability_score: float = 0.0
    significance_survival_rate: float = 0.0
    direction_consistent: bool = False
    key_insights: List[str] = None
    
    def __post_init__(self):
        if self.key_insights is None:
            self.key_insights = []


@dataclass 
class CrossSymbolInterpretation:
    """Complete interpretation of signal across all assets and time."""
    asset_behavior: AssetBehavior
    time_stability: TimeStability
    signal_strength: SignalStrength
    signal_existence: str
    asset_behavior_desc: str
    time_stability_desc: str
    strength_desc: str
    conclusion: str
    recommendations: List[str]
    asset_profiles: List[AssetSignalProfile] = None
    
    def __post_init__(self):
        if self.asset_profiles is None:
            self.asset_profiles = []


class DeterministicClassifier:
    """Layer 1: Deterministic classification rules - no randomness, no ambiguity."""
    
    def __init__(self):
        # Fixed thresholds - no magic numbers
        self.P_VALUE_SIGNIFICANT = 0.05
        self.P_VALUE_BORDERLINE = 0.10
        self.R_SQUARED_MEANINGFUL = 0.01
        self.SURVIVAL_RATE_STABLE = 0.5
        self.SIGNAL_RATIO_UNIVERSAL = 0.8
        self.SIGNAL_RATIO_SELECTIVE = 0.3
    
    def classify_existence(self, p_values: List[float], decisions: List[str]) -> SignalExistence:
        """Deterministic signal existence classification."""
        # Rule 1: Any significant p-value
        if any(p < self.P_VALUE_SIGNIFICANT for p in p_values):
            return SignalExistence.PARTIAL
        
        # Rule 2: Any borderline p-value
        elif any(p < self.P_VALUE_BORDERLINE for p in p_values):
            return SignalExistence.WEAK
        
        # Rule 3: No meaningful p-values
        else:
            return SignalExistence.NONE
    
    def classify_asset_behavior(self, p_values: List[float], decisions: List[str]) -> AssetBehavior:
        """Deterministic asset behavior classification with consistency."""
        total_assets = len(decisions)
        if total_assets == 0:
            return AssetBehavior.NONE
        
        # Count assets with signals (PROMOTE or REFINE)
        signal_assets = sum(1 for d in decisions if d in ["PROMOTE", "REFINE"])
        signal_ratio = signal_assets / total_assets
        
        # Check if any borderline p-values exist
        has_borderline = any(p < self.P_VALUE_BORDERLINE for p in p_values)
        
        # Deterministic rules with consistency
        if signal_ratio >= self.SIGNAL_RATIO_UNIVERSAL:
            return AssetBehavior.UNIVERSAL
        elif signal_ratio >= self.SIGNAL_RATIO_SELECTIVE:
            return AssetBehavior.SELECTIVE
        elif has_borderline:
            # If WEAK existence, ensure at least SELECTIVE behavior (not NONE)
            return AssetBehavior.SELECTIVE
        else:
            return AssetBehavior.NONE
    
    def classify_time_stability(self, survival_rates: List[float], direction_consistent: List[bool]) -> TimeStability:
        """Deterministic time stability classification."""
        if not survival_rates:
            return TimeStability.NOT_ESTABLISHED
        
        avg_survival = sum(survival_rates) / len(survival_rates)
        consistent_ratio = sum(direction_consistent) / len(direction_consistent)
        
        # Deterministic rules
        if avg_survival >= self.SURVIVAL_RATE_STABLE and consistent_ratio >= 0.7:
            return TimeStability.STABLE
        elif consistent_ratio >= 0.7:
            return TimeStability.FRAGILE
        else:
            return TimeStability.NOT_ESTABLISHED
    
    def classify_strength(self, r_squared_values: List[float], decisions: List[str], existence: SignalExistence) -> SignalStrength:
        """Deterministic signal strength classification with consistency."""
        if not r_squared_values:
            return SignalStrength.WEAK
        
        avg_r_squared = sum(r_squared_values) / len(r_squared_values)
        
        # Count significant decisions
        significant_assets = sum(1 for d in decisions if d in ["PROMOTE", "REFINE"])
        
        # Deterministic rules with consistency
        if existence == SignalExistence.WEAK:
            # WEAK existence must get at least MODERATE strength
            return SignalStrength.MODERATE
        elif significant_assets >= 2 and avg_r_squared >= self.R_SQUARED_MEANINGFUL:
            return SignalStrength.STRONG
        elif significant_assets >= 1 and avg_r_squared >= 0.005:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK


class DeterministicTemplates:
    """Layer 2: Deterministic template system - no contradictions."""
    
    @staticmethod
    def get_existence_text(existence: SignalExistence) -> str:
        """Deterministic existence description."""
        templates = {
            SignalExistence.YES: "Signal exists reliably across tested assets.",
            SignalExistence.PARTIAL: "Signal exists in some assets but not universally.",
            SignalExistence.WEAK: "Weak evidence of signal, not statistically reliable.",
            SignalExistence.NONE: "No reliable signal detected."
        }
        return templates[existence]
    
    @staticmethod
    def get_asset_behavior_text(behavior: AssetBehavior, asset_details: Dict[str, str]) -> str:
        """Deterministic asset behavior description."""
        if behavior == AssetBehavior.UNIVERSAL:
            return "Universal. Signal works consistently across all analyzed assets."
        elif behavior == AssetBehavior.SELECTIVE:
            parts = []
            for strength, assets in asset_details.items():
                if assets:
                    parts.append(f"{strength.capitalize()}: {', '.join(assets)}")
            return "Selective.\n- " + "\n- ".join(parts) + "."
        else:
            return "No meaningful signal across assets."
    
    @staticmethod
    def get_time_stability_text(stability: TimeStability) -> str:
        """Deterministic time stability description."""
        templates = {
            TimeStability.STABLE: "Stable. Direction and significance both survive train/test splits.",
            TimeStability.FRAGILE: "Fragile. Direction is consistent but significance does not survive train/test split.",
            TimeStability.NOT_ESTABLISHED: "Not established. Stability analysis does not confirm robustness."
        }
        return templates[stability]
    
    @staticmethod
    def get_strength_text(strength: SignalStrength) -> str:
        """Deterministic strength description."""
        templates = {
            SignalStrength.STRONG: "Strong explanatory power and consistent performance.",
            SignalStrength.MODERATE: "Moderate explanatory power with consistent direction.",
            SignalStrength.WEAK: "Low explanatory power. Limited practical value."
        }
        return templates[strength]
    
    @staticmethod
    def get_conclusion(summary: SignalSummary) -> str:
        """Deterministic conclusion based on summary state."""
        if summary.existence == SignalExistence.NONE:
            return "No reliable predictive signal detected in this formulation."
        elif summary.existence == SignalExistence.WEAK:
            return "Signal shows weak evidence but lacks statistical reliability for practical use."
        elif summary.asset_behavior == AssetBehavior.UNIVERSAL and summary.time_stability == TimeStability.STABLE:
            return "Universal and stable signal representing a fundamental market mechanism."
        elif summary.asset_behavior == AssetBehavior.SELECTIVE and summary.time_stability == TimeStability.FRAGILE:
            return "Asset-selective and time-fragile signal requiring refinement for specific assets."
        else:
            return "Signal shows selective behavior with mixed temporal stability."


class CrossSymbolInterpreter:
    """Deterministic cross-symbol interpreter - no contradictions."""
    
    def __init__(self):
        self.classifier = DeterministicClassifier()
        self.templates = DeterministicTemplates()
    
    def interpret_cross_symbol_results(self, 
                                     all_symbol_results: Dict[str, Dict[str, Any]],
                                     stability_results: Dict[str, Dict[str, Any]] = None) -> SignalSummary:
        """
        Deterministic interpretation - no randomness, no contradictions.
        Interpret signal behavior across multiple symbols and time.
        
        Args:
            all_symbol_results: Dictionary of symbol -> family_results -> decision/metrics
            stability_results: Dictionary of symbol -> stability metrics
            
        Returns:
            SignalSummary: Deterministic summary with consistency assertions
        """
        # Extract data for deterministic classification
        symbols = list(all_symbol_results.keys())
        decisions = []
        p_values = []
        r_squared_values = []
        survival_rates = []
        direction_consistent = []
        asset_details = {"strong": [], "moderate": [], "weak": [], "none": []}
        
        for symbol in symbols:
            symbol_data = all_symbol_results[symbol]
            
            # Get family data (assume single family for now)
            for family_name, family_data in symbol_data.items():
                decision = family_data.get('decision', {}).get('action', 'DROP')
                decisions.append(decision)
                
                # Extract p-values (simplified - use best p-value)
                experiments = family_data.get('experiments', [])
                if experiments:
                    best_p = min(exp.get('p_value', 1.0) for exp in experiments)
                    p_values.append(best_p)
                
                # Extract R²
                family_metrics = family_data.get('family_metrics', {})
                r_squared = family_metrics.get('explanatory_power', 0.0)
                r_squared_values.append(r_squared)
                
                # Extract stability data
                if stability_results and symbol in stability_results:
                    stability_data = stability_results[symbol]
                    if family_name in stability_data:
                        family_stability = stability_data[family_name]
                        survival_rates.append(getattr(family_stability, 'significance_survival_rate', 0.0))
                        direction_consistent.append(getattr(family_stability, 'direction_consistent', False))
                
                # Classify individual asset strength
                if decision == "PROMOTE":
                    asset_details["strong"].append(symbol)
                elif decision == "REFINE" and r_squared >= 0.01:
                    asset_details["moderate"].append(symbol)
                elif decision == "REFINE":
                    asset_details["weak"].append(symbol)
                else:
                    asset_details["none"].append(symbol)
        
        # Layer 1: Deterministic classification
        existence = self.classifier.classify_existence(p_values, decisions)
        asset_behavior = self.classifier.classify_asset_behavior(p_values, decisions)
        time_stability = self.classifier.classify_time_stability(survival_rates, direction_consistent)
        strength = self.classifier.classify_strength(r_squared_values, decisions, existence)
        
        # Create deterministic summary with consistency assertions
        summary = SignalSummary(
            existence=existence,
            asset_behavior=asset_behavior,
            time_stability=time_stability,
            strength=strength
        )
        
        return summary
    
    def format_interpretation_output(self, summary: SignalSummary, family_name: str = "") -> str:
        """Format deterministic summary for CLI output."""
        output = []
        
        # Header
        if family_name:
            output.append(f"\n🧠 SIGNAL SUMMARY — {family_name.upper().replace('_', ' ')}")
        else:
            output.append(f"\n🧠 SIGNAL SUMMARY")
        output.append(f"{'='*80}")
        
        # Layer 2: Deterministic templates
        output.append(f"\nSignal existence:")
        output.append(f"{self.templates.get_existence_text(summary.existence)}")
        
        output.append(f"\nAsset behavior:")
        # For asset behavior, we need to extract asset details - simplified for now
        asset_details = {"strong": [], "moderate": [], "weak": [], "none": []}
        output.append(f"{self.templates.get_asset_behavior_text(summary.asset_behavior, asset_details)}")
        
        output.append(f"\nTime stability:")
        output.append(f"{self.templates.get_time_stability_text(summary.time_stability)}")
        
        output.append(f"\nStrength:")
        output.append(f"{self.templates.get_strength_text(summary.strength)}")
        
        # Combined classification
        combined_label = f"{summary.asset_behavior.value} + {summary.time_stability.value} + {summary.strength.value}"
        output.append(f"\nCombined classification:")
        output.append(f"{combined_label}")
        
        # Conclusion
        output.append(f"\nConclusion:")
        output.append(f"{self.templates.get_conclusion(summary)}")
        
        return "\n".join(output)
