"""
Deterministic Signal Classifier

Maps results + stability → enums with no contradictions.
This is Layer 1 of the interpretation system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .enums import (
    SignalExistence, AssetBehavior, TimeStability, Strength, AssetSignalStrength,
    is_valid_combination
)


@dataclass
class ClassificationInputs:
    """Structured inputs for deterministic classification."""
    p_values: List[float]
    decisions: List[str]  # PROMOTE, REFINE, DROP
    r_squared_values: List[float]
    survival_rates: List[float]
    direction_consistent: List[bool]
    asset_symbols: List[str]


class DeterministicClassifier:
    """
    Deterministic classification with fixed thresholds and no contradictions.
    
    Rules are explicit and mathematical - no magic numbers, no randomness.
    """
    
    # Fixed thresholds - these are constants, not hyperparameters
    P_VALUE_SIGNIFICANT = 0.05
    P_VALUE_BORDERLINE = 0.10
    P_VALUE_WEAK = 0.15
    
    R_SQUARED_MEANINGFUL = 0.01
    R_SQUARED_MODERATE = 0.005
    R_SQUARED_WEAK = 0.001
    
    SURVIVAL_RATE_STABLE = 0.5
    SURVIVAL_RATE_FRAGILE = 0.1
    
    DIRECTION_CONSISTENCY_THRESHOLD = 0.7
    
    SIGNAL_RATIO_UNIVERSAL = 0.8
    SIGNAL_RATIO_SELECTIVE = 0.3
    
    def classify_existence(self, inputs: ClassificationInputs) -> SignalExistence:
        """
        Deterministic signal existence classification based on decisions.
        
        Rules:
        - STRONG: All assets have PROMOTE decisions
        - PARTIAL: Some assets have PROMOTE or REFINE decisions
        - WEAK: Borderline evidence but no clear signal
        - NONE: No meaningful signal
        """
        promote_count = sum(1 for d in inputs.decisions if d == "PROMOTE")
        refine_count = sum(1 for d in inputs.decisions if d == "REFINE")
        total_assets = len(inputs.decisions)
        
        if promote_count == total_assets and total_assets > 0:
            return SignalExistence.STRONG
        elif promote_count + refine_count > 0:
            return SignalExistence.PARTIAL
        else:
            return SignalExistence.NONE
    
    def classify_asset_behavior(self, inputs: ClassificationInputs) -> AssetBehavior:
        """
        Deterministic asset behavior classification.
        
        Rules:
        - UNIVERSAL: 80%+ assets have signal (PROMOTE/REFINE)
        - SELECTIVE: 30%+ assets have signal
        - SINGLE: Only 1 asset has signal
        """
        total_assets = len(inputs.decisions)
        if total_assets == 0:
            return AssetBehavior.SINGLE
        
        signal_assets = sum(1 for d in inputs.decisions if d in ["PROMOTE", "REFINE"])
        signal_ratio = signal_assets / total_assets
        
        if signal_ratio >= self.SIGNAL_RATIO_UNIVERSAL:
            return AssetBehavior.UNIVERSAL
        elif signal_ratio >= self.SIGNAL_RATIO_SELECTIVE:
            return AssetBehavior.SELECTIVE
        elif signal_assets == 1:
            return AssetBehavior.SINGLE
        else:
            return AssetBehavior.SINGLE
    
    def classify_time_stability(self, inputs: ClassificationInputs) -> TimeStability:
        """
        Deterministic time stability classification.
        
        Rules:
        - STABLE: High survival rate (> 0.5) AND direction consistency (> 0.7)
        - FRAGILE: Direction consistency (> 0.7) but low survival rate (< 0.5)
        - NOT_ESTABLISHED: No stability data or poor direction consistency
        """
        if not inputs.survival_rates:
            return TimeStability.NOT_ESTABLISHED
        
        avg_survival = sum(inputs.survival_rates) / len(inputs.survival_rates)
        consistent_ratio = sum(inputs.direction_consistent) / len(inputs.direction_consistent)
        
        if avg_survival >= self.SURVIVAL_RATE_STABLE and consistent_ratio >= self.DIRECTION_CONSISTENCY_THRESHOLD:
            return TimeStability.STABLE
        elif consistent_ratio >= self.DIRECTION_CONSISTENCY_THRESHOLD and avg_survival < self.SURVIVAL_RATE_STABLE:
            return TimeStability.FRAGILE
        else:
            return TimeStability.NOT_ESTABLISHED
    
    def classify_strength(self, inputs: ClassificationInputs, existence: SignalExistence) -> Strength:
        """
        Deterministic signal strength classification with hard caps.
        
        Rules:
        - Hard R² caps to prevent overclaiming
        - Apply survival rate caps
        - Be conservative in strength assessment
        """
        if not inputs.r_squared_values:
            return Strength.WEAK
        
        avg_r_squared = sum(inputs.r_squared_values) / len(inputs.r_squared_values)
        median_r_squared = sorted(inputs.r_squared_values)[len(inputs.r_squared_values) // 2]
        
        promote_count = sum(1 for d in inputs.decisions if d == "PROMOTE")
        refine_count = sum(1 for d in inputs.decisions if d == "REFINE")
        drop_count = sum(1 for d in inputs.decisions if d == "DROP")
        
        # Hard R² caps to prevent overclaiming
        if avg_r_squared < 0.01:
            base_strength = Strength.WEAK
        elif avg_r_squared < 0.05:
            base_strength = Strength.MODERATE
        else:
            base_strength = Strength.STRONG
        
        # Apply stricter survival rate caps
        if inputs.survival_rates and max(inputs.survival_rates) == 0.0:
            if base_strength == Strength.STRONG:
                base_strength = Strength.MODERATE
        
        # Additional cap: If drop_count > 0 and promote_count == 0, max_strength = MODERATE
        if drop_count > 0 and promote_count == 0:
            if base_strength == Strength.STRONG:
                base_strength = Strength.MODERATE
        
        return base_strength
    
    def classify_asset_strength(self, symbol: str, decision: str, r_squared: float) -> AssetSignalStrength:
        """
        Classify individual asset signal strength.
        
        Rules:
        - STRONGEST: PROMOTE decision (clear signal)
        - MODERATE: REFINE decision with meaningful R² (partial signal)
        - WEAK: REFINE decision with low R² (weak/partial signal)
        - NONE: DROP decision (no signal)
        """
        if decision == "PROMOTE":
            return AssetSignalStrength.STRONG
        elif decision == "REFINE":
            if r_squared >= self.R_SQUARED_MEANINGFUL:
                return AssetSignalStrength.MODERATE
            elif r_squared >= self.R_SQUARED_WEAK:
                return AssetSignalStrength.WEAK
            else:
                return AssetSignalStrength.WEAK
        else:  # DROP
            return AssetSignalStrength.NONE
    
    def classify_all(self, inputs: ClassificationInputs) -> Dict[str, Any]:
        """
        Classify all dimensions with consistency validation.
        
        Returns:
            Dict with existence, asset_behavior, time_stability, strength
        """
        # Classify each dimension
        existence = self.classify_existence(inputs)
        asset_behavior = self.classify_asset_behavior(inputs)
        time_stability = self.classify_time_stability(inputs)
        strength = self.classify_strength(inputs, existence)
        
        # Validate combination
        if not is_valid_combination(existence, asset_behavior, time_stability, strength):
            raise ValueError(
                f"Invalid combination: {existence.value} + {asset_behavior.value} + "
                f"{time_stability.value} + {strength.value}"
            )
        
        # Classify individual assets
        asset_strengths = {}
        for i, symbol in enumerate(inputs.asset_symbols):
            if i < len(inputs.decisions) and i < len(inputs.r_squared_values):
                asset_strengths[symbol] = self.classify_asset_strength(
                    symbol, inputs.decisions[i], inputs.r_squared_values[i]
                )
        
        return {
            'existence': existence,
            'asset_behavior': asset_behavior,
            'time_stability': time_stability,
            'strength': strength,
            'asset_strengths': asset_strengths
        }
