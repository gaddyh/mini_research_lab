"""
Decision engines for experiment evaluation and promotion.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from .experiment import ExperimentResult, ExperimentSpec, DecisionEngine, ParameterizedExperiment


@dataclass
class ExperimentDecision:
    """Decision for a single experiment."""
    action: str  # PROMOTE, REFINE, DROP
    confidence: float
    reason_codes: List[str]
    reason: str


@dataclass
class FamilyDecision:
    """Decision for an experiment family."""
    action: str  # PROMOTE, REFINE, DROP
    confidence: float
    reason_codes: List[str]
    reason: str
    selected_experiments: Dict[str, str]  # experiment_id -> selection_role


class StandardDecisionEngine(DecisionEngine):
    """Standard decision engine with configurable rules."""
    
    def __init__(self, 
                 promote_threshold: float = 0.7,
                 refine_threshold: float = 0.4,
                 stability_requirement: float = 70):
        self.promote_threshold = promote_threshold
        self.refine_threshold = refine_threshold
        self.stability_requirement = stability_requirement
    
    def make_experiment_decision(self, result: ExperimentResult, score: Dict[str, float]) -> Dict[str, any]:
        """Make decision for a single experiment."""
        reg = result.regression
        
        # Collect reason codes
        reason_codes = []
        
        # Check significance
        if reg.p_value < 0.05:
            reason_codes.append("significant")
        elif reg.p_value < 0.10:
            reason_codes.append("borderline_significant")
        else:
            reason_codes.append("not_significant")
        
        # Check effect size
        if reg.r_squared > 0.01:
            reason_codes.append("meaningful_effect_size")
        else:
            reason_codes.append("tiny_effect_size")
        
        # Check bucket support
        bucket_support = score.get('bucket_shape', 0) > 0
        if bucket_support:
            reason_codes.append("bucket_support")
        else:
            reason_codes.append("no_bucket_support")
        
        # Determine action - more lenient for strong signals like volatility
        if (reg.p_value < 0.01 and reg.r_squared > 0.01):
            action = "PROMOTE"
            confidence = 0.8
        elif (reg.p_value < 0.05 and reg.r_squared > 0.005):
            action = "REFINE"
            confidence = 0.6
        elif (reg.p_value < 0.10 and reg.r_squared > 0.003):
            action = "REFINE"
            confidence = 0.5
        else:
            action = "DROP"
            confidence = 0.3
        
        # Generate reason text
        reason = self._generate_reason_text(action, reason_codes)
        
        return {
            'action': action,
            'confidence': confidence,
            'reason_codes': reason_codes,
            'reason': reason
        }
    
    def make_family_decision(self, family_results: Dict[str, ExperimentResult], family_scores: Dict[str, float]) -> Dict[str, any]:
        """Make decision for an experiment family."""
        
        # Extract key metrics
        total_count = len(family_results)
        significant_count = sum(1 for result in family_results.values() if result.regression.p_value < 0.05)
        
        # Check direction consistency
        direction_consistent = self._check_direction_consistency(family_results)
        
        # Check stability (if available)
        stability_scores = []
        for result in family_results.values():
            if hasattr(result, 'stability_score'):
                stability_scores.append(result.stability_score)
        
        avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0
        
        # Collect reason codes
        reason_codes = []
        
        if significant_count >= 2:
            reason_codes.append("multiple_significant_variants")
        elif significant_count >= 1:
            reason_codes.append("some_significant_variants")
        else:
            reason_codes.append("insufficient_signal_strength")
        
        if direction_consistent:
            reason_codes.append("consistent_direction")
        else:
            reason_codes.append("inconsistent_direction")
        
        # Check for meaningful effect sizes
        meaningful_effects = sum(1 for result in family_results.values() 
                              if result.regression.r_squared > 0.01)
        if meaningful_effects >= 2:
            reason_codes.append("meaningful_r_squared")
        elif meaningful_effects >= 1:
            reason_codes.append("some_meaningful_r_squared")
        else:
            reason_codes.append("weak_r_squared")
        
        # Add stability info only if relevant for context, not as decision factor
        if avg_stability >= self.stability_requirement:
            reason_codes.append("high_stability")
        elif avg_stability >= 40:
            reason_codes.append("moderate_stability")
        
        # Add signal validity indicator for PROMOTE decisions
        if (direction_consistent and (significant_count / total_count) >= 0.7 and meaningful_effects >= 1):
            reason_codes.append("strong_signal_validity")
        
        # Determine action - focus on signal validity, not variation stability
        total_score = family_scores.get('total', 0)
        
        # Calculate significance ratio
        significance_ratio = significant_count / total_count
        
        # Check for meaningful effect sizes across variants
        meaningful_effects = sum(1 for result in family_results.values() 
                              if result.regression.r_squared > 0.01)
        
        # NEW LOGIC: Focus on signal validity with more lenient REFINE criteria
        if (direction_consistent and significance_ratio >= 0.7 and 
            meaningful_effects >= 1 and total_score >= self.promote_threshold):
            action = "PROMOTE"
            confidence = 0.8
        elif (direction_consistent and significance_ratio >= 0.5 and 
              meaningful_effects >= 1):
            action = "REFINE"
            confidence = 0.6
        elif (direction_consistent and significance_ratio >= 0.4):
            action = "REFINE"
            confidence = 0.5
        else:
            action = "DROP"
            confidence = 0.3
        
        # Generate reason text
        reason = self._generate_reason_text(action, reason_codes)
        
        # Select representative experiments
        selected_experiments = self._select_representative_experiments(family_results, family_scores)
        
        return {
            'action': action,
            'confidence': confidence,
            'reason_codes': reason_codes,
            'reason': reason,
            'selected_experiments': selected_experiments
        }
    
    def _check_direction_consistency(self, results: Dict[str, ExperimentResult]) -> bool:
        """Check if all significant results have consistent direction."""
        significant_results = [result for result in results.values() 
                              if result.regression.p_value < 0.05]
        
        if len(significant_results) < 2:
            return True  # Not enough data to detect inconsistency
        
        # Check if all coefficients have the same sign
        first_sign = 1 if significant_results[0].regression.coef > 0 else -1
        for result in significant_results[1:]:
            current_sign = 1 if result.regression.coef > 0 else -1
            if current_sign != first_sign:
                return False
        
        return True
    
    def _generate_reason_text(self, action: str, reason_codes: List[str]) -> str:
        """Generate human-readable reason text."""
        if action == "PROMOTE":
            return "Strong signal validity: consistent direction, high significance ratio, and meaningful effect sizes across variants."
        elif action == "REFINE":
            return "Moderate signal validity: consistent direction but needs refinement for stronger significance or effect sizes."
        else:
            return "Poor signal validity: inconsistent direction, low significance, or insufficient effect sizes."
    
    def _select_representative_experiments(self, family_results: Dict[str, ExperimentResult], 
                                         family_scores: Dict[str, float]) -> Dict[str, str]:
        """Select representative experiments from the family."""
        individual_scores = family_scores.get('individual_scores', {})
        
        if not individual_scores:
            return {}
        
        # Find best and worst experiments
        sorted_experiments = sorted(individual_scores.items(), 
                                  key=lambda x: x[1]['total'], reverse=True)
        
        best_exp = sorted_experiments[0][0] if sorted_experiments else None
        worst_exp = sorted_experiments[-1][0] if sorted_experiments else None
        
        # Find representative (closest to average)
        avg_score = family_scores.get('avg_score', 0)
        representative_exp = min(sorted_experiments, 
                              key=lambda x: abs(x[1]['total'] - avg_score))[0] if sorted_experiments else None
        
        selected = {}
        if best_exp:
            selected[best_exp] = "best_candidate"
        if worst_exp and worst_exp != best_exp:
            selected[worst_exp] = "worst_candidate"
        if representative_exp and representative_exp not in selected:
            selected[representative_exp] = "representative"
        
        return selected


class HypothesisAwareDecisionEngine(StandardDecisionEngine):
    """Decision engine that checks hypothesis consistency."""
    
    def __init__(self, hypothesis_directions: Dict[str, int], **kwargs):
        super().__init__(**kwargs)
        self.hypothesis_directions = hypothesis_directions  # base_name -> expected_sign (+1 or -1)
    
    def make_family_decision(self, family_results: Dict[str, ExperimentResult], family_scores: Dict[str, float]) -> Dict[str, any]:
        """Make decision with hypothesis consistency checking."""
        
        # Get base name from first experiment
        base_name = list(family_results.keys())[0].split('_')[0]
        expected_sign = self.hypothesis_directions.get(base_name, 0)
        
        if expected_sign == 0:
            # Fall back to standard decision making
            return super().make_family_decision(family_results, family_scores)
        
        # Check if significant results contradict hypothesis
        contradictory_significant = 0
        for result in family_results.values():
            if result.regression.p_value < 0.05:
                actual_sign = 1 if result.regression.coef > 0 else -1
                if actual_sign != expected_sign:
                    contradictory_significant += 1
        
        # If significant results contradict hypothesis, recommend DROP
        if contradictory_significant > 0:
            return {
                'action': 'DROP',
                'confidence': 0.9,
                'reason_codes': ['contradicts_hypothesis', 'significant_wrong_direction'],
                'reason': f"Significant results contradict expected direction for {base_name}.",
                'selected_experiments': {}
            }
        
        # Otherwise use standard decision making
        return super().make_family_decision(family_results, family_scores)
