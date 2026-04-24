"""
Enhanced decision engine with dynamic confidence scoring based on cross-symbol consistency.
"""

from __future__ import annotations
from .decisions import HypothesisAwareDecisionEngine


class EnhancedDecisionEngine(HypothesisAwareDecisionEngine):
    """Decision engine with dynamic confidence scoring based on signal consistency across symbols."""
    
    def __init__(self, hypothesis_directions, **kwargs):
        super().__init__(hypothesis_directions, **kwargs)
        self.family_results_cache = {}  # Cache results across symbols for confidence calculation
    
    def make_family_decision(self, family_results: Dict[str, ExperimentResult], family_scores: Dict[str, float]) -> Decision:
        """Make decision for experiment family with dynamic confidence scoring."""
        # Handle empty results
        if not family_results:
            return Decision(
                action="DROP",
                confidence=0.0,
                reason="No experiments completed successfully"
            )
        
        # Cache results for cross-symbol analysis
        base_name = list(family_results.keys())[0].split('_')[0]
        if base_name not in self.family_results_cache:
            self.family_results_cache[base_name] = []
        self.family_results_cache[base_name].append(family_results.copy())
        
        # Calculate dynamic confidence based on cross-symbol consistency
        confidence = self._calculate_dynamic_confidence(base_name, family_results, family_scores)
        
        # Get standard decision
        decision = super().make_family_decision(family_results, family_scores)
        
        # Override confidence with dynamic calculation
        decision['confidence'] = confidence
        
        return decision
    
    def _calculate_dynamic_confidence(self, base_name, family_results, family_scores):
        """Calculate confidence based on signal consistency across symbols."""
        if base_name not in self.family_results_cache or len(self.family_results_cache[base_name]) < 2:
            # Not enough cross-symbol data, use standard confidence
            return min(family_scores.get('total', 0) * 0.8 + 0.2, 1.0)
        
        # Count how many symbols show strong signals for this family
        symbol_decisions = []
        for cached_results in self.family_results_cache[base_name]:
            # Determine if this symbol shows strong signal
            significant_count = sum(1 for result in cached_results.values() 
                                  if result.regression.p_value < 0.05)
            direction_consistent = self._check_direction_consistency(cached_results)
            meaningful_effects = sum(1 for result in cached_results.values() 
                                  if result.regression.r_squared > 0.01)
            
            if (direction_consistent and significant_count >= 2 and meaningful_effects >= 1):
                symbol_decisions.append("strong")
            elif (direction_consistent and significant_count >= 1):
                symbol_decisions.append("moderate")
            else:
                symbol_decisions.append("weak")
        
        # Calculate confidence based on consistency
        strong_count = symbol_decisions.count("strong")
        total_symbols = len(symbol_decisions)
        
        if strong_count == total_symbols:
            return 1.0  # All symbols show strong signal
        elif strong_count >= total_symbols * 0.67:
            return 0.9  # 2/3 or more symbols show strong signal
        elif strong_count >= total_symbols * 0.33:
            return 0.8  # 1/3 or more symbols show strong signal
        else:
            return 0.6   # Few symbols show strong signal
