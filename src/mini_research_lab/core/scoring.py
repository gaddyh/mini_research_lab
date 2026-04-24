"""
Scoring engines for experiment evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
from .experiment import ExperimentResult, ExperimentSpec, ScoringEngine, ParameterizedExperiment


@dataclass
class ExperimentScore:
    """Score components for an experiment."""
    significance: float
    effect_direction: float
    effect_size: float
    r_squared: float
    bucket_shape: float
    diagnostics_penalty: float
    total: float
    label: str


@dataclass
class FamilyScore:
    """Score components for an experiment family."""
    avg_score: float
    max_score: float
    stability_score: float
    consistency_score: float
    explanatory_power: float
    total: float
    label: str


class StandardScoringEngine(ScoringEngine):
    """Standard scoring engine with configurable criteria."""
    
    def __init__(self, 
                 significance_threshold: float = 0.05,
                 r_squared_threshold: float = 0.01,
                 bucket_weight: float = 0.2,
                 diagnostics_weight: float = 0.1):
        self.significance_threshold = significance_threshold
        self.r_squared_threshold = r_squared_threshold
        self.bucket_weight = bucket_weight
        self.diagnostics_weight = diagnostics_weight
    
    def score_experiment(self, result: ExperimentResult) -> Dict[str, float]:
        """Score a single experiment result."""
        reg = result.regression
        
        # Significance score
        if reg.p_value < 0.001:
            significance = 1.0
        elif reg.p_value < 0.01:
            significance = 0.8
        elif reg.p_value < 0.05:
            significance = 0.6
        elif reg.p_value < 0.10:
            significance = 0.3
        else:
            significance = 0.0
        
        # Effect direction score (conditional on significance)
        if reg.p_value < self.significance_threshold:
            effect_direction = 0.5  # Correct direction gets full points later
        else:
            effect_direction = 0.0
        
        # Effect size score
        abs_coef = abs(reg.coef)
        if abs_coef > 0.1:
            effect_size = 0.3
        elif abs_coef > 0.05:
            effect_size = 0.2
        elif abs_coef > 0.02:
            effect_size = 0.1
        else:
            effect_size = 0.0
        
        # R-squared score
        if reg.r_squared > 0.05:
            r_squared = 0.3
        elif reg.r_squared > 0.02:
            r_squared = 0.2
        elif reg.r_squared > 0.01:
            r_squared = 0.1
        else:
            r_squared = 0.0
        
        # Bucket shape score
        bucket_shape = self._score_bucket_shape(result)
        
        # Diagnostics penalty
        diagnostics_penalty = self._calculate_diagnostics_penalty(result)
        
        # Total score
        total = (significance + effect_direction + effect_size + r_squared + 
                bucket_shape - diagnostics_penalty)
        
        # Score label
        label = self._get_score_label(reg.p_value, reg.r_squared, bucket_shape > 0)
        
        return {
            'significance': significance,
            'effect_direction': effect_direction, 
            'effect_size': effect_size,
            'r_squared': r_squared,
            'bucket_shape': bucket_shape,
            'diagnostics_penalty': diagnostics_penalty,
            'total': total,
            'label': label
        }
    
    def score_family(self, results: Dict[str, ExperimentResult]) -> Dict[str, float]:
        """Score a family of experiment results."""
        if not results:
            return {'total': 0.0, 'label': 'no_data'}
        
        # Score individual experiments
        individual_scores = {}
        for exp_name, result in results.items():
            individual_scores[exp_name] = self.score_experiment(result)
        
        # Calculate family metrics
        scores = [score['total'] for score in individual_scores.values()]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        
        # Consistency score (based on score variance)
        if len(scores) > 1:
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            consistency = max(0, 1 - variance)  # Lower variance = higher consistency
        else:
            consistency = 0.5
        
        # Explanatory power (based on average R²)
        avg_r2 = sum(result.regression.r_squared for result in results.values()) / len(results)
        explanatory_power = min(1.0, avg_r2 * 20)  # Scale R² to 0-1 range
        
        # Total family score
        total = (avg_score * 0.4 + max_score * 0.3 + 
                consistency * 0.2 + explanatory_power * 0.1)
        
        # Family label
        label = self._get_family_label(total, consistency, explanatory_power)
        
        return {
            'avg_score': avg_score,
            'max_score': max_score,
            'consistency_score': consistency,
            'explanatory_power': explanatory_power,
            'total': total,
            'label': label,
            'individual_scores': individual_scores
        }
    
    def _score_bucket_shape(self, result: ExperimentResult) -> float:
        """Score bucket shape based on monotonic relationship."""
        if result.bucketed_analysis is None:
            return 0.0
        
        y_means = result.bucketed_analysis['y_mean'].values
        if len(y_means) < 2:
            return 0.0
        
        # Check for monotonic relationship
        increasing = all(y_means[i] <= y_means[i+1] for i in range(len(y_means)-1))
        decreasing = all(y_means[i] >= y_means[i+1] for i in range(len(y_means)-1))
        
        if increasing or decreasing:
            return 0.2
        else:
            return 0.0
    
    def _calculate_diagnostics_penalty(self, result: ExperimentResult) -> float:
        """Calculate penalty based on diagnostic issues."""
        penalty = 0.0
        
        # Penalty for low observations
        if result.regression.n_obs < 100:
            penalty += 0.1
        elif result.regression.n_obs < 500:
            penalty += 0.05
        
        # Penalty for extreme coefficients
        if abs(result.regression.coef) > 1.0:
            penalty += 0.1
        
        return penalty
    
    def _get_score_label(self, p_value: float, r_squared: float, has_bucket_support: bool) -> str:
        """Get score label based on statistical evidence."""
        if p_value < 0.001 and r_squared > 0.05 and has_bucket_support:
            return "strong_evidence"
        elif p_value < 0.01 and r_squared > 0.02:
            return "moderate_evidence"
        elif p_value < 0.05:
            return "statistically_significant"
        elif p_value < 0.10:
            return "borderline_significant"
        else:
            return "not_significant"
    
    def _get_family_label(self, total_score: float, consistency: float, explanatory_power: float) -> str:
        """Get family score label."""
        if total_score > 0.8 and consistency > 0.7:
            return "strong_family"
        elif total_score > 0.6 and consistency > 0.5:
            return "moderate_family"
        elif total_score > 0.4:
            return "weak_family"
        else:
            return "no_evidence"
