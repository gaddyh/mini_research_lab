"""
Stability analysis engines for temporal validation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .experiment import ExperimentResult, StabilityAnalyzer


@dataclass
class StabilityConfig:
    """Configuration for stability analysis."""
    direction_weight: float = 40
    significance_weight: float = 30
    r2_weight: float = 20
    decay_weight: float = 10
    significance_threshold_train: float = 0.05
    significance_threshold_test: float = 0.10
    r2_survival_threshold: float = 0.5
    decay_stable_threshold: float = 0.7
    decay_weak_threshold: float = 0.3


class StandardStabilityAnalyzer(StabilityAnalyzer):
    """Standard stability analyzer with configurable thresholds."""
    
    def __init__(self, config: Optional[StabilityConfig] = None):
        self.config = config or StabilityConfig()
    
    def analyze_stability(self, train_result: ExperimentResult, test_result: ExperimentResult) -> 'StabilityAnalysisResult':
        """Analyze temporal stability of experiment results."""
        
        # Extract regression metrics
        train_reg = train_result.regression
        test_reg = test_result.regression
        
        # Direction consistency (most important)
        direction_stable = (train_reg.coef * test_reg.coef) > 0  # Same sign
        
        # Significance survival
        significance_stable = (train_reg.p_value < self.config.significance_threshold_train and 
                            test_reg.p_value < self.config.significance_threshold_test)
        
        # Strength decay
        decay_ratio = abs(test_reg.coef) / abs(train_reg.coef) if train_reg.coef != 0 else 0
        
        if decay_ratio > self.config.decay_stable_threshold:
            decay_score = self.config.decay_weight
            decay_label = "stable"
        elif decay_ratio > self.config.decay_weak_threshold:
            decay_score = self.config.decay_weight // 2
            decay_label = "weak_decay"
        else:
            decay_score = 0
            decay_label = "collapses"
        
        # R² survival
        r2_stable = test_reg.r_squared >= (self.config.r2_survival_threshold * train_reg.r_squared)
        
        # Stability score
        stability_score = (
            self.config.direction_weight * (1 if direction_stable else 0) +
            self.config.significance_weight * (1 if significance_stable else 0) +
            self.config.r2_weight * (1 if r2_stable else 0) +
            decay_score
        )
        
        # Stability label
        if stability_score >= 70:
            stability_label = "high"
        elif stability_score >= 40:
            stability_label = "moderate"
        else:
            stability_label = "low"
        
        return StabilityAnalysisResult(
            train_result=train_result,
            test_result=test_result,
            train_coef=train_reg.coef,
            test_coef=test_reg.coef,
            train_p_value=train_reg.p_value,
            test_p_value=test_reg.p_value,
            train_r_squared=train_reg.r_squared,
            test_r_squared=test_reg.r_squared,
            direction_stable=direction_stable,
            significance_stable=significance_stable,
            r2_stable=r2_stable,
            decay_ratio=decay_ratio,
            decay_label=decay_label,
            stability_score=stability_score,
            stability_label=stability_label,
            config=self.config
        )


@dataclass
class StabilityAnalysisResult:
    """Complete stability analysis result."""
    train_result: ExperimentResult
    test_result: ExperimentResult
    train_coef: float
    test_coef: float
    train_p_value: float
    test_p_value: float
    train_r_squared: float
    test_r_squared: float
    direction_stable: bool
    significance_stable: bool
    r2_stable: bool
    decay_ratio: float
    decay_label: str
    stability_score: int
    stability_label: str
    config: StabilityConfig
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'train_coef': self.train_coef,
            'test_coef': self.test_coef,
            'train_p_value': self.train_p_value,
            'test_p_value': self.test_p_value,
            'train_r_squared': self.train_r_squared,
            'test_r_squared': self.test_r_squared,
            'direction_stable': self.direction_stable,
            'significance_stable': self.significance_stable,
            'r2_stable': self.r2_stable,
            'decay_ratio': self.decay_ratio,
            'decay_label': self.decay_label,
            'stability_score': self.stability_score,
            'stability_label': self.stability_label
        }


class FamilyStabilityAnalyzer:
    """Analyzer for family-level stability across multiple experiments."""
    
    def __init__(self, analyzer: StabilityAnalyzer):
        self.analyzer = analyzer
    
    def analyze_family_stability(self, family_results: Dict[str, Tuple[ExperimentResult, ExperimentResult]]) -> 'FamilyStabilityResult':
        """Analyze stability for an entire experiment family."""
        
        individual_stabilities = {}
        stability_scores = []
        
        for exp_name, (train_result, test_result) in family_results.items():
            stability = self.analyzer.analyze_stability(train_result, test_result)
            individual_stabilities[exp_name] = stability
            stability_scores.append(stability.stability_score)
        
        # Calculate family-level metrics
        avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0
        best_stability = max(stability_scores) if stability_scores else 0
        worst_stability = min(stability_scores) if stability_scores else 0
        
        # Count stable experiments by category
        high_stability_count = sum(1 for s in stability_scores if s >= 70)
        moderate_stability_count = sum(1 for s in stability_scores if 40 <= s < 70)
        low_stability_count = sum(1 for s in stability_scores if s < 40)
        
        # Direction consistency across family
        direction_consistent = self._check_family_direction_consistency(individual_stabilities)
        
        # Significance survival across family
        significance_survival_rate = sum(1 for s in individual_stabilities.values() 
                                       if s.significance_stable) / len(individual_stabilities) if individual_stabilities else 0
        
        return FamilyStabilityResult(
            individual_stabilities=individual_stabilities,
            avg_stability_score=avg_stability,
            best_stability_score=best_stability,
            worst_stability_score=worst_stability,
            high_stability_count=high_stability_count,
            moderate_stability_count=moderate_stability_count,
            low_stability_count=low_stability_count,
            direction_consistent=direction_consistent,
            significance_survival_rate=significance_survival_rate
        )
    
    def _check_family_direction_consistency(self, stabilities: Dict[str, StabilityAnalysisResult]) -> bool:
        """Check if all direction-stable experiments have consistent direction."""
        direction_stable_experiments = [s for s in stabilities.values() if s.direction_stable]
        
        if len(direction_stable_experiments) < 2:
            return True  # Not enough data to detect inconsistency
        
        # Check if all have the same coefficient sign
        first_sign = 1 if direction_stable_experiments[0].train_coef > 0 else -1
        for stability in direction_stable_experiments[1:]:
            current_sign = 1 if stability.train_coef > 0 else -1
            if current_sign != first_sign:
                return False
        
        return True


@dataclass
class FamilyStabilityResult:
    """Family-level stability analysis result."""
    individual_stabilities: Dict[str, StabilityAnalysisResult]
    avg_stability_score: float
    best_stability_score: int
    worst_stability_score: int
    high_stability_count: int
    moderate_stability_count: int
    low_stability_count: int
    direction_consistent: bool
    significance_survival_rate: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'avg_stability_score': self.avg_stability_score,
            'best_stability_score': self.best_stability_score,
            'worst_stability_score': self.worst_stability_score,
            'high_stability_count': self.high_stability_count,
            'moderate_stability_count': self.moderate_stability_count,
            'low_stability_count': self.low_stability_count,
            'direction_consistent': self.direction_consistent,
            'significance_survival_rate': self.significance_survival_rate,
            'individual_stabilities': {
                name: stability.to_dict() 
                for name, stability in self.individual_stabilities.items()
            }
        }
