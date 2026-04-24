"""
Core module for experiment management and analysis.
"""

from .experiment import (
    ExperimentSpec,
    ParameterizedExperiment,
    ExperimentResult,
    StabilityResult,
    ExperimentRunner,
    StabilityAnalyzer,
    ScoringEngine,
    DecisionEngine
)

from .scoring import (
    StandardScoringEngine,
    ExperimentScore,
    FamilyScore
)

from .decisions import (
    StandardDecisionEngine,
    HypothesisAwareDecisionEngine,
    ExperimentDecision,
    FamilyDecision
)

from .stability import (
    StabilityConfig,
    StandardStabilityAnalyzer,
    StabilityAnalysisResult,
    FamilyStabilityAnalyzer,
    FamilyStabilityResult
)

__all__ = [
    # Core data models
    'ExperimentSpec',
    'ParameterizedExperiment', 
    'ExperimentResult',
    'StabilityResult',
    
    # Core interfaces
    'ExperimentRunner',
    'StabilityAnalyzer',
    'ScoringEngine',
    'DecisionEngine',
    
    # Scoring
    'StandardScoringEngine',
    'ExperimentScore',
    'FamilyScore',
    
    # Decisions
    'StandardDecisionEngine',
    'HypothesisAwareDecisionEngine',
    'ExperimentDecision',
    'FamilyDecision',
    
    # Stability
    'StabilityConfig',
    'StandardStabilityAnalyzer',
    'StabilityAnalysisResult',
    'FamilyStabilityAnalyzer',
    'FamilyStabilityResult'
]
