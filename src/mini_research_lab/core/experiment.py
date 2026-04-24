"""
Core experiment data models and interfaces.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol
from abc import ABC, abstractmethod
import pandas as pd


@dataclass(frozen=True)
class ExperimentSpec:
    """Single experiment specification."""
    name: str
    x_col: str
    y_col: str
    title: str
    description: str = ""


@dataclass(frozen=True)
class ParameterizedExperiment:
    """Template for generating multiple experiment variations."""
    base_name: str
    x_col_pattern: str  # Use {lookback} as placeholder
    y_col: str
    lookbacks: List[int]
    title_template: str
    description_template: str


@dataclass
class ExperimentResult:
    """Results from running a single experiment."""
    spec: ExperimentSpec
    x_describe: pd.Series
    y_describe: pd.Series
    regression: 'RegressionSummary'
    model: any  # statsmodels OLS result
    bucketed_analysis: Optional[pd.DataFrame] = None


@dataclass
class StabilityResult:
    """Temporal stability analysis results."""
    train_result: ExperimentResult
    test_result: ExperimentResult
    direction_stable: bool
    significance_stable: bool
    r2_stable: bool
    decay_ratio: float
    decay_label: str
    stability_score: int
    stability_label: str


class ExperimentRunner(Protocol):
    """Protocol for experiment execution engines."""
    
    def run_experiment(self, spec: ExperimentSpec) -> ExperimentResult:
        """Run a single experiment and return results."""
        ...
    
    def run_parameterized_experiment(self, param_exp: ParameterizedExperiment) -> Dict[str, ExperimentResult]:
        """Run all variations of a parameterized experiment."""
        ...


class StabilityAnalyzer(ABC):
    """Abstract base class for stability analysis."""
    
    @abstractmethod
    def analyze_stability(self, train_result: ExperimentResult, test_result: ExperimentResult) -> StabilityResult:
        """Analyze temporal stability of experiment results."""
        pass


class ScoringEngine(ABC):
    """Abstract base class for experiment scoring."""
    
    @abstractmethod
    def score_experiment(self, result: ExperimentResult) -> Dict[str, float]:
        """Score a single experiment result."""
        pass
    
    @abstractmethod
    def score_family(self, results: Dict[str, ExperimentResult]) -> Dict[str, float]:
        """Score a family of experiment results."""
        pass


class DecisionEngine(ABC):
    """Abstract base class for decision making."""
    
    @abstractmethod
    def make_experiment_decision(self, result: ExperimentResult, score: Dict[str, float]) -> Dict[str, any]:
        """Make decision for a single experiment."""
        pass
    
    @abstractmethod
    def make_family_decision(self, family_results: Dict[str, ExperimentResult], family_scores: Dict[str, float]) -> Dict[str, any]:
        """Make decision for an experiment family."""
        pass
