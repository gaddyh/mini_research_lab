"""
Signal Summary Builder

Combines classifier and templates to build final human-readable summaries.
This is the main interface for the interpretation layer.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from .enums import SignalExistence, AssetBehavior, TimeStability, Strength, AssetSignalStrength
from .classifier import DeterministicClassifier, ClassificationInputs
from .templates import (
    ExistenceTemplates, AssetBehaviorTemplates, TimeStabilityTemplates,
    StrengthTemplates, ConclusionTemplates, RecommendationTemplates,
    CombinedClassificationTemplate
)


@dataclass
class SignalSummary:
    """Final signal summary with all classifications and text."""
    existence: SignalExistence
    asset_behavior: AssetBehavior
    time_stability: TimeStability
    strength: Strength
    asset_strengths: Dict[str, AssetSignalStrength]
    
    # Generated text
    existence_text: str
    asset_behavior_text: str
    time_stability_text: str
    strength_text: str
    conclusion: str
    combined_classification: str
    recommendations: List[str]


class SummaryBuilder:
    """
    Main interface for building signal summaries.
    
    This class orchestrates:
    1. Data extraction and validation
    2. Deterministic classification
    3. Template-based text generation
    4. Final summary assembly
    """
    
    def __init__(self):
        self.classifier = DeterministicClassifier()
    
    def build_summary(self, 
                     all_symbol_results: Dict[str, Dict[str, Any]],
                     stability_results: Optional[Dict[str, Dict[str, Any]]] = None,
                     family_name: str = "") -> SignalSummary:
        """
        Build complete signal summary from experimental results.
        
        Args:
            all_symbol_results: Dictionary of symbol -> family_results -> decision/metrics
            stability_results: Dictionary of symbol -> stability metrics
            family_name: Name of the signal family for context
            
        Returns:
            SignalSummary: Complete deterministic summary
        """
        # Extract and validate inputs
        inputs = self._extract_classification_inputs(all_symbol_results, stability_results)
        
        # Classify all dimensions
        classifications = self.classifier.classify_all(inputs)
        
        # Generate text using templates
        existence_text = ExistenceTemplates.get_text(classifications['existence'])
        
        # Prepare asset details for behavior template
        asset_details = self._prepare_asset_details(classifications['asset_strengths'])
        asset_behavior_text = AssetBehaviorTemplates.get_text(
            classifications['asset_behavior'], asset_details
        )
        
        time_stability_text = TimeStabilityTemplates.get_text(classifications['time_stability'])
        strength_text = StrengthTemplates.get_text(classifications['strength'])
        
        conclusion = ConclusionTemplates.get_conclusion(
            classifications['existence'],
            classifications['asset_behavior'],
            classifications['time_stability'],
            classifications['strength']
        )
        
        combined_classification = CombinedClassificationTemplate.get_label(
            classifications['existence'],
            classifications['asset_behavior'],
            classifications['time_stability'],
            classifications['strength']
        )
        
        recommendations = RecommendationTemplates.get_recommendations(
            classifications['existence'],
            classifications['asset_behavior'],
            classifications['time_stability'],
            classifications['strength'],
            asset_details
        )
        
        # Assemble final summary
        return SignalSummary(
            existence=classifications['existence'],
            asset_behavior=classifications['asset_behavior'],
            time_stability=classifications['time_stability'],
            strength=classifications['strength'],
            asset_strengths=classifications['asset_strengths'],
            existence_text=existence_text,
            asset_behavior_text=asset_behavior_text,
            time_stability_text=time_stability_text,
            strength_text=strength_text,
            conclusion=conclusion,
            combined_classification=combined_classification,
            recommendations=recommendations
        )
    
    def _extract_classification_inputs(self, 
                                      all_symbol_results: Dict[str, Dict[str, Any]],
                                      stability_results: Optional[Dict[str, Dict[str, Any]]] = None) -> ClassificationInputs:
        """
        Extract and validate inputs for classification.
        
        This method handles the messy data extraction and provides clean inputs
        to the deterministic classifier.
        """
        symbols = list(all_symbol_results.keys())
        p_values = []
        decisions = []
        r_squared_values = []
        survival_rates = []
        direction_consistent = []
        
        for symbol in symbols:
            symbol_data = all_symbol_results[symbol]
            
            # Get family data (handle multiple families)
            for family_name, family_data in symbol_data.items():
                # Extract decision
                decision = family_data.get('decision', {}).get('action', 'DROP')
                decisions.append(decision)
                
                # Extract p-values (use best p-value from experiments)
                experiments = family_data.get('experiments', [])
                if experiments:
                    best_p = min(exp.get('p_value', 1.0) for exp in experiments)
                    p_values.append(best_p)
                else:
                    p_values.append(1.0)  # Default to non-significant
                
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
                    else:
                        survival_rates.append(0.0)
                        direction_consistent.append(False)
                else:
                    survival_rates.append(0.0)
                    direction_consistent.append(False)
                
                # Only process first family for now (simplify for single-family analysis)
                break
        
        return ClassificationInputs(
            p_values=p_values,
            decisions=decisions,
            r_squared_values=r_squared_values,
            survival_rates=survival_rates,
            direction_consistent=direction_consistent,
            asset_symbols=symbols
        )
    
    def _prepare_asset_details(self, asset_strengths: Dict[str, AssetSignalStrength]) -> Dict[str, List[str]]:
        """
        Prepare asset details for template rendering.
        
        Groups assets by their signal strength for clean display.
        """
        asset_details = {
            "strongest": [],
            "strong": [],
            "moderate": [],
            "weak": [],
            "none": []
        }
        
        for symbol, strength in asset_strengths.items():
            if strength == AssetSignalStrength.STRONG:
                asset_details["strongest"].append(symbol)
            elif strength == AssetSignalStrength.MODERATE:
                asset_details["moderate"].append(symbol)
            elif strength == AssetSignalStrength.WEAK:
                asset_details["weak"].append(symbol)
            else:
                asset_details["none"].append(symbol)
        
        return asset_details
    
    def format_summary_output(self, summary: SignalSummary, family_name: str = "") -> str:
        """
        Format summary for CLI output.
        
        This creates the final human-readable display with proper formatting.
        """
        output = []
        
        # Header
        if family_name:
            output.append(f"\n🧠 SIGNAL SUMMARY — {family_name.upper().replace('_', ' ')}")
        else:
            output.append(f"\n🧠 SIGNAL SUMMARY")
        output.append(f"{'='*80}")
        
        # Signal existence
        output.append(f"\nSignal existence:")
        output.append(f"{summary.existence_text}")
        
        # Asset behavior
        output.append(f"\nAsset behavior:")
        output.append(f"{summary.asset_behavior_text}")
        
        # Time stability
        output.append(f"\nTime stability:")
        output.append(f"{summary.time_stability_text}")
        
        # Strength
        output.append(f"\nStrength:")
        output.append(f"{summary.strength_text}")
        
        # Combined classification
        output.append(f"\nCombined classification:")
        output.append(f"{summary.combined_classification}")
        
        # Conclusion
        output.append(f"\nConclusion:")
        output.append(f"{summary.conclusion}")
        
        # Recommendations
        if summary.recommendations:
            output.append(f"\n💡 Recommendations:")
            for i, rec in enumerate(summary.recommendations, 1):
                output.append(f"{i}. {rec}")
        
        return "\n".join(output)
