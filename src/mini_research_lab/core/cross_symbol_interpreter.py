"""
Cross-Symbol Interpretation Layer

Clean interface to the new deterministic interpretation system.
This file now acts as a wrapper around the interpretation/ module.
"""

from typing import Dict, Any, Optional
from ..interpretation.summary_builder import SummaryBuilder, SignalSummary


class CrossSymbolInterpreter:
    """
    Clean interface for cross-symbol signal interpretation.
    
    This class now delegates to the new interpretation system while maintaining
    backward compatibility with the existing CLI.
    """
    
    def __init__(self):
        self.summary_builder = SummaryBuilder()
    
    def interpret_cross_symbol_results(self, 
                                     all_symbol_results: Dict[str, Dict[str, Any]],
                                     stability_results: Optional[Dict[str, Dict[str, Any]]] = None) -> SignalSummary:
        """
        Interpret signal behavior across multiple symbols and time.
        
        Args:
            all_symbol_results: Dictionary of symbol -> family_results -> decision/metrics
            stability_results: Dictionary of symbol -> stability metrics
            
        Returns:
            SignalSummary: Complete deterministic summary
        """
        # Extract family name from results (assume single family for now)
        family_name = ""
        if all_symbol_results:
            first_symbol = list(all_symbol_results.keys())[0]
            first_symbol_data = all_symbol_results[first_symbol]
            if first_symbol_data:
                family_name = list(first_symbol_data.keys())[0]
        
        return self.summary_builder.build_summary(
            all_symbol_results=all_symbol_results,
            stability_results=stability_results,
            family_name=family_name
        )
    
    def format_interpretation_output(self, summary: SignalSummary, family_name: str = "") -> str:
        """
        Format interpretation for CLI output.
        
        Args:
            summary: SignalSummary from interpretation
            family_name: Name of signal family for display
            
        Returns:
            Formatted string for CLI display
        """
        return self.summary_builder.format_summary_output(summary, family_name)
