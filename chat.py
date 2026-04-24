#!/usr/bin/env python3
"""
Chat Interface for Experiment Results

Thin LLM layer over deterministic CLI results.
User questions → LLM → tools → explanations
"""

import json
import os
import sys
from typing import Dict, Any, Optional
from pathlib import Path


class ExperimentTools:
    """Tools for accessing experiment results."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
    
    def get_summary(self, symbol: str, family: str) -> Dict[str, Any]:
        """Get family summary for a symbol."""
        # Try multiple possible paths for the summary file
        possible_paths = [
            self.reports_dir / "tables" / f"{symbol}_{family}_comparison" / "family_summary.json",
            self.reports_dir / "tables" / f"{symbol}_{family}_family_summary.json",
            self.reports_dir / f"{symbol}_{family}_family_summary.json"
        ]
        
        for summary_file in possible_paths:
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    return json.load(f)
        
        return {"error": f"Summary not found for {symbol} {family}"}
    
    def get_experiment(self, symbol: str, experiment_name: str) -> Dict[str, Any]:
        """Get specific experiment results."""
        experiment_file = self.reports_dir / f"{symbol}_{experiment_name}.json"
        if experiment_file.exists():
            with open(experiment_file, 'r') as f:
                return json.load(f)
        return {"error": f"Experiment not found: {symbol} {experiment_name}"}
    
    def get_stability(self, symbol: str, family: str) -> Dict[str, Any]:
        """Get stability analysis for a symbol."""
        # Try multiple possible paths for the stability file
        possible_paths = [
            self.reports_dir / "tables" / f"{family}_stability_comparison" / "stability_analysis.json",
            self.reports_dir / "tables" / f"{symbol}_{family}_stability_comparison" / "stability_analysis.json",
            self.reports_dir / "tables" / f"{symbol}_{family}_stability_analysis.json",
            self.reports_dir / f"{symbol}_{family}_stability_analysis.json"
        ]
        
        for path in possible_paths:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract stability data for the specific symbol
                    if 'family_stability' in data:
                        stability_data = data['family_stability']
                        
                        # Look for symbol-specific stability data in individual experiments
                        if 'individual_stabilities' in stability_data:
                            # Check each individual experiment for symbol data
                            for exp_name, exp_data in stability_data['individual_stabilities'].items():
                                if isinstance(exp_data, dict):
                                    # Check if this experiment has symbol-specific data
                                    if 'train_p_value' in exp_data and 'test_p_value' in exp_data:
                                        # Calculate stability metrics for this symbol
                                        train_sig = exp_data.get('train_p_value', 1.0) < 0.05
                                        test_sig = exp_data.get('test_p_value', 1.0) < 0.05
                                        direction_stable = exp_data.get('direction_stable', False)
                                        
                                        return {
                                            'significance_survival_rate': 100.0 if (train_sig and test_sig) else 0.0,
                                            'direction_consistent': direction_stable,
                                            'stability_score': exp_data.get('stability_score', 'N/A'),
                                            'stability_label': exp_data.get('stability_label', 'N/A'),
                                            'train_p_value': exp_data.get('train_p_value', 'N/A'),
                                            'test_p_value': exp_data.get('test_p_value', 'N/A'),
                                            'train_r_squared': exp_data.get('train_r_squared', 'N/A'),
                                            'test_r_squared': exp_data.get('test_r_squared', 'N/A')
                                        }
                        
                        # Return overall family stability if symbol-specific not found
                        return {
                            'significance_survival_rate': stability_data.get('significance_survival_rate', 'N/A'),
                            'direction_consistent': stability_data.get('direction_consistent', 'N/A'),
                            'avg_stability_score': stability_data.get('avg_stability_score', 'N/A')
                        }
                    
                    return data
                    
                except Exception as e:
                    print(f"Error loading stability from {path}: {e}")
                    continue
        
        return {"error": f"Stability data not found for {symbol} {family}"}
    
    def list_available_data(self) -> Dict[str, Any]:
        """List all available symbols and families."""
        data = {"symbols": {}, "families": set()}
        
        # Scan reports directory for JSON files
        for json_file in self.reports_dir.glob("*.json"):
            self._process_json_file(json_file, data)
        
        # Also scan tables subdirectory for JSON files
        tables_dir = self.reports_dir / "tables"
        if tables_dir.exists():
            for json_file in tables_dir.rglob("*.json"):  # Recursive search
                self._process_json_file(json_file, data)
        
        # Convert sets to lists for JSON serialization
        for symbol in data["symbols"]:
            data["symbols"][symbol] = sorted(list(data["symbols"][symbol]))
        data["families"] = sorted(list(data["families"]))
        
        return data
    
    def _process_json_file(self, json_file: Path, data: Dict[str, Any]) -> None:
        """Process a single JSON file to extract symbol and family info."""
        # Extract symbol and family from directory path, not filename
        dir_parts = json_file.parent.name.split("_")
        if len(dir_parts) >= 2:
            symbol = dir_parts[0]
            
            # Remove timestamp suffixes like "_10d_to", "_20d_to", "_50d_to"
            family_parts = []
            for part in dir_parts[1:]:
                if not part.endswith("d_to") and not part.isdigit():
                    family_parts.append(part)
            family = "_".join(family_parts)
            
            # Also check for family summaries in tables directory
            if not family or len(family) < 2:
                if "comparison" in json_file.parent.name:
                    table_parts = json_file.parent.name.split("_")
                    if len(table_parts) >= 2:
                        family_from_table = "_".join(table_parts[1:-1])  # Exclude "comparison"
                        if family_from_table:
                            family = family_from_table
            
            if symbol not in data["symbols"]:
                data["symbols"][symbol] = set()
            if family and len(family) >= 2:
                data["symbols"][symbol].add(family)
                data["families"].add(family)


class ChatInterface:
    """Main chat interface."""
    
    def __init__(self):
        self.tools = ExperimentTools()
        self.context = {}
    
    def _is_family_match(self, question: str, family: str) -> bool:
        """Check if user's family term matches the actual family name."""
        question_words = question.split()
        family_lower = family.lower()
        
        # Extract base family name from full name (remove timestamps)
        base_family = family_lower
        for suffix in ["_10d_to_1d", "_20d_to_1d", "_50d_to_1d", "_family_summary"]:
            if family_lower.endswith(suffix):
                base_family = family_lower[:-len(suffix)]
                break
        
        # Check if any word in question matches the base family
        for word in question_words:
            if word == "for":
                continue
            if base_family.startswith(word) or word.startswith(base_family):
                return True
        
        return False
    
    def _extract_base_family(self, family: str) -> str:
        """Extract base family name from full family name."""
        family_lower = family.lower()
        
        # Remove timestamp suffixes
        for suffix in ["_10d_to_1d", "_20d_to_1d", "_50d_to_1d", "_family_summary"]:
            if family_lower.endswith(suffix):
                return family_lower[:-len(suffix)]
        
        return family_lower
    
    def process_question(self, question: str) -> str:
        """
        Process user question using available tools.
        
        For now, this is a simple rule-based system.
        In a full implementation, this would call an LLM.
        """
        question_lower = question.lower()
        
        # Check what data is available
        available_data = self.tools.list_available_data()
        
        # Help command
        if "help" in question_lower or "what" in question_lower and "available" in question_lower:
            return self._format_available_data(available_data)
        
        # Summary questions
        if "summary" in question_lower:
            # Parse "summary for [symbol] [family]" pattern
            words = question_lower.split()
            if "for" in words:
                for symbol in available_data["symbols"]:
                    if symbol.lower() in question_lower:
                        # Try to find matching family (handle partial matches)
                        matched_family = None
                        for family in available_data["symbols"][symbol]:
                            # Check if user's family term is contained in the actual family name
                            if self._is_family_match(question_lower, family):
                                matched_family = family
                                break
                        
                        if matched_family:
                            # Extract base family name for file lookup
                            base_family_name = self._extract_base_family(matched_family)
                            summary = self.tools.get_summary(symbol, base_family_name)
                            return self._format_summary(symbol, matched_family, summary)
        
        # Experiment questions
        if "experiment" in question_lower:
            # Parse "experiment for [symbol] [family]" pattern
            words = question_lower.split()
            if "for" in words:
                for symbol in available_data["symbols"]:
                    if symbol.lower() in question_lower:
                        # Try to find matching family (handle partial matches)
                        matched_family = None
                        for family in available_data["symbols"][symbol]:
                            if self._is_family_match(question_lower, family):
                                matched_family = family
                                break
                        
                        if matched_family:
                            base_family_name = self._extract_base_family(matched_family)
                            summary = self.tools.get_summary(symbol, base_family_name)
                            if "selected_experiments" in summary:
                                best_exp = summary["selected_experiments"].get("best_candidate")
                                if best_exp:
                                    exp_data = self.tools.get_experiment(symbol, best_exp)
                                    return self._format_experiment(symbol, best_exp, exp_data)
        
        # Stability questions
        if "stability" in question_lower:
            # Parse "stability for [symbol] [family]" pattern
            words = question_lower.split()
            if "for" in words:
                for symbol in available_data["symbols"]:
                    if symbol.lower() in question_lower:
                        # Try to find matching family (handle partial matches)
                        matched_family = None
                        for family in available_data["symbols"][symbol]:
                            if self._is_family_match(question_lower, family):
                                matched_family = family
                                break
                        
                        if matched_family:
                            base_family_name = self._extract_base_family(matched_family)
                            stability = self.tools.get_stability(symbol, base_family_name)
                            return self._format_stability(symbol, matched_family, stability)
        
        # Why questions (explanations)
        if "why" in question_lower:
            return self._answer_why_question(question, available_data)
        
        # Default response
        return self._default_response(available_data)
    
    def _format_available_data(self, data: Dict[str, Any]) -> str:
        """Format available data for display."""
        response = ["📊 Available Data:", ""]
        
        for symbol, families in data["symbols"].items():
            response.append(f"🎯 {symbol}:")
            for family in families:
                response.append(f"  - {family}")
            response.append("")
        
        response.append("💡 Ask me about:")
        response.append("  - 'summary for [symbol] [family]'")
        response.append("  - 'experiment for [symbol] [family]'")
        response.append("  - 'stability for [symbol] [family]'")
        response.append("  - 'why is [symbol] [decision]'")
        
        return "\n".join(response)
    
    def _format_summary(self, symbol: str, family: str, summary: Dict[str, Any]) -> str:
        """Format family summary for display."""
        if "error" in summary:
            return f"❌ {summary['error']}"
        
        response = [
            f"📊 {symbol.upper()} - {family.upper().replace('_', ' ')} SUMMARY",
            "=" * 60,
            ""
        ]
        
        if "decision" in summary:
            decision = summary["decision"]
            response.append(f"🎯 Decision: {decision.get('action', 'UNKNOWN')}")
            response.append(f"📈 Confidence: {decision.get('confidence', 'N/A')}")
            response.append(f"💭 Reason: {decision.get('reason', 'N/A')}")
            response.append("")
        
        if "family_metrics" in summary:
            metrics = summary["family_metrics"]
            response.append("📊 Family Metrics:")
            response.append(f"  - Average Score: {metrics.get('average_score', 'N/A')}")
            response.append(f"  - Best Score: {metrics.get('best_score', 'N/A')}")
            response.append(f"  - Consistency: {metrics.get('consistency', 'N/A')}")
            response.append(f"  - Explanatory Power: {metrics.get('explanatory_power', 'N/A')}")
            response.append("")
        
        if "selected_experiments" in summary:
            selected = summary["selected_experiments"]
            response.append("🏆 Selected Experiments:")
            for role, exp_name in selected.items():
                response.append(f"  - {role}: {exp_name}")
            response.append("")
        
        return "\n".join(response)
    
    def _format_experiment(self, symbol: str, experiment_name: str, exp_data: Dict[str, Any]) -> str:
        """Format experiment results for display."""
        if "error" in exp_data:
            return f"❌ {exp_data['error']}"
        
        response = [
            f"🧪 {symbol.upper()} - {experiment_name.upper().replace('_', ' ')}",
            "=" * 60,
            ""
        ]
        
        if "results" in exp_data:
            results = exp_data["results"]
            response.append("📈 Results:")
            response.append(f"  - Coefficient: {results.get('coefficient', 'N/A')}")
            response.append(f"  - P-value: {results.get('p_value', 'N/A')}")
            response.append(f"  - R-squared: {results.get('r_squared', 'N/A')}")
            response.append(f"  - Observations: {results.get('observations', 'N/A')}")
            response.append("")
            
            # Significance indicator
            p_value = results.get('p_value', 1.0)
            if p_value < 0.05:
                response.append("✅ Statistically Significant")
            else:
                response.append("❌ Not Statistically Significant")
            response.append("")
        
        if "config" in exp_data:
            config = exp_data["config"]
            response.append("⚙️ Configuration:")
            for key, value in config.items():
                response.append(f"  - {key}: {value}")
        
        return "\n".join(response)
    
    def _format_stability(self, symbol: str, family: str, stability: Dict[str, Any]) -> str:
        """Format stability analysis for display."""
        if "error" in stability:
            return f"❌ {stability['error']}"
        
        response = [
            f"🔍 {symbol.upper()} - {family.upper().replace('_', ' ')} STABILITY",
            "=" * 60,
            ""
        ]
        
        response.append("📊 Stability Metrics:")
        response.append(f"  - Average Stability Score: {stability.get('average_stability_score', 'N/A')}")
        response.append(f"  - Best Stability Score: {stability.get('best_stability_score', 'N/A')}")
        response.append(f"  - Direction Consistent: {'✅' if stability.get('direction_consistent', False) else '❌'}")
        response.append(f"  - Significance Survival Rate: {stability.get('significance_survival_rate', 'N/A')}%")
        response.append("")
        
        if "stability_breakdown" in stability:
            breakdown = stability["stability_breakdown"]
            response.append("📈 Stability Breakdown:")
            response.append(f"  - High Stability: {breakdown.get('high_stability', 0)} experiments")
            response.append(f"  - Moderate Stability: {breakdown.get('moderate_stability', 0)} experiments")
            response.append(f"  - Low Stability: {breakdown.get('low_stability', 0)} experiments")
        
        return "\n".join(response)
    
    def _answer_why_question(self, question: str, available_data: Dict[str, Any]) -> str:
        """Answer 'why' questions about decisions."""
        question_lower = question.lower()
        
        # Look for symbol and decision in question
        for symbol in available_data["symbols"]:
            if symbol.lower() in question_lower:
                for family in available_data["symbols"][symbol]:
                    if family.lower() in question_lower:
                        summary = self.tools.get_summary(symbol, family)
                        if "decision" in summary:
                            decision = summary["decision"]["action"]
                            reason = summary["decision"].get("reason", "")
                            
                            response = [
                                f"🤔 Why {symbol.upper()} was {decision.upper()} for {family.upper().replace('_', ' ')}:",
                                "",
                                f"📝 Reason: {reason}",
                                ""
                            ]
                            
                            # Add context based on decision type
                            if decision == "PROMOTE":
                                response.append("💡 This means the signal showed strong statistical significance and consistency across experiments.")
                            elif decision == "REFINE":
                                response.append("💡 This means the signal showed promise but needs improvement in one or more areas (significance, consistency, or effect size).")
                            elif decision == "DROP":
                                response.append("💡 This means the signal failed to meet minimum criteria for reliability.")
                            
                            # Add stability context if available
                            stability = self.tools.get_stability(symbol, family)
                            if "significance_survival_rate" in stability:
                                survival_rate = stability["significance_survival_rate"]
                                if survival_rate == 0.0:
                                    response.append("")
                                    response.append("⚠️  Important: Stability analysis shows 0% significance survival rate, meaning the signal does not generalize over time.")
                                elif survival_rate < 50.0:
                                    response.append("")
                                    response.append(f"⚠️  Important: Stability analysis shows only {survival_rate}% significance survival rate, indicating temporal fragility.")
                            
                            return "\n".join(response)
        
        return "❓ I couldn't find the specific decision you're asking about. Try 'help' to see available data."
    
    def _default_response(self, available_data: Dict[str, Any]) -> str:
        """Default response when question doesn't match patterns."""
        response = [
            "❓ I didn't understand that question.",
            "",
            "💡 Try asking about:",
            "  - 'summary for SPY ma_distance_reversion'",
            "  - 'experiment for AAPL ma_distance_reversion'",
            "  - 'stability for MSFT ma_distance_reversion'",
            "  - 'why is NVDA DROP for ma_distance_reversion'",
            "  - 'help' to see all available data",
            "",
            f"📊 Available symbols: {', '.join(available_data['symbols'].keys())}"
        ]
        return "\n".join(response)


def main():
    """Main chat interface."""
    print("🤖 Experiment Results Chat Interface")
    print("=" * 50)
    print("Ask questions about your experiment results!")
    print("Type 'quit' to exit, 'help' for available data.")
    print()
    
    chat = ChatInterface()
    
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print()
            response = chat.process_question(question)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


if __name__ == "__main__":
    main()
