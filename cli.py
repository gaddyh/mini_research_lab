#!/usr/bin/env python3
"""
Command Line Interface for Mini Research Lab.
"""

import argparse
import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from tests.test_parameterized_user_input import run_experiments_for_symbol
from mini_research_lab.user_config import UserConfig


def main():
    """CLI main function."""
    parser = argparse.ArgumentParser(description="Mini Research Lab - Financial Signal Analysis")
    
    # Arguments
    parser.add_argument("--symbols", nargs="+", help="Symbols to analyze (e.g., AAPL MSFT SPY)")
    parser.add_argument("--family", help="Single family to analyze (e.g., volatility_clustering)")
    parser.add_argument("--start-date", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="today", help="End date (YYYY-MM-DD or 'today')")
    parser.add_argument("--train-end-date", default="2020-12-31", help="Train end date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Create user config from CLI args
    config = UserConfig(
        symbols=args.symbols or ["AAPL"],  # Default to AAPL if not specified
        start_date=args.start_date,
        end_date=args.end_date,
        train_end_date=args.train_end_date,
        families=[args.family] if args.family else ["mean_reversion", "momentum", "volatility_clustering", "ma_distance_reversion"]  # Single family if specified, otherwise all families
    )
    
    # Validate config
    issues = config.validate()
    if issues:
        print("⚠️  Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    
    # Print configuration
    config.print_summary()
    
    # Initialize scoring and decision engines
    from mini_research_lab.core import StandardScoringEngine
    from mini_research_lab.core.enhanced_decisions import EnhancedDecisionEngine
    
    scoring_engine = StandardScoringEngine()
    
    # Define hypothesis directions
    hypothesis_directions = {
        "mean_reversion": -1,  # Expect negative coefficients
        "momentum": 1,         # Expect positive coefficients  
        "volatility_clustering": 1,  # Expect positive coefficients
        "ma_distance_reversion": -1  # Expect negative coefficients
    }
    
    decision_engine = EnhancedDecisionEngine(hypothesis_directions)
    
    # Run experiments for each symbol
    all_symbol_results = {}
    
    for symbol in config.symbols:
        results = run_experiments_for_symbol(symbol, config, scoring_engine, decision_engine)
        if results:
            all_symbol_results[symbol] = results
    
    # Print overall summary
    print(f"\n{'='*80}")
    print(f"📊 OVERALL SUMMARY - ALL SYMBOLS")
    print(f"{'='*80}")
    
    for symbol, symbol_results in all_symbol_results.items():
        print(f"\n🎯 {symbol}:")
        for family_name, family_results in symbol_results.items():
            print(f"  {family_name}: {len(family_results)} experiments completed")
    
    # Cross-symbol summary
    print(f"\n{'='*80}")
    print(f"📊 CROSS-SYMBOL SUMMARY")
    print(f"{'='*80}")
    
    # Collect all family decisions across symbols
    family_decisions = {}
    for symbol, symbol_results in all_symbol_results.items():
        for family_name, family_results in symbol_results.items():
            # Get family decision from the results
            from tests.test_parameterized_user_input import create_family_summary_json
            from mini_research_lab.core import StandardScoringEngine
            from mini_research_lab.core.enhanced_decisions import EnhancedDecisionEngine
            
            # Create a dummy param_exp for the family
            if family_name == "mean_reversion":
                from mini_research_lab.experiment_specs import ParameterizedExperiment
                param_exp = ParameterizedExperiment(
                    base_name=family_name,
                    x_col_pattern="ret_{lookback}d",
                    y_col="fwd_ret_1d",
                    lookbacks=[1, 3, 5, 10, 20],
                    title_template="Mean Reversion ({lookback}d)",
                    description_template="Mean reversion using {lookback}d return"
                )
            elif family_name == "momentum":
                from mini_research_lab.experiment_specs import ParameterizedExperiment
                param_exp = ParameterizedExperiment(
                    base_name=family_name,
                    x_col_pattern="ret_{lookback}d",
                    y_col="fwd_ret_3d",
                    lookbacks=[3, 5, 10, 20],
                    title_template="Momentum ({lookback}d)",
                    description_template="Momentum using {lookback}d return"
                )
            elif family_name == "volatility_clustering":
                from mini_research_lab.experiment_specs import ParameterizedExperiment
                param_exp = ParameterizedExperiment(
                    base_name=family_name,
                    x_col_pattern="abs_ret_{lookback}d",
                    y_col="fwd_abs_ret_1d",
                    lookbacks=[1, 3, 5],
                    title_template="Volatility Clustering ({lookback}d)",
                    description_template="Volatility clustering using {lookback}d absolute return"
                )
            elif family_name == "ma_distance_reversion":
                from mini_research_lab.experiment_specs import ParameterizedExperiment
                param_exp = ParameterizedExperiment(
                    base_name=family_name,
                    x_col_pattern="dist_from_ma{lookback}",
                    y_col="fwd_ret_1d",
                    lookbacks=[10, 20, 50],
                    title_template="MA Distance Reversion ({lookback}d)",
                    description_template="MA distance reversion using {lookback}d distance"
                )
            else:
                continue
            
            # Create ExperimentResult objects from family results
            experiment_results = {}
            for exp_name, result in family_results.items():
                from tests.test_parameterized_user_input import generate_variations
                from mini_research_lab.core import ExperimentResult
                variations = generate_variations(param_exp)
                spec = next(s for s in variations if s.name == exp_name)
                experiment_results[exp_name] = ExperimentResult(
                    spec=spec,
                    x_describe=result["x_describe"],
                    y_describe=result["y_describe"],
                    regression=result["regression"],
                    model=result["model"],
                    bucketed_analysis=result.get("bucketed_analysis")
                )
            
            # Score and get decision
            scoring_engine = StandardScoringEngine()
            hypothesis_directions = {
                "mean_reversion": -1,
                "momentum": 1,
                "volatility_clustering": 1,
                "ma_distance_reversion": -1
            }
            decision_engine = EnhancedDecisionEngine(hypothesis_directions)
            
            family_scores = scoring_engine.score_family(experiment_results)
            family_decision = decision_engine.make_family_decision(experiment_results, family_scores)
            
            if family_name not in family_decisions:
                family_decisions[family_name] = []
            family_decisions[family_name].append(family_decision.get('action', 'N/A'))
    
    # Print cross-symbol summary
    for family_name, decisions in family_decisions.items():
        promote_count = decisions.count("PROMOTE")
        refine_count = decisions.count("REFINE") 
        drop_count = decisions.count("DROP")
        
        if promote_count == len(decisions):
            status = "✔ Strong across market"
            confidence = "HIGH"
        elif promote_count >= len(decisions) * 0.67:
            status = "✔ Mostly strong"
            confidence = "HIGH"
        elif refine_count >= len(decisions) * 0.67:
            status = "⚠ Signal is weak or regime-dependent"
            confidence = "MEDIUM"
        elif drop_count >= len(decisions) * 0.67:
            status = "⚠ Signal is weak or regime-dependent"
            confidence = "MEDIUM"
        else:
            status = "✗ Inconsistent"
            confidence = "LOW"
        
        print(f"\n{family_name.upper().replace('_', ' ')}:")
        print(f"  {promote_count}/{len(decisions)} → PROMOTE")
        print(f"  {refine_count}/{len(decisions)} → REFINE") 
        print(f"  {drop_count}/{len(decisions)} → DROP")
        print(f"  {status}")
        print(f"  Confidence: {confidence}")
    
    print(f"\n✅ All experiments completed!")
    print(f"📁 Results saved to reports/tables/ and reports/figures/")
    print(f"📄 JSON files copied to reports/ directory")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
