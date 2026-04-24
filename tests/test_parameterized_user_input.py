"""
Parameterized experiment runner with user input configuration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add src directory to Python path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from mini_research_lab.lab import MiniResearchLab
from mini_research_lab.data_loader import download_prices
from mini_research_lab.features import add_strategy_features, add_event_based_features
from mini_research_lab.experiment_specs import parameterized_experiments, generate_variations, generate_dynamic_experiments
from mini_research_lab.plotting import plot_experiment_bundle
from mini_research_lab.user_config import load_config, UserConfig

# Import new modular components
from mini_research_lab.core import (
    StandardScoringEngine,
    StandardDecisionEngine,
    HypothesisAwareDecisionEngine,
    ExperimentSpec,
    ExperimentResult
)
from mini_research_lab.core.enhanced_decisions import EnhancedDecisionEngine


def copy_json_to_reports(source_path: Path, target_name: str):
    """Copy JSON file to reports/ directory, overwriting if exists."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    target_path = reports_dir / target_name
    
    # Copy and overwrite
    import shutil
    shutil.copy2(source_path, target_path)
    print(f"Copied JSON to: {target_path}")


def create_experiment_json(exp_name: str, param_exp, result: dict, tables_dir: Path, figures_dir: Path, symbol: str) -> dict:
    """Create experiment JSON with structured data."""
    
    # Convert regression summary to dict if needed
    reg_dict = result["regression"].to_dict() if hasattr(result["regression"], 'to_dict') else result["regression"]
    
    # Create experiment JSON
    experiment_json = {
        "experiment_id": exp_name,
        "family": param_exp.base_name,
        "asset": symbol,
        "x_col": result["x_describe"].series_name if hasattr(result["x_describe"], 'series_name') else exp_name,
        "y_col": result["y_describe"].series_name if hasattr(result["y_describe"], 'series_name') else "fwd_ret_1d",
        "parameters": {
            "lookback_days": next((lb for lb in param_exp.lookbacks if f"_{lb}d" in exp_name), None),
            "forward_days": 1,
            "threshold": None
        },
        "hypothesis": {
            "expected_pattern": "negative_coefficient" if "mean_reversion" in param_exp.base_name or "ma_distance_reversion" in param_exp.base_name else "positive_coefficient",
            "expected_coef_sign": "negative" if "mean_reversion" in param_exp.base_name or "ma_distance_reversion" in param_exp.base_name else "positive",
            "description": param_exp.description_template.format(lookback="X", bucket="X")
        },
        "regression": {
            "intercept": reg_dict["intercept"],
            "coef": reg_dict["coef"],
            "std_err": reg_dict["std_err"],
            "t_value": reg_dict["t_value"],
            "p_value": reg_dict["p_value"],
            "r_squared": reg_dict["r_squared"],
            "n_obs": reg_dict["n_obs"],
            "ci_low": reg_dict["ci_low"],
            "ci_high": reg_dict["ci_high"]
        },
        "x_distribution": {
            "count": result["x_describe"].count,
            "mean": result["x_describe"].mean,
            "std": result["x_describe"].std,
            "min": result["x_describe"].min,
            "max": result["x_describe"].max
        },
        "y_distribution": {
            "count": result["y_describe"].count,
            "mean": result["y_describe"].mean,
            "std": result["y_describe"].std,
            "min": result["y_describe"].min,
            "max": result["y_describe"].max
        },
        "bucketed_analysis": result.get("bucketed_analysis", {}).to_dict() if result.get("bucketed_analysis") is not None else None
    }
    
    return convert_to_json_serializable(experiment_json)


def convert_to_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy types
        return obj.item()
    else:
        return obj


def create_family_summary_json(param_exp, results: dict, comparison_dir: Path, symbol: str) -> dict:
    """Create family summary JSON using modular components."""
    
    # Initialize scoring and decision engines
    scoring_engine = StandardScoringEngine()
    
    # Define hypothesis directions
    hypothesis_directions = {
        "mean_reversion": -1,  # Expect negative coefficients
        "momentum": 1,         # Expect positive coefficients  
        "volatility_clustering": 1,  # Expect positive coefficients
        "ma_distance_reversion": -1  # Expect negative coefficients
    }
    
    decision_engine = EnhancedDecisionEngine(hypothesis_directions)
    
    # Score all experiments
    scored_experiments = []
    for exp_name, result in results.items():
        # Create ExperimentResult object for scoring
        variations = generate_variations(param_exp)
        spec = next(s for s in variations if s.name == exp_name)
        
        exp_result = ExperimentResult(
            spec=spec,
            x_describe=result["x_describe"],
            y_describe=result["y_describe"], 
            regression=result["regression"],
            model=result["model"],
            bucketed_analysis=result.get("bucketed_analysis")
        )
        
        score = scoring_engine.score_experiment(exp_result)
        decision = decision_engine.make_experiment_decision(exp_result, score)
        
        scored_experiments.append({
            "experiment_id": exp_name,
            "coef": result["regression"].coef,
            "p_value": result["regression"].p_value,
            "r_squared": result["regression"].r_squared,
            "score": score,
            "experiment_decision": decision
        })
    
    # Create ExperimentResult objects for family scoring
    experiment_results = {}
    for exp_name, result in results.items():
        spec = next(s for s in generate_variations(param_exp) if s.name == exp_name)
        experiment_results[exp_name] = ExperimentResult(
            spec=spec,
            x_describe=result["x_describe"],
            y_describe=result["y_describe"],
            regression=result["regression"],
            model=result["model"],
            bucketed_analysis=result.get("bucketed_analysis")
        )
    
    # Make family decision
    family_scores = scoring_engine.score_family(experiment_results)
    family_decision = decision_engine.make_family_decision(experiment_results, family_scores)
    
    # Create family summary
    family_summary = {
        "family_id": param_exp.base_name,
        "experiment_family": param_exp.base_name,
        "asset": symbol,
        "total_variants": len(results),
        "successful_variants": len(results),
        "experiment_spec": {
            "base_name": param_exp.base_name,
            "x_col_pattern": param_exp.x_col_pattern,
            "y_col": param_exp.y_col,
            "lookbacks": param_exp.lookbacks,
            "title_template": param_exp.title_template,
            "description_template": param_exp.description_template
        },
        "experiments": scored_experiments,
        "family_metrics": family_scores,
        "llm_context": {
            "should_interpret_as_trading_signal": False,  # Conservative approach
            "important_cautions": [
                f"Average R-squared: {family_scores.get('avg_score', 0):.4f} indicates limited predictive power.",
                "Statistical significance does not imply tradable edge.",
                "Transaction costs would likely eliminate any small edge."
            ],
            "requested_output_style": "formal_research_summary"
        },
        "decision": family_decision
    }
    
    return convert_to_json_serializable(family_summary)


def run_experiments_for_symbol(symbol: str, config: UserConfig, scoring_engine, decision_engine, mode: str = "level", horizon: str = "1d"):
    """Run experiments for a single symbol."""
    
    print(f"\n{'='*80}")
    print(f"📊 RUNNING EXPERIMENTS FOR {symbol}")
    print(f"{'='*80}")
    
    # Download and prepare data
    try:
        prices = download_prices(symbol, start=config.start_date, end=config.get_actual_end_date())
        if prices.empty:
            print(f"⚠️  No data downloaded for {symbol}")
            return None
        
        # Choose feature generator based on mode
        if mode == "event":
            df = add_event_based_features(prices)
        else:
            df = add_strategy_features(prices)
        print(f"✅ Data loaded for {symbol}: {len(df)} observations from {df.index[0]} to {df.index[-1]}")
        
    except Exception as e:
        print(f"❌ Error downloading data for {symbol}: {e}")
        return None
    
    lab = MiniResearchLab(df)
    
    # Get dynamic experiments based on mode and horizon
    all_param_exps = generate_dynamic_experiments(horizon=horizon, mode=mode)
    
    # Filter by user-specified families
    param_exps = [exp for exp in all_param_exps if exp.base_name in config.families]
    
    print(f"\n📋 Experiment families to run: {', '.join(config.families)}")
    
    # Run each specified family
    all_results = {}
    all_family_summaries = {}
    
    for param_exp in param_exps:
        print(f"\n🎯 Running parameterized experiment: {param_exp.base_name}")
        print(f"Will test lookbacks: {param_exp.lookbacks}")
        print()
        
        # Run all variations using the lab
        results = lab.run_parameterized_experiment(param_exp)
        
        print(f"Completed {len(results)} variations:")
        for exp_name in results.keys():
            print(f"  - {exp_name}")
        
        # Save results for each variation
        for exp_name, result in results.items():
            # Create folder for this variation
            tables_dir = Path("reports/tables") / f"{symbol}_{exp_name}"
            tables_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summaries
            with open(tables_dir / "x_summary.txt", "w") as f:
                f.write("=== X Variable Summary ===\n")
                x_dict = result["x_describe"].to_dict()
                for key, value in x_dict.items():
                    f.write(f"{key}: {value}\n")

            with open(tables_dir / "y_summary.txt", "w") as f:
                f.write("=== Y Variable Summary ===\n")
                y_dict = result["y_describe"].to_dict()
                for key, value in y_dict.items():
                    f.write(f"{key}: {value}\n")

            with open(tables_dir / "regression.txt", "w") as f:
                f.write("=== Regression Summary ===\n")
                reg_dict = result["regression"].to_dict()
                for key, value in reg_dict.items():
                    f.write(f"{key}: {value}\n")

            # Save full model summary as text
            with open(tables_dir / "model_summary.txt", "w") as f:
                f.write(str(result["model"].summary()))

            # Save bucketed analysis
            if 'bucketed_analysis' in result:
                bucketed = result['bucketed_analysis']
                with open(tables_dir / "bucketed_analysis.txt", "w") as f:
                    f.write("=== BUCKETED RELATIONSHIP ANALYSIS ===\n")
                    f.write(f"X Variable: {exp_name}\n")
                    f.write(f"Number of buckets: {len(bucketed)}\n\n")
                    
                    # Check for monotonic relationship
                    y_means = bucketed['y_mean'].values
                    if len(y_means) >= 2:
                        increasing = all(y_means[i] <= y_means[i+1] for i in range(len(y_means)-1))
                        decreasing = all(y_means[i] >= y_means[i+1] for i in range(len(y_means)-1))
                        
                        if increasing:
                            f.write("Pattern: MONOTONIC INCREASING (higher X → higher Y)\n")
                        elif decreasing:
                            f.write("Pattern: MONOTONIC DECREASING (higher X → lower Y)\n")
                        else:
                            f.write("Pattern: NON-MONOTONIC (relationship varies across buckets)\n")
            
            # Generate plots for this variation
            variations = generate_variations(param_exp)
            spec = next(s for s in variations if s.name == exp_name)
            figures_dir = Path("reports/figures") / f"{symbol}_{exp_name}"
            saved_files = plot_experiment_bundle(
                df, spec.x_col, spec.y_col, 
                title_prefix="",
                output_dir=figures_dir
            )

            # Create experiment JSON using refactored components
            experiment_json = create_experiment_json(
                exp_name=exp_name,
                param_exp=param_exp,
                result=result,
                tables_dir=tables_dir,
                figures_dir=figures_dir,
                symbol=symbol
            )

            with open(tables_dir / "experiment.json", "w") as f:
                json.dump(experiment_json, f, indent=2)
            
            # Copy JSON to reports/ directory
            copy_json_to_reports(tables_dir / "experiment.json", f"{symbol}_{exp_name}.json")

        print(f"\nAll results saved to reports/tables/ and reports/figures/")

        # Create comparison summary file
        comparison_dir = Path("reports/tables") / f"{symbol}_{param_exp.base_name}_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed comparison summary
        with open(comparison_dir / "comparison_summary.txt", "w") as f:
            f.write(f"=== {param_exp.base_name.upper()} COMPARISON SUMMARY ===\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Experiment: {param_exp.description_template.format(lookback='X', bucket='X')}\n")
            f.write(f"Lookbacks tested: {param_exp.lookbacks}\n")
            f.write(f"Successful variations: {len(results)}\n\n")
            
            for exp_name, result in results.items():
                reg = result["regression"].to_dict()
                significance = "✓ SIGNIFICANT" if reg['p_value'] < 0.05 else "✗ NOT SIGNIFICANT"
                f.write(f"  - {exp_name}: coef={reg['coef']:+.4f}, p={reg['p_value']:.4f}, R²={reg['r_squared']:.4f} [{significance}]\n")

        # Also print summary to console
        print("\n=== COMPARISON SUMMARY ===")
        print("Lookback | Coefficient | P-value | R-squared | Observations")
        print("-" * 60)
        for exp_name, result in results.items():
            reg = result["regression"].to_dict()
            lookback = exp_name.split("_")[-3]  # Gets "1d", "3d", "5d", etc.
            print(f"{lookback:8} | {reg['coef']:11.4f} | {reg['p_value']:7.4f} | {reg['r_squared']:9.4f} | {reg['n_obs']:11}")

        # Create and save family summary JSON using refactored components
        family_summary = create_family_summary_json(param_exp, results, comparison_dir, symbol)
        with open(comparison_dir / "family_summary.json", "w") as f:
            json.dump(family_summary, f, indent=2)
        
        # Copy family summary to reports/ directory
        copy_json_to_reports(comparison_dir / "family_summary.json", f"{symbol}_{param_exp.base_name}_family_summary.json")

        print(f"\nComparison summary saved to: {comparison_dir / 'comparison_summary.txt'}")
        print(f"Family summary saved to: {comparison_dir / 'family_summary.json'}")
        
        # Print comprehensive console summary
        print(f"\n{'='*80}")
        print(f"📊 {symbol} - {param_exp.base_name.upper()} COMPARISON & DECISION SUMMARY")
        print(f"{'='*80}")
        
        # Comparison table
        print(f"\n📈 EXPERIMENT COMPARISON:")
        print("-" * 80)
        print(f"{'Experiment':<25} {'Coef':<8} {'P-value':<10} {'R²':<8} {'Significance':<12} {'Decision':<10}")
        print("-" * 80)
        
        # Get scored experiments for decisions
        scored_experiments_list = []
        for exp_name, result in results.items():
            variations = generate_variations(param_exp)
            spec = next(s for s in variations if s.name == exp_name)
            
            exp_result = ExperimentResult(
                spec=spec,
                x_describe=result["x_describe"],
                y_describe=result["y_describe"],
                regression=result["regression"],
                model=result["model"],
                bucketed_analysis=result.get("bucketed_analysis")
            )
            
            score = scoring_engine.score_experiment(exp_result)
            decision = decision_engine.make_experiment_decision(exp_result, score)
            
            scored_experiments_list.append({
                "experiment_id": exp_name,
                "experiment_decision": decision
            })
        
        for exp_name, result in results.items():
            reg = result["regression"].to_dict()
            significance = "✓ SIGNIFICANT" if reg['p_value'] < 0.05 else "✗ NOT SIGNIFICANT"
            
            # Get decision from scored experiments
            scored_exp = next((exp for exp in scored_experiments_list if exp['experiment_id'] == exp_name), None)
            decision = scored_exp['experiment_decision']['action'] if scored_exp else "N/A"
            
            print(f"{exp_name:<25} {reg['coef']:+8.4f} {reg['p_value']:10.4f} {reg['r_squared']:8.4f} {significance:<12} {decision:<10}")
        
        # Family decision summary
        family_decision = family_summary.get('decision', {})
        print(f"\n🎯 FAMILY DECISION: {family_decision.get('action', 'N/A')}")
        print(f"Confidence: {family_decision.get('confidence', 0):.1f}")
        print(f"Reason: {family_decision.get('reason', 'N/A')}")
        
        if family_decision.get('reason_codes'):
            print(f"Reason Codes: {', '.join(family_decision['reason_codes'])}")
        
        # Selected experiments
        if family_decision.get('selected_experiments'):
            print(f"\n🏆 SELECTED EXPERIMENTS:")
            for exp_id, role in family_decision['selected_experiments'].items():
                print(f"  {exp_id}: {role}")
        
        # Family metrics
        if 'family_metrics' in family_summary:
            metrics = family_summary['family_metrics']
            print(f"\n📊 FAMILY METRICS:")
            print(f"  Average Score: {metrics.get('avg_score', 0):.3f}")
            print(f"  Best Score: {metrics.get('max_score', 0):.3f}")
            print(f"  Consistency: {metrics.get('consistency_score', 0):.3f}")
            print(f"  Explanatory Power: {metrics.get('explanatory_power', 0):.3f}")
            print(f"  Family Label: {metrics.get('label', 'N/A')}")
        
        # Store results for this family
        all_results[param_exp.base_name] = results
        
        # Store family summary for cross-symbol interpretation
        all_family_summaries[param_exp.base_name] = family_summary
    
    return all_results, all_family_summaries


def main():
    """Main function using user configuration."""
    
    # Load user configuration
    config = load_config("user_config.json")
    config.print_summary()
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("⚠️  Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return
    
    # Initialize scoring and decision engines
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
    
    print(f"\n✅ All experiments completed!")
    print(f"📁 Results saved to reports/tables/ and reports/figures/")
    print(f"📄 JSON files copied to reports/ directory")


if __name__ == "__main__":
    main()
