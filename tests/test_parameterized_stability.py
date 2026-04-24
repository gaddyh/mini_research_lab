"""
Refactored stability analysis using modular components.
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
from mini_research_lab.features import add_return_features
from mini_research_lab.experiment_specs import parameterized_experiments, generate_variations

# Import new modular components
from mini_research_lab.core import (
    StandardStabilityAnalyzer,
    FamilyStabilityAnalyzer,
    StabilityConfig,
    ExperimentSpec,
    ExperimentResult
)


def copy_json_to_reports(source_path: Path, target_name: str):
    """Copy JSON file to reports/ directory, overwriting if exists."""
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    target_path = reports_dir / target_name
    
    # Copy and overwrite
    import shutil
    shutil.copy2(source_path, target_path)
    print(f"Copied JSON to: {target_path}")


def split_data_by_time(df: pd.DataFrame):
    """Split data into train (2015-2020) and test (2021-today) periods."""
    train_start = "2015-01-01"
    train_end = "2020-12-31"
    test_start = "2021-01-01"
    
    train_df = df.loc[train_start:train_end].copy()
    test_df = df.loc[test_start:].copy()
    
    return train_df, test_df


def run_experiment_with_stability(param_exp, train_df, test_df, exp_name, stability_analyzer):
    """Run single experiment on both train and test periods using modular components."""
    
    # Extract x_col and y_col from experiment name
    variations = generate_variations(param_exp)
    spec = next(s for s in variations if s.name == exp_name)
    
    # Train period
    train_lab = MiniResearchLab(train_df)
    train_result_dict = train_lab.run_experiment(spec.x_col, spec.y_col)
    
    # Test period
    test_lab = MiniResearchLab(test_df)
    test_result_dict = test_lab.run_experiment(spec.x_col, spec.y_col)
    
    # Create ExperimentResult objects
    train_result = ExperimentResult(
        spec=spec,
        x_describe=train_result_dict["x_describe"],
        y_describe=train_result_dict["y_describe"],
        regression=train_result_dict["regression"],
        model=train_result_dict["model"],
        bucketed_analysis=train_result_dict.get("bucketed_analysis")
    )
    
    test_result = ExperimentResult(
        spec=spec,
        x_describe=test_result_dict["x_describe"],
        y_describe=test_result_dict["y_describe"],
        regression=test_result_dict["regression"],
        model=test_result_dict["model"],
        bucketed_analysis=test_result_dict.get("bucketed_analysis")
    )
    
    # Analyze stability using modular component
    stability = stability_analyzer.analyze_stability(train_result, test_result)
    
    return train_result_dict, test_result_dict, stability


def main():
    """Main function using refactored stability analysis components."""
    
    # Download and prepare data
    try:
        prices = download_prices("AAPL", start="2015-01-01")
        if prices.empty:
            raise ValueError("No price data downloaded for AAPL")
        df = add_return_features(prices)
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Using fallback synthetic data for demonstration...")
        
        # Create synthetic fallback data for demonstration
        import numpy as np
        dates = pd.date_range("2015-01-01", "2024-12-31", freq='D')
        n_days = len(dates)
        
        # Generate synthetic price data with realistic characteristics
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0, 0.02, n_days)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))  # Start at 100, random walk
        
        df = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        df.set_index('date', inplace=True)
        df = add_return_features(df)
    
    # Split into train and test periods
    train_df, test_df = split_data_by_time(df)
    
    # Get parameterized experiments
    param_exps = parameterized_experiments()
    
    # Initialize stability analyzer with custom config
    stability_config = StabilityConfig(
        direction_weight=40,
        significance_weight=30,
        r2_weight=20,
        decay_weight=10,
        significance_threshold_train=0.05,
        significance_threshold_test=0.10,
        r2_survival_threshold=0.5,
        decay_stable_threshold=0.7,
        decay_weak_threshold=0.3
    )
    
    stability_analyzer = StandardStabilityAnalyzer(stability_config)
    family_stability_analyzer = FamilyStabilityAnalyzer(stability_analyzer)
    
    print("Available parameterized experiments:")
    for i, param_exp in enumerate(param_exps):
        print(f"{i+1}. {param_exp.base_name}: {len(param_exp.lookbacks)} variations")
        print(f"   Lookbacks: {param_exp.lookbacks}")
        print(f"   Pattern: {param_exp.x_col_pattern} -> {param_exp.y_col}")
        print()
    
    # Run all 4 experiment families with stability analysis
    all_families_results = {}
    
    for param_exp in param_exps:
        print(f"\n{'='*60}")
        print(f"Running parameterized experiment: {param_exp.base_name}")
        print(f"Will test lookbacks: {param_exp.lookbacks}")
        print()
        
        # Run all variations for this family
        train_results = {}
        test_results = {}
        stabilities = {}
        
        variations = []
        for lookback in param_exp.lookbacks:
            exp_name = f"{param_exp.base_name}_{lookback}d_to_1d"
            variations.append(exp_name)
        
        for exp_name in variations:
            print(f"Running {exp_name}...")
            train_result, test_result, stability = run_experiment_with_stability(
                param_exp, train_df, test_df, exp_name, stability_analyzer
            )
            
            train_results[exp_name] = train_result
            test_results[exp_name] = test_result
            stabilities[exp_name] = stability
            
            print(f"  Train: coef={train_result['regression'].coef:.4f}, p={train_result['regression'].p_value:.4f}, R²={train_result['regression'].r_squared:.4f}")
            print(f"  Test:  coef={test_result['regression'].coef:.4f}, p={test_result['regression'].p_value:.4f}, R²={test_result['regression'].r_squared:.4f}")
            print(f"  Stability: {stability.stability_label} (score: {stability.stability_score})")
            print()
        
        print(f"\nCompleted {len(variations)} variations with stability analysis.")
        
        # Create output directory
        comparison_dir = Path("reports/tables") / f"{param_exp.base_name}_stability_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Save stability summary
        with open(comparison_dir / "stability_summary.txt", "w") as f:
            f.write(f"=== {param_exp.base_name.upper()} STABILITY ANALYSIS ===\n\n")
            
            for exp_name, stability in stabilities.items():
                f.write(f"{exp_name}:\n")
                f.write(f"  Stability Label: {stability.stability_label}\n")
                f.write(f"  Stability Score: {stability.stability_score}\n")
                f.write(f"  Direction Stable: {stability.direction_stable}\n")
                f.write(f"  Significance Stable: {stability.significance_stable}\n")
                f.write(f"  R² Stable: {stability.r2_stable}\n")
                f.write(f"  Decay Ratio: {stability.decay_ratio:.3f} ({stability.decay_label})\n")
                f.write(f"  Train Coef: {stability.train_coef:.4f}\n")
                f.write(f"  Test Coef: {stability.test_coef:.4f}\n\n")
        
        # Create ExperimentResult objects for family stability analysis
        family_train_test_pairs = {}
        for exp_name in variations:
            variations_list = generate_variations(param_exp)
            spec = next(s for s in variations_list if s.name == exp_name)
            
            train_exp_result = ExperimentResult(
                spec=spec,
                x_describe=train_results[exp_name]["x_describe"],
                y_describe=train_results[exp_name]["y_describe"],
                regression=train_results[exp_name]["regression"],
                model=train_results[exp_name]["model"],
                bucketed_analysis=train_results[exp_name].get("bucketed_analysis")
            )
            
            test_exp_result = ExperimentResult(
                spec=spec,
                x_describe=test_results[exp_name]["x_describe"],
                y_describe=test_results[exp_name]["y_describe"],
                regression=test_results[exp_name]["regression"],
                model=test_results[exp_name]["model"],
                bucketed_analysis=test_results[exp_name].get("bucketed_analysis")
            )
            
            family_train_test_pairs[exp_name] = (train_exp_result, test_exp_result)
        
        family_stability = family_stability_analyzer.analyze_family_stability(family_train_test_pairs)
        
        # Save detailed results as JSON
        stability_json = {
            "experiment_family": param_exp.base_name,
            "train_period": {"start": "2015-01-01", "end": "2020-12-31"},
            "test_period": {"start": "2021-01-01", "end": "today"},
            "stability_config": {
                "direction_weight": stability_config.direction_weight,
                "significance_weight": stability_config.significance_weight,
                "r2_weight": stability_config.r2_weight,
                "decay_weight": stability_config.decay_weight,
                "significance_threshold_train": stability_config.significance_threshold_train,
                "significance_threshold_test": stability_config.significance_threshold_test,
                "r2_survival_threshold": stability_config.r2_survival_threshold,
                "decay_stable_threshold": stability_config.decay_stable_threshold,
                "decay_weak_threshold": stability_config.decay_weak_threshold
            },
            "experiments": {},
            "family_stability": family_stability.to_dict()
        }
        
        for exp_name in variations:
            stability_json["experiments"][exp_name] = {
                "train_results": {
                    "coef": train_results[exp_name]["regression"].coef,
                    "p_value": train_results[exp_name]["regression"].p_value,
                    "r_squared": train_results[exp_name]["regression"].r_squared,
                    "n_obs": train_results[exp_name]["regression"].n_obs
                },
                "test_results": {
                    "coef": test_results[exp_name]["regression"].coef,
                    "p_value": test_results[exp_name]["regression"].p_value,
                    "r_squared": test_results[exp_name]["regression"].r_squared,
                    "n_obs": test_results[exp_name]["regression"].n_obs
                },
                "stability": stabilities[exp_name].to_dict()
            }
        
        with open(comparison_dir / "stability_analysis.json", "w") as f:
            json.dump(stability_json, f, indent=2, default=str)
        
        # Copy stability analysis to reports/ directory
        copy_json_to_reports(comparison_dir / "stability_analysis.json", f"{param_exp.base_name}_stability_analysis.json")
        
        print(f"\nStability analysis saved to: {comparison_dir}")
        
        # Store results for this family
        all_families_results[param_exp.base_name] = {
            "param_exp": param_exp,
            "train_results": train_results,
            "test_results": test_results,
            "stabilities": stabilities,
            "variations": variations,
            "family_stability": family_stability
        }
    
    # Generate master stability summary
    master_stability_dir = Path("reports/tables") / "master_stability_analysis"
    master_stability_dir.mkdir(parents=True, exist_ok=True)
    
    # Create master stability summary JSON
    master_stability_json = {
        "analysis_type": "all_families_stability",
        "timestamp": pd.Timestamp.now().isoformat(),
        "families": {}
    }
    
    for family_name, family_data in all_families_results.items():
        family_stability = family_data["family_stability"]
        master_stability_json["families"][family_name] = {
            "variations": len(family_data['variations']),
            "avg_stability_score": family_stability.avg_stability_score,
            "best_stability_score": family_stability.best_stability_score,
            "worst_stability_score": family_stability.worst_stability_score,
            "high_stability_count": family_stability.high_stability_count,
            "moderate_stability_count": family_stability.moderate_stability_count,
            "low_stability_count": family_stability.low_stability_count,
            "direction_consistent": family_stability.direction_consistent,
            "significance_survival_rate": family_stability.significance_survival_rate
        }
    
    # Save master stability JSON
    master_json_path = master_stability_dir / "all_families_stability.json"
    with open(master_json_path, "w") as f:
        json.dump(master_stability_json, f, indent=2, default=str)
    
    # Copy master stability JSON to reports/ directory
    copy_json_to_reports(master_json_path, "all_families_stability.json")
    
    with open(master_stability_dir / "all_families_summary.txt", "w") as f:
        f.write("=== ALL FAMILIES STABILITY ANALYSIS ===\n\n")
        
        for family_name, family_data in all_families_results.items():
            f.write(f"\n{family_name.upper()}:\n")
            family_stability = family_data["family_stability"]
            
            f.write(f"  Variations: {len(family_data['variations'])}\n")
            f.write(f"  Average Stability Score: {family_stability.avg_stability_score:.1f}\n")
            f.write(f"  Best Stability Score: {family_stability.best_stability_score}\n")
            f.write(f"  Worst Stability Score: {family_stability.worst_stability_score}\n")
            f.write(f"  High Stability Count: {family_stability.high_stability_count}\n")
            f.write(f"  Moderate Stability Count: {family_stability.moderate_stability_count}\n")
            f.write(f"  Low Stability Count: {family_stability.low_stability_count}\n")
            f.write(f"  Direction Consistent: {family_stability.direction_consistent}\n")
            f.write(f"  Significance Survival Rate: {family_stability.significance_survival_rate:.2f}\n")
        
        f.write(f"\nMaster stability analysis saved to: {master_stability_dir}")
    
    # Print comprehensive console summary for all families
    print(f"\n{'='*80}")
    print(f"📊 ALL FAMILIES STABILITY ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    for family_name, family_data in all_families_results.items():
        family_stability = family_data["family_stability"]
        
        print(f"\n🎯 {family_name.upper()} STABILITY:")
        print("-" * 50)
        print(f"  Variations: {len(family_data['variations'])}")
        print(f"  Average Stability Score: {family_stability.avg_stability_score:.1f}")
        print(f"  Best Stability Score: {family_stability.best_stability_score}")
        print(f"  Worst Stability Score: {family_stability.worst_stability_score}")
        print(f"  High Stability Count: {family_stability.high_stability_count}")
        print(f"  Moderate Stability Count: {family_stability.moderate_stability_count}")
        print(f"  Low Stability Count: {family_stability.low_stability_count}")
        print(f"  Direction Consistent: {family_stability.direction_consistent}")
        print(f"  Significance Survival Rate: {family_stability.significance_survival_rate:.2f}")
        
        # Individual experiment stability table
        print(f"\n📈 INDIVIDUAL EXPERIMENT STABILITY:")
        print(f"{'Experiment':<25} {'Train Coef':<12} {'Test Coef':<11} {'Direction':<10} {'Significance':<12} {'Score':<8} {'Label':<8}")
        print("-" * 80)
        
        for exp_name, stability in family_data["stabilities"].items():
            direction = "✓ Consistent" if stability.direction_stable else "✗ Inconsistent"
            significance = "✓ Survives" if stability.significance_stable else "✗ Collapses"
            print(f"{exp_name:<25} {stability.train_coef:+12.4f} {stability.test_coef:+11.4f} {direction:<10} {significance:<12} {stability.stability_score:<8} {stability.stability_label:<8}")
    
    # Overall ranking table
    print(f"\n🏆 OVERALL FAMILY RANKING:")
    print(f"{'Family':<20} {'Avg Score':<12} {'Best Score':<12} {'High Count':<12} {'Direction':<10} {'Overall':<10}")
    print("-" * 80)
    
    # Sort families by average stability score
    sorted_families = sorted(
        all_families_results.items(),
        key=lambda x: x[1]["family_stability"].avg_stability_score,
        reverse=True
    )
    
    rank = 1
    for family_name, family_data in sorted_families:
        family_stability = family_data["family_stability"]
        overall = "🏆 EXCELLENT" if family_stability.avg_stability_score >= 80 else \
                  "✅ GOOD" if family_stability.avg_stability_score >= 60 else \
                  "⚠️  MODERATE" if family_stability.avg_stability_score >= 40 else \
                  "❌ POOR"
        
        print(f"{rank:2}. {family_name:<20} {family_stability.avg_stability_score:<12.1f} {family_stability.best_stability_score:<12} {family_stability.high_stability_count:<12} {family_stability.direction_consistent:<10} {overall:<10}")
        rank += 1
    
    print(f"\n{'='*60}")
    print("All 4 experiment families completed with stability analysis!")
    print("Results saved to reports/tables/*/stability_comparison/")
    print("Master summary saved to reports/tables/master_stability_analysis/")


if __name__ == "__main__":
    main()
