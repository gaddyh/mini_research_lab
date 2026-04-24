"""
Parameterized experiment runner with stability analysis across train/test periods.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Add src directory to Python path for imports
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from mini_research_lab.lab import MiniResearchLab
from mini_research_lab.data_loader import download_prices
from mini_research_lab.features import add_return_features
from mini_research_lab.experiment_specs import parameterized_experiments


def split_data_by_time(df: pd.DataFrame):
    """Split data into train (2015-2020) and test (2021-today) periods."""
    train_start = "2015-01-01"
    train_end = "2020-12-31"
    test_start = "2021-01-01"
    
    train_df = df.loc[train_start:train_end].copy()
    test_df = df.loc[test_start:].copy()
    
    return train_df, test_df


def analyze_stability(train_result: dict, test_result: dict):
    """Analyze stability of signal across train and test periods."""
    
    train_reg = train_result["regression"].to_dict() if hasattr(train_result["regression"], 'to_dict') else train_result["regression"]
    test_reg = test_result["regression"].to_dict() if hasattr(test_result["regression"], 'to_dict') else test_result["regression"]
    
    train_coef = train_reg["coef"]
    train_p = train_reg["p_value"]
    train_r2 = train_reg["r_squared"]
    
    test_coef = test_reg["coef"]
    test_p = test_reg["p_value"]
    test_r2 = test_reg["r_squared"]
    
    # Direction consistency (most important)
    direction_stable = (train_coef * test_coef) > 0  # Same sign
    
    # Significance survival
    significance_stable = (train_p < 0.05) and (test_p < 0.10)
    
    # Strength decay
    decay_ratio = abs(test_coef) / abs(train_coef) if train_coef != 0 else 0
    
    if decay_ratio > 0.7:
        decay_score = 10
        decay_label = "stable"
    elif decay_ratio > 0.3:
        decay_score = 5
        decay_label = "weak_decay"
    else:
        decay_score = 0
        decay_label = "collapses"
    
    # R² survival
    r2_stable = test_r2 >= (0.5 * train_r2)
    
    # Stability score
    stability_score = (
        40 * (1 if direction_stable else 0) +
        30 * (1 if significance_stable else 0) +
        20 * (1 if r2_stable else 0) +
        decay_score
    )
    
    # Stability label
    if stability_score >= 70:
        stability_label = "high"
    elif stability_score >= 40:
        stability_label = "moderate"
    else:
        stability_label = "low"
    
    return {
        "train_coef": train_coef,
        "test_coef": test_coef,
        "direction_stable": direction_stable,
        "significance_stable": significance_stable,
        "r2_stable": r2_stable,
        "decay_ratio": decay_ratio,
        "decay_label": decay_label,
        "stability_score": stability_score,
        "stability_label": stability_label
    }


def run_experiment_with_stability(param_exp, train_df, test_df, exp_name):
    """Run single experiment on both train and test periods."""
    
    # Train period
    train_lab = MiniResearchLab(train_df)
    
    # Extract x_col and y_col from experiment name
    variations = []
    for lookback in param_exp.lookbacks:
        if param_exp.base_name == "ma_distance_reversion":
            x_col = param_exp.x_col_pattern.format(lookback=lookback)
        else:
            x_col = param_exp.x_col_pattern.format(lookback=lookback)
        
        y_col = param_exp.y_col
        exp_name = f"{param_exp.base_name}_{lookback}d_to_1d"
        variations.append((exp_name, x_col, y_col))
    
    # Find the matching variation
    exp_spec = None
    for var_name, var_x_col, var_y_col in variations:
        if var_name == exp_name:
            exp_spec = type('ExperimentSpec', (), {
                'name': var_name,
                'x_col': var_x_col, 
                'y_col': var_y_col
            })
            break
    
    if exp_spec is None:
        raise ValueError(f"Experiment specification not found for: {exp_name}")
    
    train_result = train_lab.run_experiment(exp_spec.x_col, exp_spec.y_col)
    
    # Test period
    test_lab = MiniResearchLab(test_df)
    test_result = test_lab.run_experiment(exp_spec.x_col, exp_spec.y_col)
    
    # Analyze stability
    stability = analyze_stability(train_result, test_result)
    
    return train_result, test_result, stability


def main():
    """Main function to run all 4 experiment families with stability analysis."""
    
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
            train_result, test_result, stability = run_experiment_with_stability(param_exp, train_df, test_df, exp_name)
            
            train_results[exp_name] = train_result
            test_results[exp_name] = test_result
            stabilities[exp_name] = stability
            
            print(f"  Train: coef={train_result['regression'].coef:.4f}, p={train_result['regression'].p_value:.4f}, R²={train_result['regression'].r_squared:.4f}")
            print(f"  Test:  coef={test_result['regression'].coef:.4f}, p={test_result['regression'].p_value:.4f}, R²={test_result['regression'].r_squared:.4f}")
            print(f"  Stability: {stability['stability_label']} (score: {stability['stability_score']})")
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
                f.write(f"  Stability Label: {stability['stability_label']}\n")
                f.write(f"  Stability Score: {stability['stability_score']}\n")
                f.write(f"  Direction Stable: {stability['direction_stable']}\n")
                f.write(f"  Significance Stable: {stability['significance_stable']}\n")
                f.write(f"  R² Stable: {stability['r2_stable']}\n")
                f.write(f"  Decay Ratio: {stability['decay_ratio']:.3f} ({stability['decay_label']})\n")
                f.write(f"  Train Coef: {stability['train_coef']:.4f}\n")
                f.write(f"  Test Coef: {stability['test_coef']:.4f}\n\n")
        
        # Save detailed results as JSON
        stability_json = {
            "experiment_family": param_exp.base_name,
            "train_period": {"start": "2015-01-01", "end": "2020-12-31"},
            "test_period": {"start": "2021-01-01", "end": "today"},
            "experiments": {}
        }
        
        for exp_name in variations:
            stability_json["experiments"][exp_name] = {
                "train_results": train_results[exp_name],
                "test_results": test_results[exp_name],
                "stability": stabilities[exp_name]
            }
        
        with open(comparison_dir / "stability_analysis.json", "w") as f:
            json.dump(stability_json, f, indent=2, default=str)
        
        print(f"\nStability analysis saved to: {comparison_dir}")
        
        # Store results for this family
        all_families_results[param_exp.base_name] = {
            "param_exp": param_exp,
            "train_results": train_results,
            "test_results": test_results,
            "stabilities": stabilities,
            "variations": variations
        }
    
    # Generate master stability summary
    master_stability_dir = Path("reports/tables") / "master_stability_analysis"
    master_stability_dir.mkdir(parents=True, exist_ok=True)
    
    with open(master_stability_dir / "all_families_summary.txt", "w") as f:
        f.write("=== ALL FAMILIES STABILITY ANALYSIS ===\n\n")
        
        for family_name, family_data in all_families_results.items():
            f.write(f"\n{family_name.upper()}:\n")
            
            # Calculate family-level stability metrics
            stability_scores = [stability['stability_score'] for stability in family_data['stabilities'].values()]
            avg_stability = sum(stability_scores) / len(stability_scores) if stability_scores else 0
            
            direction_stable_count = sum(1 for stability in family_data['stabilities'].values() if stability['direction_stable'])
            significance_stable_count = sum(1 for stability in family_data['stabilities'].values() if stability['significance_stable'])
            
            f.write(f"  Variations: {len(family_data['variations'])}\n")
            f.write(f"  Average Stability Score: {avg_stability:.1f}\n")
            f.write(f"  Direction Stable: {direction_stable_count}/{len(family_data['variations'])}\n")
            f.write(f"  Significance Stable: {significance_stable_count}/{len(family_data['variations'])}\n")
            
            # Best and worst stability
            if stability_scores:
                best_stability = max(stability_scores)
                worst_stability = min(stability_scores)
                f.write(f"  Best Stability Score: {best_stability} (Low collapse risk)\n")
                f.write(f"  Worst Stability Score: {worst_stability} (High collapse risk)\n")
        
        f.write(f"\nMaster stability analysis saved to: {master_stability_dir}")
    
    print(f"\n{'='*60}")
    print("All 4 experiment families completed with stability analysis!")
    print("Results saved to reports/tables/*/stability_comparison/")
    print("Master summary saved to reports/tables/master_stability_analysis/")
    
    # Run all variations
    train_results = {}
    test_results = {}
    stabilities = {}
    
    variations = []
    for lookback in param_exp.lookbacks:
        exp_name = f"{param_exp.base_name}_{lookback}d_to_1d"
        variations.append(exp_name)
    
    for exp_name in variations:
        print(f"Running {exp_name}...")
        train_result, test_result, stability = run_experiment_with_stability(param_exp, train_df, test_df, exp_name)
        
        train_results[exp_name] = train_result
        test_results[exp_name] = test_result
        stabilities[exp_name] = stability
        
        print(f"  Train: coef={train_result['regression'].coef:.4f}, p={train_result['regression'].p_value:.4f}, R²={train_result['regression'].r_squared:.4f}")
        print(f"  Test:  coef={test_result['regression'].coef:.4f}, p={test_result['regression'].p_value:.4f}, R²={test_result['regression'].r_squared:.4f}")
        print(f"  Stability: {stability['stability_label']} (score: {stability['stability_score']})")
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
            f.write(f"  Stability Label: {stability['stability_label']}\n")
            f.write(f"  Stability Score: {stability['stability_score']}\n")
            f.write(f"  Direction Stable: {stability['direction_stable']}\n")
            f.write(f"  Significance Stable: {stability['significance_stable']}\n")
            f.write(f"  R² Stable: {stability['r2_stable']}\n")
            f.write(f"  Decay Ratio: {stability['decay_ratio']:.3f} ({stability['decay_label']})\n")
            f.write(f"  Train Coef: {stability['train_coef']:.4f}\n")
            f.write(f"  Test Coef: {stability['test_coef']:.4f}\n\n")
    
    # Save detailed results as JSON
    stability_json = {
        "experiment_family": param_exp.base_name,
        "train_period": {"start": "2015-01-01", "end": "2020-12-31"},
        "test_period": {"start": "2021-01-01", "end": "today"},
        "experiments": {}
    }
    
    for exp_name in variations:
        stability_json["experiments"][exp_name] = {
            "train_results": train_results[exp_name],
            "test_results": test_results[exp_name],
            "stability": stabilities[exp_name]
        }
    
    with open(comparison_dir / "stability_analysis.json", "w") as f:
        json.dump(stability_json, f, indent=2, default=str)
    
    print(f"\nStability analysis saved to: {comparison_dir}")


if __name__ == "__main__":
    main()
