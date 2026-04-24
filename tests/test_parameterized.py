from pathlib import Path
import pandas as pd

from mini_research_lab import (
    download_prices,
    add_return_features,
    MiniResearchLab,
    parameterized_experiments,
    plot_experiment_bundle,
    generate_variations,
)

# Download and prepare data
prices = download_prices("AAPL", start="2018-01-01")
df = add_return_features(prices)

lab = MiniResearchLab(df)

# Get parameterized experiments
param_exps = parameterized_experiments()

print("Available parameterized experiments:")
for i, param_exp in enumerate(param_exps):
    print(f"{i+1}. {param_exp.base_name}: {len(param_exp.lookbacks)} variations")
    print(f"   Lookbacks: {param_exp.lookbacks}")
    print(f"   Pattern: {param_exp.x_col_pattern} -> {param_exp.y_col}")
    print()

# Run the first parameterized experiment (mean reversion with multiple lookbacks)
param_exp = param_exps[0]  # mean_reversion
print(f"Running parameterized experiment: {param_exp.base_name}")
print(f"Will test lookbacks: {param_exp.lookbacks}")
print()

# Run all variations
results = lab.run_parameterized_experiment(param_exp)

print(f"Completed {len(results)} variations:")
for exp_name in results.keys():
    print(f"  - {exp_name}")

# Save results for each variation
for exp_name, result in results.items():
    # Create folder for this variation
    tables_dir = Path("reports/tables") / exp_name
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
            
            f.write("Bucket | X Range           | X Mean   | Y Mean   | Y Std    | Count | Y Mean ± 1 Std\n")
            f.write("-" * 85 + "\n")
            
            for _, row in bucketed.iterrows():
                f.write(f"{row['bucket']:6} | {row['x_range']:<16} | {row['x_mean']:8.4f} | {row['y_mean']:8.4f} | {row['y_std']:8.4f} | {row['count']:5} | [{row['y_mean_minus_1std']:7.4f}, {row['y_mean_plus_1std']:7.4f}]\n")
            
            f.write("\n" + "=" * 85 + "\n")
            f.write("INTERPRETATION:\n")
            f.write("-" * 85 + "\n")
            
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
            
            # Find extreme buckets
            max_y_bucket = bucketed.loc[bucketed['y_mean'].idxmax()]
            min_y_bucket = bucketed.loc[bucketed['y_mean'].idxmin()]
            
            f.write(f"Highest Y: Bucket {max_y_bucket['bucket']} (X range: {max_y_bucket['x_range']}) → Y mean: {max_y_bucket['y_mean']:.4f}\n")
            f.write(f"Lowest Y: Bucket {min_y_bucket['bucket']} (X range: {min_y_bucket['x_range']}) → Y mean: {min_y_bucket['y_mean']:.4f}\n")

    # Generate plots for this variation
    variations = generate_variations(param_exp)
    spec = next(s for s in variations if s.name == exp_name)
    figures_dir = Path("reports/figures") / exp_name
    saved_files = plot_experiment_bundle(
        df, spec.x_col, spec.y_col, 
        title_prefix="",
        output_dir=figures_dir
    )

print(f"\nAll results saved to reports/tables/ and reports/figures/")

# Create comparison summary file
comparison_dir = Path("reports/tables") / f"{param_exp.base_name}_comparison"
comparison_dir.mkdir(parents=True, exist_ok=True)

# Save detailed comparison summary
with open(comparison_dir / "comparison_summary.txt", "w") as f:
    f.write(f"=== {param_exp.base_name.upper()} COMPARISON SUMMARY ===\n")
    f.write(f"Experiment: {param_exp.description_template.format(lookback='X')}\n")
    f.write(f"Lookbacks tested: {param_exp.lookbacks}\n")
    f.write(f"Successful variations: {len(results)}\n\n")
    
    f.write("DETAILED RESULTS:\n")
    f.write("-" * 80 + "\n")
    f.write(f"{'Experiment':<25} | {'Coefficient':<11} | {'P-value':<7} | {'R-squared':<9} | {'Obs':<6}\n")
    f.write("-" * 80 + "\n")
    
    for exp_name, result in results.items():
        reg = result["regression"].to_dict()
        f.write(f"{exp_name:<25} | {reg['coef']:11.4f} | {reg['p_value']:7.4f} | {reg['r_squared']:9.4f} | {reg['n_obs']:6}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("SIGNIFICANCE ANALYSIS:\n")
    f.write("-" * 80 + "\n")
    
    significant_results = []
    for exp_name, result in results.items():
        reg = result["regression"].to_dict()
        if reg['p_value'] < 0.05:
            lookback = exp_name.split("_")[-2].replace("d", "d")
            significant_results.append((exp_name, lookback, reg['coef'], reg['p_value'], reg['r_squared']))
    
    if significant_results:
        f.write("STATISTICALLY SIGNIFICANT RESULTS (p < 0.05):\n")
        for exp_name, lookback, coef, p_val, r2 in significant_results:
            direction = "NEGATIVE (mean reversion)" if coef < 0 else "POSITIVE (momentum)"
            f.write(f"  - {exp_name}: {direction}, p={p_val:.4f}, R²={r2:.4f}\n")
    else:
        f.write("No statistically significant results found (p < 0.05)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("EFFECT SIZE ANALYSIS:\n")
    f.write("-" * 80 + "\n")
    
    # Sort by absolute coefficient for effect size
    sorted_results = sorted(results.items(), 
                         key=lambda x: abs(x[1]["regression"].to_dict()['coef']), 
                         reverse=True)
    
    f.write("RESULTS BY EFFECT SIZE (|coefficient|):\n")
    for i, (exp_name, result) in enumerate(sorted_results, 1):
        reg = result["regression"].to_dict()
        lookback = exp_name.split("_")[-2].replace("d", "d")
        f.write(f"  {i}. {exp_name} ({lookback}): {reg['coef']:+.4f} (p={reg['p_value']:.4f})\n")

# Save master summary file listing all experiments
master_summary_path = Path("reports/tables") / "master_experiment_summary.txt"
with open(master_summary_path, "a") as f:  # Append mode
    f.write(f"\n{'='*60}\n")
    f.write(f"EXPERIMENT SUITE: {param_exp.base_name.upper()}\n")
    f.write(f"Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Variations run: {len(results)}\n")
    f.write(f"Lookbacks: {param_exp.lookbacks}\n")
    
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
    lookback = exp_name.split("_")[-3]  # Gets "1d", "3d", "5d", etc. (index -3, not -2)
    print(f"{lookback:8} | {reg['coef']:11.4f} | {reg['p_value']:7.4f} | {reg['r_squared']:9.4f} | {reg['n_obs']:11}")

print(f"\nComparison summary saved to: {comparison_dir / 'comparison_summary.txt'}")
print(f"Master summary updated: {master_summary_path}")
