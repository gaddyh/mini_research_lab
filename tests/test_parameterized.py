from pathlib import Path
import pandas as pd
import json
import re

from mini_research_lab import (
    download_prices,
    add_return_features,
    MiniResearchLab,
    parameterized_experiments,
    plot_experiment_bundle,
    generate_variations,
)

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(v) for v in obj)
    else:
        return obj

def create_experiment_json(exp_name, param_exp, result, tables_dir, figures_dir):
    """Create comprehensive JSON output for experiment."""
    
    # Extract parameters from experiment name
    match = re.match(f"{param_exp.base_name}_(\\d+)d_to_(\\d+)d", exp_name)
    if match:
        lookback_days = int(match.group(1))
        forward_days = int(match.group(2))
    else:
        lookback_days = None
        forward_days = None
    
    # Get variation spec
    variations = generate_variations(param_exp)
    spec = next(s for s in variations if s.name == exp_name)
    
    # Extract regression data
    reg_dict = result["regression"].to_dict()
    reg_dict = {k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v 
                for k, v in reg_dict.items()}
    
    # Extract distribution data
    x_dict = result["x_describe"].to_dict()
    x_dict = {k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v 
              for k, v in x_dict.items()}
    y_dict = result["y_describe"].to_dict()
    y_dict = {k: float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v 
              for k, v in y_dict.items()}
    
    # Extract bucketed data
    bucketed_data = []
    pattern_label = "unknown"
    highest_y_bucket = None
    lowest_y_bucket = None
    
    if 'bucketed_analysis' in result:
        bucketed_df = result['bucketed_analysis']
        pattern_label = "non_monotonic"  # Default, will be updated
        
        # Check for monotonic pattern
        y_means = bucketed_df['y_mean'].values
        if len(y_means) >= 2:
            increasing = all(y_means[i] <= y_means[i+1] for i in range(len(y_means)-1))
            decreasing = all(y_means[i] >= y_means[i+1] for i in range(len(y_means)-1))
            
            if increasing:
                pattern_label = "monotonic_increasing"
            elif decreasing:
                pattern_label = "monotonic_decreasing"
        
        highest_y_bucket = bucketed_df.loc[bucketed_df['y_mean'].idxmax(), 'bucket']
        lowest_y_bucket = bucketed_df.loc[bucketed_df['y_mean'].idxmin(), 'bucket']
        
        for _, row in bucketed_df.iterrows():
            x_range_parts = row['x_range'].split(' to ')
            bucketed_data.append({
                "bucket": int(row['bucket']),
                "x_min": float(x_range_parts[0]),
                "x_max": float(x_range_parts[1]),
                "x_mean": float(row['x_mean']),
                "y_mean": float(row['y_mean']),
                "y_std": float(row['y_std']),
                "count": int(row['count'])
            })
    
    # Extract diagnostics from model summary
    diagnostics = {}
    if hasattr(result["model"], 'summary'):
        summary_str = str(result["model"].summary())
        # Extract Durbin-Watson
        dw_match = re.search(r'Durbin-Watson:\s+(\d+\.\d+)', summary_str)
        if dw_match:
            diagnostics["durbin_watson"] = float(dw_match.group(1))
        
        # Extract Kurtosis
        kurt_match = re.search(r'Kurtosis:\s+(\d+\.\d+)', summary_str)
        if kurt_match:
            diagnostics["kurtosis"] = float(kurt_match.group(1))
        
        # Extract Skew
        skew_match = re.search(r'Skew:\s+(-?\d+\.\d+)', summary_str)
        if skew_match:
            diagnostics["skew"] = float(skew_match.group(1))
    
    diagnostics["fat_tail_flag"] = diagnostics.get("kurtosis", 0) > 3
    diagnostics["autocorrelation_flag"] = abs(diagnostics.get("durbin_watson", 2) - 2) > 0.5
    
    # Determine hypothesis
    if "mean_reversion" in param_exp.base_name:
        expected_pattern = "mean_reversion"
        expected_coef_sign = "negative"
        description = f"Recent {lookback_days}-day return should negatively predict next {forward_days}-day return."
    elif "momentum" in param_exp.base_name:
        expected_pattern = "momentum"
        expected_coef_sign = "positive"
        description = f"Recent {lookback_days}-day return should positively predict next {forward_days}-day return."
    elif "volatility" in param_exp.base_name:
        expected_pattern = "volatility_clustering"
        expected_coef_sign = "positive"
        description = f"Recent {lookback_days}-day volatility should positively predict next {forward_days}-day volatility."
    else:
        expected_pattern = "unknown"
        expected_coef_sign = "unknown"
        description = spec.description
    
    # Create artifacts paths
    artifacts = {
        "regression_file": str(tables_dir / "regression.txt"),
        "model_summary_file": str(tables_dir / "model_summary.txt"),
        "x_summary_file": str(tables_dir / "x_summary.txt"),
        "y_summary_file": str(tables_dir / "y_summary.txt"),
        "bucketed_analysis_file": str(tables_dir / "bucketed_analysis.txt"),
        "scatter_plot": str(figures_dir / f"{spec.x_col}_vs_{spec.y_col}_scatter.png"),
        "x_histogram": str(figures_dir / f"{spec.x_col}_histogram.png"),
        "y_histogram": str(figures_dir / f"{spec.y_col}_histogram.png")
    }
    
    # Build JSON structure
    experiment_json = {
        "experiment_id": exp_name,
        "family": param_exp.base_name,
        "asset": "AAPL",
        "x_col": spec.x_col,
        "y_col": spec.y_col,
        
        "parameters": {
            "lookback_days": lookback_days,
            "forward_days": forward_days,
            "threshold": None
        },
        
        "hypothesis": {
            "expected_pattern": expected_pattern,
            "expected_coef_sign": expected_coef_sign,
            "description": description
        },
        
        "regression": {
            "intercept": reg_dict.get("intercept"),
            "coef": reg_dict.get("coef"),
            "std_err": reg_dict.get("std_err"),
            "t_value": reg_dict.get("t_value"),
            "p_value": reg_dict.get("p_value"),
            "r_squared": reg_dict.get("r_squared"),
            "n_obs": reg_dict.get("n_obs"),
            "ci_low": reg_dict.get("ci_low"),
            "ci_high": reg_dict.get("ci_high")
        },
        
        "x_distribution": {
            "count": x_dict.get("count"),
            "mean": x_dict.get("mean"),
            "std": x_dict.get("std"),
            "min": x_dict.get("min"),
            "p25": x_dict.get("25%"),
            "p50": x_dict.get("50%"),
            "p75": x_dict.get("75%"),
            "max": x_dict.get("max"),
            "iqr": x_dict.get("iqr"),
            "lower_fence": x_dict.get("lower_fence"),
            "upper_fence": x_dict.get("upper_fence"),
            "n_lower_outliers": x_dict.get("n_lower_outliers"),
            "n_upper_outliers": x_dict.get("n_upper_outliers"),
            "skew_hint": x_dict.get("skew_hint")
        },
        
        "y_distribution": {
            "count": y_dict.get("count"),
            "mean": y_dict.get("mean"),
            "std": y_dict.get("std"),
            "min": y_dict.get("min"),
            "p25": y_dict.get("25%"),
            "p50": y_dict.get("50%"),
            "p75": y_dict.get("75%"),
            "max": y_dict.get("max"),
            "iqr": y_dict.get("iqr"),
            "lower_fence": y_dict.get("lower_fence"),
            "upper_fence": y_dict.get("upper_fence"),
            "n_lower_outliers": y_dict.get("n_lower_outliers"),
            "n_upper_outliers": y_dict.get("n_upper_outliers"),
            "skew_hint": y_dict.get("skew_hint")
        },
        
        "bucketed_relationship": {
            "n_buckets": len(bucketed_data),
            "pattern_label": pattern_label,
            "highest_y_bucket": highest_y_bucket,
            "lowest_y_bucket": lowest_y_bucket,
            "buckets": bucketed_data
        },
        
        "diagnostics": diagnostics,
        
        "artifacts": artifacts
    }
    
    # Convert all types to JSON-serializable
    return convert_to_json_serializable(experiment_json)

def calculate_experiment_score(exp_name, result, param_exp, all_coefs, all_r2s):
    """Calculate deterministic score for an experiment."""
    
    reg_dict = result["regression"].to_dict()
    coef = reg_dict.get("coef", 0)
    p_val = reg_dict.get("p_value", 1.0)
    r2 = reg_dict.get("r_squared", 0.0)
    n_obs = reg_dict.get("n_obs", 0)
    
    # 1. Significance Score (0-30 points)
    if p_val < 0.01:
        significance_score = 30
    elif p_val < 0.05:
        significance_score = 20
    elif p_val < 0.10:
        significance_score = 10
    else:
        significance_score = 0
    
    # 2. Effect Direction Score (0-20 points) - CONDITIONAL ON SIGNIFICANCE
    if p_val < 0.1:  # Only score direction if there's some significance
        if "mean_reversion" in param_exp.base_name:
            # Expected negative coefficient
            if coef < 0:
                effect_direction_score = 20
            else:
                effect_direction_score = -10  # Penalty for wrong direction
        elif "momentum" in param_exp.base_name:
            # Expected positive coefficient
            if coef > 0:
                effect_direction_score = 20
            else:
                effect_direction_score = -10  # Penalty for wrong direction
        elif "volatility" in param_exp.base_name:
            # Expected positive coefficient
            if coef > 0:
                effect_direction_score = 20
            else:
                effect_direction_score = -10
        else:
            effect_direction_score = 0
    else:
        # No significance = no direction points (don't reward noise)
        effect_direction_score = 0
    
    # 3. Effect Size Score (0-20 points, normalized within family)
    abs_coefs = [abs(c) for c in all_coefs]
    if max(abs_coefs) > 0:
        normalized_coef = abs(coef) / max(abs_coefs)
        effect_size_score = int(normalized_coef * 20)
    else:
        effect_size_score = 0
    
    # 4. R-squared Score (0-15 points)
    if r2 > 0.005:
        r_squared_score = 15
    elif r2 > 0.002:
        r_squared_score = 10
    elif r2 > 0.001:
        r_squared_score = 5
    else:
        r_squared_score = 0
    
    # 5. Bucket Shape Score (0-20 points)
    bucket_shape_score = 0
    if 'bucketed_analysis' in result:
        bucketed_df = result['bucketed_analysis']
        y_means = bucketed_df['y_mean'].values
        
        if len(y_means) >= 2:
            # Check for expected pattern
            if "mean_reversion" in param_exp.base_name:
                # Expected: Y decreases as X increases
                decreasing = all(y_means[i] >= y_means[i+1] for i in range(len(y_means)-1))
                if decreasing:
                    bucket_shape_score = 20
                else:
                    bucket_shape_score = 5  # Some points for having data
            elif "momentum" in param_exp.base_name:
                # Expected: Y increases as X increases
                increasing = all(y_means[i] <= y_means[i+1] for i in range(len(y_means)-1))
                if increasing:
                    bucket_shape_score = 20
                else:
                    bucket_shape_score = 5
            elif "volatility" in param_exp.base_name:
                # Expected: high X → high Y (positive relationship)
                increasing = all(y_means[i] <= y_means[i+1] for i in range(len(y_means)-1))
                if increasing:
                    bucket_shape_score = 20
                else:
                    bucket_shape_score = 5
    else:
        bucket_shape_score = 0
    
    # 6. Diagnostics Penalty (0 to -15 points)
    diagnostics_penalty = 0
    
    # Extract diagnostics from model summary
    if hasattr(result["model"], 'summary'):
        summary_str = str(result["model"].summary())
        
        # Check for fat tails (kurtosis > 3)
        kurt_match = re.search(r'Kurtosis:\s+(\d+\.\d+)', summary_str)
        if kurt_match and float(kurt_match.group(1)) > 3:
            diagnostics_penalty -= 2  # Small penalty, fat tails are normal
        
        # Check for autocorrelation (Durbin-Watson far from 2)
        dw_match = re.search(r'Durbin-Watson:\s+(\d+\.\d+)', summary_str)
        if dw_match and abs(float(dw_match.group(1)) - 2) > 0.5:
            diagnostics_penalty -= 5
        
        # Check sample size
        if n_obs < 1000:
            diagnostics_penalty -= 5
        elif n_obs < 500:
            diagnostics_penalty -= 10
    
    # Calculate total score
    total_score = (significance_score + effect_direction_score + 
                   effect_size_score + r_squared_score + 
                   bucket_shape_score + diagnostics_penalty)
    
    # Determine score label with R² constraint
    if r2 < 0.01:
        # Low R² means weak signal regardless of total score
        if total_score >= 50 and p_val < 0.05:
            score_label = "statistically_significant_but_weak"
        elif total_score >= 30 and p_val < 0.1:
            score_label = "minimal_evidence_weak"
        else:
            score_label = "not_promising"
    else:
        # Higher R² allows stronger labels
        if total_score >= 70:
            score_label = "strong_candidate"
        elif total_score >= 50:
            score_label = "interesting_but_weak"
        elif total_score >= 30:
            score_label = "minimal_evidence"
        elif total_score >= 10:
            score_label = "very_weak"
        else:
            score_label = "not_promising"
    
    return {
        "total": total_score,
        "significance": significance_score,
        "effect_direction": effect_direction_score,
        "effect_size": effect_size_score,
        "r_squared": r_squared_score,
        "bucket_shape": bucket_shape_score,
        "diagnostics_penalty": diagnostics_penalty
    }, score_label

def get_experiment_decision(exp_data, param_exp):
    """Get deterministic decision for a single experiment."""
    
    reg_summary = exp_data["regression"]
    # Handle both dict and RegressionSummary object
    if hasattr(reg_summary, 'to_dict'):
        reg_dict = reg_summary.to_dict()
    else:
        reg_dict = reg_summary
    
    coef = reg_dict.get("coef", 0)
    p_val = reg_dict.get("p_value", 1.0)
    r2 = reg_dict.get("r_squared", 0.0)
    
    reason_codes = []
    decision = "REFINE"  # default
    
    # Check significance
    if p_val < 0.05:
        reason_codes.append("significant")
    elif p_val < 0.10:
        reason_codes.append("borderline_significant")
    else:
        reason_codes.append("not_significant")
    
    # Check direction
    if "mean_reversion" in param_exp.base_name or "ma_distance_reversion" in param_exp.base_name:
        if coef < 0:
            reason_codes.append("correct_direction")
        else:
            reason_codes.append("wrong_direction")
    elif "momentum" in param_exp.base_name:
        if coef > 0:
            reason_codes.append("correct_direction")
        else:
            reason_codes.append("wrong_direction")
    elif "volatility" in param_exp.base_name:
        if coef > 0:
            reason_codes.append("correct_direction")
        else:
            reason_codes.append("wrong_direction")
    
    # Check R²
    if r2 > 0.005:
        reason_codes.append("decent_r_squared")
    elif r2 > 0.001:
        reason_codes.append("low_r_squared")
    else:
        reason_codes.append("tiny_r_squared")
    
    # Check bucket shape if available
    bucket_support = False
    if 'bucketed_analysis' in exp_data:
        bucketed_df = exp_data['bucketed_analysis']
        y_means = bucketed_df['y_mean'].values
        
        if len(y_means) >= 2:
            if "mean_reversion" in param_exp.base_name or "ma_distance_reversion" in param_exp.base_name:
                decreasing = all(y_means[i] >= y_means[i+1] for i in range(len(y_means)-1))
                if decreasing:
                    reason_codes.append("bucket_supports_mean_reversion")
                    bucket_support = True
            elif "momentum" in param_exp.base_name:
                increasing = all(y_means[i] <= y_means[i+1] for i in range(len(y_means)-1))
                if increasing:
                    reason_codes.append("bucket_supports_momentum")
                    bucket_support = True
            elif "volatility" in param_exp.base_name:
                increasing = all(y_means[i] <= y_means[i+1] for i in range(len(y_means)-1))
                if increasing:
                    reason_codes.append("bucket_supports_volatility")
                    bucket_support = True
    
    # Apply decision rules
    if (p_val < 0.01 and 
        "correct_direction" in reason_codes and 
        r2 >= 0.01):
        # Strong statistical evidence doesn't need bucket support
        decision = "PROMOTE"
        reason_codes.append("strong_statistical_evidence")
    elif (p_val < 0.05 and 
          "correct_direction" in reason_codes and 
          r2 >= 0.01 and 
          bucket_support):
        # Moderate evidence with bucket support
        decision = "PROMOTE"
        reason_codes.append("moderate_evidence_with_bucket_support")
    elif (p_val > 0.10 or 
          "wrong_direction" in reason_codes or 
          r2 < 0.0005 or 
          ("not_significant" in reason_codes and not bucket_support)):
        decision = "DROP"
    else:
        decision = "REFINE"
    
    return decision, reason_codes

def get_family_decision(scored_experiments, param_exp):
    """Get deterministic family-level decision with hypothesis consistency check."""
    
    if not scored_experiments:
        return "DROP", [], 0.0, "No experiments"
    
    # Calculate family metrics
    total_count = len(scored_experiments)
    
    # Step 1: Count valid significant results (p < 0.05 AND correct direction)
    valid_significant_count = 0
    significant_count = 0
    wrong_direction_significant = 0
    
    for exp in scored_experiments:
        coef = exp["coef"]
        p_val = exp["p_value"]
        
        if p_val < 0.05:
            significant_count += 1
            
            # Check if direction matches hypothesis
            direction_matches = False
            if ("mean_reversion" in param_exp.base_name or "ma_distance_reversion" in param_exp.base_name) and coef < 0:
                direction_matches = True
            elif "momentum" in param_exp.base_name and coef > 0:
                direction_matches = True
            elif "volatility" in param_exp.base_name and coef > 0:
                direction_matches = True
            
            if direction_matches:
                valid_significant_count += 1
            else:
                wrong_direction_significant += 1
    
    # Step 2: Check for hypothesis contradiction
    if wrong_direction_significant > 0:
        decision = "DROP"
        confidence = 0.2
        reason_codes = ["contradicts_hypothesis", "significant_wrong_direction"]
        reason = "Significant results contradict the hypothesis, indicating the opposite effect."
        return decision, reason_codes, confidence, reason
    
    # Step 3: Apply decision rules based on valid significant results
    stability_score = valid_significant_count / total_count
    
    # Get top scores
    sorted_experiments = sorted(scored_experiments, key=lambda x: x["score"]["total"], reverse=True)
    top_score = sorted_experiments[0]["score"]["total"]
    
    # Check consistency of direction (for non-significant too)
    correct_direction_count = 0
    for exp in scored_experiments:
        coef = exp["coef"]
        if ("mean_reversion" in param_exp.base_name or "ma_distance_reversion" in param_exp.base_name) and coef < 0:
            correct_direction_count += 1
        elif "momentum" in param_exp.base_name and coef > 0:
            correct_direction_count += 1
        elif "volatility" in param_exp.base_name and coef > 0:
            correct_direction_count += 1
    
    direction_consistency = correct_direction_count / total_count
    
    # Check R² stability
    r2s = [exp["r_squared"] for exp in scored_experiments]
    max_r2 = max(r2s) if r2s else 0
    r2_decays = max_r2 > r2s[-1] * 2 if len(r2s) > 1 else False
    
    reason_codes = []
    confidence = 0.0
    
    # Build reason codes
    if valid_significant_count >= 1:
        reason_codes.append("valid_significant_variants")
    if significant_count >= 1:
        reason_codes.append("some_significant_variants")
    if valid_significant_count / total_count >= 0.5:
        reason_codes.append("good_stability")
    else:
        reason_codes.append("low_stability")
    
    if direction_consistency >= 0.8:
        reason_codes.append("consistent_direction")
    else:
        reason_codes.append("inconsistent_direction")
    
    if max_r2 > 0.005:
        reason_codes.append("decent_explanatory_power")
    elif max_r2 > 0.001:
        reason_codes.append("weak_explanatory_power")
    else:
        reason_codes.append("tiny_explanatory_power")
    
    if r2_decays:
        reason_codes.append("effect_decays_with_lookback")
    
    # Calculate average stability score across experiments
    stability_scores = []
    for exp in scored_experiments:
        if 'stability' in exp:
            stability_scores.append(exp['stability']['stability_score'])
    
    avg_stability_score = sum(stability_scores) / len(stability_scores) if stability_scores else 0
    
    # Apply corrected decision rules with stability requirements
    if (valid_significant_count >= 2 and stability_score >= 0.5 and 
        direction_consistency >= 0.8 and max_r2 > 0.001 and avg_stability_score >= 70):
        decision = "PROMOTE"
        confidence = 0.8
        reason_codes.append("meets_promote_criteria")
    elif (valid_significant_count >= 1 and wrong_direction_significant == 0 and avg_stability_score >= 40):
        decision = "REFINE"
        confidence = 0.6
        reason_codes.append("meets_refine_criteria")
    else:
        decision = "DROP"
        confidence = 0.3
        reason_codes.append("meets_drop_criteria")
    
    # Generate reason text
    if decision == "PROMOTE":
        reason = "Multiple variants show stable, significant evidence with consistent direction matching hypothesis."
    elif decision == "REFINE":
        reason = "Some variants show evidence matching hypothesis, but the signal is weak or unstable."
    else:
        reason = "No valid significant results matching hypothesis to justify continuation."
    
    return decision, reason_codes, confidence, reason

def select_experiments(scored_experiments):
    """Select best, worst, and representative experiments."""
    
    if not scored_experiments:
        return {"best": None, "worst": None, "representative": None}
    
    # Sort by total score
    sorted_experiments = sorted(scored_experiments, key=lambda x: x["score"]["total"], reverse=True)
    
    # Best: highest score
    best = sorted_experiments[0]
    
    # Worst: lowest score
    worst = sorted_experiments[-1]
    
    # Representative: median score
    mid_idx = len(sorted_experiments) // 2
    representative = sorted_experiments[mid_idx]
    
    return {
        "best": best,
        "worst": worst,
        "representative": representative
    }

def create_family_summary_json(param_exp, results, comparison_dir):
    """Create family-level summary JSON for all variations."""
    
    # Extract parameter grid
    lookback_days = param_exp.lookbacks
    forward_days = list(set([int(re.search(r'_to_(\d+)d', k).group(1)) for k in results.keys()]))
    
    # Collect all coefficients and R² for normalization
    all_coefs = []
    all_r2s = []
    for exp_name, result in results.items():
        reg_dict = result["regression"].to_dict()
        all_coefs.append(reg_dict.get("coef", 0))
        all_r2s.append(reg_dict.get("r_squared", 0.0))
    
    # Build experiments list with scores
    experiments = []
    scored_experiments = []
    for exp_name, result in results.items():
        reg_dict = result["regression"].to_dict()
        
        # Extract lookback/forward from experiment name
        match = re.match(f"{param_exp.base_name}_(\\d+)d_to_(\\d+)d", exp_name)
        if match:
            lb_days = int(match.group(1))
            fd_days = int(match.group(2))
        else:
            lb_days = None
            fd_days = None
        
        # Calculate score
        score, score_label = calculate_experiment_score(exp_name, result, param_exp, all_coefs, all_r2s)
        
        # Determine pattern label
        coef = reg_dict.get("coef", 0)
        p_val = reg_dict.get("p_value", 1.0)
        r2 = reg_dict.get("r_squared", 0.0)
        
        if p_val < 0.05:
            if abs(coef) > 0.05:
                pattern_label = "strong_effect"
            elif abs(coef) > 0.02:
                pattern_label = "moderate_effect"
            else:
                pattern_label = "weak_effect"
        else:
            pattern_label = "not_significant"
        
        # Add direction to pattern label
        if coef < 0:
            pattern_label += "_mean_reversion"
        elif coef > 0:
            pattern_label += "_momentum"
        else:
            pattern_label += "_neutral"
        
        # Determine best bucket if available
        best_bucket = "unknown"
        if 'bucketed_analysis' in result:
            bucketed_df = result['bucketed_analysis']
            max_idx = bucketed_df['y_mean'].idxmax()
            max_row = bucketed_df.loc[max_idx]
            bucket_range = max_row['x_range']
            if "neg" in bucket_range or "-" in bucket_range.split()[0]:
                best_bucket = "negative_returns"
            elif "pos" in bucket_range or float(bucket_range.split()[0]) > 0:
                best_bucket = "positive_returns"
            else:
                best_bucket = "small_returns"
        
        experiment_data = {
            "experiment_id": exp_name,
            "lookback_days": lb_days,
            "forward_days": fd_days,
            "coef": float(coef),
            "p_value": float(p_val),
            "r_squared": float(r2),
            "n_obs": int(reg_dict.get("n_obs", 0)),
            "pattern_label": pattern_label,
            "best_bucket": best_bucket,
            "notes": f"Effect size: {abs(coef):.4f}, Significance: {p_val:.4f}",
            "score": score,
            "score_label": score_label
        }
        
        experiments.append(experiment_data)
        scored_experiments.append(experiment_data)
    
    # Sort experiments by lookback days for analysis
    experiments.sort(key=lambda x: x.get("lookback_days", 0))
    
    # Add experiment-level decisions
    for exp in experiments:
        exp_result = results[exp["experiment_id"]]  # Use results dict, not result
        decision, reason_codes = get_experiment_decision(exp_result, param_exp)
        exp["experiment_decision"] = decision
        exp["experiment_reason_codes"] = reason_codes
    
    # Select best, worst, and representative experiments
    selected = select_experiments(scored_experiments)
    
    # Get family-level decision
    family_decision, family_reason_codes, confidence, family_reason = get_family_decision(scored_experiments, param_exp)
    
    # Comparison analysis
    significant_experiments = [exp for exp in experiments if exp["p_value"] < 0.05]
    
    if experiments:
        best_by_p_value = min(experiments, key=lambda x: x["p_value"])["experiment_id"]
        best_by_abs_coef = max(experiments, key=lambda x: abs(x["coef"]))["experiment_id"]
        best_by_r_squared = max(experiments, key=lambda x: x["r_squared"])["experiment_id"]
    else:
        best_by_p_value = best_by_abs_coef = best_by_r_squared = None
    
    # Analyze trends across lookbacks
    coefs = [exp["coef"] for exp in experiments if exp.get("lookback_days")]
    r2s = [exp["r_squared"] for exp in experiments if exp.get("lookback_days")]
    
    if len(coefs) >= 2:
        # Check if coefficients decay toward zero
        coef_trend = "decays_toward_zero" if abs(coefs[-1]) < abs(coefs[0]) else "increases"
        r2_trend = "decays_toward_zero" if r2s[-1] < r2s[0] else "increases"
    else:
        coef_trend = r2_trend = "insufficient_data"
    
    # Stability assessment
    significant_count = len(significant_experiments)
    total_count = len(experiments)
    stability_ratio = significant_count / total_count if total_count > 0 else 0
    
    if stability_ratio >= 0.8:
        stability_label = "high"
    elif stability_ratio >= 0.5:
        stability_label = "medium"
    elif stability_ratio >= 0.2:
        stability_label = "low"
    else:
        stability_label = "very_low"
    
    # Overall strength assessment
    avg_r2 = sum(r2s) / len(r2s) if r2s else 0
    if avg_r2 >= 0.01:
        overall_strength = "strong"
    elif avg_r2 >= 0.005:
        overall_strength = "moderate"
    elif avg_r2 >= 0.001:
        overall_strength = "weak"
    else:
        overall_strength = "very_weak"
    
    # Determine hypothesis
    if "mean_reversion" in param_exp.base_name or "ma_distance_reversion" in param_exp.base_name:
        if "ma_distance_reversion" in param_exp.base_name:
            hypothesis = "Distance from moving average should negatively predict future returns."
        else:
            hypothesis = "Recent returns should negatively predict future returns."
        expected_pattern = "negative_coefficient"
    elif "momentum" in param_exp.base_name:
        hypothesis = "Recent returns should positively predict future returns."
        expected_pattern = "positive_coefficient"
    elif "volatility" in param_exp.base_name:
        hypothesis = "Recent volatility should positively predict future volatility."
        expected_pattern = "positive_coefficient"
    else:
        hypothesis = "Unknown hypothesis."
        expected_pattern = "unknown"
    
    # Build family summary
    family_summary = {
        "family": param_exp.base_name,
        "asset": "AAPL",
        "hypothesis": hypothesis,
        "expected_pattern": expected_pattern,
        
        "parameter_grid": {
            "lookback_days": lookback_days,
            "forward_days": forward_days
        },
        
        "experiments": experiments,
        
        "selected_experiments": {
            "best": {
                "experiment_id": selected["best"]["experiment_id"] if selected["best"] else None,
                "score": selected["best"]["score"] if selected["best"] else None,
                "score_label": selected["best"]["score_label"] if selected["best"] else None,
                "selection_role": "best_candidate"
            },
            "worst": {
                "experiment_id": selected["worst"]["experiment_id"] if selected["worst"] else None,
                "score": selected["worst"]["score"] if selected["worst"] else None,
                "score_label": selected["worst"]["score_label"] if selected["worst"] else None,
                "selection_role": "worst_candidate"
            },
            "representative": {
                "experiment_id": selected["representative"]["experiment_id"] if selected["representative"] else None,
                "score": selected["representative"]["score"] if selected["representative"] else None,
                "score_label": selected["representative"]["score_label"] if selected["representative"] else None,
                "selection_role": "representative_case"
            }
        },
        
        "comparison": {
            "best_by_p_value": best_by_p_value,
            "best_by_abs_coef": best_by_abs_coef,
            "best_by_r_squared": best_by_r_squared,
            "significant_count": significant_count,
            "total_count": total_count,
            "coef_trend_across_lookbacks": coef_trend,
            "r2_trend_across_lookbacks": r2_trend,
            "stability_label": stability_label,
            "overall_strength": overall_strength
        },
        
        "llm_context": {
            "should_interpret_as_trading_signal": False,  # Conservative: statistical significance ≠ tradable signal
            "important_cautions": [
                f"Average R-squared: {avg_r2:.4f} indicates limited predictive power.",
                f"Only {significant_count}/{total_count} experiments showed statistical significance.",
                "Statistical significance does not imply tradable edge.",
                "Transaction costs would likely eliminate any small edge."
            ],
            "requested_output_style": "formal_research_summary"
        },
        
        "decision": {
            "action": family_decision,
            "confidence": confidence,
            "reason_codes": family_reason_codes,
            "reason": family_reason
        }
    }
    
    return convert_to_json_serializable(family_summary)

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

# Run the fourth parameterized experiment (MA distance reversion with multiple lookbacks)
param_exp = param_exps[0]  # ma_distance_reversion
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

    # Create experiment JSON
    experiment_json = create_experiment_json(
        exp_name=exp_name,
        param_exp=param_exp,
        result=result,
        tables_dir=tables_dir,
        figures_dir=figures_dir
    )

    with open(tables_dir / "experiment.json", "w") as f:
        json.dump(experiment_json, f, indent=2)

print(f"\nAll results saved to reports/tables/ and reports/figures/")
# ... (rest of the code remains the same)
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

# Create and save family summary JSON
family_summary = create_family_summary_json(param_exp, results, comparison_dir)
with open(comparison_dir / "family_summary.json", "w") as f:
    json.dump(family_summary, f, indent=2)

print(f"\nComparison summary saved to: {comparison_dir / 'comparison_summary.txt'}")
print(f"Family summary saved to: {comparison_dir / 'family_summary.json'}")
print(f"Master summary updated: {master_summary_path}")
