from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from .summaries import DescribeSummary, RegressionSummary
from .experiment_specs import ParameterizedExperiment, generate_variations


class MiniResearchLab:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def describe_series(self, col: str) -> DescribeSummary:
        if col not in self.df.columns:
            raise ValueError(f"Column not found: {col}")
        return DescribeSummary.from_series(self.df[col], series_name=col)

    def run_simple_regression(self, x_col: str, y_col: str):
        missing = [col for col in [x_col, y_col] if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        temp = self.df[[x_col, y_col]].dropna().copy()
        if temp.empty:
            raise ValueError("No rows left after dropping NaN values.")

        X = sm.add_constant(temp[[x_col]])
        y = temp[y_col]

        model = sm.OLS(y, X).fit()
        return model

    def summarize_regression(self, x_col: str, y_col: str) -> RegressionSummary:
        model = self.run_simple_regression(x_col, y_col)
        return RegressionSummary.from_model(x_col=x_col, y_col=y_col, model=model)

    def run_experiment(self, x_col: str, y_col: str) -> dict:
        model = self.run_simple_regression(x_col, y_col)

        return {
            "x_describe": self.describe_series(x_col),
            "y_describe": self.describe_series(y_col),
            "regression": RegressionSummary.from_model(x_col=x_col, y_col=y_col, model=model),
            "model": model,
        }

    def analyze_bucketed_relationship(self, x_col: str, y_col: str, n_buckets: int = 5) -> pd.DataFrame:
        """Analyze relationship between X and Y by bucketing X variable."""
        missing = [col for col in [x_col, y_col] if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        temp = self.df[[x_col, y_col]].dropna().copy()
        if temp.empty:
            raise ValueError("No rows left after dropping NaN values.")

        # Create buckets based on X variable quantiles
        temp['x_bucket'] = pd.qcut(temp[x_col], n_buckets, labels=False, duplicates='drop')
        
        # Calculate statistics for each bucket
        bucket_stats = []
        for bucket_num in range(n_buckets):
            bucket_data = temp[temp['x_bucket'] == bucket_num]
            if len(bucket_data) > 0:
                x_min, x_max = bucket_data[x_col].min(), bucket_data[x_col].max()
                y_mean = bucket_data[y_col].mean()
                y_std = bucket_data[y_col].std()
                count = len(bucket_data)
                
                bucket_stats.append({
                    'bucket': bucket_num + 1,
                    'x_range': f"{x_min:.4f} to {x_max:.4f}",
                    'x_mean': bucket_data[x_col].mean(),
                    'y_mean': y_mean,
                    'y_std': y_std,
                    'count': count,
                    'y_mean_plus_1std': y_mean + y_std,
                    'y_mean_minus_1std': y_mean - y_std
                })
        
        return pd.DataFrame(bucket_stats)

    def run_parameterized_experiment(self, param_exp: ParameterizedExperiment) -> dict[str, dict]:
        """Run all variations of a parameterized experiment."""
        variations = generate_variations(param_exp)
        results = {}
        
        for spec in variations:
            try:
                experiment_result = self.run_experiment(spec.x_col, spec.y_col)
                # Add bucketed analysis
                experiment_result['bucketed_analysis'] = self.analyze_bucketed_relationship(spec.x_col, spec.y_col)
                results[spec.name] = experiment_result
            except ValueError as e:
                # Skip variations that don't have the required columns
                print(f"Skipping {spec.name}: {e}")
                continue
                
        return results