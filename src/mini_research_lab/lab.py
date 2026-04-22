from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from .summaries import DescribeSummary, RegressionSummary


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