from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class DescribeSummary:
    series_name: str
    count: int
    mean: float
    std: float
    min: float
    q25: float
    median: float
    q75: float
    max: float
    iqr: float
    lower_fence: float
    upper_fence: float
    n_lower_outliers: int
    n_upper_outliers: int
    skew_hint: str

    @classmethod
    def from_series(cls, series: pd.Series, series_name: Optional[str] = None) -> "DescribeSummary":
        s = series.dropna()
        if s.empty:
            raise ValueError("Series is empty after dropping NaN values.")

        desc = s.describe()

        q25 = float(desc["25%"])
        q75 = float(desc["75%"])
        iqr = q75 - q25

        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr

        mean = float(desc["mean"])
        median = float(desc["50%"])

        if mean > median:
            skew_hint = "right-skew candidate"
        elif mean < median:
            skew_hint = "left-skew candidate"
        else:
            skew_hint = "roughly symmetric candidate"

        return cls(
            series_name=series_name or (series.name or "series"),
            count=int(desc["count"]),
            mean=mean,
            std=float(desc["std"]),
            min=float(desc["min"]),
            q25=q25,
            median=median,
            q75=q75,
            max=float(desc["max"]),
            iqr=iqr,
            lower_fence=lower_fence,
            upper_fence=upper_fence,
            n_lower_outliers=int((s < lower_fence).sum()),
            n_upper_outliers=int((s > upper_fence).sum()),
            skew_hint=skew_hint,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "series": self.series_name,
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "25%": self.q25,
            "50%": self.median,
            "75%": self.q75,
            "max": self.max,
            "iqr": self.iqr,
            "lower_fence": self.lower_fence,
            "upper_fence": self.upper_fence,
            "n_lower_outliers": self.n_lower_outliers,
            "n_upper_outliers": self.n_upper_outliers,
            "skew_hint": self.skew_hint,
        }


@dataclass
class RegressionSummary:
    x_col: str
    y_col: str
    intercept: float
    coef: float
    std_err: float
    t_value: float
    p_value: float
    r_squared: float
    n_obs: int
    ci_low: float
    ci_high: float

    @classmethod
    def from_model(cls, x_col: str, y_col: str, model) -> "RegressionSummary":
        params = model.params
        bse = model.bse
        tvalues = model.tvalues
        pvalues = model.pvalues
        conf_int = model.conf_int()

        return cls(
            x_col=x_col,
            y_col=y_col,
            intercept=float(params["const"]),
            coef=float(params[x_col]),
            std_err=float(bse[x_col]),
            t_value=float(tvalues[x_col]),
            p_value=float(pvalues[x_col]),
            r_squared=float(model.rsquared),
            n_obs=int(model.nobs),
            ci_low=float(conf_int.loc[x_col, 0]),
            ci_high=float(conf_int.loc[x_col, 1]),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "x_col": self.x_col,
            "y_col": self.y_col,
            "intercept": self.intercept,
            "coef": self.coef,
            "std_err": self.std_err,
            "t_value": self.t_value,
            "p_value": self.p_value,
            "r_squared": self.r_squared,
            "n_obs": self.n_obs,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
        }