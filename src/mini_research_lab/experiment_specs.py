from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    x_col: str
    y_col: str
    title: str
    description: str = ""


@dataclass(frozen=True)
class ParameterizedExperiment:
    base_name: str
    x_col_pattern: str  # Use {lookback} as placeholder, e.g., "ret_{lookback}d"
    y_col: str
    lookbacks: list[int]
    title_template: str  # Use {lookback} as placeholder
    description_template: str  # Use {lookback} as placeholder


def generate_variations(param_exp: ParameterizedExperiment) -> list[ExperimentSpec]:
    """Generate ExperimentSpec variations from a ParameterizedExperiment."""
    variations = []
    for lookback in param_exp.lookbacks:
        variations.append(ExperimentSpec(
            name=f"{param_exp.base_name}_{lookback}d_to_1d",
            x_col=param_exp.x_col_pattern.format(lookback=lookback),
            y_col=param_exp.y_col,
            title=param_exp.title_template.format(lookback=lookback),
            description=param_exp.description_template.format(lookback=lookback)
        ))
    return variations


def default_experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="mean_reversion_3d_to_1d",
            x_col="ret_3d",
            y_col="fwd_ret_1d",
            title="Do extreme 3-day moves revert next day?",
            description="Mean reversion mini-project using recent 3-day return vs next 1-day return.",
        ),
        ExperimentSpec(
            name="momentum_5d_to_3d",
            x_col="ret_5d",
            y_col="fwd_ret_3d",
            title="Do recent 5-day gains continue over the next 3 days?",
            description="Momentum mini-project using recent 5-day return vs next 3-day return.",
        ),
        ExperimentSpec(
            name="vol_clustering_1d",
            x_col="abs_ret_1d",
            y_col="fwd_abs_ret_1d",
            title="Do big moves tend to be followed by big moves?",
            description="Volatility clustering mini-project using absolute returns.",
        ),
        ExperimentSpec(
            name="distance_from_ma20_to_next_return",
            x_col="dist_from_ma20",
            y_col="fwd_ret_1d",
            title="Does distance from 20-day moving average predict next-day return?",
            description="Moving-average distance mini-project for mean reversion intuition.",
        ),
    ]


def parameterized_experiments() -> list[ParameterizedExperiment]:
    """Return list of parameterized experiments for generating variations."""
    return [
        ParameterizedExperiment(
            base_name="mean_reversion",
            x_col_pattern="ret_{lookback}d",
            y_col="fwd_ret_1d",
            lookbacks=[1, 3, 5, 10, 20],
            title_template="Do extreme {lookback}-day moves revert next day?",
            description_template="Mean reversion mini-project using recent {lookback}-day return vs next 1-day return."
        ),
        ParameterizedExperiment(
            base_name="momentum",
            x_col_pattern="ret_{lookback}d",
            y_col="fwd_ret_3d",
            lookbacks=[3, 5, 10, 20],
            title_template="Do recent {lookback}-day gains continue over the next 3 days?",
            description_template="Momentum mini-project using recent {lookback}-day return vs next 3-day return."
        ),
        ParameterizedExperiment(
            base_name="volatility_clustering",
            x_col_pattern="abs_ret_{lookback}d",
            y_col="fwd_abs_ret_1d",
            lookbacks=[1, 3, 5],
            title_template="Do big {lookback}-day moves tend to be followed by big 1-day moves?",
            description_template="Volatility clustering mini-project using {lookback}-day absolute returns."
        ),
    ]