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
    x_col_pattern: str  # Use {lookback} or {bucket} as placeholder
    y_col: str
    lookbacks: list[int] | list[str]  # Can be days or bucket names
    title_template: str  # Use {lookback} or {bucket} as placeholder
    description_template: str  # Use {lookback} or {bucket} as placeholder


def generate_variations(param_exp: ParameterizedExperiment) -> list[ExperimentSpec]:
    """Generate ExperimentSpec variations from a ParameterizedExperiment."""
    variations = []
    for lookback in param_exp.lookbacks:
        if isinstance(lookback, str):
            # Handle string lookbacks (like RSI buckets)
            variations.append(ExperimentSpec(
                name=f"{param_exp.base_name}_{lookback}",
                x_col=param_exp.x_col_pattern.format(bucket=lookback),
                y_col=param_exp.y_col,
                title=param_exp.title_template.format(bucket=lookback),
                description=param_exp.description_template.format(bucket=lookback)
            ))
        else:
            # Handle numeric lookbacks (like days)
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


def generate_dynamic_experiments(horizon: str = "1d", mode: str = "level") -> list[ParameterizedExperiment]:
    """Generate experiments with dynamic horizon and mode support."""
    
    # Map horizon to forward return column
    horizon_map = {
        "1d": "fwd_ret_1d",
        "3d": "fwd_ret_3d", 
        "5d": "fwd_ret_5d",
        "10d": "fwd_ret_10d",
        "20d": "fwd_ret_20d"
    }
    fwd_ret_col = horizon_map.get(horizon, "fwd_ret_1d")
    
    experiments = [
        # Standard experiments (always included)
        ParameterizedExperiment(
            base_name="mean_reversion",
            x_col_pattern="ret_{lookback}d",
            y_col=fwd_ret_col,
            lookbacks=[1, 3, 5, 10, 20],
            title_template=f"Do extreme {{lookback}}-day moves revert next {horizon}?",
            description_template=f"Mean reversion mini-project using recent {{lookback}}-day return vs next {horizon} return."
        ),
        ParameterizedExperiment(
            base_name="momentum",
            x_col_pattern="ret_{lookback}d",
            y_col=fwd_ret_col,
            lookbacks=[3, 5, 10, 20],
            title_template=f"Do recent {{lookback}}-day gains continue over the next {horizon}?",
            description_template=f"Momentum mini-project using recent {{lookback}}-day return vs next {horizon} return."
        ),
        ParameterizedExperiment(
            base_name="volatility_clustering",
            x_col_pattern="abs_ret_{lookback}d",
            y_col=f" fwd_abs_ret_{horizon[0]}" if horizon[0] in ['1', '3', '5'] else fwd_ret_col,  # Use absolute returns for short horizons
            lookbacks=[1, 3, 5],
            title_template=f"Do big {{lookback}}-day moves tend to be followed by big {horizon} moves?",
            description_template=f"Volatility clustering mini-project using {{lookback}}-day absolute returns vs next {horizon} return."
        ),
        ParameterizedExperiment(
            base_name="ma_distance_reversion",
            x_col_pattern="dist_from_ma{lookback}",
            y_col=fwd_ret_col,
            lookbacks=[10, 20, 50],
            title_template=f"Does distance from {{lookback}}-day moving average predict next {horizon} return?",
            description_template=f"Moving-average distance mini-project for mean reversion using {{lookback}}-day MA vs next {horizon} return."
        ),
    ]
    
    # Add strategy experiments based on mode
    if mode == "event":
        # Event-based experiments
        experiments.extend([
            ParameterizedExperiment(
                base_name="rsi_mean_reversion_event",
                x_col_pattern="rsi_{lookback}_oversold_recovery",
                y_col=fwd_ret_col,
                lookbacks=[7, 14],
                title_template=f"Do RSI {{lookback}}-day oversold recovery events predict next {horizon} returns?",
                description_template=f"RSI event-based mean reversion using {{lookback}}-day oversold recovery signals vs next {horizon} return."
            ),
            ParameterizedExperiment(
                base_name="donchian_breakout_event",
                x_col_pattern="donchian_{lookback}d_upper_breakout_strong",
                y_col=fwd_ret_col,
                lookbacks=[10, 20],
                title_template=f"Do {{lookback}}-day Donchian upper breakout events predict next {horizon} returns?",
                description_template=f"Donchian event-based breakout using {{lookback}}-day upper breakout signals vs next {horizon} return."
            ),
            ParameterizedExperiment(
                base_name="ma_crossover_event",
                x_col_pattern="golden_cross_10_20",
                y_col=fwd_ret_col,
                lookbacks=[1],  # Single event type
                title_template=f"Do golden cross events predict next {horizon} returns?",
                description_template=f"Moving average crossover event analysis using golden cross signals vs next {horizon} return."
            ),
        ])
    else:
        # Level-based experiments (original strategy experiments)
        experiments.extend([
            ParameterizedExperiment(
                base_name="rsi_mean_reversion",
                x_col_pattern="rsi_{lookback}_oversold_recovery",
                y_col=fwd_ret_col,
                lookbacks=[7, 14],
                title_template=f"Do RSI {{lookback}}-day oversold recovery signals predict next {horizon} returns?",
                description_template=f"RSI mean reversion mini-project using {{lookback}}-day oversold recovery signals vs next {horizon} return."
            ),
            ParameterizedExperiment(
                base_name="rsi_bucket_analysis",
                x_col_pattern="rsi_14_{bucket}_bucket",
                y_col=fwd_ret_col,
                lookbacks=["oversold", "neutral_low", "neutral_high", "overbought"],
                title_template=f"Do RSI {{bucket}} levels predict next {horizon} returns?",
                description_template=f"RSI bucket analysis mini-project using {{bucket}} RSI levels vs next {horizon} return."
            ),
            ParameterizedExperiment(
                base_name="donchian_breakout_5d",
                x_col_pattern="donchian_{lookback}d_upper_breakout_strong",
                y_col="fwd_ret_5d",
                lookbacks=[10, 20],
                title_template="Do {lookback}-day Donchian upper breakouts predict next 5-day returns?",
                description_template="Donchian breakout mini-project using {lookback}-day upper breakout signals vs next 5-day return."
            ),
            ParameterizedExperiment(
                base_name="donchian_breakout_10d",
                x_col_pattern="donchian_{lookback}d_upper_breakout_strong",
                y_col="fwd_ret_10d",
                lookbacks=[10, 20],
                title_template="Do {lookback}-day Donchian upper breakouts predict next 10-day returns?",
                description_template="Donchian breakout mini-project using {lookback}-day upper breakout signals vs next 10-day return."
            ),
            ParameterizedExperiment(
                base_name="donchian_breakout_20d",
                x_col_pattern="donchian_{lookback}d_upper_breakout_strong",
                y_col="fwd_ret_20d",
                lookbacks=[10, 20],
                title_template="Do {lookback}-day Donchian upper breakouts predict next 20-day returns?",
                description_template="Donchian breakout mini-project using {lookback}-day upper breakout signals vs next 20-day return."
            ),
        ])
    
    return experiments


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
        ParameterizedExperiment(
            base_name="ma_distance_reversion",
            x_col_pattern="dist_from_ma{lookback}",
            y_col="fwd_ret_1d",
            lookbacks=[10, 20, 50],
            title_template="Does distance from {lookback}-day moving average predict next-day return?",
            description_template="Moving-average distance mini-project for mean reversion using {lookback}-day MA."
        ),
        ParameterizedExperiment(
            base_name="rsi_mean_reversion",
            x_col_pattern="rsi_{lookback}_oversold_recovery",
            y_col="fwd_ret_1d",
            lookbacks=[7, 14],
            title_template="Do RSI {lookback}-day oversold recovery signals predict next-day returns?",
            description_template="RSI mean reversion mini-project using {lookback}-day oversold recovery signals vs next 1-day return."
        ),
        ParameterizedExperiment(
            base_name="rsi_bucket_analysis",
            x_col_pattern="rsi_14_{bucket}_bucket",
            y_col="fwd_ret_1d",
            lookbacks=["oversold", "neutral_low", "neutral_high", "overbought"],
            title_template="Do RSI {bucket} levels predict next-day returns?",
            description_template="RSI bucket analysis mini-project using {bucket} RSI levels vs next 1-day return."
        ),
        ParameterizedExperiment(
            base_name="donchian_breakout_5d",
            x_col_pattern="donchian_{lookback}d_upper_breakout_strong",
            y_col="fwd_ret_5d",
            lookbacks=[10, 20],
            title_template="Do {lookback}-day Donchian upper breakouts predict next 5-day returns?",
            description_template="Donchian breakout mini-project using {lookback}-day upper breakout signals vs next 5-day return."
        ),
        ParameterizedExperiment(
            base_name="donchian_breakout_10d",
            x_col_pattern="donchian_{lookback}d_upper_breakout_strong",
            y_col="fwd_ret_10d",
            lookbacks=[10, 20],
            title_template="Do {lookback}-day Donchian upper breakouts predict next 10-day returns?",
            description_template="Donchian breakout mini-project using {lookback}-day upper breakout signals vs next 10-day return."
        ),
        ParameterizedExperiment(
            base_name="donchian_breakout_20d",
            x_col_pattern="donchian_{lookback}d_upper_breakout_strong",
            y_col="fwd_ret_20d",
            lookbacks=[10, 20],
            title_template="Do {lookback}-day Donchian upper breakouts predict next 20-day returns?",
            description_template="Donchian breakout mini-project using {lookback}-day upper breakout signals vs next 20-day return."
        ),
    ]