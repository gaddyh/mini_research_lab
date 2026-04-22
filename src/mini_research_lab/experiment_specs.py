from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    x_col: str
    y_col: str
    title: str
    description: str = ""


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