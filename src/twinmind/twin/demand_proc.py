"""Order-book process driven by the Model B demand forecast."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import simpy


class DemandProcess:
    """Generates customer orders by sampling the forecast quantiles.

    Nothing here is a fixed arrival rate: the daily order volume is drawn from
    the q10/q50/q90 band produced by Model B, scaled by the scenario's
    ``price_multiplier`` through a simple price-elasticity term.
    """

    def __init__(
        self,
        env: simpy.Environment,
        forecast_quantiles: np.ndarray,
        price_multiplier: float = 1.0,
        price_elasticity: float = -1.2,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.env = env
        self.forecast_quantiles = forecast_quantiles
        self.price_multiplier = price_multiplier
        self.price_elasticity = price_elasticity
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_daily_demand(self, day: int) -> float:
        """Draw one day's demand from the forecast band."""
        raise NotImplementedError("DemandProcess.sample_daily_demand is not implemented yet")

    def run(self) -> Iterator[simpy.Event]:
        """Emit orders into the twin's order book."""
        raise NotImplementedError("DemandProcess.run is not implemented yet")
