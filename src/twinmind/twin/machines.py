"""SimPy machine processes: production, failure, repair and preventive maintenance."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import simpy

from twinmind.utils.config import MachineConfig


class Machine:
    """One station on the line, with a failure hazard supplied by Model A."""

    def __init__(
        self,
        env: simpy.Environment,
        config: MachineConfig,
        hazard_fn: Any,
        speed_setpoint: float = 1.0,
        pm_interval_hours: float = 168.0,
    ) -> None:
        self.env = env
        self.config = config
        self.hazard_fn = hazard_fn
        self.speed_setpoint = speed_setpoint
        self.pm_interval_hours = pm_interval_hours

    def run(self) -> Iterator[simpy.Event]:
        """Main production process."""
        raise NotImplementedError("Machine.run is not implemented yet")

    def failure_process(self) -> Iterator[simpy.Event]:
        """Sample time-to-failure from the Model A hazard and trigger repairs."""
        raise NotImplementedError("Machine.failure_process is not implemented yet")

    def maintenance_process(self) -> Iterator[simpy.Event]:
        """Take the machine down every ``pm_interval_hours`` of operation."""
        raise NotImplementedError("Machine.maintenance_process is not implemented yet")
