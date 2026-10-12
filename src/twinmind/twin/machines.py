"""SimPy station processes: production, replayed failure, repair and maintenance.

THE FAILURE MECHANISM, AND WHY THERE IS NO FAILURE PROCESS.

The obvious design gives each machine a second SimPy process that waits a
sampled time-to-failure and interrupts production. That costs two extra events
and an interrupt per lot, and the twin's budget is 0.25 s for roughly 22,000 lot
cycles, so it cannot be afforded.

It is also unnecessary, because failure here is **deterministic given the
trajectory**. A machine carries a C-MAPSS engine id and an age in cycles, and it
fails when the age reaches that engine's real failure cycle -- a number read
from the dataset. So at the start of each lot the station can compute exactly
how many hours of processing remain before failure and compare that with the
lot's processing time. If the lot would run past the failure point, the machine
runs up to it and breaks. One event, no interrupt, and the failure time is
identical to what a racing process would have produced.

BLOCKING AND STARVING ARE NOT MODELLED, THEY EMERGE.

Buffers are ``simpy.Container`` objects with finite capacity. A station's
``get`` blocks when its input buffer is empty (starving) and its ``put`` blocks
when the downstream buffer is full (blocking). Neither is detected or
special-cased anywhere; they are what the Container does, which is the point of
using a discrete-event engine rather than a rate calculation.

Containers hold a *level*, not objects. A ``Store`` of per-unit Python objects
would be the natural-looking choice and is the single worst thing that could be
done to the hot loop here -- 45,000 objects allocated and freed per replication.
"""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import simpy

from twinmind.twin.prng import ChunkedBinomial, ChunkedNormal
from twinmind.utils.config import MachineConfig


@dataclass
class StationState:
    """Everything one station accumulates over a replication.

    Plain floats and ints rather than a list of events: the verification suite
    needs totals, the figures need a handful of sampled traces, and logging
    every state change would cost more than the simulation.
    """

    trajectory_slot: int = 0
    trajectory_id: int = 0
    failure_cycle: float = 0.0
    age_cycles: float = 0.0
    hours_since_renewal: float = 0.0

    processing_hours: float = 0.0
    downtime_unplanned: float = 0.0
    downtime_planned: float = 0.0
    blocked_hours: float = 0.0
    starved_hours: float = 0.0

    unplanned_failures: int = 0
    preventive_events: int = 0
    lots_completed: int = 0
    units_in: float = 0.0
    units_out: float = 0.0
    scrap_units: float = 0.0
    #: Units taken from the inbound buffer but not yet delivered onward -- the
    #: lot physically inside the machine. Without this, flow conservation is
    #: short by whatever is mid-process when the horizon ends (measured: 20
    #: units, two lots), because that material is in no buffer and has not been
    #: scrapped or finished.
    in_process: float = 0.0

    #: Warning bookkeeping for component validation 1: did the monitor warn
    #: before this machine's failure, and how many hours of notice did it give?
    warnings_raised: int = 0
    warned_before_failure: int = 0
    unwarned_failures: int = 0
    warning_lead_hours: list[float] = field(default_factory=list)
    pending_warning_at: float | None = None

    @property
    def downtime_hours(self) -> float:
        """Planned plus unplanned hours this station spent down."""
        return self.downtime_planned + self.downtime_unplanned


class Station:
    """One station on the line, failing according to a replayed trajectory."""

    def __init__(
        self,
        env: simpy.Environment,
        config: MachineConfig,
        inbound: simpy.Container,
        outbound: simpy.Container,
        state: StationState,
        *,
        speed_setpoint: float,
        pm_interval_hours: float,
        lot_size: int,
        throughput_multiplier: float,
        processing_cv: float,
        wear_exponent: float,
        life_scale_hours_per_cycle: float,
        defect_base: float,
        defect_speed_exponent: float,
        repair_multiplier: float,
        warning_threshold: float,
        hazard_source: str,
        health_lookup: Any,
        shift_calendar: Any,
        rng_processing: np.random.Generator,
        rng_repair: np.random.Generator,
        rng_defects: np.random.Generator,
        rng_life: np.random.Generator,
    ) -> None:
        self.env = env
        self.config = config
        self.inbound = inbound
        self.outbound = outbound
        self.state = state
        self.lot_size = lot_size
        self.speed = speed_setpoint
        self.pm_interval = pm_interval_hours
        self.processing_cv = processing_cv
        self.wear_exponent = wear_exponent
        self.life_scale = life_scale_hours_per_cycle
        self.defect_base = defect_base
        self.defect_speed_exponent = defect_speed_exponent
        self.repair_multiplier = repair_multiplier
        self.warning_threshold = warning_threshold
        self.hazard_source = hazard_source
        self.health = health_lookup
        self.calendar = shift_calendar
        self.rng_processing = rng_processing
        self.rng_repair = rng_repair
        self.rng_defects = rng_defects
        self.rng_life = rng_life

        # Effective rate: nominal x speed x the line-wide throughput target.
        rate = config.nominal_rate_units_per_hour * speed_setpoint * throughput_multiplier
        self.hours_per_lot = lot_size / max(rate, 1e-9)
        # Ageing is driven by speed alone, not by the throughput multiplier: the
        # multiplier is a production target, the setpoint is how hard the metal
        # is actually being driven.
        self.cycles_per_hour = (speed_setpoint**wear_exponent) / max(
            life_scale_hours_per_cycle, 1e-9
        )
        self.defect_probability = min(defect_base * (speed_setpoint**defect_speed_exponent), 1.0)

        # Lognormal parameters computed once rather than per lot. mu subtracts
        # sigma^2/2 so the MEAN is hours_per_lot whatever the CV; setting mu to
        # log(mean) instead -- the easy mistake -- would make a higher CV
        # silently slow the line, and the processing-CV sensitivity result would
        # then be measuring the wrong thing.
        if processing_cv > 0.0:
            self._sigma = float(np.sqrt(np.log1p(processing_cv * processing_cv)))
            self._mu = float(np.log(self.hours_per_lot) - 0.5 * self._sigma * self._sigma)
        else:
            self._sigma = 0.0
            self._mu = float(np.log(self.hours_per_lot))
        self._normals = ChunkedNormal(rng_processing)
        self._defect_draws = ChunkedBinomial(rng_defects, lot_size, self.defect_probability)

        # Cached staffed-window edge. The staffing check runs at every lot
        # boundary -- roughly 12,000 times a replication -- and a binary search
        # per call was measurable, so the current window's end is remembered and
        # the search only repeats once it is passed.
        self._staffed_until = -1.0

    # ----------------------------------------------------------------- run --
    def run(self) -> Generator[simpy.Event, Any, Any]:
        """Main production process: one iteration per lot."""
        lot = float(self.lot_size)
        while True:
            # Wait out any unstaffed period before touching the buffers, so an
            # idle night shift is not counted as starving.
            now = self.env.now
            if now >= self._staffed_until:
                gap = self.calendar.hours_until_staffed(now)
                if gap > 0.0:
                    yield self.env.timeout(gap)
                    continue
                self._staffed_until = self.calendar.next_shift_end(now)

            due = self._maintenance_due()
            if due is not None:
                yield from self._maintain(due)
                continue

            waited_from = self.env.now
            yield self.inbound.get(lot)
            self.state.starved_hours += self.env.now - waited_from
            self.state.units_in += lot
            self.state.in_process += lot

            processing = self._processing_hours()
            hours_to_failure = self._hours_to_failure()

            if self.hazard_source == "learned" and processing > hours_to_failure:
                # The lot is interrupted by a real failure partway through.
                yield self.env.timeout(hours_to_failure)
                self._advance_age(hours_to_failure)
                # The lot in the machine is destroyed by the failure. It leaves
                # in_process as scrap, so conservation still balances.
                self.state.in_process -= lot
                self.state.scrap_units += lot
                yield from self._repair()
                continue

            yield self.env.timeout(processing)
            self._advance_age(processing)
            self.state.processing_hours += processing

            good = self._good_units(lot)

            blocked_from = self.env.now
            if good > 0:
                yield self.outbound.put(good)
            self.state.blocked_hours += self.env.now - blocked_from

            # in_process is released only once the output has actually landed
            # downstream. Releasing it before the put would lose the units for
            # as long as a full buffer blocks the station -- measured as a 9-to-
            # 20 unit hole in flow conservation, which is exactly the size of
            # one good lot waiting on a full buffer.
            self.state.in_process -= lot
            self.state.scrap_units += lot - good

            self.state.units_out += good
            self.state.lots_completed += 1

            if (
                self.hazard_source == "fixed_mtbf"
                and self.state.age_cycles >= self.state.failure_cycle
            ):
                yield from self._repair()

    # --------------------------------------------------------- maintenance --
    def _maintenance_due(self) -> str | None:
        """``"time"``, ``"warning"`` or ``None``.

        Time-based PM fires on hours since the last renewal. The early-warning
        branch only exists in ``learned`` mode: in ``fixed_mtbf`` mode there is
        no monitoring system, which is exactly the contrast the week-6 ablation
        is built on.
        """
        if self.state.hours_since_renewal >= self.pm_interval:
            return "time"
        if self.hazard_source != "learned":
            return None
        if self.state.pending_warning_at is not None:
            # Scheduled for the next end of shift; take it once we are there.
            if self.env.now >= self.state.pending_warning_at:
                return "warning"
            return None
        _, _, p_fail = self._beliefs()
        if p_fail >= self.warning_threshold:
            self.state.warnings_raised += 1
            self.state.pending_warning_at = self.calendar.next_shift_end(self.env.now)
        return None

    def _maintain(self, reason: str) -> Generator[simpy.Event, Any, Any]:
        """Take the machine down for planned maintenance and renew it."""
        hours = self.config.pm_hours
        if reason == "warning":
            lead = self.state.failure_cycle - self.state.age_cycles
            self.state.warning_lead_hours.append(max(lead, 0.0) / max(self.cycles_per_hour, 1e-9))
            self.state.warned_before_failure += 1
        yield self.env.timeout(hours)
        self.state.downtime_planned += hours
        self.state.preventive_events += 1
        self._renew()

    def _repair(self) -> Generator[simpy.Event, Any, Any]:
        """Unplanned repair after a failure, then renewal."""
        mean = self.config.repair_hours_mean * self.repair_multiplier
        sd = max(self.config.repair_hours_sd, 1e-9)
        hours = float(np.clip(self.rng_repair.normal(mean, sd), 0.25, 10.0 * mean))
        yield self.env.timeout(hours)
        self.state.downtime_unplanned += hours
        self.state.unplanned_failures += 1
        if self.state.pending_warning_at is None:
            self.state.unwarned_failures += 1
        self._renew()

    def _renew(self) -> None:
        """As-good-as-new: a fresh trajectory from the pool at age zero.

        A real overhaul is rarely as-good-as-new, but modelling imperfect repair
        needs a repair-effectiveness parameter with nothing behind it, and the
        C-MAPSS pool only contains complete run-to-failure trajectories -- there
        is no such thing as a half-renewed engine in the data. Documented in the
        model card as a limitation rather than papered over with a third
        invented constant.
        """
        self.state.age_cycles = 0.0
        self.state.hours_since_renewal = 0.0
        self.state.pending_warning_at = None
        if self.hazard_source == "fixed_mtbf":
            self.state.failure_cycle = float(self.health.draw_lifetime(self.rng_life))
        else:
            slot, engine_id, failure_cycle = self.health.draw_engine(self.rng_life)
            self.state.trajectory_slot = slot
            self.state.trajectory_id = engine_id
            self.state.failure_cycle = float(failure_cycle)

    # -------------------------------------------------------------- detail --
    def _processing_hours(self) -> float:
        """Lognormal lot processing time with the configured CV.

        Parameterised so the *mean* is ``hours_per_lot`` regardless of CV --
        ``mu = log(mean) - sigma^2/2``. Setting mu to log(mean) directly, which
        is the easy mistake, would make a higher CV silently slow the line down
        and the processing-CV sensitivity result would be measuring the wrong
        thing.
        """
        if self._sigma <= 0.0:
            return self.hours_per_lot
        return float(np.exp(self._mu + self._sigma * self._normals.next()))

    def _good_units(self, lot: float) -> float:
        """Units surviving the defect draw, binomial over the lot."""
        if self.defect_probability <= 0.0:
            return lot
        return float(lot - self._defect_draws.next())

    def _hours_to_failure(self) -> float:
        """Processing hours remaining before the replayed trajectory fails."""
        remaining = self.state.failure_cycle - self.state.age_cycles
        if remaining <= 0.0:
            return 0.0
        return remaining / max(self.cycles_per_hour, 1e-9)

    def _advance_age(self, hours: float) -> None:
        """Consume life. Only processing time ages a machine, not idle time."""
        self.state.age_cycles += hours * self.cycles_per_hour
        self.state.hours_since_renewal += hours

    def _beliefs(self) -> tuple[float, float, float]:
        """What the monitoring system currently thinks: (rul_mean, rul_std, p_fail)."""
        return self.health.predictions(self.state.trajectory_slot, int(self.state.age_cycles))
