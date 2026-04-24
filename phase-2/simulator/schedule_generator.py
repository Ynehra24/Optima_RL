"""
schedule_generator.py — Synthetic daily truck schedule generator.

Generates a plausible daily schedule of inbound and outbound trucks
for the cross-docking hub, calibrated using:
  - FAF5 routing matrix (which O-D pairs carry what volume)
  - CFS 2022 cargo profiles (perishable%, value, SLA urgency)
  - Cargo 2000 delay distributions (via DelaySampler)

Truck arrivals follow a 3-wave daily pattern:
  Morning wave:  06:00-09:00 (25% of daily volume)
  Midday wave:   11:00-14:00 (40% of daily volume)
  Evening wave:  17:00-20:00 (35% of daily volume)

This mirrors the real-world pattern at cross-docking hubs where trucks
arrive in waves synchronized with highway driving schedules.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from simulator.config import SimConfig
from simulator.hub_config import HubNetwork
from simulator.delay_sampler import DelaySampler


@dataclass
class CargoUnit:
    """A single cargo unit on a truck.

    Logistics analog of a PAX (passenger) in Phase 1.
    """
    cargo_id: str
    value_score: float      # Normalized 0-1 (derived from shipment value)
    sla_urgency: int        # 0=standard, 1=priority, 2=express
    is_perishable: bool
    weight_kg: float
    dest_zone: int          # Which zone this cargo is headed to
    dest_truck_id: str      # ID of the outbound truck it should board


@dataclass
class TruckSchedule:
    """A truck's schedule entry (inbound or outbound).

    Logistics analog of a Flight in Phase 1.
    """
    truck_id: str
    direction: str              # 'inbound' or 'outbound'
    origin_zone: int
    dest_zone: int
    route_type: str             # 'short', 'medium', 'long'
    scheduled_arrival: float    # Minutes from midnight (for inbound)
    scheduled_departure: float  # Minutes from midnight (for outbound)
    cargo_units: List[CargoUnit] = field(default_factory=list)
    connecting_outbound: List[str] = field(default_factory=list)
    prev_truck_id: Optional[str] = None  # Same-route previous truck (for delay tree)
    driver_hours_remaining: float = 660.0

    @property
    def n_cargo(self) -> int:
        return len(self.cargo_units)


@dataclass
class DailySchedule:
    """Complete simulated daily truck schedule at the hub."""
    day_index: int
    inbound_trucks: List[TruckSchedule]
    outbound_trucks: List[TruckSchedule]
    # Map: outbound_truck_id → [inbound_truck_ids that feed into it]
    feeder_map: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def all_trucks(self) -> List[TruckSchedule]:
        return self.inbound_trucks + self.outbound_trucks


# ── Wave parameters (fraction of daily trucks, time window in minutes) ──
ARRIVAL_WAVES = [
    {"frac": 0.25, "start": 360,  "end": 540},   # 06:00-09:00
    {"frac": 0.40, "start": 660,  "end": 840},   # 11:00-14:00
    {"frac": 0.35, "start": 1020, "end": 1200},  # 17:00-20:00
]

# Outbound trucks depart ~MTT + processing time after their inbound wave
OUTBOUND_OFFSETS = {
    360:  90,   # Morning inbound → depart ~08:30
    660:  90,   # Midday inbound  → depart ~13:30
    1020: 90,   # Evening inbound → depart ~19:00
}


class ScheduleGenerator:
    """Generates synthetic daily truck schedules for the cross-docking hub.

    Usage:
        gen = ScheduleGenerator(cfg, hub_network, delay_sampler, cargo_profiles)
        schedule = gen.generate_day(day_index=0)
    """

    def __init__(
        self,
        cfg: SimConfig,
        hub_network: HubNetwork,
        delay_sampler: DelaySampler,
        cargo_profiles: dict,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = cfg
        self.hub = hub_network
        self.sampler = delay_sampler
        self.cargo_profiles = cargo_profiles
        self.rng = rng or np.random.default_rng(cfg.seed)

        # Pre-compute SLA urgency sampling weights from cargo profiles
        sla_dist = cargo_profiles.get("sla_distribution", {0: 0.5, 1: 0.3, 2: 0.2})
        self._sla_weights = [sla_dist.get(str(i), sla_dist.get(i, 0.0)) for i in range(3)]
        sla_total = sum(self._sla_weights)
        if sla_total > 0:
            self._sla_weights = [w / sla_total for w in self._sla_weights]
        else:
            self._sla_weights = [0.5, 0.3, 0.2]

        # Value score sampling bounds
        vp = cargo_profiles.get("value_percentiles", {})
        self._log_min = vp.get("log_min", 0.0)
        self._log_max = vp.get("log_max", 10.0)

        self._truck_counter = 0

    # ── Public API ─────────────────────────────────────────────────────

    def generate_day(self, day_index: int) -> DailySchedule:
        """Generate a complete daily schedule.

        Args:
            day_index: Day number in the episode (0-indexed)

        Returns:
            DailySchedule with inbound/outbound trucks and feeder_map
        """
        inbound_trucks = []
        outbound_trucks = []
        feeder_map: Dict[str, List[str]] = {}

        hub_zone = self.hub.hub_zone.zone_id

        for wave in ARRIVAL_WAVES:
            n_trucks = max(1, int(self.cfg.trucks_per_day * wave["frac"]))
            wave_start = wave["start"]
            wave_end = wave["end"]

            # Generate inbound trucks for this wave
            wave_inbound = []
            for _ in range(n_trucks):
                origin_zone = self._sample_origin_zone(hub_zone)
                route_type = self.hub.get_route_type(origin_zone, hub_zone)
                sched_arrival = float(self.rng.uniform(wave_start, wave_end))
                truck = self._make_inbound_truck(
                    origin_zone=origin_zone,
                    dest_zone=hub_zone,
                    route_type=route_type,
                    scheduled_arrival=sched_arrival,
                )
                wave_inbound.append(truck)
                inbound_trucks.append(truck)

            # Generate matching outbound trucks departing ~MTT after wave end
            outbound_depart_time = wave_end + self.cfg.mtt + 30
            n_outbound = max(1, n_trucks // 3)  # ~1 outbound per 3 inbound

            wave_outbound = []
            for i in range(n_outbound):
                dest_zone = self._sample_dest_zone(hub_zone)
                route_type = self.hub.get_route_type(hub_zone, dest_zone)
                # Stagger departures by 10 minutes
                sched_departure = outbound_depart_time + i * 10.0
                truck = self._make_outbound_truck(
                    origin_zone=hub_zone,
                    dest_zone=dest_zone,
                    route_type=route_type,
                    scheduled_departure=sched_departure,
                )
                wave_outbound.append(truck)
                outbound_trucks.append(truck)

            # Link inbound cargo to outbound trucks and build feeder_map
            self._link_cargo(wave_inbound, wave_outbound, feeder_map)

        return DailySchedule(
            day_index=day_index,
            inbound_trucks=sorted(inbound_trucks, key=lambda t: t.scheduled_arrival),
            outbound_trucks=sorted(outbound_trucks, key=lambda t: t.scheduled_departure),
            feeder_map=feeder_map,
        )

    # ── Private helpers ────────────────────────────────────────────────

    def _new_truck_id(self, prefix: str = "T") -> str:
        self._truck_counter += 1
        return f"{prefix}{self._truck_counter:04d}"

    def _sample_origin_zone(self, hub_zone: int) -> int:
        """Sample an origin zone (not the hub itself)."""
        candidates = [z.zone_id for z in self.hub.zones if z.zone_id != hub_zone]
        if not candidates:
            return hub_zone
        return int(self.rng.choice(candidates))

    def _sample_dest_zone(self, hub_zone: int) -> int:
        """Sample a destination zone using routing matrix probabilities."""
        dest = self.hub.sample_destination(hub_zone, self.rng)
        if dest is None:
            candidates = [z.zone_id for z in self.hub.zones if z.zone_id != hub_zone]
            return int(self.rng.choice(candidates)) if candidates else hub_zone
        return dest

    def _make_inbound_truck(
        self,
        origin_zone: int,
        dest_zone: int,
        route_type: str,
        scheduled_arrival: float,
    ) -> TruckSchedule:
        """Create an inbound truck with sampled cargo."""
        truck_id = self._new_truck_id("IN")
        n_cargo = self.sampler.sample_cargo_count(mean=30)
        driver_hours = self.sampler.sample_driver_hours_remaining()

        return TruckSchedule(
            truck_id=truck_id,
            direction="inbound",
            origin_zone=origin_zone,
            dest_zone=dest_zone,
            route_type=route_type,
            scheduled_arrival=scheduled_arrival,
            scheduled_departure=scheduled_arrival + 60.0,  # placeholder
            cargo_units=[],          # filled by _link_cargo
            driver_hours_remaining=driver_hours,
        )

    def _make_outbound_truck(
        self,
        origin_zone: int,
        dest_zone: int,
        route_type: str,
        scheduled_departure: float,
    ) -> TruckSchedule:
        """Create an outbound truck (cargo assigned later)."""
        truck_id = self._new_truck_id("OUT")
        driver_hours = self.sampler.sample_driver_hours_remaining()

        return TruckSchedule(
            truck_id=truck_id,
            direction="outbound",
            origin_zone=origin_zone,
            dest_zone=dest_zone,
            route_type=route_type,
            scheduled_arrival=scheduled_departure - 30.0,  # arrives to bay before departure
            scheduled_departure=scheduled_departure,
            cargo_units=[],
            driver_hours_remaining=driver_hours,
        )

    def _link_cargo(
        self,
        inbound_trucks: List[TruckSchedule],
        outbound_trucks: List[TruckSchedule],
        feeder_map: Dict[str, List[str]],
    ):
        """Distribute inbound cargo to outbound trucks.

        Each inbound truck's cargo is split across outbound trucks
        proportionally (round-robin with some randomness).
        Builds the feeder_map: {outbound_id: [inbound_ids that feed it]}.
        """
        if not outbound_trucks:
            return

        out_ids = [t.truck_id for t in outbound_trucks]
        out_zones = {t.truck_id: t.dest_zone for t in outbound_trucks}
        out_map = {t.truck_id: t for t in outbound_trucks}

        for in_truck in inbound_trucks:
            n_cargo = self.sampler.sample_cargo_count(mean=25)
            cargo_counter = 0

            for out_truck in outbound_trucks:
                if out_truck.truck_id not in feeder_map:
                    feeder_map[out_truck.truck_id] = []

            for i in range(n_cargo):
                # Assign to a random outbound truck
                out_truck_id = str(self.rng.choice(out_ids))
                out_truck = out_map[out_truck_id]

                cargo = self._sample_cargo_unit(
                    cargo_id=f"{in_truck.truck_id}_C{i:03d}",
                    dest_zone=out_truck.dest_zone,
                    dest_truck_id=out_truck_id,
                )
                in_truck.cargo_units.append(cargo)
                out_truck.cargo_units.append(cargo)  # track on both sides

                if in_truck.truck_id not in feeder_map.get(out_truck_id, []):
                    feeder_map.setdefault(out_truck_id, []).append(in_truck.truck_id)
                    out_truck.connecting_outbound.append(in_truck.truck_id)

    def _sample_cargo_unit(
        self,
        cargo_id: str,
        dest_zone: int,
        dest_truck_id: str,
    ) -> CargoUnit:
        """Sample a cargo unit with calibrated attributes."""
        perishable_frac = self.cargo_profiles.get(
            "perishable_frac", self.cfg.perishable_frac
        )
        is_perishable = bool(self.rng.random() < perishable_frac)

        sla_urgency = int(self.rng.choice([0, 1, 2], p=self._sla_weights))

        # Perishable cargo is always at least priority
        if is_perishable and sla_urgency == 0:
            sla_urgency = 1

        # Value score: log-uniform in calibrated range, normalized to [0,1]
        log_val = float(self.rng.uniform(self._log_min, self._log_max))
        value_score = (log_val - self._log_min) / (self._log_max - self._log_min + 1e-9)
        value_score = float(np.clip(value_score, 0.0, 1.0))

        # Weight: roughly 50-500 kg per unit
        weight_kg = float(max(1.0, self.rng.exponential(scale=80.0)))

        return CargoUnit(
            cargo_id=cargo_id,
            value_score=value_score,
            sla_urgency=sla_urgency,
            is_perishable=is_perishable,
            weight_kg=weight_kg,
            dest_zone=dest_zone,
            dest_truck_id=dest_truck_id,
        )


def load_cargo_profiles(calibrated_dir: str) -> dict:
    """Load cargo profiles from calibrated JSON, with fallback defaults."""
    path = os.path.join(calibrated_dir, "cargo_profiles.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print("[schedule_generator] WARNING: cargo_profiles.json not found. Using defaults.")
    return {
        "perishable_frac": 0.174,
        "sla_distribution": {0: 0.55, 1: 0.27, 2: 0.18},
        "value_percentiles": {"log_min": 0.0, "log_max": 10.0},
    }
