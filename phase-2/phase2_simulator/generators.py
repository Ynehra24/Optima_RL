"""
Synthetic data generators for the logistics cross-docking simulator.

Mirrors simulator/generators.py exactly — same 4 sections, same function
signatures, same internal structure.

1. generate_hubs()          replaces generate_airports()
2. generate_truck_plans()   replaces generate_tail_plans()
3. generate_cargo_units()   replaces generate_pax_itineraries()
4. DelaySampler             same class, logistics-renamed methods

Data source: dynamic_supply_chain_logistics_dataset.csv
  — used to seed delay distributions, cargo attributes, and route lengths
  — CSV is loaded once at module import and sliced per day_index
"""

from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phase2_simulator.config import SimConfig
from phase2_simulator.models import (
    CargoUnit,
    Hub,
    ScheduledTruck,
    TruckPlan,
)


# ---------------------------------------------------------------------------
# CSV loading — load once, slice per simulation day
# ---------------------------------------------------------------------------
_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "phase2_dataset",
    "dynamic_supply_chain_logistics_dataset.csv",
)

_dataset: Optional[pd.DataFrame] = None


def _load_dataset() -> pd.DataFrame:
    """Load (and cache) the logistics dataset CSV."""
    global _dataset
    if _dataset is None:
        _dataset = pd.read_csv(_CSV_PATH, parse_dates=["timestamp"])
        _dataset = _dataset.reset_index(drop=True)
    return _dataset


def _get_day_rows(day_index: int, rows_per_day: int = 24) -> pd.DataFrame:
    """Return the CSV rows corresponding to *day_index* (one day = 24 rows).

    The CSV has hourly rows starting 2021-01-01.  We slice 24 rows per day.
    If the index overflows the CSV, we wrap around (modulo).
    """
    df = _load_dataset()
    start = (day_index * rows_per_day) % len(df)
    end = start + rows_per_day
    if end <= len(df):
        return df.iloc[start:end].reset_index(drop=True)
    # Wrap around
    part1 = df.iloc[start:].reset_index(drop=True)
    part2 = df.iloc[: end - len(df)].reset_index(drop=True)
    return pd.concat([part1, part2], ignore_index=True)


# ======================================================================
# 1. Hub Network Generator  (replaces generate_airports)
# ======================================================================
def generate_hubs(cfg: SimConfig) -> Dict[str, Hub]:
    """Create a synthetic hub network.

    Main hubs get shorter minimum transfer times (better infrastructure).
    Mirrors generate_airports() from simulator/generators.py.
    """
    hubs: Dict[str, Hub] = {}
    n = cfg.hub.num_lanes        # total lanes ≈ num_airports
    n_main = cfg.hub.num_hubs    # main hubs ≈ hub airports

    for i in range(n):
        is_main = i < n_main
        code = f"MAIN{i}" if is_main else f"SPOKE{i:04d}"
        mtt = cfg.min_transfer_time_main if is_main else cfg.min_transfer_time_spoke
        hubs[code] = Hub(
            hub_id=code,
            is_main=is_main,
            min_transfer_time=mtt,
            num_bays=cfg.num_bays if is_main else max(4, cfg.num_bays // 4),
        )

    return hubs


# ======================================================================
# 2. Truck Plan Generator  (replaces generate_tail_plans)
# ======================================================================
def _route_duration(route_type: str) -> float:
    """Return a sampled scheduled route duration (minutes, door-to-door)."""
    if route_type == "short":
        return random.uniform(60, 180)
    elif route_type == "long":
        return random.uniform(360, 600)
    else:  # medium
        return random.uniform(180, 360)


def _classify_route(duration: float, cfg: SimConfig) -> str:
    short_max_min = cfg.short_route_max * 60 / 100   # rough conversion
    long_min_min = cfg.long_route_min * 60 / 100
    if duration <= short_max_min:
        return "short"
    elif duration >= long_min_min:
        return "long"
    return "medium"


def generate_truck_plans(
    cfg: SimConfig,
    hubs: Dict[str, Hub],
    day_index: int = 0,
    rng: np.random.Generator | None = None,
) -> Tuple[List[TruckPlan], List[ScheduledTruck]]:
    """Generate one day of truck plans and scheduled trucks.

    Directly mirrors generate_tail_plans() from simulator/generators.py.
    CSV data for the given day seeds ETA variation and congestion levels.

    Returns:
        truck_plans : list of TruckPlan
        trucks      : list of ScheduledTruck
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed + day_index)

    # Pull day's CSV slice for seeding
    day_rows = _get_day_rows(day_index)
    avg_eta_lag = float(day_rows["eta_variation_hours"].mean())  # hours
    avg_congestion = float(day_rows["traffic_congestion_level"].mean())  # 0-10

    hub_codes = list(hubs.keys())
    main_codes = [c for c, h in hubs.items() if h.is_main]
    spoke_codes = [c for c, h in hubs.items() if not h.is_main]

    num_routes = cfg.hub.num_routes
    trucks_target = cfg.hub.trucks_per_day
    mean_legs = max(2, trucks_target / num_routes)
    day_offset = day_index * 1440  # minutes offset for this day

    truck_plans: List[TruckPlan] = []
    trucks: List[ScheduledTruck] = []
    truck_counter = 0

    for r_idx in range(num_routes):
        route_id = f"R{r_idx:04d}"
        n_legs = max(1, int(rng.poisson(mean_legs - 1)) + 1)
        n_legs = min(n_legs, 8)

        # Starting hub: main hubs more likely
        if rng.random() < 0.6:
            current_hub = rng.choice(main_codes)
        else:
            current_hub = rng.choice(hub_codes)

        base_hub = current_hub
        first_dep = day_offset + rng.uniform(0, cfg.first_departure_spread * 1440)

        leg_ids: List[str] = []
        current_time = first_dep

        # Scheduled dock time = departure − loading_unloading_time proxy
        dock_lead = float(day_rows["loading_unloading_time"].median()) * 60  # convert if needed
        dock_lead = max(15.0, min(dock_lead, 90.0))  # clamp to realistic range

        for leg_idx in range(n_legs):
            # Pick destination
            if leg_idx == n_legs - 1:
                dest = rng.choice(main_codes)
            else:
                if rng.random() < 0.4:
                    dest = rng.choice(main_codes)
                else:
                    dest = rng.choice(spoke_codes)

            attempts = 0
            while dest == current_hub and attempts < 10:
                dest = rng.choice(hub_codes)
                attempts += 1
            if dest == current_hub:
                dest = main_codes[0] if current_hub != main_codes[0] else spoke_codes[0]

            # Haul type
            route_types = ["short", "medium", "long"]
            route_weights = [0.35, 0.50, 0.15]
            rtype = rng.choice(route_types, p=route_weights)
            duration = _route_duration(rtype)
            rtype = _classify_route(duration, cfg)

            # Add ETA lag noise from CSV (eta_variation_hours → minutes)
            duration_with_lag = duration + abs(avg_eta_lag) * 60 * rng.random() * 0.3

            scheduled_dep = current_time
            scheduled_dock = max(0.0, scheduled_dep - dock_lead)
            scheduled_arr = scheduled_dep + duration_with_lag

            truck_id = f"T{truck_counter:05d}-D{day_index}"
            truck_number = f"T{truck_counter:05d}"

            st = ScheduledTruck(
                truck_id=truck_id,
                truck_number=truck_number,
                origin_hub=current_hub,
                dest_hub=dest,
                scheduled_dock=scheduled_dock,
                scheduled_departure=scheduled_dep,
                scheduled_arrival=scheduled_arr,
                route_id=route_id,
                leg_index=leg_idx,
                route_type=rtype,
                cargo_capacity=cfg.hub.avg_cargo_per_truck,
            )
            trucks.append(st)
            leg_ids.append(truck_id)
            truck_counter += 1

            # Turnaround at destination hub
            turnaround = max(cfg.min_turnaround, rng.normal(45, 10))
            current_time = scheduled_arr + turnaround
            current_hub = dest

        truck_plans.append(TruckPlan(route_id=route_id, legs=leg_ids, base_hub=base_hub))

    return truck_plans, trucks


# ======================================================================
# 3. Cargo Unit Generator  (replaces generate_pax_itineraries)
# ======================================================================
def _build_transfer_matrix(
    trucks: List[ScheduledTruck],
    hubs: Dict[str, Hub],
    rng: np.random.Generator,
    transfer_buffer: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """Build a transfer matrix: for each inbound truck, what fraction of its
    cargo connects to each outbound truck at the same hub.

    Directly mirrors _build_routing_matrix() from simulator/generators.py.
    """
    # Group departing trucks by hub
    departures_by_hub: Dict[str, List[ScheduledTruck]] = defaultdict(list)
    for t in trucks:
        departures_by_hub[t.origin_hub].append(t)

    for hub_id in departures_by_hub:
        departures_by_hub[hub_id].sort(key=lambda t: t.scheduled_departure)

    transfer_map: Dict[str, List[Tuple[str, float]]] = {}

    for t_in in trucks:
        dest_hub = t_in.dest_hub
        is_main = hubs.get(dest_hub, Hub(hub_id=dest_hub)).is_main

        candidate_deps = departures_by_hub.get(dest_hub, [])
        if not candidate_deps:
            continue

        connections = []
        for t_out in candidate_deps:
            if t_out.truck_id == t_in.truck_id:
                continue
            window = t_out.scheduled_departure - t_in.scheduled_arrival
            mtt = hubs[dest_hub].min_transfer_time if dest_hub in hubs else 35
            # Valid transfer: window between (mtt + buffer) and 4 hours
            if (mtt + transfer_buffer) <= window <= 240:
                ideal = 60.0
                weight = math.exp(-((window - ideal) ** 2) / (2 * 40 ** 2))
                if is_main:
                    weight *= 3.0  # main hubs have more connecting cargo
                connections.append((t_out.truck_id, weight))

        if connections:
            total_w = sum(w for _, w in connections)
            if total_w > 0:
                connections = [(tid, w / total_w) for tid, w in connections]
            transfer_map[t_in.truck_id] = connections

    return transfer_map


def generate_cargo_units(
    cfg: SimConfig,
    trucks: List[ScheduledTruck],
    hubs: Dict[str, Hub],
    day_index: int = 0,
    rng: np.random.Generator | None = None,
) -> List[CargoUnit]:
    """Generate synthetic cargo units for a given day's truck schedule.

    Mirrors generate_pax_itineraries() from simulator/generators.py.
    Two types:
      1. Direct cargo  — single-leg, no transfer needed
      2. Connecting cargo — two legs, requires transfer at intermediate hub

    CSV columns used:
      shipping_costs          → value_score (normalised)
      risk_classification     → sla_urgency  {Low: 0, Moderate: 1, High: 2}
      iot_temperature         → is_perishable (|temp| > perishable_temp_max)
      cargo_condition_status  → is_perishable (secondary signal)
      historical_demand       → base cargo count per truck
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed + day_index)

    day_rows = _get_day_rows(day_index)

    # Pre-compute day-level stats from CSV
    max_shipping_cost = float(day_rows["shipping_costs"].max()) + 1e-6
    min_shipping_cost = float(day_rows["shipping_costs"].min())
    risk_map = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2}

    transfer_map = _build_transfer_matrix(
        trucks, hubs, rng,
        transfer_buffer=cfg.hub.transfer_buffer_minutes or 10,
    )

    truck_map = {t.truck_id: t for t in trucks}
    cargo_list: List[CargoUnit] = []
    cargo_counter = 0

    for idx, truck in enumerate(trucks):
        # Pull a representative CSV row for this truck
        csv_row = day_rows.iloc[idx % len(day_rows)]

        total_capacity = int(truck.cargo_capacity * cfg.hub.avg_load_factor)
        connecting_slots = int(total_capacity * cfg.hub.connecting_cargo_fraction)
        direct_slots = total_capacity - connecting_slots

        # Value score: normalised shipping cost
        raw_cost = float(csv_row.get("shipping_costs", 300.0))
        value_score = (raw_cost - min_shipping_cost) / (max_shipping_cost - min_shipping_cost)
        value_score = float(np.clip(value_score, 0.0, 1.0))

        # SLA urgency from risk_classification
        risk_str = str(csv_row.get("risk_classification", "Moderate Risk"))
        sla_urgency = risk_map.get(risk_str, 1)

        # Perishability: flag if temperature reading or cargo_condition_status indicates it
        iot_temp = float(csv_row.get("iot_temperature", 10.0))
        cargo_cond = float(csv_row.get("cargo_condition_status", 0.5))
        is_perishable = (abs(iot_temp) < cfg.perishable_temp_max) or (cargo_cond < 0.2)

        # --- Direct cargo ---
        remaining = direct_slots
        while remaining > 0:
            batch = min(remaining, int(rng.integers(1, 5)))
            cargo = CargoUnit(
                cargo_id=f"C{cargo_counter:08d}",
                unit_count=batch,
                legs=[truck.truck_id],
                origin_hub=truck.origin_hub,
                destination_hub=truck.dest_hub,
                value_score=value_score,
                sla_urgency=sla_urgency,
                is_perishable=bool(is_perishable),
            )
            cargo_list.append(cargo)
            cargo_counter += 1
            remaining -= batch

    # --- Connecting cargo via transfer matrix ---
    for inbound_tid, connections in transfer_map.items():
        if inbound_tid not in truck_map:
            continue
        inbound_truck = truck_map[inbound_tid]
        total_inbound = int(inbound_truck.cargo_capacity * cfg.hub.avg_load_factor)
        total_connecting = int(total_inbound * cfg.hub.connecting_cargo_fraction)

        # CSV row for this inbound truck
        inbound_idx = trucks.index(inbound_truck) if inbound_truck in trucks else 0
        csv_row = day_rows.iloc[inbound_idx % len(day_rows)]
        raw_cost = float(csv_row.get("shipping_costs", 300.0))
        v_score = (raw_cost - min_shipping_cost) / (max_shipping_cost - min_shipping_cost)
        v_score = float(np.clip(v_score, 0.0, 1.0))
        risk_str = str(csv_row.get("risk_classification", "Moderate Risk"))
        sla_urg = risk_map.get(risk_str, 1)
        iot_temp = float(csv_row.get("iot_temperature", 10.0))
        cargo_cond = float(csv_row.get("cargo_condition_status", 0.5))
        perishable = (abs(iot_temp) < cfg.perishable_temp_max) or (cargo_cond < 0.2)

        for outbound_tid, fraction in connections:
            if outbound_tid not in truck_map:
                continue
            outbound_truck = truck_map[outbound_tid]
            n_cargo = max(1, int(total_connecting * fraction))
            n_cargo = min(n_cargo, 30)

            remaining = n_cargo
            while remaining > 0:
                batch = min(remaining, int(rng.integers(1, 4)))
                cargo = CargoUnit(
                    cargo_id=f"C{cargo_counter:08d}",
                    unit_count=batch,
                    legs=[inbound_tid, outbound_tid],
                    origin_hub=inbound_truck.origin_hub,
                    destination_hub=outbound_truck.dest_hub,
                    value_score=v_score,
                    sla_urgency=sla_urg,
                    is_perishable=bool(perishable),
                )
                cargo_list.append(cargo)
                cargo_counter += 1
                remaining -= batch

    return cargo_list


# ======================================================================
# 4. Delay Distribution Sampler  (replaces DelaySampler)
# ======================================================================
class DelaySampler:
    """Sample intrinsic delays based on route type and dataset-informed parameters.

    Directly mirrors simulator/generators.py DelaySampler.
    Methods renamed for logistics domain:
      sample_departure_delay()  — same
      sample_road_delay()       — replaces sample_airtime_delay()
      sample_bay_dwell_delay()  — replaces sample_ground_delay()

    CSV columns used:
      eta_variation_hours      → scales departure delay
      traffic_congestion_level → scales road delay
      weather_condition_severity → additional road delay noise
      loading_unloading_time   → bay dwell delay baseline
      handling_equipment_availability → bay dwell delay modifier
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator | None = None,
                 day_index: int = 0):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.random_seed)
        # Pre-compute day-level CSV aggregates
        day_rows = _get_day_rows(day_index)
        self._avg_eta_lag = float(day_rows["eta_variation_hours"].mean())
        self._avg_congestion = float(day_rows["traffic_congestion_level"].mean()) / 10.0
        self._avg_weather = float(day_rows["weather_condition_severity"].mean())
        self._avg_loading = float(day_rows["loading_unloading_time"].mean())
        self._avg_equipment = float(day_rows["handling_equipment_availability"].mean())

    def sample_departure_delay(self, truck: ScheduledTruck) -> float:
        """Sample intrinsic departure delay for a truck leg.

        Log-normal distribution (same as aviation simulator) calibrated
        with ETA variation from the CSV.
        """
        cv = self._get_cv(truck.route_type)
        hub = self.cfg.hub
        mu = hub.departure_delay_mu if hub.departure_delay_mu is not None else self.cfg.departure_delay_mu
        base_sigma = hub.departure_delay_sigma if hub.departure_delay_sigma is not None else self.cfg.departure_delay_sigma
        sigma = base_sigma * (1 + cv)

        # Scale by average ETA lag from the CSV
        eta_scale = 1.0 + abs(self._avg_eta_lag) * 0.1
        raw = float(self.rng.lognormal(mu, sigma)) * eta_scale
        return float(np.clip(raw, 0, 240))

    def sample_road_delay(self, truck: ScheduledTruck) -> float:
        """Sample road-time delay (replaces airtime delay).

        Driven by traffic congestion and weather severity from the CSV.
        Can be negative (traffic lighter than expected = early arrival).
        """
        cv = self._get_cv(truck.route_type)
        sigma = self.cfg.road_delay_sigma * (1 + cv)

        # Congestion bias (higher congestion → positive delay skew)
        congestion_bias = self._avg_congestion * 10.0  # 0-10 minutes
        weather_noise = self._avg_weather * 2.0        # 0-~2 minutes

        delay = float(self.rng.normal(
            self.cfg.road_delay_mu + congestion_bias, sigma
        )) + float(self.rng.normal(0, weather_noise))
        return float(np.clip(delay, -20, 60))

    def sample_bay_dwell_delay(self) -> float:
        """Sample bay dwell delay (replaces ground delay).

        Driven by loading/unloading time and equipment availability from CSV.
        """
        # Equipment unavailability increases dwell time
        equipment_factor = max(0.1, 1.0 - self._avg_equipment)
        base_extra = self._avg_loading * equipment_factor * 0.5

        delay = float(
            self.rng.normal(self.cfg.bay_dwell_mu + base_extra, self.cfg.bay_dwell_sigma)
        )
        return max(0.0, delay)

    def _get_cv(self, route_type: str) -> float:
        """Coefficient of variance by route type."""
        if route_type == "short":
            return self.cfg.cv_short
        elif route_type == "long":
            return self.cfg.cv_long
        return self.cfg.cv_medium
