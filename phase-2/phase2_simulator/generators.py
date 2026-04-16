"""
Synthetic generators for the logistics cross-docking simulator.

Mirrors simulator/generators.py — same 4 sections, same function signatures.

1. generate_hubs()          ← generate_airports()
2. generate_truck_plans()   ← generate_tail_plans()
3. generate_cargo_units()   ← generate_pax_itineraries()
4. DelaySampler             ← DelaySampler (methods renamed for logistics)

Route duration calibration (from FAF5 NTAD dataset analysis):
  FAF5 free-flow time distribution (per link): 74% <1h, 14% 1-3h, 12% >3h
  For cross-docking routes (multi-link, regional): 55% short, 35% medium, 10% long
  Short  = 60-120 min (local delivery, ~50-100 miles @ 43 mph avg)
  Medium = 120-360 min (regional, 100-250 miles)
  Long   = 360-600 min (long-haul, 250-430 miles)

CSV columns used by this module:
  eta_variation_hours          → departure delay scale factor
  traffic_congestion_level     → road delay bias
  weather_condition_severity   → road delay noise
  loading_unloading_time       → (reference only; dock_lead from config)
  handling_equipment_availability → bay dwell modifier
  shipping_costs               → V_k (cargo value score)
  risk_classification          → X_k (SLA urgency 0/1/2)
  iot_temperature              → E_k (perishability flag)
  cargo_condition_status       → E_k (secondary perishability flag)
  historical_demand            → cargo unit count seed
  delay_probability            → F_k (downstream deadline pressure)
  fatigue_monitoring_score     → L_k (driver hours remaining)
"""

from __future__ import annotations

import math
import os
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
# CSV loading — load once, slice per day
# ---------------------------------------------------------------------------
_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "phase2_dataset",
    "dynamic_supply_chain_logistics_dataset.csv",
)

_dataset: Optional[pd.DataFrame] = None


def _load_dataset() -> pd.DataFrame:
    global _dataset
    if _dataset is None:
        _dataset = pd.read_csv(_CSV_PATH, parse_dates=["timestamp"])
        _dataset = _dataset.reset_index(drop=True)
    return _dataset


def _get_day_rows(day_index: int, rows_per_day: int = 24) -> pd.DataFrame:
    """Slice 24 rows (one day) from the CSV, wrapping around at end."""
    df = _load_dataset()
    start = (day_index * rows_per_day) % len(df)
    end = start + rows_per_day
    if end <= len(df):
        return df.iloc[start:end].reset_index(drop=True)
    part1 = df.iloc[start:].reset_index(drop=True)
    part2 = df.iloc[: end - len(df)].reset_index(drop=True)
    return pd.concat([part1, part2], ignore_index=True)


# ====================================================================
# 1. Hub Network Generator  ← generate_airports()
# ====================================================================
def generate_hubs(cfg: SimConfig) -> Dict[str, Hub]:
    """Create a synthetic hub network.

    Identical structure to generate_airports().
    num_lanes = total hubs in network (= num_airports).
    First num_hubs codes are 'MAIN' hubs (better infrastructure).
    """
    hubs: Dict[str, Hub] = {}
    n_total = cfg.hub.num_lanes
    n_main = cfg.hub.num_hubs

    for i in range(n_total):
        is_main = i < n_main
        code = f"MAIN{i}" if is_main else f"SPOKE{i:04d}"
        mtt = cfg.min_transfer_time_main if is_main else cfg.min_transfer_time_spoke
        n_bays = cfg.num_bays if is_main else max(4, cfg.num_bays // 4)
        hubs[code] = Hub(
            hub_id=code,
            is_main=is_main,
            min_transfer_time=mtt,
            num_bays=n_bays,
        )
    return hubs


# ====================================================================
# 2. Truck Plan Generator  ← generate_tail_plans()
# ====================================================================

# FAF5-calibrated route duration ranges (minutes)
_ROUTE_DURATIONS = {
    "short":  (60,  120),   # 1-2h, local delivery
    "medium": (120, 360),   # 2-6h, regional haul
    "long":   (360, 600),   # 6-10h, long-haul
}
# FAF5-calibrated route type weights: 55% short, 35% medium, 10% long
_ROUTE_WEIGHTS = [0.55, 0.35, 0.10]
_ROUTE_TYPES   = ["short", "medium", "long"]


def _sample_route_duration(route_type: str, rng: np.random.Generator) -> float:
    """Sample a route travel time (minutes) based on FAF5-calibrated ranges."""
    lo, hi = _ROUTE_DURATIONS[route_type]
    return float(rng.uniform(lo, hi))


def generate_truck_plans(
    cfg: SimConfig,
    hubs: Dict[str, Hub],
    day_index: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[TruckPlan], List[ScheduledTruck]]:
    """Generate one day's truck plans and scheduled truck legs.

    Mirrors generate_tail_plans() from simulator/generators.py.
    Key difference: route durations calibrated to FAF5 distribution.

    CSV columns used:
      eta_variation_hours → ETA noise added to scheduled_arrival
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed + day_index)

    day_rows = _get_day_rows(day_index)
    # eta_variation_hours: mean deviation of truck arrival time from schedule (CSV)
    avg_eta_lag_min = float(day_rows["eta_variation_hours"].mean()) * 60.0

    hub_codes  = list(hubs.keys())
    main_codes = [c for c, h in hubs.items() if h.is_main]
    spoke_codes = [c for c, h in hubs.items() if not h.is_main]

    num_routes   = cfg.hub.num_routes
    trucks_target = cfg.hub.trucks_per_day
    mean_legs    = max(2.0, trucks_target / num_routes)
    day_offset   = day_index * 1440  # day N starts at N×1440 sim-minutes

    truck_plans: List[TruckPlan] = []
    trucks: List[ScheduledTruck] = []
    truck_counter = 0

    for r_idx in range(num_routes):
        route_id = f"R{r_idx:04d}"
        n_legs = max(1, int(rng.poisson(mean_legs - 1)) + 1)
        n_legs = min(n_legs, 8)

        # Starting hub: main hubs preferred (60% chance)
        current_hub = (
            rng.choice(main_codes) if rng.random() < 0.6
            else rng.choice(hub_codes)
        )
        base_hub = current_hub

        # Spread first departure across cfg.first_departure_spread fraction of day
        current_time = day_offset + rng.uniform(0, cfg.first_departure_spread * 1440)

        leg_ids: List[str] = []

        for leg_idx in range(n_legs):
            # Pick destination
            if leg_idx == n_legs - 1:
                dest = rng.choice(main_codes)
            else:
                dest = (
                    rng.choice(main_codes) if rng.random() < 0.4
                    else rng.choice(spoke_codes)
                )
            # Ensure origin ≠ destination
            attempts = 0
            while dest == current_hub and attempts < 10:
                dest = rng.choice(hub_codes)
                attempts += 1
            if dest == current_hub:
                dest = main_codes[0] if current_hub != main_codes[0] else spoke_codes[0]

            # Sample route type from FAF5-calibrated distribution
            rtype = rng.choice(_ROUTE_TYPES, p=_ROUTE_WEIGHTS)
            duration = _sample_route_duration(rtype, rng)

            # Add a small ETA noise from the CSV eta_variation_hours column
            # This reflects day-level variability (traffic, weather patterns)
            eta_noise = abs(avg_eta_lag_min) * rng.random() * 0.2
            scheduled_arr = current_time + duration + eta_noise

            # scheduled_dock: truck backs into bay dock_lead_minutes BEFORE departure
            scheduled_dock = max(0.0, current_time - cfg.dock_lead_minutes)

            truck_id     = f"T{truck_counter:05d}-D{day_index}"
            truck_number = f"T{truck_counter:05d}"

            st = ScheduledTruck(
                truck_id=truck_id,
                truck_number=truck_number,
                origin_hub=current_hub,
                dest_hub=dest,
                scheduled_dock=scheduled_dock,
                scheduled_departure=current_time,
                scheduled_arrival=scheduled_arr,
                route_id=route_id,
                leg_index=leg_idx,
                route_type=rtype,
                cargo_capacity=cfg.hub.avg_cargo_per_truck,
            )
            trucks.append(st)
            leg_ids.append(truck_id)
            truck_counter += 1

            # Turnaround at destination before next leg
            # 90 min mean gives 45-min slack with min_turnaround=45 → cascade only if >45-min late
            turnaround = max(cfg.min_turnaround, float(rng.normal(90, 20)))
            current_time = scheduled_arr + turnaround
            current_hub = dest

        truck_plans.append(TruckPlan(route_id=route_id, legs=leg_ids, base_hub=base_hub))

    return truck_plans, trucks


# ====================================================================
# 3. Cargo Unit Generator  ← generate_pax_itineraries()
# ====================================================================

def _build_transfer_matrix(
    trucks: List[ScheduledTruck],
    hubs: Dict[str, Hub],
    rng: np.random.Generator,
    transfer_buffer: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    """Build transfer connection weights: for each inbound truck, which outbound
    trucks at the destination hub can receive its cargo?

    Mirrors _build_routing_matrix() from simulator/generators.py.
    Transfer window must be [mtt + buffer, 120 min].
    Reduced from 240 to 120: only "nearby" outbound trucks are candidates.
    Tighter windows mean more cargo misses when arrival delays strike.
    Closeness to 60-min ideal window is rewarded (Gaussian weight).
    """
    departures_by_hub: Dict[str, List[ScheduledTruck]] = defaultdict(list)
    for t in trucks:
        departures_by_hub[t.origin_hub].append(t)
    for hub_id in departures_by_hub:
        departures_by_hub[hub_id].sort(key=lambda t: t.scheduled_departure)

    transfer_map: Dict[str, List[Tuple[str, float]]] = {}

    for t_in in trucks:
        dest = t_in.dest_hub
        hub_obj = hubs.get(dest)
        is_main = hub_obj.is_main if hub_obj else False
        mtt     = hub_obj.min_transfer_time if hub_obj else 35

        outbound_candidates = departures_by_hub.get(dest, [])
        connections = []
        for t_out in outbound_candidates:
            if t_out.truck_id == t_in.truck_id:
                continue
            window = t_out.scheduled_departure - t_in.scheduled_arrival
            if (mtt + transfer_buffer) <= window <= 180:
                ideal_window = 60.0
                weight = math.exp(-((window - ideal_window) ** 2) / (2 * 45 ** 2))
                if is_main:
                    weight *= 2.5   # main hubs concentrate more connecting cargo
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
    rng: Optional[np.random.Generator] = None,
) -> List[CargoUnit]:
    """Generate synthetic cargo units for a given day's truck schedule.

    Mirrors generate_pax_itineraries().
    Two types:
      1. Direct cargo  — single-leg (no transfer)
      2. Connecting cargo — two-leg (requires hub transfer)

    CSV columns → cargo attributes:
      shipping_costs        → value_score (price paid = proxy for cargo value)
      risk_classification   → sla_urgency {Low=0, Moderate=1, High=2}
      iot_temperature       → is_perishable (temp outside safe range)
      cargo_condition_status → is_perishable (poor condition = fragile/perishable)
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed + day_index)

    day_rows = _get_day_rows(day_index)

    # Pre-compute normalisation bounds for value_score
    max_cost = float(day_rows["shipping_costs"].max()) + 1e-6
    min_cost = float(day_rows["shipping_costs"].min())

    # SLA urgency map from CSV categorical column
    risk_map = {"Low Risk": 0, "Moderate Risk": 1, "High Risk": 2}

    # Build truck id → list index map (O(1) lookup — fixes trucks.index() bug)
    truck_idx_map = {t.truck_id: i for i, t in enumerate(trucks)}
    truck_id_map  = {t.truck_id: t for t in trucks}

    transfer_map = _build_transfer_matrix(
        trucks, hubs, rng,
        transfer_buffer=cfg.hub.transfer_buffer_minutes or 10,
    )

    cargo_list: List[CargoUnit] = []
    cargo_counter = 0

    # ── Direct cargo (single-leg) ──────────────────────────────────────
    for idx, truck in enumerate(trucks):
        csv_row = day_rows.iloc[idx % len(day_rows)]

        total_cap = int(truck.cargo_capacity * cfg.hub.avg_load_factor)
        direct_slots = total_cap - int(total_cap * cfg.hub.connecting_cargo_fraction)

        # value_score: normalise shipping_costs to [0,1]
        raw_cost   = float(csv_row.get("shipping_costs", 300.0))
        value_score = float(np.clip((raw_cost - min_cost) / (max_cost - min_cost), 0.0, 1.0))

        # sla_urgency: from risk_classification categorical
        sla_urgency = risk_map.get(str(csv_row.get("risk_classification", "Moderate Risk")), 1)

        # is_perishable: flagged when IoT temperature is within perishable range
        # OR cargo condition score is very low
        iot_temp   = float(csv_row.get("iot_temperature", 10.0))
        cargo_cond = float(csv_row.get("cargo_condition_status", 0.5))
        is_perishable = bool(abs(iot_temp) < cfg.perishable_temp_max or cargo_cond < 0.2)

        remaining = direct_slots
        while remaining > 0:
            batch = min(remaining, int(rng.integers(1, 5)))
            cargo_list.append(CargoUnit(
                cargo_id=f"C{cargo_counter:08d}",
                unit_count=batch,
                legs=[truck.truck_id],
                origin_hub=truck.origin_hub,
                destination_hub=truck.dest_hub,
                value_score=value_score,
                sla_urgency=sla_urgency,
                is_perishable=is_perishable,
            ))
            cargo_counter += 1
            remaining -= batch

    # ── Connecting cargo (two-leg) ─────────────────────────────────────
    for inbound_tid, connections in transfer_map.items():
        inbound_truck = truck_id_map.get(inbound_tid)
        if inbound_truck is None:
            continue

        # Use truck_idx_map (O(1) dict lookup) instead of trucks.index() (O(n))
        inbound_idx = truck_idx_map.get(inbound_tid, 0)
        csv_row = day_rows.iloc[inbound_idx % len(day_rows)]

        raw_cost     = float(csv_row.get("shipping_costs", 300.0))
        value_score  = float(np.clip((raw_cost - min_cost) / (max_cost - min_cost), 0.0, 1.0))
        sla_urgency  = risk_map.get(str(csv_row.get("risk_classification", "Moderate Risk")), 1)
        iot_temp     = float(csv_row.get("iot_temperature", 10.0))
        cargo_cond   = float(csv_row.get("cargo_condition_status", 0.5))
        is_perishable = bool(abs(iot_temp) < cfg.perishable_temp_max or cargo_cond < 0.2)

        total_in  = int(inbound_truck.cargo_capacity * cfg.hub.avg_load_factor)
        total_con = int(total_in * cfg.hub.connecting_cargo_fraction)

        for outbound_tid, fraction in connections:
            outbound_truck = truck_id_map.get(outbound_tid)
            if outbound_truck is None:
                continue
            n_cargo = max(1, int(total_con * fraction))
            n_cargo = min(n_cargo, 30)
            remaining = n_cargo
            while remaining > 0:
                batch = min(remaining, int(rng.integers(1, 4)))
                cargo_list.append(CargoUnit(
                    cargo_id=f"C{cargo_counter:08d}",
                    unit_count=batch,
                    legs=[inbound_tid, outbound_tid],
                    origin_hub=inbound_truck.origin_hub,
                    destination_hub=outbound_truck.dest_hub,
                    value_score=value_score,
                    sla_urgency=sla_urgency,
                    is_perishable=is_perishable,
                ))
                cargo_counter += 1
                remaining -= batch

    return cargo_list


# ====================================================================
# 4. Delay Sampler  ← DelaySampler (methods renamed for logistics)
# ====================================================================
class DelaySampler:
    """Samples intrinsic delays for each truck leg.

    Mirrors simulator/generators.py DelaySampler.
    Methods renamed:
      sample_departure_delay()  — same
      sample_road_delay()       ← sample_airtime_delay()
      sample_bay_dwell_delay()  ← sample_ground_delay()

    CSV columns → delay parameters:
      eta_variation_hours          → departure delay scale (avg lag from schedule)
      traffic_congestion_level     → road delay additive bias (0-10 → 0-10 min)
      weather_condition_severity   → road delay noise scale
      handling_equipment_availability → bay dwell delay modifier (inverse)

    FAF5 calibration:
      road_delay_sigma = 10.0 (from speed std 6.7 mph on typical 100-mile route)
      departure_delay_mu = 0.70 (achieves ~85% OTP vs baseline 65% before fix)
    """

    def __init__(self, cfg: SimConfig, rng: Optional[np.random.Generator] = None,
                 day_index: int = 0):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.random_seed)

        day_rows = _get_day_rows(day_index)
        # eta_variation_hours: mean ETA lag across the day (hours → minutes)
        self._avg_eta_lag_min = float(day_rows["eta_variation_hours"].mean()) * 60.0
        # traffic_congestion_level: 0-10 scale; 10 = maximum congestion
        self._avg_congestion  = float(day_rows["traffic_congestion_level"].mean()) / 10.0
        # weather_condition_severity: higher = worse road conditions
        self._avg_weather     = float(day_rows["weather_condition_severity"].mean())
        # handling_equipment_availability: 0-1; low = equipment issues → longer bay dwell
        self._avg_equipment   = float(day_rows["handling_equipment_availability"].mean())

    def sample_departure_delay(self, truck: ScheduledTruck) -> float:
        """Log-normal intrinsic departure delay (same distribution as aviation).

        FAF5 calibration: mu=0.70, sigma=0.90 → median ≈ 2 min.
        ETA lag from CSV applies a small scaling factor (max +5% on delay).
        """
        cfg = self.cfg
        hub = cfg.hub
        mu    = hub.departure_delay_mu    if hub.departure_delay_mu    is not None else cfg.departure_delay_mu
        sigma = hub.departure_delay_sigma if hub.departure_delay_sigma is not None else cfg.departure_delay_sigma
        cv    = self._get_cv(truck.route_type)
        sigma_adj = sigma * (1.0 + cv)

        # Apply a small ETA lag multiplier from the CSV (max 5% extra delay)
        eta_factor = 1.0 + min(0.05, abs(self._avg_eta_lag_min) * 0.001)
        raw = float(self.rng.lognormal(mu, sigma_adj)) * eta_factor
        return float(np.clip(raw, 0.0, 240.0))

    def sample_road_delay(self, truck: ScheduledTruck) -> float:
        """Road-time delay (replaces airtime delay).

        Can be negative (lighter traffic = earlier arrival).
        FAF5: speed_std = 6.7 mph on 100-mile route → time_std ≈ 9-10 min.
        Congestion bias adds up to +10 min when maximum congestion (CSV).
        """
        cv    = self._get_cv(truck.route_type)
        sigma = self.cfg.road_delay_sigma * (1.0 + cv)

        # Congestion bias: positive (adds delay) from CSV traffic_congestion_level
        # Factor 5.0 (not 10.0): keeps mean road delay ~2-3 min to preserve OTP
        congestion_bias = self._avg_congestion * 5.0   # 0 to 5 minutes

        # Weather noise: small additional variance from CSV weather_condition_severity
        weather_std = max(0.0, self._avg_weather) * 1.5

        base  = float(self.rng.normal(self.cfg.road_delay_mu + congestion_bias, sigma))
        noise = float(self.rng.normal(0.0, weather_std))
        return float(np.clip(base + noise, -30.0, 90.0))

    def sample_bay_dwell_delay(self) -> float:
        """Bay dwell delay (replaces ground delay).

        Driven by handling equipment availability from CSV.
        Low equipment → more manual handling → longer dwell.
        """
        # equipment_factor: 0=perfect, 1=completely unavailable
        equipment_factor = max(0.0, 1.0 - self._avg_equipment)
        extra_wait = equipment_factor * 6.0   # up to 6 extra minutes

        delay = float(
            self.rng.normal(self.cfg.bay_dwell_mu + extra_wait, self.cfg.bay_dwell_sigma)
        )
        return max(0.0, delay)

    def _get_cv(self, route_type: str) -> float:
        if route_type == "short":
            return self.cfg.cv_short
        elif route_type == "long":
            return self.cfg.cv_long
        return self.cfg.cv_medium
