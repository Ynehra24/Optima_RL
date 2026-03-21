"""
Synthetic data generators for:
  1. Tail plans (aircraft routings)
  2. PAX itineraries (passenger connection profiles)
  3. Delay distributions

These generators produce data that is *statistically representative* of real
airline operations, calibrated to match the scale parameters in AirlineProfile.

Reference: Section 6.1 of the paper.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from simulator.config import SimConfig
from simulator.models import (
    Airport,
    PaxItinerary,
    ScheduledFlight,
    TailPlan,
)


# ======================================================================
# 1. Airport Network Generator
# ======================================================================
def generate_airports(cfg: SimConfig) -> Dict[str, Airport]:
    """Create a synthetic airport network.

    Hub airports get lower MCTs (better infrastructure).
    """
    airports: Dict[str, Airport] = {}
    n = cfg.airline.num_airports
    n_hubs = cfg.airline.num_hubs

    for i in range(n):
        is_hub = i < n_hubs
        code = f"HUB{i}" if is_hub else f"SPK{i:04d}"
        mct = cfg.mct_hub if is_hub else cfg.mct_default
        airports[code] = Airport(code=code, is_hub=is_hub, mct=mct)

    return airports


# ======================================================================
# 2. Tail Plan Generator
# ======================================================================
def _flight_duration(haul: str) -> float:
    """Return a sampled scheduled flight duration (minutes)."""
    if haul == "short":
        return random.uniform(60, 120)
    elif haul == "long":
        return random.uniform(300, 480)
    else:  # medium
        return random.uniform(120, 300)


def _classify_haul(duration: float, cfg: SimConfig) -> str:
    if duration <= cfg.short_haul_max:
        return "short"
    elif duration >= cfg.long_haul_min:
        return "long"
    return "medium"


def generate_tail_plans(
    cfg: SimConfig,
    airports: Dict[str, Airport],
    day_index: int = 0,
    rng: np.random.Generator | None = None,
) -> Tuple[List[TailPlan], List[ScheduledFlight]]:
    """Generate one day of tail plans and scheduled flights.

    Each tail (aircraft) starts at a base airport, then flies a sequence
    of legs returning ideally close to a hub by end-of-day.

    Returns:
        tail_plans: list of TailPlan
        flights: list of ScheduledFlight
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed + day_index)

    airport_codes = list(airports.keys())
    hub_codes = [c for c, a in airports.items() if a.is_hub]
    spoke_codes = [c for c, a in airports.items() if not a.is_hub]

    num_tails = cfg.airline.num_aircraft
    avg_legs = cfg.avg_legs_per_tail
    flights_target = cfg.airline.flights_per_day

    # Distribute legs per tail so total ≈ flights_per_day
    # Mean legs per tail = flights_per_day / num_aircraft
    mean_legs = max(2, flights_target / num_tails)
    day_offset = day_index * 1440  # minutes offset for this day

    tail_plans: List[TailPlan] = []
    flights: List[ScheduledFlight] = []
    flight_counter = 0

    for t_idx in range(num_tails):
        tail_id = f"T{t_idx:04d}"
        # Number of legs for this tail (at least 1)
        n_legs = max(1, int(rng.poisson(mean_legs - 1)) + 1)
        # Cap to avoid unrealistic schedules
        n_legs = min(n_legs, 8)

        # Starting airport: hubs are more likely origins
        if rng.random() < 0.6:
            current_airport = rng.choice(hub_codes)
        else:
            current_airport = rng.choice(airport_codes)

        base_airport = current_airport

        # First departure time spread across the day
        first_dep = day_offset + rng.uniform(0, cfg.first_departure_spread * 1440)

        leg_ids: List[str] = []
        current_time = first_dep

        for leg_idx in range(n_legs):
            # Pick destination: hubs are more likely, especially for last leg
            if leg_idx == n_legs - 1:
                # Try to return to a hub
                dest = rng.choice(hub_codes)
            else:
                if rng.random() < 0.4:
                    dest = rng.choice(hub_codes)
                else:
                    dest = rng.choice(spoke_codes)

            # Avoid same origin-destination
            attempts = 0
            while dest == current_airport and attempts < 10:
                dest = rng.choice(airport_codes)
                attempts += 1
            if dest == current_airport:
                dest = hub_codes[0] if current_airport != hub_codes[0] else spoke_codes[0]

            # Determine haul type and duration
            haul_types = ["short", "medium", "long"]
            haul_weights = [0.4, 0.45, 0.15]
            haul = rng.choice(haul_types, p=haul_weights)
            duration = _flight_duration(haul)
            haul = _classify_haul(duration, cfg)

            scheduled_dep = current_time
            scheduled_arr = scheduled_dep + duration

            flight_id = f"F{flight_counter:05d}-D{day_index}"
            flight_number = f"F{flight_counter:05d}"

            sf = ScheduledFlight(
                flight_id=flight_id,
                flight_number=flight_number,
                origin=current_airport,
                destination=dest,
                scheduled_departure=scheduled_dep,
                scheduled_arrival=scheduled_arr,
                tail_id=tail_id,
                leg_index=leg_idx,
                haul_type=haul,
                seat_capacity=cfg.airline.avg_seats,
            )
            flights.append(sf)
            leg_ids.append(flight_id)
            flight_counter += 1

            # Turnaround time at destination
            turnaround = max(cfg.min_turnaround, rng.normal(50, 10))
            current_time = scheduled_arr + turnaround
            current_airport = dest

        tail_plans.append(TailPlan(tail_id=tail_id, legs=leg_ids, base_airport=base_airport))

    return tail_plans, flights


# ======================================================================
# 3. PAX Itinerary Generator
# ======================================================================
def _build_routing_matrix(
    flights: List[ScheduledFlight],
    airports: Dict[str, Airport],
    rng: np.random.Generator,
    connection_buffer: int = 7,
) -> Dict[str, List[Tuple[str, float]]]:
    """Build a routing matrix ρ_ij: for each arriving flight, what fraction
    of its PAX connect to each departing flight at the same airport.

    Returns a dict mapping arriving flight_id -> [(departing flight_id, fraction), ...]
    """
    # Group departing flights by airport
    departures_by_airport: Dict[str, List[ScheduledFlight]] = defaultdict(list)
    for f in flights:
        departures_by_airport[f.origin].append(f)

    # Sort departures by time at each airport
    for apt in departures_by_airport:
        departures_by_airport[apt].sort(key=lambda f: f.scheduled_departure)

    routing: Dict[str, List[Tuple[str, float]]] = {}

    for f_in in flights:
        dest_airport = f_in.destination
        is_hub = airports.get(dest_airport, Airport(code=dest_airport)).is_hub

        # Only generate connections at the arrival airport
        candidate_deps = departures_by_airport.get(dest_airport, [])
        if not candidate_deps:
            continue

        connections = []
        for f_out in candidate_deps:
            if f_out.flight_id == f_in.flight_id:
                continue
            # Connection window: time between arrival and departure
            window = f_out.scheduled_departure - f_in.scheduled_arrival
            mct = airports[dest_airport].mct if dest_airport in airports else 35
            # Valid connection: window at least MCT + buffer, max 3 hours
            if (mct + connection_buffer) <= window <= 180:
                # Weight: prefer 50-80 min windows (realistic connecting times)
                # Gaussian weighting centred at ~60 min, σ=35
                ideal = 60.0
                weight = math.exp(-((window - ideal) ** 2) / (2 * 35**2))
                if is_hub:
                    weight *= 3.0  # hubs have more connections
                connections.append((f_out.flight_id, weight))

        if connections:
            # Normalise weights to get fractions, then scale by connecting fraction
            total_w = sum(w for _, w in connections)
            if total_w > 0:
                connections = [(fid, w / total_w) for fid, w in connections]
            routing[f_in.flight_id] = connections

    return routing


def generate_pax_itineraries(
    cfg: SimConfig,
    flights: List[ScheduledFlight],
    airports: Dict[str, Airport],
    rng: np.random.Generator | None = None,
) -> List[PaxItinerary]:
    """Generate synthetic PAX itineraries.

    Two types of passengers:
      1. Direct PAX: fly a single leg (not relevant to HNH but part of load)
      2. Connecting PAX: fly two legs with a connection at an intermediate airport

    The generator uses a routing matrix to create realistic connection patterns.

    Reference: Section 6.1 "PAX profiles" in the paper.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.random_seed)

    routing = _build_routing_matrix(
        flights, airports, rng,
        connection_buffer=cfg.airline.connection_buffer if cfg.airline.connection_buffer is not None else 7,
    )

    # Build a lookup for flights
    flight_map = {f.flight_id: f for f in flights}

    pax_list: List[PaxItinerary] = []
    pax_counter = 0

    for f in flights:
        total_seats = int(f.seat_capacity * cfg.airline.avg_load_factor)
        connecting_seats = int(total_seats * cfg.airline.connecting_pax_fraction)
        direct_seats = total_seats - connecting_seats

        # --- Direct PAX ---
        # We group them into bookings of 1-3 people
        remaining = direct_seats
        while remaining > 0:
            grp = min(remaining, int(rng.integers(1, 4)))
            pax = PaxItinerary(
                pax_id=f"P{pax_counter:07d}",
                group_size=grp,
                legs=[f.flight_id],
                origin=f.origin,
                destination=f.destination,
            )
            pax_list.append(pax)
            pax_counter += 1
            remaining -= grp

        # --- Connecting PAX (incoming connections) ---
        # These are PAX arriving on some feeder flight and connecting to f
        # We'll generate them from the routing matrix perspective:
        # For each flight arriving at f.origin that routes PAX to f
        pass  # handled below

    # Now generate connecting PAX using routing matrix
    for arriving_fid, connections in routing.items():
        if arriving_fid not in flight_map:
            continue
        arr_flight = flight_map[arriving_fid]
        total_arr_seats = int(arr_flight.seat_capacity * cfg.airline.avg_load_factor)
        total_connecting = int(total_arr_seats * cfg.airline.connecting_pax_fraction)

        for dep_fid, fraction in connections:
            if dep_fid not in flight_map:
                continue
            dep_flight = flight_map[dep_fid]
            n_pax = max(1, int(total_connecting * fraction))
            # Cap to avoid overloading
            n_pax = min(n_pax, 20)

            remaining = n_pax
            while remaining > 0:
                grp = min(remaining, int(rng.integers(1, 3)))
                pax = PaxItinerary(
                    pax_id=f"P{pax_counter:07d}",
                    group_size=grp,
                    legs=[arriving_fid, dep_fid],
                    origin=arr_flight.origin,
                    destination=dep_flight.destination,
                )
                pax_list.append(pax)
                pax_counter += 1
                remaining -= grp

    return pax_list


# ======================================================================
# 4. Delay Distribution Sampler
# ======================================================================
class DelaySampler:
    """Sample intrinsic delays based on haul type and flight characteristics.

    From the paper Section 6.1:
    - Delay distributions estimated from actual vs planned times
    - When samples limited, use clustering by haul type
    - Coefficient of variance varies by short/medium/long haul
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng(cfg.random_seed)

    def sample_departure_delay(self, flight: ScheduledFlight) -> float:
        """Sample intrinsic departure delay for a flight.

        Uses a log-normal distribution that produces the right-skewed delay
        profile observed in real airline operations (many small delays, heavy
        right tail).  Parameters come from per-airline calibration (stored in
        AirlineProfile) or fall back to SimConfig defaults.
        """
        cv = self._get_cv(flight.haul_type)
        airline = self.cfg.airline
        mu = airline.departure_delay_mu if airline.departure_delay_mu is not None else self.cfg.departure_delay_mu
        base_sigma = airline.departure_delay_sigma if airline.departure_delay_sigma is not None else self.cfg.departure_delay_sigma
        sigma = base_sigma * (1 + cv)

        # Log-normal gives right-skewed distribution (many small, few large)
        raw = float(self.rng.lognormal(mu, sigma))
        return float(np.clip(raw, 0, 180))

    def sample_airtime_delay(self, flight: ScheduledFlight) -> float:
        """Sample air-time delay (can be negative = early arrival).

        Symmetric noise around the scheduled air-time to model wind,
        routing, and ATC variability.
        """
        cv = self._get_cv(flight.haul_type)
        sigma = self.cfg.airtime_delay_sigma * (1 + cv)
        delay = float(self.rng.normal(self.cfg.airtime_delay_mu, sigma))
        return float(np.clip(delay, -15, 30))

    def sample_ground_delay(self) -> float:
        """Sample ground-time delay (turnaround, taxi)."""
        delay = float(
            self.rng.normal(self.cfg.ground_delay_mu, self.cfg.ground_delay_sigma)
        )
        return max(0, delay)

    def _get_cv(self, haul_type: str) -> float:
        """Coefficient of variance by haul type."""
        if haul_type == "short":
            return self.cfg.cv_short
        elif haul_type == "long":
            return self.cfg.cv_long
        return self.cfg.cv_medium
