"""
Configuration and hyperparameters for the logistics cross-docking simulator.

Two hub profiles are defined:
  - HUB_SMALL : ~400 truck movements/day, 1 hub, moderate throughput
  - HUB_LARGE : ~1 400 truck movements/day, 4 hubs, high throughput

Direct counterpart of simulator/config.py.
Aviation → Logistics term mapping (for reference):
  AirlineProfile          → HubProfile
  flights_per_day         → trucks_per_day
  num_aircraft            → num_routes
  num_airports            → num_hubs
  connecting_pax_fraction → connecting_cargo_fraction
  baseline_misconnect_rate→ baseline_failed_transfer_rate
  baseline_otp            → baseline_schedule_otp
  departure_delay_mu/sigma→ departure_delay_mu/sigma  (same mechanics)
  mct_hub / mct_default   → transfer_slack_main / transfer_slack_spoke
  delta_p                 → delta_C  (cargo delay cap)
  delta_f                 → delta_F  (operator delay cap)
  alpha                   → alpha   (cargo vs operator trade-off)
  beta                    → beta    (local vs global trade-off)
  NEW: lambda_congestion, B_thresh, T_sla (logistics-only, from PDF §7)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Hub profile — captures the scale of a cross-docking hub network
# ---------------------------------------------------------------------------
@dataclass
class HubProfile:
    """Scale parameters for a synthetic logistics hub network."""

    name: str
    trucks_per_day: int           # total truck movements (in + out) per day
    num_routes: int               # number of distinct truck routes (≈ aircraft)
    num_hubs: int                 # number of cross-docking hubs in the network
    num_lanes: int                # total origin/destination lanes (≈ airports)

    # Fraction of inbound cargo that needs to transfer to an outbound truck
    connecting_cargo_fraction: float = 0.35

    # Average load factor (fraction of truck capacity filled)
    avg_load_factor: float = 0.80

    # Average cargo units per truck
    avg_cargo_per_truck: int = 120

    # Baseline failed-transfer rate (fraction of connecting cargo that misses)
    baseline_failed_transfer_rate: float = 0.05

    # Baseline schedule OTP (outbound trucks departing within 15 min of schedule)
    baseline_schedule_otp: float = 0.85

    # Per-hub departure delay distribution overrides (None = use SimConfig defaults)
    departure_delay_mu: float | None = None
    departure_delay_sigma: float | None = None

    # Transfer slack above minimum (analogous to connection_buffer in aviation)
    transfer_buffer_minutes: int | None = None


# Pre-defined hub profiles
HUB_SMALL = HubProfile(
    name="Hub-Small",
    trucks_per_day=400,
    num_routes=120,
    num_hubs=1,
    num_lanes=80,
    connecting_cargo_fraction=0.35,
    avg_load_factor=0.80,
    avg_cargo_per_truck=120,
    baseline_failed_transfer_rate=0.05,
    baseline_schedule_otp=0.85,
    departure_delay_mu=1.10,       # calibrated for OTP ≈ 85 %
    departure_delay_sigma=1.0,
    transfer_buffer_minutes=10,
)

HUB_LARGE = HubProfile(
    name="Hub-Large",
    trucks_per_day=1400,
    num_routes=500,
    num_hubs=4,
    num_lanes=280,
    connecting_cargo_fraction=0.40,
    avg_load_factor=0.82,
    avg_cargo_per_truck=140,
    baseline_failed_transfer_rate=0.08,
    baseline_schedule_otp=0.82,
    departure_delay_mu=1.35,
    departure_delay_sigma=1.0,
    transfer_buffer_minutes=5,
)


# ---------------------------------------------------------------------------
# Simulator configuration
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    """All tuneable knobs for the logistics cross-docking microsimulator.

    Mirrors simulator/config.py field-for-field.
    New logistics-only fields (from PDF §7) are grouped at the bottom.
    """

    hub: HubProfile = field(default_factory=lambda: HUB_SMALL)

    # --- Simulation time ---
    num_days: int = 7                   # days to simulate per episode
    epoch_length_hours: float = 24.0    # 1 epoch = 1 day (schedule repeats daily)
    random_seed: int = 42

    # --- Hold action space (discrete, minutes) ---
    # Inherited directly from Phase 1 (PDF §7: {0,5,10,15,20,25,30})
    hold_actions: List[int] = field(
        default_factory=lambda: [0, 5, 10, 15, 20, 25, 30]
    )
    max_hold_minutes: int = 30

    # --- Delay distributions (log-normal parameters for departure delay) ---
    # Same mechanics as aviation; log-normal gives right-skewed profile
    departure_delay_mu: float = 1.10
    departure_delay_sigma: float = 1.0
    # Road-time delay (replaces airtime_delay): normal, can be negative = early
    road_delay_mu: float = 0.0
    road_delay_sigma: float = 5.0
    # Bay dwell delay (replaces ground_delay): normal
    bay_dwell_mu: float = 0.0
    bay_dwell_sigma: float = 2.0

    # --- Route distance classification thresholds (km) ---
    short_route_max: int = 150          # ≤ 150 km
    long_route_min: int = 500           # ≥ 500 km
    # Coefficient of variance multiplier per route length
    cv_short: float = 0.10
    cv_medium: float = 0.07
    cv_long: float = 0.04

    # --- Transfer timing ---
    # Minimum transfer time at hubs (analogous to MCT at airports)
    min_transfer_time_main: int = 20    # main hub (better ops, shorter dwell)
    min_transfer_time_spoke: int = 35   # spoke hub (default)

    # --- Reward / utility normalisation ---
    # Δ_C: max tolerable cargo delivery delay (minutes) — replaces delta_p
    delta_C: float = 480.0             # 8 hours
    # Δ_F: max tolerable operator departure delay — replaces delta_f
    delta_F: float = 60.0

    # --- RL knobs (PDF §7 — same defaults as Phase 1) ---
    alpha: float = 0.75                # cargo-utility vs operator-utility
    beta: float = 0.75                 # local vs global reward

    # --- Global utility window ---
    global_window_hours: float = 24.0  # W = 24 h (PDF §7)

    # --- On-time threshold (minutes) ---
    ontime_threshold: float = 15.0

    # --- Route plan ---
    avg_legs_per_route: int = 4        # truck legs per day per route
    min_turnaround: int = 30           # minimum turnaround at hub (minutes)
    first_departure_spread: float = 0.6

    # -----------------------------------------------------------------------
    # NEW logistics-only knobs (no aviation analog — PDF §7)
    # -----------------------------------------------------------------------
    # λ: congestion sensitivity weight in OL(τ) formula
    lambda_congestion: float = 0.30

    # B_thresh: bay utilisation threshold above which congestion penalty fires
    B_thresh: float = 0.80

    # T_sla: SLA tolerance window (replaces 15-min grace period, in minutes)
    # Set low for express tiers, higher for standard freight
    T_sla: float = 30.0

    # Number of docking bays at the hub (for BG calculation)
    num_bays: int = 20

    # SLA urgency levels: {0: standard, 1: next-day, 2: same-day express}
    sla_urgency_levels: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Perishability temperature threshold (°C) — cargo flagged if IoT reading
    # is outside [-5, T_perishable_max]
    perishable_temp_max: float = 8.0

    # Next-cycle penalty (minutes) when cargo misses its outbound truck
    # Equivalent to passenger rebooking delay in aviation
    next_cycle_penalty_minutes: float = 1440.0  # 24 hours

    # --- Reproducibility ---
    def copy(self, **overrides) -> "SimConfig":
        import dataclasses
        return dataclasses.replace(self, **overrides)
