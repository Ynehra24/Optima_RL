"""
Configuration and hyperparameters for the logistics cross-docking simulator.

Two hub profiles:
  HUB_SMALL : ~400 truck movements/day, 1 main hub, 20 bays
  HUB_LARGE : ~1400 truck movements/day, 4 main hubs, 60 bays

CALIBRATION SOURCE — FAF5 (NTAD Freight Analysis Framework):
  487,394 US road links analysed:
    Speed_Limit mean = 47.5 mph
    AB_FinalSpeed mean = 43.2 mph  (actual achieved speed)
    Speed loss mean = 4.3 mph, std = 6.7 mph
    Free-flow travel time: 74% of links < 1h, 14% at 1-3h, 12% > 3h
  These values drive road_delay_sigma and route duration thresholds below.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# HubProfile  (aviation analog: AirlineProfile)
# ---------------------------------------------------------------------------
@dataclass
class HubProfile:
    """Scale parameters for a synthetic logistics hub network.

    Maps to AirlineProfile field-for-field:
      trucks_per_day          ← flights_per_day
      num_routes              ← num_aircraft
      num_hubs                ← num_hubs
      num_lanes               ← num_airports
      connecting_cargo_fraction ← connecting_pax_fraction
      avg_load_factor         ← avg_load_factor
      avg_cargo_per_truck     ← avg_seats
      baseline_failed_transfer_rate ← baseline_misconnect_rate
      baseline_schedule_otp   ← baseline_otp
    """

    name: str

    # Network scale
    trucks_per_day: int           # total outbound truck legs per day across all routes
    num_routes: int               # number of truck route chains (= num_aircraft / tail plans)
    num_hubs: int                 # number of cross-docking hubs in the network
    num_lanes: int                # origin + destination lanes (= num_airports)

    # Cargo profile
    connecting_cargo_fraction: float = 0.35   # fraction of cargo that needs a transfer
    avg_load_factor: float = 0.80             # fraction of cargo capacity filled
    avg_cargo_per_truck: int = 120            # total cargo unit slots per truck

    # Baselines (used by validation.py to check plausibility)
    baseline_failed_transfer_rate: float = 0.05
    baseline_schedule_otp: float = 0.85

    # Delay distribution overrides (None → use SimConfig global defaults)
    departure_delay_mu: float | None = None
    departure_delay_sigma: float | None = None

    # Transfer buffer above minimum (= connection_buffer in aviation)
    transfer_buffer_minutes: int | None = None


# Pre-defined profiles
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
    departure_delay_mu=0.55,     # recalibrated: OTP ≈ 85 % (was 0.70 → 81%)
    departure_delay_sigma=0.85,
    transfer_buffer_minutes=10,  # increased from 5 to reduce failed transfers back to ~5%
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
    departure_delay_mu=0.82,     # increased from 0.75 to hit 82% OTP target
    departure_delay_sigma=0.85,
    transfer_buffer_minutes=10,  # increased from 5 to reduce failed transfers
)


# ---------------------------------------------------------------------------
# SimConfig  (mirrors simulator/config.py SimConfig exactly)
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    """All tuneable knobs for the logistics cross-docking microsimulator.

    CALIBRATION NOTES (against FAF5 dataset):
      departure_delay_mu  = 0.70  → lognormal median ≈ 2.0 min → OTP ≈ 85 %
      road_delay_sigma    = 10.0  → from FAF5 speed std (6.7 mph on 100-mile route
                                     ≈ 9-min travel-time std → 10 min chosen)
      short_route_max     = 120   → 2h (FAF5: short-haul clusters at 60-120 min)
      long_route_min      = 360   → 6h (FAF5: long-haul > 360 min, 10% of routes)
    """

    hub: HubProfile = field(default_factory=lambda: HUB_SMALL)

    # ── Simulation time ────────────────────────────────────────────────────
    num_days: int = 7
    epoch_length_hours: float = 24.0
    random_seed: int = 42

    # ── Hold action space {0,5,10,15,20,25,30} min (PDF §7 — identical to Phase 1)
    hold_actions: List[int] = field(
        default_factory=lambda: [0, 5, 10, 15, 20, 25, 30]
    )
    max_hold_minutes: int = 30

    # ── Delay distributions ────────────────────────────────────────────────
    # Departure delay: recalibrated from 0.70 to 0.55 to close OTP gap (81% → 85%)
    # lognormal(0.55, 0.85): median=1.73 min, P(>15)=0.8% → OTP ~85% after cascades
    departure_delay_mu: float = 0.55
    departure_delay_sigma: float = 0.85

    # Road delay: reverted to 10.0 to reduce arrival variance and lower failed transfers
    # FAF5: speed_std=6.7mph on 100-mile route → time_std ≈ 9-12 min
    road_delay_mu: float = 0.0
    road_delay_sigma: float = 10.0

    # Bay dwell delay: reduced sigma 4 → 2.5 (was adding too much departure delay noise)
    # bay_dwell only adds to DEPARTURE delay; keeping it smaller preserves OTP
    bay_dwell_mu: float = 0.0
    bay_dwell_sigma: float = 2.5

    # ── Route classification thresholds (minutes door-to-door) ───────────
    # FAF5 calibration: 55 % short, 35 % medium, 10 % long (cross-docking reality)
    short_route_max: int = 120     # ≤ 2 h
    long_route_min: int = 360      # ≥ 6 h
    # Coefficient of variance by route length
    cv_short: float = 0.12
    cv_medium: float = 0.08
    cv_long: float = 0.05

    # ── Transfer timing (= MCT analogs) ──────────────────────────────────
    # Increased from 20/35 to 25/40 to create slightly tighter transfer windows 
    # but not as extreme as 30/50.
    min_transfer_time_main: int = 25    
    min_transfer_time_spoke: int = 40

    # ── Reward / utility normalisation (PDF §5) ───────────────────────────
    delta_C: float = 480.0    # Δ_C — max tolerable cargo delivery delay (min); replaces delta_p
    delta_F: float = 60.0     # Δ_F — max tolerable operator departure delay (min); replaces delta_f

    # ── RL knobs (PDF §7, identical to Phase 1 defaults) ─────────────────
    alpha: float = 0.75     # cargo utility vs operator utility trade-off
    beta: float = 0.75      # local vs global reward trade-off

    # ── Global utility window ─────────────────────────────────────────────
    global_window_hours: float = 24.0   # W = 24 h (PDF §7)

    # ── On-time threshold (minutes) ───────────────────────────────────────
    ontime_threshold: float = 15.0

    # ── Route plan ────────────────────────────────────────────────────────
    avg_legs_per_route: int = 4
    min_turnaround: int = 45          # min turnaround at hub (minutes)
    first_departure_spread: float = 0.6
    dock_lead_minutes: float = 45.0   # cross-docking unload time before departure

    # ── Bay management ────────────────────────────────────────────────────
    num_bays: int = 20

    # ── NEW logistics-only knobs (PDF §7, no aviation analog) ────────────
    lambda_congestion: float = 0.30    # λ: bay-blockage penalty weight in OL(τ)
    B_thresh: float = 0.80             # B_thresh: BG above which congestion penalty activates
    bay_curfew_threshold: float = 0.95 # hard cap: if BG > this, force hold = 0

    T_sla: float = 30.0                # SLA tolerance window (replaces 15-min grace, min)
    sla_urgency_levels: List[int] = field(default_factory=lambda: [0, 1, 2])

    # Perishability: cargo flagged when IoT temp outside safe range (CSV: iot_temperature)
    perishable_temp_max: float = 8.0
    # Perishable exponential decay rate (higher = faster penalty growth with delay)
    perishable_decay_rate: float = 0.004

    # Next-cycle penalty for missed transfer (24h = logistics equivalent of rebook delay)
    next_cycle_penalty_minutes: float = 1440.0

    # ── Reproducibility ───────────────────────────────────────────────────
    def copy(self, **overrides) -> "SimConfig":
        import dataclasses
        return dataclasses.replace(self, **overrides)
