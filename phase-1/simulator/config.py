"""
Configuration and hyperparameters for the airline network microsimulator.

Two airline profiles are defined matching the paper:
  - Air-East: ~460 flights/day, 130+ aircraft, 130+ destinations, 1 hub
  - Air-West: ~1600 flights/day, 800+ aircraft, 340+ destinations, 8 hubs
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Airline profile – captures the scale of an airline network
# ---------------------------------------------------------------------------
@dataclass
class AirlineProfile:
    """Scale parameters for a synthetic airline."""

    name: str
    flights_per_day: int
    num_aircraft: int
    num_airports: int
    num_hubs: int
    # Average connecting PAX fraction (of total PAX that are connecting)
    connecting_pax_fraction: float = 0.30
    # Average load factor (fraction of seats filled)
    avg_load_factor: float = 0.85
    # Average seats per aircraft
    avg_seats: int = 160
    # Baseline missed-connection rate (fraction of connecting PAX)
    baseline_misconnect_rate: float = 0.03
    # Average OTP (on-time performance, arrival within 15 min)
    baseline_otp: float = 0.88
    # Per-airline delay distribution overrides (None = use SimConfig defaults)
    departure_delay_mu: float | None = None
    departure_delay_sigma: float | None = None
    # Per-airline connection buffer override (minutes above MCT)
    connection_buffer: int | None = None


# Pre-defined airline profiles from the paper
AIR_EAST = AirlineProfile(
    name="Air-East",
    flights_per_day=460,
    num_aircraft=130,
    num_airports=130,
    num_hubs=1,
    connecting_pax_fraction=0.30,
    avg_load_factor=0.85,
    avg_seats=160,
    baseline_misconnect_rate=0.03,
    baseline_otp=0.885,
    departure_delay_mu=1.05,        # calibrated for OTP ≈ 88.5%
    departure_delay_sigma=1.0,
    connection_buffer=7,            # MCT + 7 min
)

AIR_WEST = AirlineProfile(
    name="Air-West",
    flights_per_day=1600,
    num_aircraft=800,
    num_airports=340,
    num_hubs=8,
    connecting_pax_fraction=0.30,
    avg_load_factor=0.85,
    avg_seats=160,
    baseline_misconnect_rate=0.055,
    baseline_otp=0.86,
    departure_delay_mu=1.30,        # calibrated for OTP ≈ 86%
    departure_delay_sigma=1.0,
    connection_buffer=1,            # tight connections at multi-hub network
)


# ---------------------------------------------------------------------------
# Simulator configuration
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    """All tuneable knobs for the microsimulator."""

    airline: AirlineProfile = field(default_factory=lambda: AIR_EAST)

    # --- Simulation time ---
    num_days: int = 7                   # how many days to simulate per episode
    epoch_length_hours: float = 24.0    # 1 epoch = 1 day (flight schedule repeats daily)
    random_seed: int = 42

    # --- Hold action space (discrete, in minutes) ---
    hold_actions: List[int] = field(
        default_factory=lambda: [0, 5, 10, 15, 20, 25, 30]
    )
    max_hold_minutes: int = 30          # maximum permissible hold

    # --- Delay distributions (minutes) ---
    # Intrinsic departure delay: log-normal parameters
    # lognormal(1.05, 1.0) → median ≈2.9 min, mean ≈4.7 min
    # Calibrated: Air-East OTP ≈ 88.7%, misconnect ≈ 3.2%  (paper: 88.5%, 3.0%)
    departure_delay_mu: float = 1.05    # log-mean
    departure_delay_sigma: float = 1.0  # log-std
    # Airtime delay: normal parameters (can be negative = early arrival)
    airtime_delay_mu: float = 0.0       # symmetric noise around scheduled air-time
    airtime_delay_sigma: float = 3.0    # std (minutes)
    # Ground time delay (turnaround variability): normal
    ground_delay_mu: float = 0.0
    ground_delay_sigma: float = 1.0

    # --- Flight haul classification thresholds (in minutes of scheduled duration) ---
    short_haul_max: int = 120           # <=2 h
    long_haul_min: int = 300            # >=5 h
    # Coefficient of variance multiplier per haul type
    cv_short: float = 0.08
    cv_medium: float = 0.06
    cv_long: float = 0.04

    # --- PAX ---
    # Minimum Connection Time at airports (minutes)
    # Realistic MCTs: domestic hub ~25 min, spoke ~35 min
    mct_default: int = 35              # default MCT at spoke airports
    mct_hub: int = 25                  # MCT at hubs (lower, better infrastructure)

    # --- Reward / utility normalisation (used by context engine) ---
    delta_p: float = 240.0             # PAX delay cap (minutes) for PU normalisation
    delta_f: float = 60.0              # flight delay cap (minutes) for AU normalisation

    # --- RL knobs (defaults from paper) ---
    alpha: float = 0.75                # PAX-vs-AU trade-off
    beta: float = 0.75                 # local-vs-global trade-off

    # --- Global utility window ---
    global_window_hours: float = 24.0  # W in the paper

    # --- On-time threshold (minutes) ---
    ontime_threshold: float = 15.0

    # --- Tail-plan ---
    # Average number of flights (legs) per tail per day
    avg_legs_per_tail: int = 4
    # Minimum turnaround time between consecutive legs (minutes)
    min_turnaround: int = 35
    # Spread of first departure across the day (fraction of 24 h)
    first_departure_spread: float = 0.6

    # --- Reproducibility ---
    def copy(self, **overrides) -> "SimConfig":
        import dataclasses
        return dataclasses.replace(self, **overrides)
