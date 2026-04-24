"""
SimConfig — Central configuration dataclass for the Phase 2 Simulator.

All hyperparameters are in one place. Mirrors the paper's setup (Section 6.2)
adapted for the cross-docking logistics domain.

Usage:
    cfg = SimConfig()                   # all defaults
    cfg = SimConfig(n_bays=20, alpha=0.8)  # override specific params
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class SimConfig:
    """All simulation hyperparameters in one place.

    Defaults are calibrated to produce realistic logistics behavior
    comparable to the paper's Air-East results.
    """

    # ── Reward knobs (Section 5, paper) ──────────────────────────────
    # Same optimal values as paper: α=0.75, β=0.75
    alpha: float = 0.75
    """Cargo Utility vs Operator Utility trade-off weight (α)."""

    beta: float = 0.75
    """Local reward vs Global reward weight (β)."""

    lambda_congestion: float = 0.30
    """Bay-congestion penalty weight in global reward (λ_bay).
    Scales how much bay-blockage attribution reduces R_G.
    0.30 means bay-congestion contributes at most ~30% of global signal.
    """

    # ── RL action space ───────────────────────────────────────────────
    hold_actions: List[int] = field(
        default_factory=lambda: [0, 5, 10, 15, 20, 25, 30]
    )
    """Discrete hold durations in minutes. 7 actions, matching paper."""

    # ── Hub parameters ────────────────────────────────────────────────
    n_bays: int = 15
    """Number of docking bays at the cross-docking hub.
    Logistics hubs typically have 10-30 bays; 15 creates realistic
    congestion at 80 trucks/day given ~30 min average dwell time.
    """

    operating_start: int = 360
    """Hub opens at 06:00 (360 minutes from midnight)."""

    operating_end: int = 1320
    """Hub closes at 22:00 (1320 minutes from midnight)."""

    mtt: int = 30
    """Minimum Transfer Time in minutes.
    Minimum time needed to unload from inbound and load to outbound.
    Logistics analog of MCT (Minimum Connection Time) in aviation.
    """

    otp_threshold: int = 15
    """On-Time Performance threshold in minutes.
    Arrival within 15 min of schedule = on-time (same as paper).
    """

    # ── Delay normalizers ─────────────────────────────────────────────
    delta_c: float = 1440.0
    """Max cargo delay in minutes (24h = next-cycle penalty).
    When cargo misses connection, it waits for next truck = 24h delay.
    Aviation analog: rebooking penalty (typically +120 min in paper).
    """

    delta_f: float = 60.0
    """Operator delay normalizer in minutes.
    Departure delays > delta_f are fully penalised (AU → 0).
    """

    decay_lambda: float = 0.05
    """Exponential decay rate for perishable cargo disutility.
    σ_perishable(τ) = 1 - exp(-decay_lambda * delay).
    At decay_lambda=0.05: 60-min delay → 95% disutility.
    """

    # ── Congestion parameters ─────────────────────────────────────────
    bay_congestion_threshold: float = 0.7
    """Bay utilization above this level triggers OL congestion penalty.
    At BG > 0.7 (>10.5 of 15 bays occupied), holding is extra-penalised.
    """

    lambda_bay_local: float = 0.10
    """Weight of bay-congestion penalty in local OL(τ).
    OL(τ) = schedule_term - lambda_bay_local * congestion_term.
    """

    # ── Episode parameters ────────────────────────────────────────────
    episode_days: int = 7
    """Length of one training episode in simulated days (= 1 week)."""

    trucks_per_day: int = 80
    """Number of inbound trucks per day at the hub.
    Scale: Air-East had ~460 flights/day over a full network.
    80 inbound trucks at 1 hub is comparable in HNH decision density.
    """

    # ── Driver / regulatory constraints ───────────────────────────────
    max_driver_hours: float = 660.0
    """Maximum legal driving hours per shift in minutes (= 11 hours HOS).
    L_k state variable = remaining driver hours. Agent cannot hold
    beyond this limit (negative reward imposed).
    """

    # ── Global rolling window ─────────────────────────────────────────
    global_window_minutes: float = 1440.0
    """Rolling window for global utility averages = 24 hours."""

    # ── Cargo profile defaults (overridden by calibrated data) ────────
    perishable_frac: float = 0.174
    """Fraction of cargo that is perishable (from CFS 2022: 17.4%)."""

    # ── Paths ─────────────────────────────────────────────────────────
    data_dir: str = "phase-2/data"
    calibrated_dir: str = "phase-2/simulator/calibrated"

    # ── Random seed ───────────────────────────────────────────────────
    seed: int = 42

    # ── Multi-hub network parameters ──────────────────────────────────
    n_hubs: int = 1
    """Number of hubs (1 = single-hub legacy, 10 = full FAF5 mesh)."""

    hub_ids: List[str] = field(
        default_factory=list
    )
    """Hub IDs — auto-populated from routing_matrix.json by HubChain."""

    transit_time_mean: float = 60.0
    """Mean truck transit time between adjacent hubs in minutes (~1 hour highway)."""

    transit_time_std: float = 10.0
    """Standard deviation of inter-hub transit time in minutes."""

    inter_hub_fraction: float = 0.30
    """Fraction of outbound trucks that travel to the next hub (vs local delivery).
    These become inbound feeders at the downstream hub, creating cascade effects.
    """

    def __post_init__(self):
        os.makedirs(self.calibrated_dir, exist_ok=True)

    @property
    def n_actions(self) -> int:
        return len(self.hold_actions)

    @property
    def state_dim(self) -> int:
        """State dims: 34 local + 8 network context = 42 for multi-hub."""
        return 42 if self.n_hubs > 1 else 34

    def hold_minutes(self, action_idx: int) -> int:
        """Convert discrete action index to hold minutes."""
        return self.hold_actions[action_idx]

    def action_index(self, hold_minutes: int) -> int:
        """Convert hold minutes to action index."""
        try:
            return self.hold_actions.index(int(hold_minutes))
        except ValueError:
            return 0

