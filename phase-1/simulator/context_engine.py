"""
Context Engine — generates the state vector presented to the RL agent.

The state has 5 components (Section 4 of the paper):
  PL  : Forecasted local PAX utility for various hold times τ
  AL  : Forecasted local Airline utility for various hold times τ
  P_G : Actual global PAX utility (measured over past window W)
  A_G : Actual global Airline utility (measured over past window W)
  τ*  : Helper variable — locally optimal hold time

This module is the *interface* for Yatharth's state‐representation work;
the simulator fills in the raw data and the context engine computes
the derived values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator.config import SimConfig
from simulator.models import FlightState, PaxItinerary, ScheduledFlight


@dataclass
class FlightContext:
    """The full context / state vector for one HNH decision."""

    flight_id: str

    # Local PU vector  PL(τ) for each hold action
    PL: List[float] = field(default_factory=list)
    # Local AU vector  AL(τ) for each hold action
    AL: List[float] = field(default_factory=list)
    # Global PU  (scalar)
    PG: float = 0.0
    # Global AU  (scalar)
    AG: float = 0.0
    # Helper variable τ*
    tau_star: float = 0.0

    def to_array(self) -> np.ndarray:
        """Flatten to a 1-D numpy vector for the RL agent.

        Layout: [PL(τ0)..PL(τN), AL(τ0)..AL(τN), PG, AG, τ*]
        """
        return np.array(self.PL + self.AL + [self.PG, self.AG, self.tau_star],
                        dtype=np.float32)

    @property
    def state_dim(self) -> int:
        return len(self.PL) * 2 + 3


class ContextEngine:
    """Computes the state vector from raw simulator data.

    The context engine sits between the simulator and the RL agent.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        # Ring buffer of recent global utilities for window W
        self._global_pu_history: List[Tuple[float, float]] = []  # (time, pu)
        self._global_au_history: List[Tuple[float, float]] = []  # (time, au)

    # -----------------------------------------------------------------
    # Local PU vector
    # -----------------------------------------------------------------
    def compute_local_pu(
        self,
        flight_state: FlightState,
        connecting_pax: List[PaxItinerary],
        hold_actions: List[int],
        airports: Dict,
        flight_map: Dict[str, FlightState],
    ) -> List[float]:
        """Compute PL(τ) for each possible hold action τ.

        For each τ, estimate whether each connecting PAX makes or misses
        their connection.  PU = avg(1 - σ_i(τ)) across all PAX.

        σ_i(τ) = 0                           if delay ≤ 15 min
                  min(delay, Δ_P) / Δ_P      otherwise
        """
        if not connecting_pax:
            return [1.0] * len(hold_actions)

        connection_airport = flight_state.flight.origin
        apt = airports.get(connection_airport)
        mct = getattr(apt, "mct", self.cfg.mct_default)

        pu_values = []
        for tau in hold_actions:
            utilities = []
            for pax in connecting_pax:
                delay = self._estimate_pax_delay(
                    pax, flight_state, tau, mct, flight_map
                )
                sigma = self._pax_disutility(delay)
                utilities.append(1.0 - sigma)
            pu_values.append(float(np.mean(utilities)) if utilities else 1.0)
        return pu_values

    # -----------------------------------------------------------------
    # Local AU vector
    # -----------------------------------------------------------------
    def compute_local_au(
        self,
        flight_state: FlightState,
        hold_actions: List[int],
    ) -> List[float]:
        """Compute AL(τ) for each possible hold action.

        AU(τ) = 1 - max(0, delay_caused_by_hold) / Δ_F
        Key fix: only penalise the MARGINAL delay from holding, not the
        pre-existing intrinsic delay.  Otherwise AU always peaks at τ=0
        regardless of context, biasing every agent to never hold.
        """
        # How much arrival delay already exists without any hold
        base_delay = flight_state.total_arrival_delay
        # OTP threshold slack: how much more delay can the flight absorb
        # before breaching the 15-min on-time threshold
        slack = max(0.0, self.cfg.ontime_threshold - base_delay)

        au_values = []
        for tau in hold_actions:
            # Only the part of τ that exceeds the slack penalises AU
            marginal_delay = max(0.0, tau - slack)
            au = 1.0 - min(marginal_delay, self.cfg.delta_f) / self.cfg.delta_f
            au_values.append(au)
        return au_values

    # -----------------------------------------------------------------
    # Global PU & AU
    # -----------------------------------------------------------------
    def compute_global_pu(self, current_time: float) -> float:
        """Average PU over the past W=24 hours."""
        window_start = current_time - self.cfg.global_window_hours * 60
        vals = [pu for t, pu in self._global_pu_history if t >= window_start]
        return float(np.mean(vals)) if vals else 1.0

    def compute_global_au(self, current_time: float) -> float:
        """Average AU over the past W=24 hours."""
        window_start = current_time - self.cfg.global_window_hours * 60
        vals = [au for t, au in self._global_au_history if t >= window_start]
        return float(np.mean(vals)) if vals else 1.0

    def record_global_pu(self, time: float, pu: float) -> None:
        self._global_pu_history.append((time, pu))

    def record_global_au(self, time: float, au: float) -> None:
        self._global_au_history.append((time, au))

    # -----------------------------------------------------------------
    # Helper variable τ*
    # -----------------------------------------------------------------
    def compute_tau_star(
        self, PL: List[float], AL: List[float], hold_actions: List[int]
    ) -> float:
        """τ* = argmax_τ (α * PL(τ) + (1-α) * AL(τ))"""
        alpha = self.cfg.alpha
        scores = [alpha * pl + (1 - alpha) * al for pl, al in zip(PL, AL)]
        best_idx = int(np.argmax(scores))
        return float(hold_actions[best_idx])

    # -----------------------------------------------------------------
    # Full context for a flight
    # -----------------------------------------------------------------
    def build_context(
        self,
        flight_state: FlightState,
        connecting_pax: List[PaxItinerary],
        hold_actions: List[int],
        airports: Dict,
        flight_map: Dict[str, FlightState],
        current_time: float,
    ) -> FlightContext:
        PL = self.compute_local_pu(
            flight_state, connecting_pax, hold_actions, airports, flight_map
        )
        AL = self.compute_local_au(flight_state, hold_actions)
        PG = self.compute_global_pu(current_time)
        AG = self.compute_global_au(current_time)
        tau_star = self.compute_tau_star(PL, AL, hold_actions)

        return FlightContext(
            flight_id=flight_state.flight.flight_id,
            PL=PL, AL=AL, PG=PG, AG=AG, tau_star=tau_star,
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------
    def _estimate_pax_delay(
        self,
        pax: PaxItinerary,
        outbound_flight: FlightState,
        hold_tau: int,
        mct: int,
        flight_map: Dict[str, FlightState],
    ) -> float:
        """Estimate delay to final destination for a PAX given hold τ.

        Parameters
        ----------
        pax : PaxItinerary
            The connecting passenger (must have legs[0] = inbound, legs[1] = outbound)
        outbound_flight : FlightState
            The OUTBOUND (connecting) flight that we're deciding to hold
        hold_tau : int
            The hold duration in minutes
        mct : int
            Minimum connection time at the hub
        flight_map : Dict[str, FlightState]
            Map of all flight states

        Logic
        -----
        1. Get the inbound flight from pax.legs[0]
        2. Estimate inbound's arrival time (using total_arrival_delay)
        3. Compute connection window = outbound departure - inbound arrival
        4. If window >= mct: PAX makes connection → delay = outbound's delay
           If window < mct: PAX misses → rebooked with +120 min penalty
        """
        if len(pax.legs) < 2:
            return 0.0

        # Get the inbound (feeder) flight
        inbound_fid = pax.legs[0]
        inbound_fs = flight_map.get(inbound_fid)
        if inbound_fs is None:
            return 0.0

        # Estimate inbound arrival time
        inbound_est_arrival = (
            inbound_fs.flight.scheduled_arrival
            + inbound_fs.total_arrival_delay
        )

        # Outbound departure time: scheduled + intrinsic delay + hold
        # Key fix: include the flight's real departure delay, not just τ
        outbound_intrinsic = max(
            outbound_flight.intrinsic_departure_delay,
            outbound_flight.propagated_departure_delay,
        )
        outbound_est_dep = (
            outbound_flight.flight.scheduled_departure
            + outbound_intrinsic
            + outbound_flight.ground_departure_delay
            + hold_tau
        )

        # Connection window
        connection_window = outbound_est_dep - inbound_est_arrival

        if connection_window >= mct:
            # PAX makes the connection
            # Their delay to final destination = outbound's delay + the hold we impose
            outbound_delay = outbound_flight.total_arrival_delay + hold_tau
            return max(0, outbound_delay)
        else:
            # PAX misses and gets rebooked
            # Penalty: assume rebook adds ~120 min to their trip
            return 120.0

    def _estimate_flight_arrival_delay(
        self, flight_state: FlightState, hold_tau: int
    ) -> float:
        """Estimated arrival delay of the flight if held by τ minutes."""
        base_delay = flight_state.total_arrival_delay
        return base_delay + hold_tau

    def _pax_disutility(self, delay: float) -> float:
        """σ_i(τ) as defined in the paper."""
        if delay <= self.cfg.ontime_threshold:
            return 0.0
        return min(delay, self.cfg.delta_p) / self.cfg.delta_p

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------
    def reset(self) -> None:
        self._global_pu_history.clear()
        self._global_au_history.clear()
