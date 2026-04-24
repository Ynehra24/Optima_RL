"""
cargo_manager.py — Transfer logic and CL(τ)/OL(τ) vector computation.

Handles:
  1. Transfer check: does a cargo unit make its connection given hold τ?
  2. CL(τ) vector: cargo utility per hold action (with value, SLA, perishability)
  3. OL(τ) vector: operator utility per hold action (with bay-congestion penalty)

This is the direct analog of Phase 1's passenger utility computation,
but enriched with logistics-specific attributes:
  - value_score: high-value cargo penalized more when delayed
  - sla_urgency: express cargo has amplified disutility (1+X_k multiplier)
  - is_perishable: exponential (not linear) disutility function
  - bay-congestion: OL penalized when hub is congested
"""

from __future__ import annotations
import math
from typing import List, Optional
import numpy as np

from simulator.config import SimConfig
from simulator.schedule_generator import CargoUnit, TruckSchedule


class CargoManager:
    """Computes transfer outcomes and utility vectors for the RL agent.

    All utility values are in [0, 1]:
      - 1.0 = perfect outcome (all cargo transferred, on time)
      - 0.0 = worst outcome (all cargo missed, maximum delay)
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    # ── Transfer Check ─────────────────────────────────────────────────

    def cargo_makes_transfer(
        self,
        cargo: CargoUnit,
        inbound_actual_arrival: float,
        outbound_departure: float,
    ) -> bool:
        """Check if a cargo unit makes its connection.

        Transfer succeeds if:
            inbound_actual_arrival + MTT <= outbound_departure

        Logistics analog of Phase 1's MCT (minimum connection time) check.

        Args:
            cargo: The cargo unit being transferred
            inbound_actual_arrival: When the inbound truck actually arrives
            outbound_departure: Scheduled + hold time of the outbound truck

        Returns:
            True if cargo successfully transfers
        """
        connection_window = outbound_departure - inbound_actual_arrival
        return connection_window >= self.cfg.mtt

    # ── CL(τ) Vector ──────────────────────────────────────────────────

    def compute_CL_vector(
        self,
        outbound_truck: TruckSchedule,
        inbound_arrivals: dict,   # {truck_id: actual_arrival_time}
        connecting_inbound: List[TruckSchedule],
    ) -> np.ndarray:
        """Compute CL(τ): local cargo utility for each hold action.

        For each τ in hold_actions:
          CL(τ) = (1/N) Σ_i V_i × (1 - σ_i(τ))

        where σ_i(τ) is the disutility for cargo unit i at hold τ:
          - Linear  for standard/priority cargo: delay/Δ_C
          - Exponential for perishable cargo: 1 - exp(-λ × delay)
          - Amplified by (1 + X_k) for SLA urgency

        Args:
            outbound_truck: The truck whose hold decision is being evaluated
            inbound_arrivals: Known/estimated arrival times of inbound feeders
            connecting_inbound: Inbound trucks feeding into this outbound

        Returns:
            np.ndarray of shape (7,) — one CL value per hold action
        """
        CL = np.zeros(len(self.cfg.hold_actions))

        # Gather all cargo units that should board this outbound truck
        cargo_on_inbound = []
        for in_truck in connecting_inbound:
            in_arrival = inbound_arrivals.get(in_truck.truck_id, None)
            if in_arrival is None:
                continue
            for cargo in in_truck.cargo_units:
                if cargo.dest_truck_id == outbound_truck.truck_id:
                    cargo_on_inbound.append((cargo, in_arrival))

        if not cargo_on_inbound:
            # No connecting cargo → holding has no CL benefit
            # Return 0.5 (neutral) so the agent doesn't get confused
            return np.full(len(self.cfg.hold_actions), 0.5)

        for i, tau in enumerate(self.cfg.hold_actions):
            outbound_depart = outbound_truck.scheduled_departure + tau
            utilities = []

            for cargo, in_arrival in cargo_on_inbound:
                utility = self._cargo_utility(cargo, in_arrival, outbound_depart)
                utilities.append(utility)

            CL[i] = float(np.mean(utilities)) if utilities else 0.5

        return np.clip(CL, 0.0, 1.0)

    # ── OL(τ) Vector ──────────────────────────────────────────────────

    def compute_OL_vector(
        self,
        outbound_truck: TruckSchedule,
        bay_utilization: float,  # B_G ∈ [0, 1]
    ) -> np.ndarray:
        """Compute OL(τ): local operator utility for each hold action.

        Phase 1: OL(τ) = 1 - departure_delay / Δ_F

        Phase 2 adds bay-congestion penalty:
          OL(τ) = 1 - τ/Δ_F - λ × max(0, BG - B_thresh) × τ/30

        The penalty term increases with:
          - Hold duration τ (longer hold = more bay blockage)
          - Hub congestion BG (penalize more when hub is busy)

        Args:
            outbound_truck: The truck being evaluated
            bay_utilization: Current bay utilization rate B_G

        Returns:
            np.ndarray of shape (7,) — one OL value per hold action
        """
        OL = np.zeros(len(self.cfg.hold_actions))

        congestion_excess = max(
            0.0, bay_utilization - self.cfg.bay_congestion_threshold
        )

        for i, tau in enumerate(self.cfg.hold_actions):
            # Base schedule penalty: how much the hold degrades OTP
            schedule_penalty = tau / self.cfg.delta_f

            # Bay-congestion penalty: extra cost when hub is congested
            congestion_penalty = (
                self.cfg.lambda_bay_local
                * congestion_excess
                * (tau / 30.0)  # normalized by max hold time
            )

            OL[i] = 1.0 - schedule_penalty - congestion_penalty

        return np.clip(OL, 0.0, 1.0)

    # ── Actual Utilities (measured after events) ──────────────────────

    def compute_actual_CU(
        self,
        outbound_truck: TruckSchedule,
        inbound_arrivals: dict,
        connecting_inbound: List[TruckSchedule],
        actual_departure: float,
    ) -> float:
        """Compute the actually realized cargo utility after the hold is applied.

        This is P_L^k measured at time t+1 in the paper's notation.
        Used for the local reward component.
        """
        cargo_on_inbound = []
        for in_truck in connecting_inbound:
            in_arrival = inbound_arrivals.get(in_truck.truck_id)
            if in_arrival is None:
                continue
            for cargo in in_truck.cargo_units:
                if cargo.dest_truck_id == outbound_truck.truck_id:
                    cargo_on_inbound.append((cargo, in_arrival))

        if not cargo_on_inbound:
            return 0.5

        utilities = [
            self._cargo_utility(cargo, in_arrival, actual_departure)
            for cargo, in_arrival in cargo_on_inbound
        ]
        return float(np.mean(utilities)) if utilities else 0.5

    def compute_actual_OU(
        self,
        scheduled_departure: float,
        actual_departure: float,
    ) -> float:
        """Compute the actually realized operator utility after departure.

        OU = 1 - departure_delay / Δ_F
        Capped to [0, 1].
        """
        delay = max(0.0, actual_departure - scheduled_departure)
        ou = 1.0 - delay / self.cfg.delta_f
        return float(np.clip(ou, 0.0, 1.0))

    def count_transfers(
        self,
        outbound_truck: TruckSchedule,
        inbound_arrivals: dict,
        connecting_inbound: List[TruckSchedule],
        actual_departure: float,
    ) -> tuple:
        """Count successful and missed transfers.

        Returns:
            (n_success, n_missed) cargo transfer counts
        """
        n_success, n_missed = 0, 0
        for in_truck in connecting_inbound:
            in_arrival = inbound_arrivals.get(in_truck.truck_id)
            if in_arrival is None:
                n_missed += sum(
                    1 for c in in_truck.cargo_units
                    if c.dest_truck_id == outbound_truck.truck_id
                )
                continue
            for cargo in in_truck.cargo_units:
                if cargo.dest_truck_id != outbound_truck.truck_id:
                    continue
                if self.cargo_makes_transfer(cargo, in_arrival, actual_departure):
                    n_success += 1
                else:
                    n_missed += 1
        return n_success, n_missed

    # ── Private helpers ────────────────────────────────────────────────

    def _cargo_utility(
        self,
        cargo: CargoUnit,
        in_arrival: float,
        outbound_departure: float,
    ) -> float:
        """Compute utility ∈ [0, 1] for one cargo unit.

        If cargo makes transfer: utility = 1 (no delay to destination)
        If cargo misses: utility = (1 - disutility) where disutility is
          - Linear for standard cargo
          - Exponential for perishable cargo
          - Amplified by (1 + X_k) for SLA urgency
        """
        makes_transfer = self.cargo_makes_transfer(
            cargo, in_arrival, outbound_departure
        )

        if makes_transfer:
            return cargo.value_score  # reward proportional to value

        # Missed transfer: cargo waits for next truck (delta_c = 24h penalty)
        delay = self.cfg.delta_c  # always next-cycle penalty on miss

        # SLA urgency amplification: (1 + X_k)
        sla_multiplier = 1.0 + cargo.sla_urgency

        if cargo.is_perishable:
            # Exponential disutility for perishables
            raw_disutility = 1.0 - math.exp(-self.cfg.decay_lambda * delay)
        else:
            # Linear disutility (same as Phase 1 passengers)
            raw_disutility = min(delay / self.cfg.delta_c, 1.0)

        # Apply SLA amplification and value weighting
        disutility = min(1.0, raw_disutility * sla_multiplier)
        utility = cargo.value_score * (1.0 - disutility)

        return float(max(0.0, utility))
