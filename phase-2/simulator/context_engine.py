"""
context_engine.py — Assembles the 34-dimensional RL state vector.

This is the direct analog of the Context Engine described in paper Section 4,
extended from 17 dimensions (Phase 1) to 34 dimensions (Phase 2).

State vector layout:
  [0:7]   CL(τ)    — local cargo utility for each hold action
  [7:14]  OL(τ)    — local operator utility for each hold action
  [14]    CG       — global cargo utility (24h rolling average)
  [15]    OG       — global operator utility (24h rolling average)
  [16]    τ*       — locally optimal hold (argmax of α·CL + (1-α)·OL)
  [17]    V_k      — mean cargo value score
  [18]    Q_k      — truck volume utilization
  [19]    X_k      — worst-case SLA urgency (0/1/2, normalized)
  [20]    E_k      — fraction of connecting cargo that is perishable
  [21]    Δ_in     — ETA lag of inbound feeder (normalized)
  [22]    Δ_slack  — transfer slack (time margin, normalized)
  [23]    L_k      — driver hours remaining (normalized)
  [24]    F_k      — downstream deadline pressure (1 - X_k/2)
  [25]    N_in     — number of distinct inbound feeders (normalized)
  [26]    B_G      — bay utilization rate
  [27]    W_G      — hub throughput rate (rolling transfer success rate)
  [28]    Y_G      — failed transfer rate (24h rolling)
  [29]    Z_G      — inbound queue depth (normalized)
  [30]    D_k      — departure delay (normalized)
  [31]    A_k      — arrival delay (normalized)
  [32]    G_bay_k  — bay dwell delay (GB node input, normalized)
  [33]    G_road_k — road time delay (normalized)
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

from simulator.config import SimConfig
from simulator.schedule_generator import CargoUnit, TruckSchedule
from simulator.bay_manager import BayManager
from simulator.cargo_manager import CargoManager


@dataclass
class TruckContext:
    """Pre-assembled context for one outbound truck decision.

    Passed to LogisticsRewardCalculator and the Gym env's step().
    """
    truck_id: str
    scheduled_departure: float
    CL: np.ndarray    # shape (7,)
    OL: np.ndarray    # shape (7,)
    CG: float
    OG: float
    tau_star: float   # locally optimal hold in minutes

    # Raw features for state dims 17-33
    V_k: float
    Q_k: float
    X_k: int
    E_k: float
    delta_in: float
    delta_slack: float
    L_k: float
    F_k: float
    N_in: int
    B_G: float
    W_G: float
    Y_G: float
    Z_G: float
    D_k: float
    A_k: float
    G_bay_k: float
    G_road_k: float

    def to_state_vector(self) -> np.ndarray:
        """Assemble the 34-dimensional state vector for the RL agent."""
        state = np.zeros(34, dtype=np.float32)

        state[0:7]  = self.CL
        state[7:14] = self.OL
        state[14]   = self.CG
        state[15]   = self.OG
        state[16]   = self.tau_star / 30.0   # normalize to [0,1]

        state[17]   = self.V_k
        state[18]   = self.Q_k
        state[19]   = self.X_k / 2.0         # 0,1,2 → 0,0.5,1
        state[20]   = self.E_k
        state[21]   = np.clip(self.delta_in / 60.0, 0.0, 1.0)
        state[22]   = np.clip(self.delta_slack / 60.0, -1.0, 1.0)
        state[23]   = np.clip(self.L_k / 660.0, 0.0, 1.0)
        state[24]   = self.F_k
        state[25]   = np.clip(self.N_in / 10.0, 0.0, 1.0)

        state[26]   = self.B_G
        state[27]   = self.W_G
        state[28]   = self.Y_G
        state[29]   = np.clip(self.Z_G / self.N_in if self.N_in > 0 else 0, 0.0, 1.0)

        state[30]   = np.clip(self.D_k / 60.0, 0.0, 1.0)
        state[31]   = np.clip(self.A_k / 60.0, 0.0, 1.0)
        state[32]   = np.clip(self.G_bay_k / 30.0, 0.0, 1.0)
        state[33]   = np.clip(self.G_road_k / 60.0, 0.0, 1.0)

        return state


class ContextEngine:
    """Assembles the 34-dimensional state vector for the RL agent.

    Maintains rolling global statistics (CG, OG, W_G, Y_G) over a
    24-hour window, analogous to the paper's global utility tracking.
    """

    def __init__(self, cfg: SimConfig, bay_manager: BayManager,
                 cargo_manager: CargoManager):
        self.cfg = cfg
        self.bay_manager = bay_manager
        self.cargo_manager = cargo_manager

        # Rolling 24h windows for global stats
        self._cu_history: Deque[Tuple[float, float]] = deque()  # (time, cu)
        self._ou_history: Deque[Tuple[float, float]] = deque()  # (time, ou)
        self._transfer_history: Deque[Tuple[float, int, int]] = deque()  # (time, success, total)

    # ── Public API ─────────────────────────────────────────────────────

    def compute_context(
        self,
        outbound_truck: TruckSchedule,
        connecting_inbound: List[TruckSchedule],
        inbound_arrivals: Dict[str, float],   # known arrivals so far
        inbound_etas: Dict[str, float],       # estimated arrivals
        current_time: float,
        departure_delay: float = 0.0,
        arrival_delay: float = 0.0,
        gb_delay: float = 0.0,
        road_delay: float = 0.0,
    ) -> TruckContext:
        """Assemble TruckContext for an outbound truck decision.

        Args:
            outbound_truck: The truck about to depart
            connecting_inbound: Inbound trucks feeding into this one
            inbound_arrivals: {truck_id: actual_arrival} for trucks already here
            inbound_etas: {truck_id: estimated_arrival} for trucks en route
            current_time: Current simulation time (= scheduled departure)
            departure_delay: Accumulated departure delay so far
            arrival_delay: Arrival delay at destination (from previous leg)
            gb_delay: Bay-blockage delay for this truck
            road_delay: Road time delay accumulated

        Returns:
            TruckContext ready for to_state_vector()
        """
        # Merge known arrivals with ETAs
        all_arrivals = {**inbound_etas, **inbound_arrivals}  # actuals override ETAs

        # ── CL(τ) and OL(τ) vectors ─────────────────────────────────
        B_G = self.bay_manager.get_bay_utilization(current_time)
        CL = self.cargo_manager.compute_CL_vector(
            outbound_truck, all_arrivals, connecting_inbound
        )
        OL = self.cargo_manager.compute_OL_vector(outbound_truck, B_G)

        # ── τ* helper variable (paper Section 4) ─────────────────────
        combined = self.cfg.alpha * CL + (1 - self.cfg.alpha) * OL
        best_idx = int(np.argmax(combined))
        tau_star = float(self.cfg.hold_actions[best_idx])

        # ── Global rolling utilities ──────────────────────────────────
        CG, OG = self._get_global_utilities(current_time)
        W_G, Y_G = self._get_throughput_stats(current_time)

        # ── Cargo attributes ──────────────────────────────────────────
        connecting_cargo = [
            c for t in connecting_inbound
            for c in t.cargo_units
            if c.dest_truck_id == outbound_truck.truck_id
        ]

        V_k = self._mean_value_score(connecting_cargo)
        Q_k = self._volume_utilization(outbound_truck, connecting_cargo)
        X_k = max((c.sla_urgency for c in connecting_cargo), default=0)
        E_k = self._perishable_fraction(connecting_cargo)
        F_k = 1.0 - X_k / 2.0   # downstream deadline pressure

        # ── Feeder delay features ─────────────────────────────────────
        delta_in = self._feeder_eta_lag(connecting_inbound, all_arrivals, current_time)
        delta_slack = self._transfer_slack(connecting_inbound, all_arrivals, current_time,
                                           outbound_truck.scheduled_departure)
        N_in = len(connecting_inbound)
        L_k = outbound_truck.driver_hours_remaining

        # ── Queue depth ───────────────────────────────────────────────
        Z_G = float(self.bay_manager.get_queue_depth())

        return TruckContext(
            truck_id=outbound_truck.truck_id,
            scheduled_departure=outbound_truck.scheduled_departure,
            CL=CL, OL=OL, CG=CG, OG=OG, tau_star=tau_star,
            V_k=V_k, Q_k=Q_k, X_k=X_k, E_k=E_k,
            delta_in=delta_in, delta_slack=delta_slack,
            L_k=L_k, F_k=F_k, N_in=N_in,
            B_G=B_G, W_G=W_G, Y_G=Y_G, Z_G=Z_G,
            D_k=departure_delay, A_k=arrival_delay,
            G_bay_k=gb_delay, G_road_k=road_delay,
        )

    def record_outcome(
        self,
        event_time: float,
        cargo_utility: float,
        operator_utility: float,
        n_success: int,
        n_total: int,
    ):
        """Record a completed transfer event for rolling global stats.

        Called by EventQueue after each outbound truck's departure.
        """
        self._cu_history.append((event_time, cargo_utility))
        self._ou_history.append((event_time, operator_utility))
        self._transfer_history.append((event_time, n_success, n_total))
        self._prune_history(event_time)

    def reset(self):
        """Clear rolling history for a new episode."""
        self._cu_history.clear()
        self._ou_history.clear()
        self._transfer_history.clear()

    # ── Private helpers ────────────────────────────────────────────────

    def _prune_history(self, current_time: float):
        """Remove entries older than the 24h rolling window."""
        cutoff = current_time - self.cfg.global_window_minutes
        for hist in [self._cu_history, self._ou_history, self._transfer_history]:
            while hist and hist[0][0] < cutoff:
                hist.popleft()

    def _get_global_utilities(self, current_time: float) -> Tuple[float, float]:
        """Compute 24h rolling average global CG and OG."""
        self._prune_history(current_time)
        if not self._cu_history:
            return 0.5, 0.5
        CG = float(np.mean([v for _, v in self._cu_history]))
        OG = float(np.mean([v for _, v in self._ou_history]))
        return CG, OG

    def _get_throughput_stats(self, current_time: float) -> Tuple[float, float]:
        """Compute W_G (success rate) and Y_G (failure rate) over 24h."""
        self._prune_history(current_time)
        if not self._transfer_history:
            return 0.8, 0.05   # neutral priors

        total_success = sum(s for _, s, _ in self._transfer_history)
        total_cargo = sum(t for _, _, t in self._transfer_history)
        if total_cargo == 0:
            return 0.8, 0.05
        W_G = total_success / total_cargo
        Y_G = 1.0 - W_G
        return float(W_G), float(Y_G)

    def _mean_value_score(self, cargo: List[CargoUnit]) -> float:
        if not cargo:
            return 0.5
        return float(np.mean([c.value_score for c in cargo]))

    def _volume_utilization(self, truck: TruckSchedule,
                            cargo: List[CargoUnit]) -> float:
        """Estimate truck volume utilization as fraction of capacity."""
        if not cargo:
            return 0.0
        total_weight = sum(c.weight_kg for c in cargo)
        # Typical 40t truck payload = 24,000 kg
        return float(np.clip(total_weight / 24000.0, 0.0, 1.0))

    def _perishable_fraction(self, cargo: List[CargoUnit]) -> float:
        if not cargo:
            return 0.0
        return float(sum(1 for c in cargo if c.is_perishable) / len(cargo))

    def _feeder_eta_lag(self, connecting_inbound: List[TruckSchedule],
                        arrivals: Dict[str, float], current_time: float) -> float:
        """Mean ETA lag of inbound feeders (how late they are expected to be)."""
        lags = []
        for t in connecting_inbound:
            eta = arrivals.get(t.truck_id, t.scheduled_arrival)
            lag = max(0.0, eta - t.scheduled_arrival)
            lags.append(lag)
        return float(np.mean(lags)) if lags else 0.0

    def _transfer_slack(self, connecting_inbound: List[TruckSchedule],
                        arrivals: Dict[str, float],
                        current_time: float,
                        scheduled_departure: float) -> float:
        """Mean transfer slack: time available for transfer after inbound arrives."""
        slacks = []
        for t in connecting_inbound:
            eta = arrivals.get(t.truck_id, t.scheduled_arrival)
            slack = scheduled_departure - eta - self.cfg.mtt
            slacks.append(slack)
        return float(np.mean(slacks)) if slacks else 0.0
