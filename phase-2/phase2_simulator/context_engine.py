"""
Context Engine — computes the state vector presented to the RL agent.

Mirrors simulator/context_engine.py field-for-field.
All formula symbols from logistics_state_space_reference.pdf §5.

LOCAL state (per outbound truck k — PDF §2):
  CL(τ) : Local Cargo Utility vector   ← PL(τ) in aviation
  OL(τ) : Local Operator Utility vector ← AL(τ) in aviation
  τ*    : argmax α·CL + (1-α)·OL
  V_k   : cargo value score
  Q_k   : cargo volume fraction
  X_k   : SLA urgency {0,1,2}
  E_k   : perishability fraction
  Δ_in  : ETA lag of inbound truck
  Δ_slack : transfer slack
  L_k   : driver hours remaining
  F_k   : downstream deadline pressure
  N_in  : number of inbound trucks

GLOBAL state (hub-level — PDF §3):
  CG    : Global Cargo Utility rolling avg    ← PG
  OG    : Global Operator Utility rolling avg ← AG
  BG    : Bay utilisation rate (NEW)
  WG    : Hub throughput rate (NEW)
  YG    : Failed transfer rate (NEW)
  ZG    : Inbound queue depth (NEW)

KEY FIXES vs previous version:
  BG: now correctly populated (was always 0.0 — TRUCK_DOCK events now scheduled)
  WG: records 1.0 on success AND 0.0 on failure → correct rolling success rate
  YG: records 1.0 on failure AND 0.0 on success → correct rolling failure rate
  Perishable cargo: uses exponential disutility instead of linear
"""

from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from phase2_simulator.config import SimConfig
from phase2_simulator.models import CargoUnit, ScheduledTruck, TruckState


# ====================================================================
# TruckContext  (replaces FlightContext)
# ====================================================================
@dataclass
class TruckContext:
    """Full context / state vector for one HOLD_DECISION event.

    Mirrors FlightContext from simulator/context_engine.py.
    Extended with all PDF §2-§3 state variables.

    State vector layout (to_array()):
      [0:7]   CL(τ) for τ in {0,5,10,15,20,25,30}
      [7:14]  OL(τ) for τ in {0,5,10,15,20,25,30}
      [14]    CG
      [15]    OG
      [16]    τ* (minutes)
      [17]    V_k
      [18]    Q_k
      [19]    X_k (cast to float)
      [20]    E_k
      [21]    Δ_in  (capped at 60, normalised /60)
      [22]    Δ_slack (capped at 120, normalised /120)
      [23]    L_k (capped at 270, normalised /270)
      [24]    F_k
      [25]    N_in (capped at 20, normalised /20)
      [26]    B_G
      [27]    W_G
      [28]    Y_G
      [29]    Z_G
      [30]    D_k (departure delay, normalised /Δ_F)
      [31]    A_k (arrival delay, normalised /Δ_C)
      [32]    G_bay_k (normalised /30)
      [33]    G_road_k (normalised /60)
    Total: 34 dimensions
    """
    truck_id: str

    # Local utility vectors (7 values each — one per hold action)
    CL: List[float] = field(default_factory=list)   # Cargo Utility ← PL
    OL: List[float] = field(default_factory=list)   # Operator Utility ← AL

    # Global utility scalars
    CG: float = 0.0    # Global Cargo Utility ← PG
    OG: float = 0.0    # Global Operator Utility ← AG

    tau_star: float = 0.0

    # Local state scalars (PDF §2)
    V_k: float = 0.5          # cargo value score
    Q_k: float = 0.0          # cargo volume fraction
    X_k: int = 0              # SLA urgency {0,1,2}
    E_k: float = 0.0          # perishability fraction
    delta_in: float = 0.0     # Δ_in: ETA lag of inbound (minutes)
    delta_slack: float = 0.0  # Δ_slack: transfer slack (minutes)
    L_k: float = 270.0        # driver hours remaining (minutes)
    F_k: float = 1.0          # downstream deadline pressure
    N_in: int = 0             # number of inbound trucks

    # Global state scalars (PDF §3 — NEW, no aviation analog)
    B_G: float = 0.0   # BG: bay utilisation rate
    W_G: float = 1.0   # WG: hub throughput rate (successful-transfer rate)
    Y_G: float = 0.0   # YG: failed transfer rate (24h rolling)
    Z_G: float = 0.0   # ZG: inbound queue depth (delayed trucks fraction)

    # Delay tree scalars (same as aviation)
    D_k: float = 0.0     # departure delay
    A_k: float = 0.0     # arrival delay
    G_bay_k: float = 0.0  # bay dwell delay
    G_road_k: float = 0.0 # road-time delay

    def to_array(self) -> np.ndarray:
        """Flatten to 34-dim numpy float32 vector for the RL agent.

        All values are normalised to approximately [0,1].
        Mirrors FlightContext.to_array() in layout.
        """
        delta_F = 60.0   # use Δ_F normalisation
        delta_C = 480.0  # use Δ_C normalisation
        local_scalars = [
            self.V_k,
            self.Q_k,
            float(self.X_k) / 2.0,              # X_k ∈ {0,1,2} → /2 → [0,1]
            self.E_k,
            float(np.clip(self.delta_in, 0, 60)) / 60.0,
            float(np.clip(self.delta_slack, 0, 120)) / 120.0,
            float(np.clip(self.L_k, 0, 270)) / 270.0,
            self.F_k,
            float(np.clip(self.N_in, 0, 20)) / 20.0,
        ]
        global_scalars = [self.B_G, self.W_G, self.Y_G, self.Z_G]
        delay_tree = [
            float(np.clip(self.D_k, 0, delta_F)) / delta_F,
            float(np.clip(self.A_k, 0, delta_C)) / delta_C,
            float(np.clip(self.G_bay_k, 0, 30)) / 30.0,
            float(np.clip(self.G_road_k, 0, 60)) / 60.0,
        ]
        vec = self.CL + self.OL + [self.CG, self.OG, self.tau_star / 30.0] \
              + local_scalars + global_scalars + delay_tree
        return np.array(vec, dtype=np.float32)

    @property
    def state_dim(self) -> int:
        return 34


# ====================================================================
# ContextEngine  (mirrors simulator/context_engine.py ContextEngine)
# ====================================================================
class ContextEngine:
    """Computes the logistics state vector from simulator data.

    Mirrors ContextEngine from simulator/context_engine.py.
    Key formula changes vs aviation:
      1. σ_i(τ) multiplied by (1+X_k) — SLA urgency amplifies penalty
      2. Perishable cargo uses exponential disutility instead of linear
      3. OL(τ) includes bay-blockage penalty λ·max(0,BG-B_thresh)·τ/30
      4. YG and WG record 0.0/1.0 on BOTH success AND failure (fixed)
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        # Rolling history buffers: list of (sim_time, value)
        self._global_cargo_history:    List[Tuple[float, float]] = []
        self._global_operator_history: List[Tuple[float, float]] = []
        self._bay_util_history:        List[Tuple[float, float]] = []
        self._throughput_history:      List[Tuple[float, float]] = []  # WG
        self._failed_transfer_history: List[Tuple[float, float]] = []  # YG
        self._inbound_queue_history:   List[Tuple[float, float]] = []  # ZG

    # ── Local Cargo Utility CL(τ)  ← compute_local_pu() ──────────────
    def compute_local_cargo_utility(
        self,
        truck_state: TruckState,
        connecting_cargo: List[CargoUnit],
        hold_actions: List[int],
        hubs: Dict,
        truck_map: Dict[str, TruckState],
    ) -> List[float]:
        """CL(τ) = (1/N) · Σ V_ki · (1 − σ_i(τ))   (PDF §5.2)

        For each hold action τ, compute mean weighted cargo utility.
        V_ki weights each cargo unit by its value score.
        σ_i(τ) is the disutility (0 = no delay impact, 1 = maximum penalty).
        """
        if not connecting_cargo:
            return [1.0] * len(hold_actions)

        dest = truck_state.truck.dest_hub
        hub_obj = hubs.get(dest)
        mtt = hub_obj.min_transfer_time if hub_obj else self.cfg.min_transfer_time_spoke

        cl_values = []
        for tau in hold_actions:
            utilities = []
            for cargo in connecting_cargo:
                delay = self._estimate_cargo_delay(cargo, truck_state, tau, mtt, truck_map)
                sigma = self._cargo_disutility(delay, cargo.sla_urgency, cargo.is_perishable)
                utilities.append(cargo.value_score * (1.0 - sigma))
            cl_values.append(float(np.mean(utilities)) if utilities else 1.0)
        return cl_values

    # ── Local Operator Utility OL(τ)  ← compute_local_au() ───────────
    def compute_local_operator_utility(
        self,
        truck_state: TruckState,
        hold_actions: List[int],
        current_bay_util: float = 0.0,
    ) -> List[float]:
        """OL(τ) = 1 − δ_k(τ)/Δ_F − λ·max(0,BG−B_thresh)·τ/30   (PDF §5.3)

        Hard curfew:
          if τ > L_k (driver hours): OL = -1.0 (infeasible)
          if BG > bay_curfew_threshold: max feasible τ = 0
        """
        lambda_  = self.cfg.lambda_congestion
        B_thresh = self.cfg.B_thresh
        delta_F  = self.cfg.delta_F
        L_k      = truck_state.driver_hours_remaining
        bay_curfew = self.cfg.bay_curfew_threshold

        ol_values = []
        for tau in hold_actions:
            # Hard bay curfew: if hub is essentially full, no hold allowed
            if current_bay_util >= bay_curfew and tau > 0:
                ol_values.append(-1.0)
                continue
            # Hard driver curfew: tau exceeds available driving hours
            if tau > L_k:
                ol_values.append(-1.0)
                continue

            dep_delay = self._estimate_truck_departure_delay(truck_state, tau)
            schedule_term = 1.0 - min(max(dep_delay, 0.0), delta_F) / delta_F
            congestion_penalty = lambda_ * max(0.0, current_bay_util - B_thresh) * tau / 30.0
            ol = schedule_term - congestion_penalty
            ol_values.append(float(np.clip(ol, -1.0, 1.0)))
        return ol_values

    # ── Global Cargo Utility CG  ← compute_global_pu() ───────────────
    def compute_global_cargo_utility(self, current_time: float) -> float:
        """Rolling 24h mean cargo utility across all completed transfers."""
        window_start = current_time - self.cfg.global_window_hours * 60.0
        vals = [v for t, v in self._global_cargo_history if t >= window_start]
        return float(np.mean(vals)) if vals else 1.0

    # ── Global Operator Utility OG  ← compute_global_au() ────────────
    def compute_global_operator_utility(self, current_time: float) -> float:
        """Rolling 24h mean operator utility across all departed trucks."""
        window_start = current_time - self.cfg.global_window_hours * 60.0
        vals = [v for t, v in self._global_operator_history if t >= window_start]
        return float(np.mean(vals)) if vals else 1.0

    # ── BG  (NEW — no aviation analog) ───────────────────────────────
    def compute_bay_utilisation(self, current_time: float) -> float:
        """BG: fraction of bays occupied in the past 60 minutes."""
        window_start = current_time - 60.0
        vals = [v for t, v in self._bay_util_history if t >= window_start]
        return float(np.mean(vals)) if vals else 0.0

    # ── WG  (NEW — hub throughput rate) ──────────────────────────────
    def compute_hub_throughput(self, current_time: float) -> float:
        """WG: rolling 24h successful-transfer rate (0=all failed, 1=all succeeded).

        FIX: records 1.0 on success AND 0.0 on failure → correct rolling average.
        Previous version recorded 1.0 only on success → always returned 1.0.
        """
        window_start = current_time - self.cfg.global_window_hours * 60.0
        vals = [v for t, v in self._throughput_history if t >= window_start]
        return float(np.mean(vals)) if vals else 1.0

    # ── YG  (NEW — failed transfer rate) ─────────────────────────────
    def compute_failed_transfer_rate(self, current_time: float) -> float:
        """YG: rolling 24h failed-transfer rate (0=none failed, 1=all failed).

        FIX: records 1.0 on failure AND 0.0 on success → correct rolling average.
        Previous version recorded 1.0 only on failure → always returned 1.0.
        """
        window_start = current_time - self.cfg.global_window_hours * 60.0
        vals = [v for t, v in self._failed_transfer_history if t >= window_start]
        return float(np.mean(vals)) if vals else 0.0

    # ── ZG  (NEW — inbound queue depth) ──────────────────────────────
    def compute_inbound_queue_depth(self, current_time: float) -> float:
        """ZG: normalised delayed-inbound fraction from past 60 min."""
        window_start = current_time - 60.0
        vals = [v for t, v in self._inbound_queue_history if t >= window_start]
        return float(np.mean(vals)) if vals else 0.0

    # ── History record methods  ───────────────────────────────────────
    def record_global_cargo_utility(self, time: float, cu: float) -> None:
        self._global_cargo_history.append((time, float(np.clip(cu, 0.0, 1.0))))

    def record_global_operator_utility(self, time: float, ou: float) -> None:
        self._global_operator_history.append((time, float(np.clip(ou, 0.0, 1.0))))

    def record_bay_utilisation(self, time: float, util: float) -> None:
        self._bay_util_history.append((time, float(np.clip(util, 0.0, 1.0))))

    def record_transfer_outcome(self, time: float, succeeded: bool) -> None:
        """Record one transfer outcome into BOTH WG and YG histories.

        FIX: Both WG and YG record on every transfer (success or failure).
        """
        val = 1.0 if succeeded else 0.0
        self._throughput_history.append((time, val))        # WG: 1=success, 0=fail
        self._failed_transfer_history.append((time, 1.0 - val))  # YG: 1=fail, 0=success

    def record_inbound_queue(self, time: float, depth: float) -> None:
        self._inbound_queue_history.append((time, float(np.clip(depth, 0.0, 1.0))))

    # ── τ*  (identical formula to aviation) ──────────────────────────
    def compute_tau_star(self, CL: List[float], OL: List[float],
                         hold_actions: List[int]) -> float:
        """τ* = argmax_τ [α·CL(τ) + (1−α)·OL(τ)]  subject to OL(τ) ≥ 0."""
        alpha = self.cfg.alpha
        scores = [alpha * cl + (1 - alpha) * ol for cl, ol in zip(CL, OL)]
        best_idx = int(np.argmax(scores))
        return float(hold_actions[best_idx])

    # ── Full context builder  ← build_context() ──────────────────────
    def build_context(
        self,
        truck_state: TruckState,
        connecting_cargo: List[CargoUnit],
        hold_actions: List[int],
        hubs: Dict,
        truck_map: Dict[str, TruckState],
        current_time: float,
    ) -> TruckContext:
        """Build the full TruckContext for one HOLD_DECISION event.

        Mirrors build_context() from simulator/context_engine.py.
        """
        current_BG = self.compute_bay_utilisation(current_time)

        CL = self.compute_local_cargo_utility(
            truck_state, connecting_cargo, hold_actions, hubs, truck_map)
        OL = self.compute_local_operator_utility(
            truck_state, hold_actions, current_bay_util=current_BG)
        CG = self.compute_global_cargo_utility(current_time)
        OG = self.compute_global_operator_utility(current_time)
        WG = self.compute_hub_throughput(current_time)
        YG = self.compute_failed_transfer_rate(current_time)
        ZG = self.compute_inbound_queue_depth(current_time)
        tau_star = self.compute_tau_star(CL, OL, hold_actions)

        # Δ_in: mean actual arrival delay of all inbound (feeder) trucks
        inbound_tids = {c.legs[0] for c in connecting_cargo if len(c.legs) >= 2}
        lags = [truck_map[tid].arrival_delay_A
                for tid in inbound_tids if tid in truck_map]
        delta_in = float(np.mean(lags)) if lags else 0.0

        # Δ_slack: scheduled departure − scheduled dock (guaranteed loading window)
        delta_slack = max(0.0,
            truck_state.truck.scheduled_departure - truck_state.truck.scheduled_dock)

        return TruckContext(
            truck_id=truck_state.truck.truck_id,
            CL=CL, OL=OL, CG=CG, OG=OG, tau_star=tau_star,
            V_k=truck_state.cargo_value_score,
            Q_k=truck_state.cargo_volume_fraction,
            X_k=truck_state.sla_urgency,
            E_k=truck_state.perishability_fraction,
            delta_in=delta_in,
            delta_slack=delta_slack,
            L_k=truck_state.driver_hours_remaining,
            F_k=truck_state.deadline_pressure,
            N_in=truck_state.n_inbound_trucks,
            B_G=current_BG, W_G=WG, Y_G=YG, Z_G=ZG,
            D_k=truck_state.departure_delay_D,
            A_k=truck_state.arrival_delay_A,
            G_bay_k=truck_state.bay_dwell_delay,
            G_road_k=truck_state.road_delay,
        )

    # ── Internal helpers ──────────────────────────────────────────────

    def _cargo_disutility(self, delay: float, sla_urgency: int = 0,
                          is_perishable: bool = False) -> float:
        """σ_i(τ) — cargo disutility given delivery delay.

        PDF §5.1 formula (base):
          σ = 0                                    if delay ≤ T_sla
          σ = (1+X_k) · min(delay, Δ_C) / Δ_C    if delay > T_sla  [linear]

        Logistics improvement (perishable cargo):
          For is_perishable=True, use exponential growth:
          σ = min(1.0, (1+X_k) · (1 − exp(−k · delay)))
          where k = perishable_decay_rate (config)
          Rationale: cold-chain cargo spoils exponentially, not linearly.
        """
        T_sla   = self.cfg.T_sla
        delta_C = self.cfg.delta_C

        if delay <= T_sla:
            return 0.0

        if is_perishable:
            # Exponential disutility: grows faster at first, caps at 1.0
            k = self.cfg.perishable_decay_rate
            raw = (1 + sla_urgency) * (1.0 - math.exp(-k * delay))
            return float(np.clip(raw, 0.0, 1.0))
        else:
            # PDF §5.1 linear formula, clipped to [0,1]
            # (1+Xk) can push sigma > 1.0 when Xk > 0 — clip to prevent
            # negative CL values (cargo can't have utility worse than total loss)
            raw = (1 + sla_urgency) * min(delay, delta_C) / delta_C
            return float(np.clip(raw, 0.0, 1.0))

    def _estimate_cargo_delay(self, cargo: CargoUnit, inbound_ts: TruckState,
                              hold_tau: int, mtt: int,
                              truck_map: Dict[str, TruckState]) -> float:
        """Estimate delivery delay for one cargo unit given hold τ.

        If cargo makes the transfer → delay = outbound truck's arrival delay.
        If cargo misses the transfer → next-cycle penalty (24h).
        Mirrors _estimate_pax_delay() from simulator/context_engine.py.
        """
        if len(cargo.legs) < 2:
            return 0.0

        est_arrival   = inbound_ts.truck.scheduled_arrival + inbound_ts.arrival_delay_A
        outbound_ts   = truck_map.get(cargo.legs[1])
        if outbound_ts is None:
            return 0.0

        outbound_dep  = outbound_ts.truck.scheduled_departure + hold_tau
        transfer_window = outbound_dep - est_arrival

        if transfer_window >= mtt:
            return max(0.0, outbound_ts.arrival_delay_A + hold_tau)
        else:
            return self.cfg.next_cycle_penalty_minutes

    def _estimate_truck_departure_delay(self, ts: TruckState, hold_tau: int) -> float:
        return ts.total_departure_delay + hold_tau

    def reset(self) -> None:
        self._global_cargo_history.clear()
        self._global_operator_history.clear()
        self._bay_util_history.clear()
        self._throughput_history.clear()
        self._failed_transfer_history.clear()
        self._inbound_queue_history.clear()

