"""
Context Engine — generates the state vector presented to the RL agent.

Direct counterpart of simulator/context_engine.py.

The state has the following components (from logistics_state_space_reference.pdf):

LOCAL state (per outbound truck k):
  CL(τ) : Local Cargo Utility vector for each hold action τ
  OL(τ) : Local Operator Utility vector for each hold action τ
  τ*    : Locally optimal hold time
  V_k   : Cargo value score ∈ [0, 1]
  Q_k   : Cargo volume fraction (connecting cargo / truck capacity)
  X_k   : SLA urgency ∈ {0, 1, 2}
  E_k   : Cargo perishability fraction
  Δ_in  : ETA lag of inbound truck (minutes)
  Δ_slack : Transfer slack (scheduled dep − scheduled dock)
  L_k   : Driver hours remaining (minutes)
  F_k   : Downstream deadline pressure ∈ [0, 1]
  N_in  : Number of inbound trucks feeding cargo to outbound truck k

GLOBAL state (hub-level, window W = 24 h):
  CG    : Global Cargo Utility (rolling avg  ← PG in aviation)
  OG    : Global Operator Utility (rolling avg ← AG in aviation)
  BG    : Bay utilisation rate (new)
  WG    : Hub throughput rate (new)
  YG    : Failed transfer rate (new)
  ZG    : Inbound queue depth (new)

DELAY TREE scalars (same as aviation):
  D_k   : Departure delay of truck k
  A_k   : Arrival delay of truck k
  G_bay_k : Bay dwell delay
  G_road_k: Road-time delay

Key formulae from PDF §5:
  σ_i(τ) = 0                                      if δ_i(τ) ≤ T_sla
  σ_i(τ) = (1+X_k)·min(δ_i(τ), Δ_C)/Δ_C          if δ_i(τ) > T_sla
  CL(τ)  = (1/N)·Σ V_ki·(1−σ_i(τ))
  OL(τ)  = 1 − δ_k(τ)/Δ_F − λ·max(0,BG−B_thresh)·τ/30
  τ*     = argmax_τ [α·CL(τ) + (1−α)·OL(τ)]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from phase2_simulator.config import SimConfig
from phase2_simulator.models import CargoUnit, ScheduledTruck, TruckState


# ======================================================================
# TruckContext  (replaces FlightContext)
# ======================================================================
@dataclass
class TruckContext:
    """The full context / state vector for one HOLD_DECISION.

    Mirrors FlightContext from simulator/context_engine.py.
    Extended with all PDF §2 and §3 state variables.
    """

    truck_id: str

    # --- Local Cargo Utility vector CL(τ) — replaces PL ---
    CL: List[float] = field(default_factory=list)
    # --- Local Operator Utility vector OL(τ) — replaces AL ---
    OL: List[float] = field(default_factory=list)
    # --- Global Cargo Utility scalar CG — replaces PG ---
    CG: float = 0.0
    # --- Global Operator Utility scalar OG — replaces AG ---
    OG: float = 0.0
    # --- Helper variable τ* ---
    tau_star: float = 0.0

    # --- Local state scalars (PDF §2, all NEW vs aviation) ---
    V_k: float = 0.5           # cargo value score
    Q_k: float = 0.0           # cargo volume fraction
    X_k: int = 0               # SLA urgency {0, 1, 2}
    E_k: float = 0.0           # perishability fraction
    delta_in: float = 0.0      # ETA lag of inbound truck (minutes)
    delta_slack: float = 0.0   # transfer slack (minutes)
    L_k: float = 270.0         # driver hours remaining (minutes)
    F_k: float = 1.0           # downstream deadline pressure
    N_in: int = 0              # number of inbound trucks

    # --- Global state scalars (PDF §3, NEW vs aviation) ---
    B_G: float = 0.0           # bay utilisation rate
    W_G: float = 1.0           # hub throughput rate
    Y_G: float = 0.0           # failed transfer rate (24h)
    Z_G: float = 0.0           # inbound queue depth

    # --- Delay tree fields (carried over from aviation) ---
    D_k: float = 0.0           # departure delay
    A_k: float = 0.0           # arrival delay
    G_bay_k: float = 0.0       # bay dwell delay
    G_road_k: float = 0.0      # road-time delay

    def to_array(self) -> np.ndarray:
        """Flatten to a 1-D numpy vector for the RL agent.

        Layout (mirrors FlightContext.to_array()):
          [CL(τ0)..CL(τN), OL(τ0)..OL(τN),
           CG, OG, τ*,
           V_k, Q_k, X_k, E_k, delta_in, delta_slack, L_k, F_k, N_in,
           B_G, W_G, Y_G, Z_G,
           D_k, A_k, G_bay_k, G_road_k]
        """
        local_scalars = [
            self.V_k, self.Q_k, float(self.X_k), self.E_k,
            self.delta_in, self.delta_slack, self.L_k, self.F_k, float(self.N_in),
        ]
        global_scalars = [self.B_G, self.W_G, self.Y_G, self.Z_G]
        delay_tree = [self.D_k, self.A_k, self.G_bay_k, self.G_road_k]
        return np.array(
            self.CL + self.OL + [self.CG, self.OG, self.tau_star]
            + local_scalars + global_scalars + delay_tree,
            dtype=np.float32,
        )

    @property
    def state_dim(self) -> int:
        """Dimension of the flattened state vector."""
        return len(self.CL) * 2 + 3 + 9 + 4 + 4  # CL + OL + [CG,OG,τ*] + local + global + delays


# ======================================================================
# ContextEngine  (replaces ContextEngine in simulator/context_engine.py)
# ======================================================================
class ContextEngine:
    """Computes the logistics state vector from raw simulator data.

    Mirrors simulator/context_engine.py ContextEngine.
    All method names and structures are preserved; formulas adapted from
    aviation (PAX) to logistics (cargo) per the PDF §5.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        # Rolling history for global utilities (time, value)
        self._global_cargo_history: List[Tuple[float, float]] = []   # CG ← PG
        self._global_operator_history: List[Tuple[float, float]] = [] # OG ← AG
        # Hub-level telemetry history
        self._bay_util_history: List[Tuple[float, float]] = []
        self._throughput_history: List[Tuple[float, float]] = []
        self._failed_transfer_history: List[Tuple[float, float]] = []
        self._inbound_queue_history: List[Tuple[float, float]] = []

    # -----------------------------------------------------------------
    # Local Cargo Utility vector CL(τ)  (replaces compute_local_pu)
    # -----------------------------------------------------------------
    def compute_local_cargo_utility(
        self,
        truck_state: TruckState,
        connecting_cargo: List[CargoUnit],
        hold_actions: List[int],
        hubs: Dict,
        truck_map: Dict[str, TruckState],
    ) -> List[float]:
        """Compute CL(τ) for each possible hold action τ.

        Formula (PDF §5.2):
          CL(τ) = (1/N) · Σ V_ki · (1 − σ_i(τ))

        Mirrors compute_local_pu() from simulator/context_engine.py.
        """
        if not connecting_cargo:
            return [1.0] * len(hold_actions)

        dest_hub = truck_state.truck.dest_hub
        hub_obj = hubs.get(dest_hub)
        mtt = hub_obj.min_transfer_time if hub_obj else self.cfg.min_transfer_time_spoke

        cl_values = []
        for tau in hold_actions:
            utilities = []
            for cargo in connecting_cargo:
                delay = self._estimate_cargo_delay(
                    cargo, truck_state, tau, mtt, truck_map
                )
                sla_urgency = cargo.sla_urgency
                sigma = self._cargo_disutility(delay, sla_urgency)
                utilities.append(cargo.value_score * (1.0 - sigma))
            cl_values.append(float(np.mean(utilities)) if utilities else 1.0)
        return cl_values

    # -----------------------------------------------------------------
    # Local Operator Utility vector OL(τ)  (replaces compute_local_au)
    # -----------------------------------------------------------------
    def compute_local_operator_utility(
        self,
        truck_state: TruckState,
        hold_actions: List[int],
        current_bay_util: float = 0.0,
    ) -> List[float]:
        """Compute OL(τ) for each possible hold action.

        Formula (PDF §5.3):
          OL(τ) = 1 − δ_k(τ)/Δ_F − λ·max(0, BG−B_thresh)·τ/30

        Hard curfew: if τ > L_k (driver hours), value = -∞ → encoded as -1.0.

        Mirrors compute_local_au() from simulator/context_engine.py.
        """
        lambda_ = self.cfg.lambda_congestion
        B_thresh = self.cfg.B_thresh
        delta_F = self.cfg.delta_F
        L_k = truck_state.driver_hours_remaining

        ol_values = []
        for tau in hold_actions:
            # Hard curfew violation
            if tau > L_k:
                ol_values.append(-1.0)
                continue
            arr_delay = self._estimate_truck_departure_delay(truck_state, tau)
            schedule_term = 1.0 - min(max(arr_delay, 0), delta_F) / delta_F
            congestion_penalty = lambda_ * max(0.0, current_bay_util - B_thresh) * tau / 30.0
            ol = schedule_term - congestion_penalty
            ol_values.append(float(np.clip(ol, -1.0, 1.0)))
        return ol_values

    # -----------------------------------------------------------------
    # Global Cargo Utility CG  (replaces compute_global_pu)
    # -----------------------------------------------------------------
    def compute_global_cargo_utility(self, current_time: float) -> float:
        """Average cargo utility over the past W=24 hours."""
        window_start = current_time - self.cfg.global_window_hours * 60
        vals = [v for t, v in self._global_cargo_history if t >= window_start]
        return float(np.mean(vals)) if vals else 1.0

    # -----------------------------------------------------------------
    # Global Operator Utility OG  (replaces compute_global_au)
    # -----------------------------------------------------------------
    def compute_global_operator_utility(self, current_time: float) -> float:
        """Average operator utility over the past W=24 hours."""
        window_start = current_time - self.cfg.global_window_hours * 60
        vals = [v for t, v in self._global_operator_history if t >= window_start]
        return float(np.mean(vals)) if vals else 1.0

    # -----------------------------------------------------------------
    # New global state scalars (PDF §3 — no aviation analog)
    # -----------------------------------------------------------------
    def compute_bay_utilisation(self, current_time: float) -> float:
        """BG: fraction of docking bays currently occupied."""
        window_start = current_time - 60  # last hour
        vals = [v for t, v in self._bay_util_history if t >= window_start]
        return float(np.mean(vals)) if vals else 0.0

    def compute_hub_throughput(self, current_time: float) -> float:
        """WG: hub throughput rate (successful transfers/hr, normalised)."""
        window_start = current_time - self.cfg.global_window_hours * 60
        vals = [v for t, v in self._throughput_history if t >= window_start]
        return float(np.mean(vals)) if vals else 1.0

    def compute_failed_transfer_rate(self, current_time: float) -> float:
        """YG: rolling failed-transfer fraction over 24h window."""
        window_start = current_time - self.cfg.global_window_hours * 60
        vals = [v for t, v in self._failed_transfer_history if t >= window_start]
        return float(np.mean(vals)) if vals else 0.0

    def compute_inbound_queue_depth(self, current_time: float) -> float:
        """ZG: normalised count of delayed inbound trucks at hub."""
        window_start = current_time - 60
        vals = [v for t, v in self._inbound_queue_history if t >= window_start]
        return float(np.mean(vals)) if vals else 0.0

    # -----------------------------------------------------------------
    # Record history  (mirrors record_global_pu / record_global_au)
    # -----------------------------------------------------------------
    def record_global_cargo_utility(self, time: float, cu: float) -> None:
        self._global_cargo_history.append((time, cu))

    def record_global_operator_utility(self, time: float, ou: float) -> None:
        self._global_operator_history.append((time, ou))

    def record_bay_utilisation(self, time: float, util: float) -> None:
        self._bay_util_history.append((time, util))

    def record_hub_throughput(self, time: float, throughput: float) -> None:
        self._throughput_history.append((time, throughput))

    def record_failed_transfer(self, time: float, rate: float) -> None:
        self._failed_transfer_history.append((time, rate))

    def record_inbound_queue(self, time: float, depth: float) -> None:
        self._inbound_queue_history.append((time, depth))

    # -----------------------------------------------------------------
    # τ*  (identical logic, renamed variables)
    # -----------------------------------------------------------------
    def compute_tau_star(
        self, CL: List[float], OL: List[float], hold_actions: List[int]
    ) -> float:
        """τ* = argmax_τ [α·CL(τ) + (1−α)·OL(τ)]  (PDF §5.4)."""
        alpha = self.cfg.alpha
        scores = [alpha * cl + (1 - alpha) * ol for cl, ol in zip(CL, OL)]
        best_idx = int(np.argmax(scores))
        return float(hold_actions[best_idx])

    # -----------------------------------------------------------------
    # Full context builder  (mirrors build_context)
    # -----------------------------------------------------------------
    def build_context(
        self,
        truck_state: TruckState,
        connecting_cargo: List[CargoUnit],
        hold_actions: List[int],
        hubs: Dict,
        truck_map: Dict[str, TruckState],
        current_time: float,
    ) -> TruckContext:
        """Build the full TruckContext for a HOLD_DECISION event.

        Mirrors build_context() from simulator/context_engine.py.
        """
        # Compute bay utilisation for OL penalty
        current_Bay_G = self.compute_bay_utilisation(current_time)

        CL = self.compute_local_cargo_utility(
            truck_state, connecting_cargo, hold_actions, hubs, truck_map
        )
        OL = self.compute_local_operator_utility(
            truck_state, hold_actions, current_bay_util=current_Bay_G
        )
        CG = self.compute_global_cargo_utility(current_time)
        OG = self.compute_global_operator_utility(current_time)
        WG = self.compute_hub_throughput(current_time)
        YG = self.compute_failed_transfer_rate(current_time)
        ZG = self.compute_inbound_queue_depth(current_time)
        tau_star = self.compute_tau_star(CL, OL, hold_actions)

        # Inbound ETA lag (Δ_in): from first connecting inbound truck
        delta_in = 0.0
        if connecting_cargo:
            inbound_tids = {c.legs[0] for c in connecting_cargo if len(c.legs) >= 2}
            lags = []
            for tid in inbound_tids:
                ts = truck_map.get(tid)
                if ts:
                    lags.append(ts.total_arrival_delay)
            delta_in = float(np.mean(lags)) if lags else 0.0

        # Transfer slack (Δ_slack): scheduled departure − scheduled dock
        delta_slack = max(0.0,
            truck_state.truck.scheduled_departure - truck_state.truck.scheduled_dock
        )

        return TruckContext(
            truck_id=truck_state.truck.truck_id,
            CL=CL,
            OL=OL,
            CG=CG,
            OG=OG,
            tau_star=tau_star,
            V_k=truck_state.cargo_value_score,
            Q_k=truck_state.cargo_volume_fraction,
            X_k=truck_state.sla_urgency,
            E_k=truck_state.perishability_fraction,
            delta_in=delta_in,
            delta_slack=delta_slack,
            L_k=truck_state.driver_hours_remaining,
            F_k=truck_state.deadline_pressure,
            N_in=truck_state.n_inbound_trucks,
            B_G=current_Bay_G,
            W_G=WG,
            Y_G=YG,
            Z_G=ZG,
            D_k=truck_state.departure_delay_D,
            A_k=truck_state.arrival_delay_A,
            G_bay_k=truck_state.bay_dwell_delay,
            G_road_k=truck_state.road_delay,
        )

    # -----------------------------------------------------------------
    # Internal helpers  (mirrors _estimate_pax_delay etc.)
    # -----------------------------------------------------------------
    def _estimate_cargo_delay(
        self,
        cargo: CargoUnit,
        inbound_truck: TruckState,
        hold_tau: int,
        mtt: int,
        truck_map: Dict[str, TruckState],
    ) -> float:
        """Estimate delivery delay for a cargo unit given hold τ.

        If cargo makes the transfer → delay = outbound truck's arrival delay.
        If cargo misses → next-cycle penalty (24h default).

        Mirrors _estimate_pax_delay() from simulator/context_engine.py.
        """
        if len(cargo.legs) < 2:
            return 0.0

        est_arrival = (
            inbound_truck.truck.scheduled_arrival
            + inbound_truck.arrival_delay_A
        )

        outbound_tid = cargo.legs[1]
        outbound_ts = truck_map.get(outbound_tid)
        if outbound_ts is None:
            return 0.0

        outbound_dep = outbound_ts.truck.scheduled_departure + hold_tau
        transfer_window = outbound_dep - est_arrival

        if transfer_window >= mtt:
            # Cargo makes the transfer
            outbound_delay = outbound_ts.arrival_delay_A + hold_tau
            return max(outbound_delay, 0.0)
        else:
            # Cargo misses — next-cycle penalty (24h)
            return self.cfg.next_cycle_penalty_minutes

    def _estimate_truck_departure_delay(
        self, truck_state: TruckState, hold_tau: int
    ) -> float:
        """Estimated departure delay of truck if held by τ minutes."""
        return truck_state.total_departure_delay + hold_tau

    def _cargo_disutility(self, delay: float, sla_urgency: int = 0) -> float:
        """σ_i(τ) as defined in PDF §5.1.

        σ_i(τ) = 0                                      if δ ≤ T_sla
        σ_i(τ) = (1+X_k)·min(δ, Δ_C)/Δ_C               if δ > T_sla

        Replaces _pax_disutility() from simulator/context_engine.py.
        The SLA urgency multiplier X_k ∈ {0,1,2} amplifies the penalty
        for high-urgency cargo that misses its transfer.
        """
        T_sla = self.cfg.T_sla
        delta_C = self.cfg.delta_C
        if delay <= T_sla:
            return 0.0
        return (1 + sla_urgency) * min(delay, delta_C) / delta_C

    # -----------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------
    def reset(self) -> None:
        self._global_cargo_history.clear()
        self._global_operator_history.clear()
        self._bay_util_history.clear()
        self._throughput_history.clear()
        self._failed_transfer_history.clear()
        self._inbound_queue_history.clear()
