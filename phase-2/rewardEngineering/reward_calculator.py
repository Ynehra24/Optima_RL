"""
Reward Calculator for the Logistics Cross-Docking domain (Phase 2).

Implements the reward engineering described in Section 5 of the paper,
adapted for the logistics Hold-or-Not-Hold problem.

DOMAIN MAPPING (Aviation → Logistics):
  PU (Passenger Utility)  →  CU (Cargo Utility)
  AU (Airline Utility)    →  OU (Operator Utility)
  P_L, P_G               →  CL, CG   (local/global cargo utility)
  A_L, A_G               →  OL, OG   (local/global operator utility)
  Flight f                →  Truck k

REWARD STRUCTURE (same formula, richer inputs):
  R_T^k = β · R_L^k + (1-β) · R_G^k

  Local:   R_L^k = α · CL^k(τ) + (1-α) · OL^k(τ)
  Global:  R_G^k = α · CG^k_attributed + (1-α) · OG^k_attributed
                   − λ_bay · BayCongestion^k_attributed

KEY DIFFERENCES FROM PHASE 1:
  1. CL(τ) already encodes value-weighting (V_k), SLA urgency (X_k),
     and perishable exponential disutility (E_k) — computed by ContextEngine.
  2. OL(τ) includes bay-congestion penalty λ·max(0, BG-B_thresh)·τ/30
     — computed by ContextEngine.
  3. The Delay Tree has a new "GB" (bay-blockage) node that lets us
     attribute bay-congestion penalties to past hold decisions.
  4. Global reward has a third sub-component: bay-congestion attribution
     that penalises holds which block docking bays for downstream trucks.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    # Only used for type hints — not needed at runtime
    from simulator.context_engine import TruckContext

from rewardEngineering.delay_tree import LogisticsDelayTree


class LogisticsRewardCalculator:
    """Computes local, global, and total rewards for the logistics HNH agent.

    API is identical to Phase 1's RewardCalculator to ensure the A2C
    agent code requires zero changes.

    Methods (same interface as Phase 1):
      - compute_local_reward(ctx, hold_minutes) → float
      - register_truck_arrival(...)              [was register_flight_arrival]
      - register_truck_departure(...)            [was register_flight_departure]
      - get_global_reward(truck_id)             → float
      - get_total_reward(ctx, hold_minutes, truck_id) → float
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.alpha = cfg.alpha   # Cargo vs Operator trade-off (α)
        self.beta = getattr(cfg, 'beta', 0.75)  # Local vs Global trade-off (β)
        self.delay_tree = LogisticsDelayTree()

        # Bay-congestion penalty weight for global reward
        # Scales how much bay-blockage attribution affects R_G
        self.lambda_bay = getattr(cfg, 'lambda_congestion', 0.30)

        # Track global utilities attributed to each truck's hold decision
        self.global_cu_attributed: Dict[str, float] = {}   # CG per truck
        self.global_ou_attributed: Dict[str, float] = {}   # OG per truck
        self.bay_congestion_attributed: Dict[str, float] = {}  # Bay penalty per truck

    # ── Local Reward ──────────────────────────────────────────────────

    def compute_local_reward(self, ctx: TruckContext, hold_minutes: float) -> float:
        """R_L^k = α · CL^k(τ) + (1-α) · OL^k(τ)

        CL and OL are pre-computed by the ContextEngine with all
        logistics-specific enrichments (value-weighting, SLA, perishability,
        bay-congestion penalty).  We simply index into them.

        This is structurally identical to Phase 1, but the underlying
        CL/OL values are richer.
        """
        action_idx = self._get_action_index(hold_minutes)

        local_cu = ctx.CL[action_idx] if action_idx < len(ctx.CL) else 0.5
        local_ou = ctx.OL[action_idx] if action_idx < len(ctx.OL) else 0.5

        return self.alpha * local_cu + (1 - self.alpha) * local_ou

    # ── Delay Tree Construction ───────────────────────────────────────

    def register_truck_arrival(
        self,
        truck_id: str,
        arrival_delay: float,
        departure_delay: float,
        road_delay: float,
        arrival_bay_delay: float,
        arrival_cu: float,
        arrival_ou: float,
    ):
        """Called when a truck arrives at its destination hub.

        Builds the arrival node in the Delay Tree (Rule 1) and
        attributes the global CU and OU to past hold decisions.

        Maps to Phase 1's register_flight_arrival().
        """
        # Rule 1: Build arrival delay sub-tree
        A_node = self.delay_tree.build_arrival_delay_tree(
            truck_id=truck_id,
            arrival_delay=arrival_delay,
            departure_delay=departure_delay,
            road_delay=road_delay,
            arrival_bay_delay=arrival_bay_delay,
        )

        if A_node:
            # Attribute CU and OU of this delayed arrival to past holds
            attributed_cu = self.delay_tree.attribute_outcome(A_node, arrival_cu)
            attributed_ou = self.delay_tree.attribute_outcome(A_node, arrival_ou)

            for t_id, cu_val in attributed_cu.items():
                self.global_cu_attributed[t_id] = (
                    self.global_cu_attributed.get(t_id, 0.0) + cu_val
                )

            for t_id, ou_val in attributed_ou.items():
                self.global_ou_attributed[t_id] = (
                    self.global_ou_attributed.get(t_id, 0.0) + ou_val
                )

    def register_truck_departure(
        self,
        truck_id: str,
        departure_delay: float,
        prev_truck_id: Optional[str],
        prev_arrival_delay: float,
        hold_duration: float,
        departure_ground_delay: float,
        bay_blockage_delay: float,
        incoming_trucks: List[tuple],
    ):
        """Called when a truck departs its origin hub.

        Builds the hold node (Rule 3) and departure node (Rule 2)
        in the Delay Tree.

        EXTENSION vs Phase 1:
          - bay_blockage_delay (GB_k) is passed to Rule 2, creating a
            new "GB" child node in the departure sub-tree.

        Maps to Phase 1's register_flight_departure().
        """
        # Rule 3: Hold delay depends on incoming feeder trucks
        if hold_duration > 0:
            self.delay_tree.build_hold_delay_tree(
                truck_id=truck_id,
                hold_duration=hold_duration,
                incoming_trucks=incoming_trucks,
            )

        # Rule 2: Departure delay depends on prev arrival, hold, ground, bay blockage
        self.delay_tree.build_departure_delay_tree(
            truck_id=truck_id,
            departure_delay=departure_delay,
            prev_truck_id=prev_truck_id,
            prev_arrival_delay=prev_arrival_delay,
            hold_duration=hold_duration,
            departure_ground_delay=departure_ground_delay,
            bay_blockage_delay=bay_blockage_delay,
        )

    def register_bay_congestion(
        self,
        held_truck_id: str,
        blocked_trucks: Dict[str, float],
    ):
        """Register bay-congestion caused by a hold decision.

        Called when a hold at truck T_k causes other trucks to wait for
        docking bays.  The bay-blockage delays are attributed back to
        T_k's hold decision as an additional global penalty.

        This has no Phase 1 analog — it's a logistics-specific extension.

        Args:
            held_truck_id: Truck whose hold caused the bay blockage.
            blocked_trucks: {truck_id: bay_blockage_delay_minutes}
        """
        attribution = self.delay_tree.attribute_bay_congestion(
            held_truck_id=held_truck_id,
            bay_blockage_delays=blocked_trucks,
        )

        for t_id, penalty in attribution.items():
            self.bay_congestion_attributed[t_id] = (
                self.bay_congestion_attributed.get(t_id, 0.0) + penalty
            )

    # ── Global Reward ─────────────────────────────────────────────────

    def get_global_reward(self, truck_id: str) -> float:
        """R_G^k = α · CG^k + (1-α) · OG^k − λ_bay · BayCongestion^k

        This is Phase 1's formula with an additional bay-congestion
        penalty term.  The penalty is scaled by λ_bay (lambda_congestion
        from config, default 0.30).

        When BayCongestion^k = 0 (no bay blocking), this reduces to
        exactly the Phase 1 formula.
        """
        cg = self.global_cu_attributed.get(truck_id, 0.0)
        og = self.global_ou_attributed.get(truck_id, 0.0)
        bay_penalty = self.bay_congestion_attributed.get(truck_id, 0.0)

        rg = self.alpha * cg + (1 - self.alpha) * og - self.lambda_bay * bay_penalty
        return rg

    # ── Total Reward ──────────────────────────────────────────────────

    def get_total_reward(
        self,
        ctx: TruckContext,
        hold_minutes: float,
        truck_id: str,
    ) -> float:
        """R_T^k = β · R_L^k + (1-β) · R_G^k

        Identical to Phase 1.  The richness comes from:
          - R_L consuming value-weighted, SLA-aware CL/OL from ContextEngine
          - R_G incorporating bay-congestion attribution via the extended DT

        Note: R_G may only be fully known after downstream events have
        been processed.  In online RL, this is the reward from the
        *previous* epoch's hold decision.
        """
        r_l = self.compute_local_reward(ctx, hold_minutes)
        r_g = self.get_global_reward(truck_id)

        return self.beta * r_l + (1 - self.beta) * r_g

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self):
        """Clear all accumulated attributions for a new episode."""
        self.delay_tree = LogisticsDelayTree()
        self.global_cu_attributed.clear()
        self.global_ou_attributed.clear()
        self.bay_congestion_attributed.clear()

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_action_index(self, hold_minutes: float) -> int:
        """Convert hold minutes to action index in config.hold_actions."""
        try:
            return self.cfg.hold_actions.index(int(hold_minutes))
        except ValueError:
            return 0
