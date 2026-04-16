"""
Logistics Cross-Docking Microsimulator — main orchestrator.

Direct counterpart of simulator/simulator.py.
Implements a discrete-event microsimulator for the logistics
Hold-or-Not-Hold (HOH) RL problem at a cross-docking hub:
  • Models truck docking, departures, cargo transfers
  • Propagates delays dynamically through route plans and cargo itineraries
  • Tracks cargo movement and failed transfers (missed connections)
  • Exposes a Gym-like step(action) → (state, reward, done, info) API

Aviation → Logistics mapping:
  AirlineNetworkSimulator → CrossDockSimulator
  _handle_departure       → _handle_dock  (truck arrives at hub)
                          + _handle_departure  (truck departs hub)
  _handle_arrival         → _handle_delivery  (truck arrives at depot)
  _handle_pax_connection  → _handle_cargo_transfer
  _rebook_pax             → _rebook_cargo  (next-cycle 24h penalty)
  MetricsTracker          → MetricsTracker (same structure + hub metrics)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phase2_simulator.config import SimConfig
from phase2_simulator.context_engine import ContextEngine, TruckContext
from phase2_simulator.event_engine import EventEngine
from phase2_simulator.generators import (
    DelaySampler,
    _get_day_rows,
    generate_cargo_units,
    generate_hubs,
    generate_truck_plans,
)
from phase2_simulator.models import (
    CargoUnit,
    EventType,
    Hub,
    ScheduledTruck,
    SimEvent,
    TruckPlan,
    TruckState,
    TruckStatus,
)


# ======================================================================
# Metrics Tracker  (mirrors simulator/simulator.py MetricsTracker)
# ======================================================================
class MetricsTracker:
    """Accumulates business and validation metrics during simulation.

    Same structure as simulator/simulator.py MetricsTracker.
    New logistics metrics: failed_transfers, bay_utilisation, throughput.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_trucks = 0
        self.docked_trucks = 0
        self.departed_trucks = 0
        self.delivered_trucks = 0
        self.ontime_departures = 0
        self.departure_delays: List[float] = []
        self.arrival_delays: List[float] = []

        self.total_cargo = 0
        self.connecting_cargo = 0
        self.failed_transfers = 0
        self.successful_transfers = 0
        self.cargo_delays: List[float] = []

        self.hold_decisions: List[float] = []
        self.rewards: List[float] = []

        # Per-day tracking
        self.daily_failed: Dict[int, int] = defaultdict(int)
        self.daily_transfers: Dict[int, int] = defaultdict(int)
        self.daily_ontime: Dict[int, int] = defaultdict(int)
        self.daily_trucks: Dict[int, int] = defaultdict(int)

        # Hub-specific metrics
        self.bay_util_samples: List[float] = []
        self.hub_throughput_samples: List[float] = []

    @property
    def schedule_otp(self) -> float:
        """Fraction of trucks departing within 15 min of schedule."""
        if self.departed_trucks == 0:
            return 1.0
        return self.ontime_departures / self.departed_trucks

    @property
    def avg_departure_delay(self) -> float:
        return float(np.mean(self.departure_delays)) if self.departure_delays else 0.0

    @property
    def avg_arrival_delay(self) -> float:
        return float(np.mean(self.arrival_delays)) if self.arrival_delays else 0.0

    @property
    def failed_transfer_rate(self) -> float:
        total = self.failed_transfers + self.successful_transfers
        return self.failed_transfers / total if total else 0.0

    @property
    def avg_bay_utilisation(self) -> float:
        return float(np.mean(self.bay_util_samples)) if self.bay_util_samples else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "total_trucks": self.total_trucks,
            "docked": self.docked_trucks,
            "departed": self.departed_trucks,
            "delivered": self.delivered_trucks,
            "schedule_OTP": round(self.schedule_otp * 100, 2),
            "avg_departure_delay_min": round(self.avg_departure_delay, 2),
            "avg_arrival_delay_min": round(self.avg_arrival_delay, 2),
            "total_connecting_cargo": self.connecting_cargo,
            "failed_transfers": self.failed_transfers,
            "successful_transfers": self.successful_transfers,
            "failed_transfer_rate_pct": round(self.failed_transfer_rate * 100, 2),
            "avg_hold_min": (
                round(float(np.mean(self.hold_decisions)), 2)
                if self.hold_decisions else 0.0
            ),
            "avg_bay_utilisation": round(self.avg_bay_utilisation, 3),
        }


# ======================================================================
# Main Simulator  (mirrors AirlineNetworkSimulator)
# ======================================================================
class CrossDockSimulator:
    """Discrete-event microsimulator for the logistics Hold-or-Not-Hold problem.

    Usage (Gym-like API — identical to aviation simulator)::

        sim = CrossDockSimulator(cfg)
        state, info = sim.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = sim.step(action)
    """

    def __init__(self, cfg: SimConfig | None = None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.random_seed)

        self.event_engine = EventEngine()
        self.context_engine = ContextEngine(self.cfg)
        self.metrics = MetricsTracker()

        # Data stores  (mirrors AirlineNetworkSimulator)
        self.hubs: Dict[str, Hub] = {}
        self.trucks: Dict[str, TruckState] = {}
        self.truck_plans: Dict[str, TruckPlan] = {}
        self.cargo: Dict[str, CargoUnit] = {}

        # Indexes
        self._trucks_by_route: Dict[str, List[str]] = defaultdict(list)
        self._inbound_cargo: Dict[str, List[str]] = defaultdict(list)   # outbound_tid → [cargo_ids]
        self._outbound_cargo: Dict[str, List[str]] = defaultdict(list)  # inbound_tid → [cargo_ids]
        self._departures_at_hub: Dict[str, List[str]] = defaultdict(list)
        self._next_route_truck: Dict[str, Optional[str]] = {}
        self._prev_route_truck: Dict[str, Optional[str]] = {}

        self._done: bool = True
        self._pending_hold_truck: Optional[str] = None
        self._pending_context: Optional[TruckContext] = None

        # Hub-level telemetry (for BG, ZG computation)
        self._active_bays: int = 0   # trucks currently docked
        self._delayed_inbound: int = 0

    # ==================================================================
    # Gym-like API  (mirrors AirlineNetworkSimulator)
    # ==================================================================
    def reset(self, seed: int | None = None) -> Tuple[TruckContext, Dict[str, Any]]:
        """Reset the simulator for a new episode.

        Mirrors AirlineNetworkSimulator.reset() exactly.
        """
        if seed is not None:
            self.cfg.random_seed = seed
        self.rng = np.random.default_rng(self.cfg.random_seed)

        self.event_engine.clear()
        self.context_engine.reset()
        self.metrics.reset()
        self.trucks.clear()
        self.truck_plans.clear()
        self.cargo.clear()
        self._trucks_by_route.clear()
        self._inbound_cargo.clear()
        self._outbound_cargo.clear()
        self._departures_at_hub.clear()
        self._next_route_truck.clear()
        self._prev_route_truck.clear()
        self._active_bays = 0
        self._delayed_inbound = 0

        # 1. Generate hub network
        self.hubs = generate_hubs(self.cfg)

        # 2. Generate truck plans and schedules for all days
        all_scheduled: List[ScheduledTruck] = []
        for day in range(self.cfg.num_days):
            delay_sampler = DelaySampler(self.cfg, self.rng, day_index=day)
            plans, schedules = generate_truck_plans(
                self.cfg, self.hubs, day_index=day, rng=self.rng
            )
            for tp in plans:
                self.truck_plans[tp.route_id] = tp
            all_scheduled.extend(schedules)

            # Sample intrinsic delays now, store in truck states
            for st in schedules:
                ts = TruckState(truck=st)
                ts.intrinsic_departure_delay = delay_sampler.sample_departure_delay(st)
                ts.road_delay = delay_sampler.sample_road_delay(st)
                ts.bay_dwell_delay = delay_sampler.sample_bay_dwell_delay()
                ts.bay_arrival_delay = delay_sampler.sample_bay_dwell_delay()
                self.trucks[st.truck_id] = ts
                self._trucks_by_route[st.route_id].append(st.truck_id)
                self._departures_at_hub[st.origin_hub].append(st.truck_id)

            # Generate cargo units for this day
            day_cargo = generate_cargo_units(
                self.cfg, schedules, self.hubs, day_index=day, rng=self.rng
            )
            for cu in day_cargo:
                self.cargo[cu.cargo_id] = cu
                if len(cu.legs) >= 2:
                    # outbound_tid → cargo connecting INTO it
                    self._inbound_cargo[cu.legs[1]].append(cu.cargo_id)
                if cu.legs:
                    self._outbound_cargo[cu.legs[0]].append(cu.cargo_id)

        # Sort routes and build prev/next chain
        for route_id in self._trucks_by_route:
            tids = self._trucks_by_route[route_id]
            tids.sort(key=lambda tid: self.trucks[tid].truck.scheduled_departure)
            for i, tid in enumerate(tids):
                self._prev_route_truck[tid] = tids[i - 1] if i > 0 else None
                self._next_route_truck[tid] = tids[i + 1] if i < len(tids) - 1 else None

        for hub_id in self._departures_at_hub:
            self._departures_at_hub[hub_id].sort(
                key=lambda tid: self.trucks[tid].truck.scheduled_departure
            )

        # Enrich truck states with cargo-level attributes from PDF §2
        self._enrich_truck_states()

        # Metrics totals
        self.metrics.total_trucks = len(self.trucks)
        self.metrics.total_cargo = sum(cu.unit_count for cu in self.cargo.values())
        self.metrics.connecting_cargo = sum(
            cu.unit_count for cu in self.cargo.values() if len(cu.legs) >= 2
        )

        # Register event handlers and schedule initial HOLD_DECISION events
        self._register_handlers()
        self._schedule_initial_events()

        self._done = False
        return self._advance_to_next_hold()

    def step(self, action: int) -> Tuple[TruckContext, float, bool, Dict[str, Any]]:
        """Apply hold action and advance simulation to next decision point.

        Mirrors AirlineNetworkSimulator.step() exactly.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        hold_minutes = self.cfg.hold_actions[action]
        tid = self._pending_hold_truck
        ts = self.trucks[tid]

        # Apply hold
        ts.hold_delay = float(hold_minutes)
        ts.hold_decided = True
        ts.hold_action = hold_minutes
        self.metrics.hold_decisions.append(float(hold_minutes))

        # Update propagated delay from previous route truck
        ts.propagated_departure_delay = self._compute_propagated_delay_dynamic(tid)

        # Compute actual times
        ts.compute_actual_times()

        # Update bay utilisation telemetry
        self._update_bay_telemetry(ts.truck.origin_hub, hold_minutes > 0)

        # Schedule truck departure event
        self.event_engine.schedule(SimEvent(
            time=ts.actual_departure,
            event_type=EventType.TRUCK_DEPARTURE,
            truck_id=tid,
        ))

        # Compute reward
        reward = self._compute_reward(tid, hold_minutes)

        # Advance to next decision
        next_state, info = self._advance_to_next_hold()
        return next_state, reward, self._done, info

    # ==================================================================
    # Dynamic delay propagation  (mirrors AirlineNetworkSimulator exactly)
    # ==================================================================
    def _compute_propagated_delay_dynamic(self, truck_id: str) -> float:
        """Propagate delay from previous truck on the same route.

        Identical logic to AirlineNetworkSimulator._compute_propagated_delay_dynamic().
        Hold delays cascade: holding truck k → delays k's departure →
        next truck on k's route gets a propagated departure delay.
        """
        prev_tid = self._prev_route_truck.get(truck_id)
        if prev_tid is None:
            return 0.0

        prev_ts = self.trucks[prev_tid]
        ts = self.trucks[truck_id]

        if prev_ts.actual_arrival is not None:
            prev_arrival = prev_ts.actual_arrival
        else:
            prev_arrival = (
                prev_ts.truck.scheduled_arrival + prev_ts.total_arrival_delay
            )

        prev_arr_delay = prev_arrival - prev_ts.truck.scheduled_arrival
        turnaround_sched = (
            ts.truck.scheduled_departure - prev_ts.truck.scheduled_arrival
        )
        slack = max(0, turnaround_sched - self.cfg.min_turnaround)
        return max(0.0, prev_arr_delay - slack)

    # ==================================================================
    # Truck state enrichment — fill PDF §2 local fields from cargo data
    # ==================================================================
    def _enrich_truck_states(self) -> None:
        """Populate cargo-derived fields on each TruckState (V_k, Q_k, X_k, E_k, N_in)."""
        for tid, ts in self.trucks.items():
            cargo_ids = self._inbound_cargo.get(tid, [])
            if not cargo_ids:
                continue
            cargo_objs = [self.cargo[cid] for cid in cargo_ids if cid in self.cargo]
            if not cargo_objs:
                continue

            # V_k: mean value score of connecting cargo
            ts.cargo_value_score = float(np.mean([c.value_score for c in cargo_objs]))
            # Q_k: connecting cargo units / truck capacity
            total_units = sum(c.unit_count for c in cargo_objs)
            ts.cargo_volume_fraction = min(1.0, total_units / max(1, ts.truck.cargo_capacity))
            # X_k: max SLA urgency among connecting cargo (worst-case)
            ts.sla_urgency = max(c.sla_urgency for c in cargo_objs)
            # E_k: fraction of connecting cargo that is perishable
            perishable_units = sum(c.unit_count for c in cargo_objs if c.is_perishable)
            ts.perishability_fraction = perishable_units / max(1, total_units)
            # N_in: distinct inbound trucks
            inbound_tids = {c.legs[0] for c in cargo_objs if len(c.legs) >= 2}
            ts.n_inbound_trucks = len(inbound_tids)
            # L_k: driver hours remaining — derived from fatigue score (inverted)
            # No per-truck CSV value here; we use a reasonable default range
            ts.driver_hours_remaining = float(self.rng.uniform(120, 270))
            # F_k: deadline pressure — inversely proportional to SLA urgency
            ts.deadline_pressure = 1.0 - (ts.sla_urgency / 2.0)

    # ==================================================================
    # Event scheduling & handlers  (mirrors AirlineNetworkSimulator)
    # ==================================================================
    def _schedule_initial_events(self) -> None:
        """Schedule HOLD_DECISION events 10 minutes before each truck's scheduled departure."""
        for tid, ts in self.trucks.items():
            hold_time = ts.truck.scheduled_departure - 10
            self.event_engine.schedule(SimEvent(
                time=max(0.0, hold_time),
                event_type=EventType.HOLD_DECISION,
                truck_id=tid,
            ))

    def _register_handlers(self) -> None:
        self.event_engine.register_handler(
            EventType.TRUCK_DOCK, self._handle_dock
        )
        self.event_engine.register_handler(
            EventType.TRUCK_DEPARTURE, self._handle_departure
        )
        self.event_engine.register_handler(
            EventType.CARGO_TRANSFER_CHECK, self._handle_cargo_transfer
        )

    def _handle_dock(self, event: SimEvent) -> Optional[List[SimEvent]]:
        """Truck arrives and docks at the hub.

        Aviation analog: _handle_departure (flight departs from gate).
        Here: truck arrives from road and occupies a bay.
        """
        tid = event.truck_id
        ts = self.trucks.get(tid)
        if ts is None or ts.status != TruckStatus.SCHEDULED:
            return None

        ts.status = TruckStatus.DOCKED
        ts.actual_dock = event.time
        self._active_bays = min(self._active_bays + 1, self.cfg.num_bays)

        self.metrics.docked_trucks += 1
        return None

    def _handle_departure(self, event: SimEvent) -> Optional[List[SimEvent]]:
        """Truck departs the hub with (or without) connecting cargo.

        Mirrors _handle_departure from simulator/simulator.py.
        """
        tid = event.truck_id
        ts = self.trucks.get(tid)
        if ts is None or ts.status not in (TruckStatus.SCHEDULED, TruckStatus.DOCKED):
            return None

        ts.status = TruckStatus.DEPARTED
        ts.compute_actual_times()
        self._active_bays = max(0, self._active_bays - 1)

        self.metrics.departed_trucks += 1
        self.metrics.departure_delays.append(ts.departure_delay_D)
        day = int(ts.actual_departure // 1440)
        self.metrics.daily_trucks[day] = self.metrics.daily_trucks.get(day, 0) + 1

        if ts.departure_delay_D <= self.cfg.ontime_threshold:
            self.metrics.ontime_departures += 1
            self.metrics.daily_ontime[day] = self.metrics.daily_ontime.get(day, 0) + 1

        # Record operator utility
        dep_delay = ts.departure_delay_D
        ou = 1.0 - min(max(dep_delay, 0), self.cfg.delta_F) / self.cfg.delta_F
        self.context_engine.record_global_operator_utility(event.time, ou)

        # Record bay utilisation
        bay_util = self._active_bays / max(1, self.cfg.num_bays)
        self.context_engine.record_bay_utilisation(event.time, bay_util)
        self.metrics.bay_util_samples.append(bay_util)

        # Schedule delivery event
        return [SimEvent(
            time=ts.actual_arrival,
            event_type=EventType.CARGO_TRANSFER_CHECK,
            truck_id=tid,
        )]

    def _handle_cargo_transfer(self, event: SimEvent) -> Optional[List[SimEvent]]:
        """Check whether connecting cargo makes it onto the outbound truck.

        Mirrors _handle_pax_connection from simulator/simulator.py.
        'Missed transfer' → next-cycle penalty (24h).
        """
        tid = event.truck_id
        ts = self.trucks.get(tid)
        if ts is None:
            return None

        # Mark truck as delivered
        if ts.status == TruckStatus.DEPARTED:
            ts.status = TruckStatus.DELIVERED
            ts.actual_arrival = event.time
            self.metrics.delivered_trucks += 1
            arr_delay = ts.arrival_delay_A
            self.metrics.arrival_delays.append(arr_delay)

            # Record cargo utility for this delivery
            cu_val = 1.0 - min(max(arr_delay, 0), self.cfg.delta_C) / self.cfg.delta_C
            self.context_engine.record_global_cargo_utility(event.time, cu_val)

        # Check cargo that was supposed to connect onto a truck departing from tid's dest_hub
        outbound_cargo_ids = self._outbound_cargo.get(tid, [])
        new_events = []

        for cid in outbound_cargo_ids:
            cu = self.cargo.get(cid)
            if cu is None or len(cu.legs) < 2 or cu.legs[0] != tid:
                continue

            inbound_ts = self.trucks.get(cu.legs[0])
            outbound_ts = self.trucks.get(cu.legs[1])
            if inbound_ts is None or outbound_ts is None:
                continue

            actual_arrival = inbound_ts.actual_arrival
            if actual_arrival is None:
                actual_arrival = inbound_ts.truck.scheduled_arrival + inbound_ts.total_arrival_delay

            if outbound_ts.actual_departure is not None:
                outbound_dep = outbound_ts.actual_departure
            elif outbound_ts.hold_decided:
                outbound_dep = (
                    outbound_ts.truck.scheduled_departure + outbound_ts.total_departure_delay
                )
            else:
                outbound_dep = (
                    outbound_ts.truck.scheduled_departure
                    + max(outbound_ts.intrinsic_departure_delay,
                          outbound_ts.propagated_departure_delay)
                    + outbound_ts.bay_dwell_delay
                )

            dest_hub = inbound_ts.truck.dest_hub
            hub_obj = self.hubs.get(dest_hub)
            mtt = hub_obj.min_transfer_time if hub_obj else self.cfg.min_transfer_time_spoke
            transfer_window = outbound_dep - actual_arrival
            day = int(actual_arrival // 1440)

            if transfer_window >= mtt:
                # Successful transfer
                self.metrics.successful_transfers += cu.unit_count
                self.metrics.daily_transfers[day] = (
                    self.metrics.daily_transfers.get(day, 0) + cu.unit_count
                )
                cu.missed_transfer = False
                cu.delay_to_destination = max(0.0, inbound_ts.arrival_delay_A)
                sigma = self.context_engine._cargo_disutility(
                    cu.delay_to_destination, cu.sla_urgency
                )
                self.context_engine.record_global_cargo_utility(event.time, 1.0 - sigma)
                # Record throughput
                self.context_engine.record_hub_throughput(event.time, 1.0)
            else:
                # Failed transfer — next-cycle penalty
                self.metrics.failed_transfers += cu.unit_count
                self.metrics.daily_failed[day] = (
                    self.metrics.daily_failed.get(day, 0) + cu.unit_count
                )
                cu.missed_transfer = True
                rebook_delay = self._rebook_cargo(cu, dest_hub, actual_arrival)
                cu.delay_to_destination = rebook_delay
                self.metrics.cargo_delays.append(rebook_delay * cu.unit_count)
                sigma = self.context_engine._cargo_disutility(rebook_delay, cu.sla_urgency)
                self.context_engine.record_global_cargo_utility(event.time, 1.0 - sigma)
                self.context_engine.record_failed_transfer(event.time, 1.0)

        # Update inbound queue depth
        self.context_engine.record_inbound_queue(
            event.time, float(self._delayed_inbound) / max(1, self.cfg.num_bays)
        )

        return new_events if new_events else None

    # ==================================================================
    # Cargo rebooking  (replaces _rebook_pax)
    # ==================================================================
    def _rebook_cargo(
        self, cargo: CargoUnit, current_hub: str, current_time: float
    ) -> float:
        """Find next available truck from current_hub to cargo's destination.

        If no same-day truck found, return the next-cycle penalty (24h).
        Mirrors _rebook_pax from simulator/simulator.py.
        """
        final_dest = cargo.destination_hub
        candidates = self._departures_at_hub.get(current_hub, [])
        best_delay = self.cfg.next_cycle_penalty_minutes

        for tid in candidates:
            ts = self.trucks[tid]
            if ts.truck.dest_hub != final_dest:
                continue
            if ts.truck.scheduled_departure <= current_time:
                continue
            est_arr = ts.truck.scheduled_arrival + ts.total_arrival_delay
            orig_arr = self.trucks[cargo.legs[-1]].truck.scheduled_arrival
            delay = est_arr - orig_arr
            if 0 < delay < best_delay:
                best_delay = delay
                cargo.rebooked_truck_id = tid

        return max(0.0, best_delay)

    # ==================================================================
    # Bay telemetry helper
    # ==================================================================
    def _update_bay_telemetry(self, hub_id: str, is_held: bool) -> None:
        """Update bay utilisation and delayed inbound count."""
        if is_held:
            self._delayed_inbound = min(
                self._delayed_inbound + 1, self.cfg.hub.num_routes
            )
        bay_util = self._active_bays / max(1, self.cfg.num_bays)
        ts_now = self.event_engine.current_time
        self.context_engine.record_bay_utilisation(ts_now, bay_util)

    # ==================================================================
    # Advance to next HOLD_DECISION  (mirrors _advance_to_next_hnh)
    # ==================================================================
    def _advance_to_next_hold(self) -> Tuple[TruckContext, Dict[str, Any]]:
        """Run simulation until next HOLD_DECISION event.

        Mirrors _advance_to_next_hnh() from simulator/simulator.py.
        """
        hold_event = self.event_engine.run_until_hold_decision()
        if hold_event is None:
            self.event_engine.drain()
            self._done = True
            return TruckContext(truck_id="DONE"), self.metrics.summary()

        tid = hold_event.truck_id
        ts = self.trucks[tid]
        self._pending_hold_truck = tid

        # Dynamic propagated delay
        ts.propagated_departure_delay = self._compute_propagated_delay_dynamic(tid)

        # Gather connecting cargo
        conn_cargo_ids = self._inbound_cargo.get(tid, [])
        conn_cargo = [self.cargo[cid] for cid in conn_cargo_ids if cid in self.cargo]

        # Bay utilisation at decision time
        bay_util = self._active_bays / max(1, self.cfg.num_bays)
        ts.bay_utilisation_at_decision = bay_util

        ctx = self.context_engine.build_context(
            truck_state=ts,
            connecting_cargo=conn_cargo,
            hold_actions=self.cfg.hold_actions,
            hubs=self.hubs,
            truck_map=self.trucks,
            current_time=hold_event.time,
        )
        self._pending_context = ctx

        info = {
            "truck_id": tid,
            "origin_hub": ts.truck.origin_hub,
            "dest_hub": ts.truck.dest_hub,
            "connecting_cargo_count": len(conn_cargo),
            "scheduled_departure": ts.truck.scheduled_departure,
            "intrinsic_delay": ts.intrinsic_departure_delay,
            "propagated_delay": ts.propagated_departure_delay,
            "tau_star": ctx.tau_star,
            "sla_urgency": ts.sla_urgency,
            "driver_hours_remaining": ts.driver_hours_remaining,
            "bay_utilisation": bay_util,
        }
        return ctx, info

    # ==================================================================
    # Reward computation  (mirrors _compute_basic_reward)
    # ==================================================================
    def _compute_reward(self, truck_id: str, hold_minutes: float) -> float:
        """Compute reward for a hold decision.

        R_T_k = β·R_L_k + (1-β)·R_G_k  (PDF §5.5)
        R_L_k = α·C_L_k + (1-α)·O_L_k
        R_G_k = α·C_G_k + (1-α)·O_G_k
        """
        ctx = self._pending_context
        if ctx is None:
            return 0.0

        alpha = self.cfg.alpha
        beta = self.cfg.beta

        # Local reward
        action_idx = self.cfg.hold_actions.index(int(hold_minutes)) if int(hold_minutes) in self.cfg.hold_actions else 0
        cl_val = ctx.CL[action_idx] if action_idx < len(ctx.CL) else 0.5
        ol_val = ctx.OL[action_idx] if action_idx < len(ctx.OL) else 0.5
        R_L = alpha * cl_val + (1 - alpha) * ol_val

        # Global reward
        R_G = alpha * ctx.CG + (1 - alpha) * ctx.OG

        reward = float(beta * R_L + (1 - beta) * R_G)
        self.metrics.rewards.append(reward)
        return reward

    # ==================================================================
    # Baseline policies  (mirrors _select_baseline_action)
    # ==================================================================
    def run_episode(
        self,
        policy: str = "no_hold",
        max_hold: int = 15,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        """Run a full episode with a baseline policy.

        Mirrors AirlineNetworkSimulator.run_episode() exactly.
        """
        ctx, info = self.reset(seed=seed)
        done = False
        while not done:
            action = self._select_baseline_action(policy, max_hold)
            ctx, reward, done, info = self.step(action)
        return self.metrics.summary()

    def _select_baseline_action(self, policy: str, max_hold: int = 15) -> int:
        """Baseline action selection.

        no_hold   : always τ=0
        heuristic : hold only when connecting cargo is at risk of missing transfer
        random    : uniform random

        Mirrors _select_baseline_action from simulator/simulator.py.
        """
        if policy == "no_hold":
            return 0

        if policy.startswith("heuristic"):
            tid = self._pending_hold_truck
            ts = self.trucks[tid]

            conn_cargo_ids = self._inbound_cargo.get(tid, [])
            if not conn_cargo_ids:
                return 0

            est_dep = (
                ts.truck.scheduled_departure
                + max(ts.intrinsic_departure_delay, ts.propagated_departure_delay)
                + ts.bay_dwell_delay
            )

            conn_hub = ts.truck.origin_hub
            hub_obj = self.hubs.get(conn_hub)
            mtt = hub_obj.min_transfer_time if hub_obj else self.cfg.min_transfer_time_spoke

            max_needed_hold = 0.0

            for cid in conn_cargo_ids:
                cu = self.cargo.get(cid)
                if cu is None or len(cu.legs) < 2:
                    continue

                inbound_tid = cu.legs[0]
                inbound_ts = self.trucks.get(inbound_tid)
                if inbound_ts is None:
                    continue

                if inbound_ts.actual_arrival is not None:
                    inbound_arr = inbound_ts.actual_arrival
                else:
                    inbound_arr = (
                        inbound_ts.truck.scheduled_arrival
                        + inbound_ts.total_arrival_delay
                    )

                window = est_dep - inbound_arr
                if window >= mtt:
                    continue

                needed = mtt - window
                if needed <= max_hold:
                    max_needed_hold = max(max_needed_hold, needed)

            if max_needed_hold <= 0:
                return 0

            for idx, tau in enumerate(self.cfg.hold_actions):
                if tau >= max_needed_hold:
                    return idx
            for idx in range(len(self.cfg.hold_actions) - 1, -1, -1):
                if self.cfg.hold_actions[idx] <= max_hold:
                    return idx
            return 0

        if policy == "random":
            return int(self.rng.integers(0, len(self.cfg.hold_actions)))
        return 0

    # ==================================================================
    # Data access helpers  (mirrors AirlineNetworkSimulator interface)
    # ==================================================================
    def get_truck_state(self, truck_id: str) -> Optional[TruckState]:
        return self.trucks.get(truck_id)

    def get_route_trucks(self, route_id: str) -> List[str]:
        return self._trucks_by_route.get(route_id, [])

    def get_inbound_cargo_trucks(self, truck_id: str) -> List[str]:
        cargo_ids = self._inbound_cargo.get(truck_id, [])
        feeders = set()
        for cid in cargo_ids:
            cu = self.cargo.get(cid)
            if cu and len(cu.legs) >= 2:
                feeders.add(cu.legs[0])
        return list(feeders)

    def get_previous_route_truck(self, truck_id: str) -> Optional[str]:
        return self._prev_route_truck.get(truck_id)

    def get_next_route_truck(self, truck_id: str) -> Optional[str]:
        return self._next_route_truck.get(truck_id)
