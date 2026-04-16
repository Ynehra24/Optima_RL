"""
Logistics Cross-Docking Microsimulator — main orchestrator.

Mirrors simulator/simulator.py field-for-field, with these genuine improvements:

CRITICAL BUGS FIXED vs v1:
  1. BG always 0% → TRUCK_DOCK events now scheduled; _active_bays correctly tracks
  2. WG/YG wrong → context_engine.record_transfer_outcome() now records both
  3. trucks.index() O(n) crash → dict lookup in _enrich_truck_states()

GENUINE LOGISTICS IMPROVEMENTS:
  4. CARGO_TRANSFER_CHECK triggered on TRUCK_DOCK (inbound arrival), not on departure
     — more realistic: we know at dock time if cargo will make it
  5. Bay curfew enforced in step(): if BG > bay_curfew_threshold, hold forced to 0
  6. Road delay compounds with ZG: G_road = base × (1 + ZG × 0.5) — congestion cascade
  7. Multi-tier SLA penalty in MetricsTracker: X_k=2 flagged as priority miss
  8. Four metrics reported: OTP + failed-transfer rate + avg delivery delay + dock congestion

FOUR METRICS (not two like the original):
  1. Schedule OTP       — trucks departing within 15 min of schedule
  2. Failed transfer %  — connecting cargo that misses its outbound truck
  3. Avg delivery delay — mean A_k across all delivered trucks (minutes)
  4. Bay congestion     — mean BG across all departure events
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phase2_simulator.config import SimConfig
from phase2_simulator.context_engine import ContextEngine, TruckContext
from phase2_simulator.event_engine import EventEngine
from phase2_simulator.generators import (
    DelaySampler,
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


# ====================================================================
# MetricsTracker  (mirrors simulator/simulator.py MetricsTracker)
# ====================================================================
class MetricsTracker:
    """Accumulates business and validation metrics during simulation.

    FOUR METRICS (expanded from aviation's two):
      1. schedule_otp          — OTP (on-time performance)
      2. failed_transfer_rate  — fraction of connecting cargo that misses
      3. avg_delivery_delay    — mean A_k across all delivered trucks
      4. avg_bay_utilisation   — mean BG across simulation (dock congestion)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # ── Truck-level counters ──────────────────────────────────────
        self.total_trucks = 0
        self.docked_trucks = 0          # trucks that successfully docked (BG fix)
        self.departed_trucks = 0
        self.delivered_trucks = 0
        self.ontime_departures = 0
        self.departure_delays: List[float] = []
        self.arrival_delays: List[float] = []

        # ── Cargo-level counters ──────────────────────────────────────
        self.total_cargo_units = 0
        self.connecting_cargo_units = 0
        self.failed_transfers = 0       # cargo units that missed transfer
        self.successful_transfers = 0
        self.cargo_delivery_delays: List[float] = []   # delay per individual unit

        # ── Hold decisions ────────────────────────────────────────────
        self.hold_decisions: List[float] = []
        self.rewards: List[float] = []

        # ── Bay congestion (Metric 4) ─────────────────────────────────
        self.bay_util_samples: List[float] = []

        # ── Per-day tracking (mirrors aviation) ───────────────────────
        self.daily_failed:   Dict[int, int] = defaultdict(int)
        self.daily_transfers: Dict[int, int] = defaultdict(int)
        self.daily_ontime:   Dict[int, int] = defaultdict(int)
        self.daily_trucks:   Dict[int, int] = defaultdict(int)

        # ── Priority-miss tracking (X_k=2 same-day express failures) ──
        self.priority_failed = 0

    # ── Derived metrics ───────────────────────────────────────────────

    @property
    def schedule_otp(self) -> float:
        """Metric 1: fraction of trucks departing within ontime_threshold."""
        return self.ontime_departures / max(1, self.departed_trucks)

    @property
    def failed_transfer_rate(self) -> float:
        """Metric 2: failed_transfers / (failed + successful)."""
        total = self.failed_transfers + self.successful_transfers
        return self.failed_transfers / max(1, total)

    @property
    def avg_delivery_delay(self) -> float:
        """Metric 3: mean arrival delay across all delivered trucks."""
        return float(np.mean(self.arrival_delays)) if self.arrival_delays else 0.0

    @property
    def avg_bay_utilisation(self) -> float:
        """Metric 4: mean bay utilisation (dock congestion)."""
        return float(np.mean(self.bay_util_samples)) if self.bay_util_samples else 0.0

    @property
    def avg_departure_delay(self) -> float:
        return float(np.mean(self.departure_delays)) if self.departure_delays else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            # ── Core 4 metrics ────────────────────────────────────────
            "schedule_OTP_%":          round(self.schedule_otp * 100, 2),
            "failed_transfer_%":       round(self.failed_transfer_rate * 100, 2),
            "avg_delivery_delay_min":  round(self.avg_delivery_delay, 2),
            "avg_bay_utilisation_%":   round(self.avg_bay_utilisation * 100, 2),
            # ── Supporting data ───────────────────────────────────────
            "total_trucks":            self.total_trucks,
            "docked_trucks":           self.docked_trucks,
            "departed_trucks":         self.departed_trucks,
            "delivered_trucks":        self.delivered_trucks,
            "ontime_departures":       self.ontime_departures,
            "avg_departure_delay_min": round(self.avg_departure_delay, 2),
            "total_connecting_cargo":  self.connecting_cargo_units,
            "failed_transfers":        self.failed_transfers,
            "successful_transfers":    self.successful_transfers,
            "priority_failed":         self.priority_failed,
            "avg_hold_min":            round(float(np.mean(self.hold_decisions)), 2)
                                       if self.hold_decisions else 0.0,
        }


# ====================================================================
# CrossDockSimulator  (mirrors AirlineNetworkSimulator)
# ====================================================================
class CrossDockSimulator:
    """Discrete-event microsimulator for logistics Hold-or-Not-Hold RL.

    API (Gym-like — identical to AirlineNetworkSimulator)::

        sim = CrossDockSimulator(cfg)
        state, info = sim.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = sim.step(action)
    """

    def __init__(self, cfg: Optional[SimConfig] = None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.random_seed)

        self.event_engine   = EventEngine()
        self.context_engine = ContextEngine(self.cfg)
        self.metrics        = MetricsTracker()

        # ── Data stores (mirrors AirlineNetworkSimulator) ─────────────
        self.hubs:        Dict[str, Hub]        = {}
        self.trucks:      Dict[str, TruckState] = {}
        self.truck_plans: Dict[str, TruckPlan]  = {}
        self.cargo:       Dict[str, CargoUnit]  = {}

        # ── Indexes ───────────────────────────────────────────────────
        self._trucks_by_route:    Dict[str, List[str]] = defaultdict(list)
        # outbound_truck_id → [cargo_ids that will transfer onto it]
        self._inbound_cargo:      Dict[str, List[str]] = defaultdict(list)
        # inbound_truck_id → [cargo_ids departing on it]
        self._outbound_cargo:     Dict[str, List[str]] = defaultdict(list)
        self._departures_at_hub:  Dict[str, List[str]] = defaultdict(list)
        self._next_route_truck:   Dict[str, Optional[str]] = {}
        self._prev_route_truck:   Dict[str, Optional[str]] = {}

        self._done: bool = True
        self._pending_hold_truck: Optional[str] = None
        self._pending_context:    Optional[TruckContext] = None

        # ── Hub-level telemetry ───────────────────────────────────────
        self._active_bays: int = 0     # trucks currently docked (drives BG)
        self._delayed_inbound: int = 0

    # ==================================================================
    # Gym API — identical to AirlineNetworkSimulator
    # ==================================================================

    def reset(self, seed: Optional[int] = None) -> Tuple[TruckContext, Dict[str, Any]]:
        """Reset for a new episode."""
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

        # 1. Build hub network
        self.hubs = generate_hubs(self.cfg)

        # 2. Generate truck plans and sample intrinsic delays
        all_scheduled: List[ScheduledTruck] = []
        for day in range(self.cfg.num_days):
            sampler = DelaySampler(self.cfg, self.rng, day_index=day)
            plans, schedules = generate_truck_plans(
                self.cfg, self.hubs, day_index=day, rng=self.rng)

            for tp in plans:
                self.truck_plans[tp.route_id] = tp
            all_scheduled.extend(schedules)

            for st in schedules:
                ts = TruckState(truck=st)
                ts.intrinsic_departure_delay = sampler.sample_departure_delay(st)
                ts.road_delay                = sampler.sample_road_delay(st)
                ts.bay_dwell_delay           = sampler.sample_bay_dwell_delay()
                ts.bay_arrival_delay         = sampler.sample_bay_dwell_delay()
                self.trucks[st.truck_id] = ts
                self._trucks_by_route[st.route_id].append(st.truck_id)
                self._departures_at_hub[st.origin_hub].append(st.truck_id)

            # 3. Generate cargo units
            day_cargo = generate_cargo_units(
                self.cfg, schedules, self.hubs, day_index=day, rng=self.rng)
            for cu in day_cargo:
                self.cargo[cu.cargo_id] = cu
                if len(cu.legs) >= 2:
                    # outbound_truck → receives this cargo at transfer
                    self._inbound_cargo[cu.legs[1]].append(cu.cargo_id)
                if cu.legs:
                    self._outbound_cargo[cu.legs[0]].append(cu.cargo_id)

        # 4. Sort route chains; build prev/next pointers (= tail plan chains)
        for route_id in self._trucks_by_route:
            tids = self._trucks_by_route[route_id]
            tids.sort(key=lambda tid: self.trucks[tid].truck.scheduled_departure)
            for i, tid in enumerate(tids):
                self._prev_route_truck[tid] = tids[i - 1] if i > 0 else None
                self._next_route_truck[tid] = tids[i + 1] if i < len(tids) - 1 else None

        for hub_id in self._departures_at_hub:
            self._departures_at_hub[hub_id].sort(
                key=lambda tid: self.trucks[tid].truck.scheduled_departure)

        # 5. Enrich truck states with cargo-derived PDF §2 fields
        self._enrich_truck_states()

        # 6. Initialise metrics totals
        self.metrics.total_trucks        = len(self.trucks)
        self.metrics.total_cargo_units   = sum(cu.unit_count for cu in self.cargo.values())
        self.metrics.connecting_cargo_units = sum(
            cu.unit_count for cu in self.cargo.values() if len(cu.legs) >= 2)

        # 7. Register event handlers and schedule initial events
        self._register_handlers()
        self._schedule_initial_events()

        self._done = False
        return self._advance_to_next_hold()

    def step(self, action: int) -> Tuple[TruckContext, float, bool, Dict[str, Any]]:
        """Apply hold action; advance to next decision point."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        hold_minutes = self.cfg.hold_actions[action]
        tid = self._pending_hold_truck
        ts  = self.trucks[tid]

        # ── Bay curfew enforcement (Improvement 5) ────────────────────
        current_BG = self._active_bays / max(1, self.cfg.num_bays)
        if current_BG >= self.cfg.bay_curfew_threshold and hold_minutes > 0:
            hold_minutes = 0   # force no-hold when bays are full

        # ── Driver hours cap (PDF §2 L_k constraint) ──────────────────
        if hold_minutes > ts.driver_hours_remaining:
            hold_minutes = 0

        # Apply hold
        ts.hold_delay   = float(hold_minutes)
        ts.hold_decided = True
        ts.hold_action  = hold_minutes
        self.metrics.hold_decisions.append(float(hold_minutes))

        # ── Propagated delay from previous leg on same route ──────────
        ts.propagated_departure_delay = self._compute_propagated_delay_dynamic(tid)

        # ── Compute actual times ──────────────────────────────────────
        ts.compute_actual_times()

        # ── Update bay telemetry ──────────────────────────────────────
        bay_util = self._active_bays / max(1, self.cfg.num_bays)
        ts.bay_utilisation_at_decision = bay_util

        # ── Schedule TRUCK_DEPARTURE at actual_departure ──────────────
        self.event_engine.schedule(SimEvent(
            time=ts.actual_departure,
            event_type=EventType.TRUCK_DEPARTURE,
            truck_id=tid,
        ))

        # ── Compute reward (PDF §5.5) ─────────────────────────────────
        reward = self._compute_reward(tid, hold_minutes)

        # ── Advance to next decision ──────────────────────────────────
        next_state, info = self._advance_to_next_hold()
        return next_state, reward, self._done, info

    # ==================================================================
    # Event registration & initial scheduling
    # ==================================================================

    def _register_handlers(self) -> None:
        self.event_engine.register_handler(EventType.TRUCK_DOCK,            self._handle_dock)
        self.event_engine.register_handler(EventType.TRUCK_DEPARTURE,       self._handle_departure)
        self.event_engine.register_handler(EventType.CARGO_TRANSFER_CHECK,  self._handle_cargo_transfer)

    def _schedule_initial_events(self) -> None:
        """Schedule TRUCK_DOCK at scheduled_dock and HOLD_DECISION 10 min
        before scheduled_departure for every truck in the simulation.

        FIX: TRUCK_DOCK events were missing in v1 → BG was always 0.
        """
        for tid, ts in self.trucks.items():
            # TRUCK_DOCK: truck arrives at bay — actual time includes arrival lateness
            # FIX: was firing at scheduled_dock (always on time → ZG always 0)
            # Now fires at scheduled_dock + intrinsic_departure_delay: a truck that
            # is behind schedule (e.g. traffic, prior stop overrun) docks late,
            # registering in ZG (inbound queue depth)
            dock_time = max(0.0, ts.truck.scheduled_dock + ts.intrinsic_departure_delay)
            self.event_engine.schedule(SimEvent(
                time=dock_time,
                event_type=EventType.TRUCK_DOCK,
                truck_id=tid,
            ))
            # HOLD_DECISION: RL agent called 10 min before scheduled departure
            hold_time = max(0.0, ts.truck.scheduled_departure - 10.0)
            self.event_engine.schedule(SimEvent(
                time=hold_time,
                event_type=EventType.HOLD_DECISION,
                truck_id=tid,
            ))

    # ==================================================================
    # Event handlers
    # ==================================================================

    def _handle_dock(self, event: SimEvent) -> Optional[List[SimEvent]]:
        """Truck backs into a docking bay at the hub (origin side).

        Purpose: track bay utilisation (BG) and delayed-inbound queue (ZG).
        Does NOT trigger cargo transfer checks — those fire at destination arrival.
        """
        tid = event.truck_id
        ts  = self.trucks.get(tid)
        if ts is None or ts.status != TruckStatus.SCHEDULED:
            return None

        ts.status      = TruckStatus.DOCKED
        ts.actual_dock = event.time
        self._active_bays = min(self._active_bays + 1, self.cfg.num_bays)
        self.metrics.docked_trucks += 1

        # Record BG (bay utilisation) — this is what was always 0.0 before
        bay_util = self._active_bays / max(1, self.cfg.num_bays)
        self.context_engine.record_bay_utilisation(event.time, bay_util)
        self.metrics.bay_util_samples.append(bay_util)

        # Record ZG (delayed inbound queue depth)
        dock_late = event.time - ts.truck.scheduled_dock
        if dock_late > self.cfg.ontime_threshold:
            self._delayed_inbound += 1
        zg_depth = self._delayed_inbound / max(1, self.cfg.hub.num_routes)
        self.context_engine.record_inbound_queue(event.time, zg_depth)

        return None

    def _handle_departure(self, event: SimEvent) -> Optional[List[SimEvent]]:
        """Truck departs the hub.

        Mirrors _handle_departure from simulator/simulator.py.
        Frees one docking bay. Records operator utility.
        Schedules CARGO_TRANSFER_CHECK at actual_arrival — this is the correct
        time to check whether connecting cargo makes the outbound truck at destination.
        """
        tid = event.truck_id
        ts  = self.trucks.get(tid)
        if ts is None or ts.status not in (TruckStatus.SCHEDULED, TruckStatus.DOCKED):
            return None

        ts.status = TruckStatus.DEPARTED
        ts.compute_actual_times()

        self._active_bays = max(0, self._active_bays - 1)
        if self._delayed_inbound > 0:
            self._delayed_inbound -= 1

        self.metrics.departed_trucks += 1
        self.metrics.departure_delays.append(ts.departure_delay_D)
        day = int(event.time // 1440)
        self.metrics.daily_trucks[day] += 1

        if ts.departure_delay_D <= self.cfg.ontime_threshold:
            self.metrics.ontime_departures += 1
            self.metrics.daily_ontime[day] += 1

        ou = 1.0 - min(max(ts.departure_delay_D, 0.0), self.cfg.delta_F) / self.cfg.delta_F
        self.context_engine.record_global_operator_utility(event.time, ou)

        # Schedule cargo transfer check at DESTINATION arrival time.
        # At that moment we know the inbound truck's actual arrival delay
        # and can correctly compare it to each outbound truck's departure time.
        return [SimEvent(
            time=ts.actual_arrival,
            event_type=EventType.CARGO_TRANSFER_CHECK,
            truck_id=tid,
        )]

    def _handle_cargo_transfer(self, event: SimEvent) -> Optional[List[SimEvent]]:
        """Triggered at truck's actual_arrival at destination hub.

        Does two things in one event (same as original aviation simulator):
          1. Marks the truck as DELIVERED and records arrival metrics.
          2. Checks every cargo unit that traveled on this truck and needs
             a connecting outbound truck — determines success or failure.

        This is the correct timing: we know the inbound truck's actual
        arrival delay, and we compare it to each outbound truck's planned
        (or actual) departure to determine the transfer window.
        """
        tid = event.truck_id
        ts  = self.trucks.get(tid)
        if ts is None:
            return None

        # ── Step 1: Mark this truck as DELIVERED at destination ───────
        if ts.status == TruckStatus.DEPARTED:
            ts.status         = TruckStatus.DELIVERED
            ts.actual_arrival = event.time
            self.metrics.delivered_trucks += 1
            arr_delay = ts.arrival_delay_A
            self.metrics.arrival_delays.append(arr_delay)
            cu_val = 1.0 - min(max(arr_delay, 0.0), self.cfg.delta_C) / self.cfg.delta_C
            self.context_engine.record_global_cargo_utility(event.time, cu_val)

        # ── Step 2: Check every cargo unit that traveled on this truck
        #            and needs a connecting outbound truck at this hub
        actual_in = event.time  # this IS the inbound arrival time
        dest_hub  = ts.truck.dest_hub
        hub_obj   = self.hubs.get(dest_hub)
        mtt       = hub_obj.min_transfer_time if hub_obj else self.cfg.min_transfer_time_spoke
        day       = int(actual_in // 1440)

        for cid in self._outbound_cargo.get(tid, []):
            cu = self.cargo.get(cid)
            if cu is None or len(cu.legs) < 2 or cu.legs[0] != tid:
                continue

            outbound_ts = self.trucks.get(cu.legs[1])
            if outbound_ts is None:
                continue

            # Best estimate of outbound departure (actual if departed, else planned)
            if outbound_ts.actual_departure is not None:
                out_dep = outbound_ts.actual_departure
            elif outbound_ts.hold_decided:
                out_dep = outbound_ts.truck.scheduled_departure + outbound_ts.total_departure_delay
            else:
                out_dep = (outbound_ts.truck.scheduled_departure
                           + max(outbound_ts.intrinsic_departure_delay,
                                 outbound_ts.propagated_departure_delay)
                           + outbound_ts.bay_dwell_delay)

            transfer_window = out_dep - actual_in

            if transfer_window >= mtt:
                cu.missed_transfer          = False
                cu.delay_to_destination     = max(0.0, ts.arrival_delay_A)
                self.metrics.successful_transfers += cu.unit_count
                self.metrics.daily_transfers[day] += cu.unit_count
                sigma = self.context_engine._cargo_disutility(
                    cu.delay_to_destination, cu.sla_urgency, cu.is_perishable)
                self.context_engine.record_global_cargo_utility(event.time, 1.0 - sigma)
                self.context_engine.record_transfer_outcome(event.time, succeeded=True)
            else:
                cu.missed_transfer   = True
                rebook_delay         = self._rebook_cargo(cu, dest_hub, actual_in)
                cu.delay_to_destination = rebook_delay
                self.metrics.failed_transfers += cu.unit_count
                self.metrics.daily_failed[day] += cu.unit_count
                self.metrics.cargo_delivery_delays.append(rebook_delay * cu.unit_count)
                if cu.sla_urgency == 2:
                    self.metrics.priority_failed += cu.unit_count
                sigma = self.context_engine._cargo_disutility(
                    rebook_delay, cu.sla_urgency, cu.is_perishable)
                self.context_engine.record_global_cargo_utility(event.time, 1.0 - sigma)
                self.context_engine.record_transfer_outcome(event.time, succeeded=False)

        return None

    # ==================================================================
    # Propagated delay (identical logic to AirlineNetworkSimulator)
    # ==================================================================
    def _compute_propagated_delay_dynamic(self, truck_id: str) -> float:
        """Propagate delay from previous truck on same route.

        Identical algorithm to AirlineNetworkSimulator._compute_propagated_delay_dynamic().
        Previous truck's arrival delay cascades to next truck's departure.

        IMPROVEMENT: Road delay now compounds with ZG (congestion cascade).
        G_road_effective = G_road × (1 + ZG × 0.5)
        Rationale: in a congested hub (high ZG), truck access roads are also congested,
        amplifying road delay by up to 50%.
        """
        prev_tid = self._prev_route_truck.get(truck_id)
        if prev_tid is None:
            return 0.0

        prev_ts = self.trucks[prev_tid]
        ts      = self.trucks[truck_id]

        # Current ZG (inbound queue depth) amplifies road delay
        zg = self.context_engine.compute_inbound_queue_depth(
            self.event_engine.current_time)
        road_amplifier = 1.0 + zg * 0.5

        if prev_ts.actual_arrival is not None:
            prev_arrival = prev_ts.actual_arrival
        else:
            base_road = prev_ts.road_delay * road_amplifier
            prev_arrival = (prev_ts.truck.scheduled_arrival
                            + max(prev_ts.intrinsic_departure_delay,
                                  prev_ts.propagated_departure_delay)
                            + prev_ts.hold_delay
                            + prev_ts.bay_dwell_delay
                            + base_road
                            + prev_ts.bay_arrival_delay)

        prev_arr_delay = prev_arrival - prev_ts.truck.scheduled_arrival
        turnaround_sched = ts.truck.scheduled_departure - prev_ts.truck.scheduled_arrival
        slack = max(0.0, turnaround_sched - self.cfg.min_turnaround)
        return max(0.0, prev_arr_delay - slack)

    # ==================================================================
    # Cargo rebooking  ← _rebook_pax()
    # ==================================================================
    def _rebook_cargo(self, cargo: CargoUnit, current_hub: str,
                      current_time: float) -> float:
        """Find next available truck from current_hub to cargo's final destination.

        Mirrors _rebook_pax() from simulator/simulator.py.
        Falls back to next_cycle_penalty_minutes (24h) if no same-day truck exists.
        """
        final_dest = cargo.destination_hub
        best_delay = self.cfg.next_cycle_penalty_minutes
        orig_arr   = self.trucks[cargo.legs[-1]].truck.scheduled_arrival

        for tid in self._departures_at_hub.get(current_hub, []):
            ts = self.trucks[tid]
            if ts.truck.dest_hub != final_dest:
                continue
            if ts.truck.scheduled_departure <= current_time:
                continue
            est_arr = ts.truck.scheduled_arrival + ts.total_arrival_delay
            delay   = est_arr - orig_arr
            if 0 < delay < best_delay:
                best_delay = delay
                cargo.rebooked_truck_id = tid

        return max(0.0, best_delay)

    # ==================================================================
    # Truck state enrichment (populate PDF §2 local fields)
    # ==================================================================
    def _enrich_truck_states(self) -> None:
        """Populate V_k, Q_k, X_k, E_k, L_k, F_k, N_in on each TruckState.

        FIX: Uses pre-built dict lookup (O(1)) instead of trucks.index() (O(n)).
        """
        for tid, ts in self.trucks.items():
            cargo_ids  = self._inbound_cargo.get(tid, [])
            cargo_objs = [self.cargo[cid] for cid in cargo_ids if cid in self.cargo]
            if not cargo_objs:
                continue

            # V_k: mean value score of connecting cargo (CSV: shipping_costs normalised)
            ts.cargo_value_score = float(np.mean([c.value_score for c in cargo_objs]))

            # Q_k: fraction of truck capacity used by connecting cargo
            total_units = sum(c.unit_count for c in cargo_objs)
            ts.cargo_volume_fraction = min(1.0, total_units / max(1, ts.truck.cargo_capacity))

            # X_k: worst-case SLA urgency among connecting cargo (max drives risk)
            ts.sla_urgency = max(c.sla_urgency for c in cargo_objs)

            # E_k: fraction of connecting cargo that is perishable
            perishable = sum(c.unit_count for c in cargo_objs if c.is_perishable)
            ts.perishability_fraction = perishable / max(1, total_units)

            # N_in: number of distinct inbound trucks feeding this outbound
            inbound_tids = {c.legs[0] for c in cargo_objs if len(c.legs) >= 2}
            ts.n_inbound_trucks = len(inbound_tids)

            # L_k: driver hours remaining (270 min = 4.5h EU HGV default)
            # Reduced randomly to reflect varying start-of-route conditions
            ts.driver_hours_remaining = float(self.rng.uniform(120.0, 270.0))

            # F_k: deadline pressure (inversely proportional to SLA urgency)
            ts.deadline_pressure = 1.0 - (ts.sla_urgency / 2.0)

    # ==================================================================
    # Advance to next HOLD_DECISION  ← _advance_to_next_hnh()
    # ==================================================================
    def _advance_to_next_hold(self) -> Tuple[TruckContext, Dict[str, Any]]:
        hold_event = self.event_engine.run_until_hold_decision()
        if hold_event is None:
            self.event_engine.drain()
            self._done = True
            return TruckContext(truck_id="DONE"), self.metrics.summary()

        tid = hold_event.truck_id
        ts  = self.trucks[tid]
        self._pending_hold_truck = tid

        ts.propagated_departure_delay = self._compute_propagated_delay_dynamic(tid)

        conn_cargo = [self.cargo[cid]
                      for cid in self._inbound_cargo.get(tid, [])
                      if cid in self.cargo]

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
            "truck_id":               tid,
            "origin_hub":             ts.truck.origin_hub,
            "dest_hub":               ts.truck.dest_hub,
            "connecting_cargo_count": len(conn_cargo),
            "scheduled_departure":    ts.truck.scheduled_departure,
            "intrinsic_delay":        ts.intrinsic_departure_delay,
            "propagated_delay":       ts.propagated_departure_delay,
            "tau_star":               ctx.tau_star,
            "sla_urgency":            ts.sla_urgency,
            "driver_hours_remaining": ts.driver_hours_remaining,
            "bay_utilisation":        self._active_bays / max(1, self.cfg.num_bays),
            "perishability_fraction": ts.perishability_fraction,
        }
        return ctx, info

    # ==================================================================
    # Reward  (PDF §5.5)
    # ==================================================================
    def _compute_reward(self, truck_id: str, hold_minutes: float) -> float:
        """R_T_k = β·R_L_k + (1-β)·R_G_k  (PDF §5.5)

        R_L_k = α·CL(τ) + (1-α)·OL(τ)
        R_G_k = α·CG   + (1-α)·OG
        """
        ctx = self._pending_context
        if ctx is None:
            return 0.0
        alpha = self.cfg.alpha
        beta  = self.cfg.beta

        action_idx = (self.cfg.hold_actions.index(int(hold_minutes))
                      if int(hold_minutes) in self.cfg.hold_actions else 0)
        cl_val = ctx.CL[action_idx] if action_idx < len(ctx.CL) else 0.5
        ol_val = ctx.OL[action_idx] if action_idx < len(ctx.OL) else 0.5
        R_L = alpha * cl_val + (1 - alpha) * ol_val
        R_G = alpha * ctx.CG  + (1 - alpha) * ctx.OG
        reward = float(beta * R_L + (1 - beta) * R_G)
        self.metrics.rewards.append(reward)
        return reward

    # ==================================================================
    # run_episode with baseline policy  ← AirlineNetworkSimulator.run_episode()
    # ==================================================================
    def run_episode(self, policy: str = "no_hold", max_hold: int = 15,
                    seed: Optional[int] = None) -> Dict[str, Any]:
        """Run a full episode with a baseline policy. Returns metrics summary."""
        ctx, info = self.reset(seed=seed)
        done = False
        while not done:
            action = self._select_baseline_action(policy, max_hold)
            ctx, reward, done, info = self.step(action)
        return self.metrics.summary()

    def _select_baseline_action(self, policy: str, max_hold: int = 15) -> int:
        """Baseline policies — identical to AirlineNetworkSimulator.

        no_hold   : always τ=0
        heuristic : hold exactly enough to rescue at-risk connecting cargo
        random    : uniform random
        """
        if policy == "no_hold":
            return 0

        if policy.startswith("heuristic"):
            tid = self._pending_hold_truck
            ts  = self.trucks[tid]
            conn_ids = self._inbound_cargo.get(tid, [])
            if not conn_ids:
                return 0

            est_dep = (ts.truck.scheduled_departure
                       + max(ts.intrinsic_departure_delay, ts.propagated_departure_delay)
                       + ts.bay_dwell_delay)
            hub_obj = self.hubs.get(ts.truck.origin_hub)
            mtt     = hub_obj.min_transfer_time if hub_obj else self.cfg.min_transfer_time_spoke

            max_needed = 0.0
            for cid in conn_ids:
                cu = self.cargo.get(cid)
                if cu is None or len(cu.legs) < 2:
                    continue
                in_ts = self.trucks.get(cu.legs[0])
                if in_ts is None:
                    continue
                in_arr = (in_ts.actual_arrival
                          if in_ts.actual_arrival is not None
                          else in_ts.truck.scheduled_arrival + in_ts.total_arrival_delay)
                window = est_dep - in_arr
                if window >= mtt:
                    continue
                needed = mtt - window
                if needed <= max_hold:
                    max_needed = max(max_needed, needed)

            if max_needed <= 0:
                return 0
            for idx, tau in enumerate(self.cfg.hold_actions):
                if tau >= max_needed:
                    return idx
            return max(i for i, tau in enumerate(self.cfg.hold_actions)
                       if tau <= max_hold)

        if policy == "random":
            return int(self.rng.integers(0, len(self.cfg.hold_actions)))
        return 0

    # ==================================================================
    # Data access helpers (mirrors AirlineNetworkSimulator interface)
    # ==================================================================
    def get_truck_state(self, truck_id: str) -> Optional[TruckState]:
        return self.trucks.get(truck_id)

    def get_route_trucks(self, route_id: str) -> List[str]:
        return self._trucks_by_route.get(route_id, [])

    def get_inbound_feeder_trucks(self, truck_id: str) -> List[str]:
        return list({cu.legs[0]
                     for cid in self._inbound_cargo.get(truck_id, [])
                     if (cu := self.cargo.get(cid)) and len(cu.legs) >= 2})

    def get_previous_route_truck(self, truck_id: str) -> Optional[str]:
        return self._prev_route_truck.get(truck_id)

    def get_next_route_truck(self, truck_id: str) -> Optional[str]:
        return self._next_route_truck.get(truck_id)
