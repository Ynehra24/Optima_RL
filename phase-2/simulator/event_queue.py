"""
event_queue.py — Discrete-event simulation engine for the logistics hub.

Direct analog of the paper's Section 6.1 microsimulator, adapted for
the cross-docking logistics domain.

Event types and their handlers:
  INBOUND_ARRIVE   → assign bay (or queue) → compute GB delay
  HOLD_DECISION    → called by Gym env → agent picks τ
  OUTBOUND_DEPART  → transfer check → record rewards → DT nodes built
  DAY_END          → compute episode-level statistics

The event queue uses a min-heap ordered by simulation time.
The Gym env drives the loop by calling advance_to_next_decision()
which processes all events up to the next HOLD_DECISION.
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from simulator.config import SimConfig
from simulator.schedule_generator import DailySchedule, TruckSchedule
from simulator.delay_sampler import DelaySampler
from simulator.bay_manager import BayManager
from simulator.cargo_manager import CargoManager
from simulator.context_engine import ContextEngine, TruckContext
from rewardEngineering.reward_calculator import LogisticsRewardCalculator


# ── Event dataclass ────────────────────────────────────────────────────

@dataclass(order=True)
class Event:
    """A discrete simulation event.

    Ordered by time (earliest first) for the min-heap.
    """
    time: float
    event_type: str = field(compare=False)
    truck_id: str = field(compare=False)
    data: Dict[str, Any] = field(default_factory=dict, compare=False)


# ── Episode statistics ─────────────────────────────────────────────────

@dataclass
class EpisodeStats:
    """Tracks statistics for one simulation episode."""
    n_hold_decisions: int = 0
    n_transfers_success: int = 0
    n_transfers_missed: int = 0
    n_rebooked: int = 0              # cargo rebooked onto later trucks
    total_departure_delay: float = 0.0
    total_rebook_delay: float = 0.0  # extra delay caused by rebooking congestion
    total_reward: float = 0.0
    n_rewards: int = 0
    n_on_time_departures: int = 0   # departed within OTP_THRESHOLD of schedule
    n_total_departures: int = 0
    bay_utilization_samples: List[float] = field(default_factory=list)

    OTP_THRESHOLD: float = 15.0  # minutes (standard logistics on-time window)

    @property
    def missed_transfer_rate(self) -> float:
        total = self.n_transfers_success + self.n_transfers_missed
        return self.n_transfers_missed / total if total > 0 else 0.0

    @property
    def OTP(self) -> float:
        """On-Time Departure Rate: % trucks departing within 15 min of schedule."""
        return (self.n_on_time_departures / self.n_total_departures * 100.0
                if self.n_total_departures > 0 else 100.0)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.n_rewards if self.n_rewards > 0 else 0.0

    @property
    def mean_bay_utilization(self) -> float:
        return float(np.mean(self.bay_utilization_samples)) if self.bay_utilization_samples else 0.0


# ── Main EventQueue class ──────────────────────────────────────────────

class EventQueue:
    """Discrete-event simulation engine for the cross-docking hub.

    Usage (called by LogisticsEnv):
        eq = EventQueue(cfg, schedule, ...)
        eq.load_day(daily_schedule)
        context, truck_id = eq.advance_to_next_decision()
        reward = eq.apply_hold(truck_id, hold_minutes)
        context, truck_id = eq.advance_to_next_decision()
    """

    def __init__(
        self,
        cfg: SimConfig,
        delay_sampler: DelaySampler,
        bay_manager: BayManager,
        cargo_manager: CargoManager,
        context_engine: ContextEngine,
        reward_calculator: LogisticsRewardCalculator,
        rng: Optional[np.random.Generator] = None,
    ):
        self.cfg = cfg
        self.sampler = delay_sampler
        self.bay_mgr = bay_manager
        self.cargo_mgr = cargo_manager
        self.ctx_engine = context_engine
        self.reward_calc = reward_calculator
        self.rng = rng or np.random.default_rng(cfg.seed)

        # Heap of pending events
        self._heap: List[Event] = []

        # Simulation state
        self._current_time: float = cfg.operating_start
        self._daily_schedule: Optional[DailySchedule] = None
        self._day_end_time: float = cfg.operating_end

        # Track truck states
        self._actual_arrivals: Dict[str, float] = {}    # {truck_id: actual_arrival}
        self._inbound_etas: Dict[str, float] = {}       # {truck_id: estimated ETA}
        self._road_delays: Dict[str, float] = {}        # {truck_id: road delay}
        self._gb_delays: Dict[str, float] = {}          # {truck_id: gb delay}
        self._departure_delays: Dict[str, float] = {}   # {truck_id: departure delay}
        self._applied_holds: Dict[str, float] = {}      # {truck_id: hold applied}
        self._pending_decisions: List[str] = []         # outbound truck_ids awaiting decisions
        self._decided_trucks: set = set()               # trucks already decided

        # ── Rebooking queue (cascade mechanism) ──────────────────────
        # When cargo misses a connection, it's "rebooked" onto a later
        # outbound truck. That truck gets extra loading time → departure
        # delay → propagates to next hub. This is the Phase 1 analog of
        # rebooked passengers causing later flights to be more crowded.
        self._rebook_queue: Dict[str, int] = {}   # {truck_id: n_rebooked_cargo}
        self._rebook_delay_per_unit: float = 2.0  # minutes extra loading per rebooked unit
        # 2.0 min/unit: rebooking 10 missed cargo → +20 min delay on later truck
        # This matches Phase 1 where rebooked PAX cause significant gate delays

        # Pending context for current decision
        self._pending_context: Optional[TruckContext] = None
        self._pending_truck_id: Optional[str] = None

        # Stats
        self.stats = EpisodeStats()

        # Inter-hub tracking: trucks injected from upstream hubs
        self._inter_hub_arrivals: Dict[str, float] = {}   # {truck_id: arrival_time}
        self._inter_hub_origins: Dict[str, str] = {}      # {truck_id: origin_hub_id}

    # ── Public API ─────────────────────────────────────────────────────

    def inject_inter_hub_arrival(self, truck_id: str, arrival_time: float,
                                  scheduled_arrival: float,
                                  origin_hub: str):
        """Inject an inter-hub truck arrival — the CASCADE mechanism.

        The key insight: feeder connections are planned around scheduled_arrival
        (the on-time ETA), but the truck actually arrives at arrival_time
        (which may be late due to upstream holds). This gap creates real
        missed transfers — exactly like Phase 1 airline connections.

        Args:
            truck_id:          ID of the truck
            arrival_time:      ACTUAL arrival time (includes hold delays)
            scheduled_arrival: PLANNED arrival time (on-time, no delays)
            origin_hub:        Hub ID where this truck originated
        """
        self._inter_hub_arrivals[truck_id] = scheduled_arrival  # planned time
        self._inter_hub_origins[truck_id] = origin_hub

        # Schedule the INBOUND_ARRIVE event at ACTUAL arrival time (late)
        heapq.heappush(self._heap, Event(
            time=arrival_time,
            event_type="INBOUND_ARRIVE",
            truck_id=truck_id,
            data={
                "truck": None,
                "is_inter_hub": True,
                "origin_hub": origin_hub,
                "scheduled_arrival": scheduled_arrival,
            },
        ))

        # ── PRE-PLAN feeder connections based on SCHEDULED arrival ────
        # This happens NOW (before the truck arrives), just like airlines
        # pre-sell connecting tickets before the feeder flight lands.
        if self._daily_schedule is not None:
            from simulator.schedule_generator import TruckSchedule
            synth_truck = TruckSchedule(
                truck_id=truck_id,
                direction="inbound",
                origin_zone=0,
                dest_zone=0,
                route_type="medium",
                scheduled_arrival=scheduled_arrival,  # PLANNED time
                scheduled_departure=0.0,
            )
            self._daily_schedule.inbound_trucks.append(synth_truck)

            # Connect to outbound trucks planned AFTER scheduled_arrival + MCT
            # NOT after actual_arrival — this is the key difference!
            mct = float(self.cfg.mtt)
            base = float(self.cfg.operating_start)
            upcoming = [
                t for t in self._daily_schedule.outbound_trucks
                if (t.scheduled_departure + base) >= (scheduled_arrival + mct)
                and t.truck_id not in self._decided_trucks
            ]
            for target in upcoming[:3]:
                feeders = self._daily_schedule.feeder_map.get(
                    target.truck_id, [])
                if truck_id not in feeders:
                    feeders.append(truck_id)
                    self._daily_schedule.feeder_map[target.truck_id] = feeders

    def get_inter_hub_delay(self) -> float:
        """Return mean lateness (min) of all inter-hub arrivals at this hub.

        Used by HubChain.get_network_state() to populate upstream_inter_hub_delay.
        Returns 0.0 if no inter-hub trucks have arrived yet.
        """
        if not self._inter_hub_arrivals:
            return 0.0
        # We don't have scheduled times for inter-hub trucks; use 0 as baseline
        # delay = how late compared to the earliest possible arrival (transit_min=40)
        return float(np.mean(list(self._inter_hub_arrivals.values())))



    def load_day(self, schedule: DailySchedule):
        """Load a daily truck schedule and populate the event heap."""
        self._daily_schedule = schedule
        self._heap.clear()
        self._day_end_time = self.cfg.operating_end + schedule.day_index * 1440.0
        base_time = schedule.day_index * 1440.0   # minutes offset for multi-day

        # Schedule inbound truck arrivals
        for truck in schedule.inbound_trucks:
            road_delay = self.sampler.sample_road_delay(truck.route_type)
            self._road_delays[truck.truck_id] = road_delay
            actual_arrival = truck.scheduled_arrival + road_delay + base_time
            self._inbound_etas[truck.truck_id] = actual_arrival

            heapq.heappush(self._heap, Event(
                time=actual_arrival,
                event_type="INBOUND_ARRIVE",
                truck_id=truck.truck_id,
                data={"truck": truck},
            ))

        # Schedule outbound departure decisions
        for truck in schedule.outbound_trucks:
            gd_delay = self.sampler.sample_ground_departure_delay()
            self._departure_delays[truck.truck_id] = gd_delay
            decision_time = truck.scheduled_departure + base_time

            heapq.heappush(self._heap, Event(
                time=decision_time,
                event_type="HOLD_DECISION",
                truck_id=truck.truck_id,
                data={"truck": truck},
            ))

        # Day-end marker
        heapq.heappush(self._heap, Event(
            time=self._day_end_time,
            event_type="DAY_END",
            truck_id="__system__",
        ))

    def advance_to_next_decision(self) -> Optional[Tuple[TruckContext, str]]:
        """Process events until the next HOLD_DECISION.

        Returns:
            (TruckContext, truck_id) for the outbound truck that needs a decision
            None if no more decisions exist (day is over)
        """
        while self._heap:
            event = heapq.heappop(self._heap)
            self._current_time = event.time

            if event.event_type == "INBOUND_ARRIVE":
                self._handle_inbound_arrive(event)

            elif event.event_type == "HOLD_DECISION":
                if event.truck_id in self._decided_trucks:
                    continue  # already decided this truck
                ctx = self._handle_hold_decision(event)
                if ctx is not None:
                    self._pending_context = ctx
                    self._pending_truck_id = event.truck_id
                    return ctx, event.truck_id

            elif event.event_type == "OUTBOUND_DEPART":
                self._handle_outbound_depart(event)

            elif event.event_type == "DAY_END":
                return None  # signal end of day

        return None  # heap exhausted

    def apply_hold(self, truck_id: str, hold_minutes: float) -> float:
        """Apply a hold decision and return the immediate local reward.

        Called by LogisticsEnv.step() after the agent picks an action.

        Args:
            truck_id: The outbound truck being held
            hold_minutes: Hold duration chosen by agent

        Returns:
            Local reward R_L for this hold decision
        """
        if self._daily_schedule is None:
            return 0.0

        self._applied_holds[truck_id] = hold_minutes
        self._decided_trucks.add(truck_id)

        # Find the truck schedule
        truck = self._find_outbound_truck(truck_id)
        if truck is None:
            return 0.0

        # Compute actual departure time using ABSOLUTE simulation time
        # self._current_time = scheduled_departure + base_time (set by HOLD_DECISION)
        gd_delay = self._departure_delays.get(truck_id, 0.0)
        actual_departure = self._current_time + hold_minutes + gd_delay

        # Apply hold to bay manager (extends bay occupation)
        blocked_trucks = self.bay_mgr.apply_hold(
            truck_id, hold_minutes, self._current_time
        )

        # Register bay congestion for global reward
        if blocked_trucks:
            self.reward_calc.register_bay_congestion(truck_id, blocked_trucks)

        # Schedule the actual departure event
        heapq.heappush(self._heap, Event(
            time=actual_departure,
            event_type="OUTBOUND_DEPART",
            truck_id=truck_id,
            data={
                "truck": truck,
                "hold_minutes": hold_minutes,
                "actual_departure": actual_departure,
                "scheduled_abs": self._current_time,  # absolute scheduled time
                "gd_delay": gd_delay,
            },
        ))

        # Register departure in delay tree
        connecting_in = self._get_connecting_inbound(truck_id)
        incoming_trucks = [
            (t.truck_id, self._road_delays.get(t.truck_id, 0.0))
            for t in connecting_in
        ]
        self.reward_calc.register_truck_departure(
            truck_id=truck_id,
            departure_delay=hold_minutes + gd_delay,
            prev_truck_id=truck.prev_truck_id,
            prev_arrival_delay=self._road_delays.get(truck.prev_truck_id or "", 0.0),
            hold_duration=hold_minutes,
            departure_ground_delay=gd_delay,
            bay_blockage_delay=self._gb_delays.get(truck_id, 0.0),
            incoming_trucks=incoming_trucks,
        )

        # Compute and return local reward
        ctx = self._pending_context
        if ctx is None:
            return 0.0

        local_reward = self.reward_calc.compute_local_reward(ctx, hold_minutes)
        self.stats.total_reward += local_reward
        self.stats.n_rewards += 1
        self.stats.n_hold_decisions += 1

        return local_reward

    def get_global_reward(self, truck_id: str) -> float:
        """Get the global reward attributed to a truck's past hold decision."""
        return self.reward_calc.get_global_reward(truck_id)

    def get_total_reward(self, truck_id: str, hold_minutes: float) -> float:
        """Compute β·R_L + (1-β)·R_G for a truck."""
        if self._pending_context is None:
            return 0.0
        return self.reward_calc.get_total_reward(
            self._pending_context, hold_minutes, truck_id
        )

    @property
    def current_time(self) -> float:
        return self._current_time

    # ── Event Handlers ─────────────────────────────────────────────────

    def _handle_inbound_arrive(self, event: Event):
        """INBOUND_ARRIVE: truck arrives at hub, gets bay or queues."""
        truck: TruckSchedule = event.data.get("truck")
        arrival_time = event.time

        # Inter-hub arrivals: feeder connections were pre-planned in
        # inject_inter_hub_arrival using scheduled_arrival. The truck now
        # physically arrives — record its ACTUAL arrival time. The gap
        # between scheduled and actual creates missed transfers.
        if truck is None:
            truck_id = event.truck_id
            self._actual_arrivals[truck_id] = arrival_time

            # Bay assignment for inter-hub truck
            _, gb_delay = self.bay_mgr.truck_arrives(
                truck_id=truck_id,
                arrival_time=arrival_time,
                processing_time=30.0,
            )
            self._gb_delays[truck_id] = gb_delay

            self.stats.bay_utilization_samples.append(
                self.bay_mgr.get_bay_utilization(arrival_time)
            )
            return

        road_delay = self._road_delays.get(truck.truck_id, 0.0)


        # Record actual arrival
        self._actual_arrivals[truck.truck_id] = arrival_time

        # Bay assignment
        arrival_bay_delay = self.sampler.sample_arrival_bay_delay()
        _, gb_delay = self.bay_mgr.truck_arrives(
            truck_id=truck.truck_id,
            arrival_time=arrival_time,
            processing_time=30.0 + arrival_bay_delay,
        )
        self._gb_delays[truck.truck_id] = gb_delay

        # Update bay utilization sample
        self.stats.bay_utilization_samples.append(
            self.bay_mgr.get_bay_utilization(arrival_time)
        )

        # Register in delay tree (Rule 1: arrival delay)
        self.reward_calc.register_truck_arrival(
            truck_id=truck.truck_id,
            arrival_delay=max(0.0, arrival_time - truck.scheduled_arrival),
            departure_delay=self._departure_delays.get(truck.truck_id, 0.0),
            road_delay=road_delay,
            arrival_bay_delay=arrival_bay_delay,
            arrival_cu=0.5,   # placeholder; actual CU computed at departure
            arrival_ou=0.5,
        )

    def _handle_hold_decision(self, event: Event) -> Optional[TruckContext]:
        """HOLD_DECISION: build context for agent to make a hold decision."""
        truck: TruckSchedule = event.data["truck"]
        current_time = event.time

        # Get connecting inbound trucks
        connecting_in = self._get_connecting_inbound(truck.truck_id)
        if not connecting_in:
            # No inbound feeders → just apply zero hold automatically
            self.apply_hold(truck.truck_id, 0)
            return None

        # Build context
        gb_delay = self._gb_delays.get(truck.truck_id, 0.0)
        road_delay = self._road_delays.get(truck.truck_id, 0.0)
        dep_delay = self._departure_delays.get(truck.truck_id, 0.0)

        ctx = self.ctx_engine.compute_context(
            outbound_truck=truck,
            connecting_inbound=connecting_in,
            inbound_arrivals=self._actual_arrivals,
            inbound_etas=self._inbound_etas,
            current_time=current_time,
            departure_delay=dep_delay,
            arrival_delay=0.0,
            gb_delay=gb_delay,
            road_delay=road_delay,
        )
        return ctx

    def _handle_outbound_depart(self, event: Event):
        """OUTBOUND_DEPART: truck departs, transfer check, rebook missed cargo.

        The rebooking cascade is the key mechanism matching Phase 1 airlines:
          1. Cargo misses connection → rebooked onto a LATER outbound truck
          2. Later truck gets extra loading time → departs later
          3. Late departure propagates to next hub → cascade
          4. Strategic holding PREVENTS this cascade (= Phase 1 OTP benefit)
        """
        truck: TruckSchedule = event.data["truck"]
        hold_minutes: float = event.data["hold_minutes"]
        actual_departure: float = event.data["actual_departure"]
        scheduled_abs: float = event.data.get("scheduled_abs", actual_departure - hold_minutes)

        # ── Rebooking congestion: add extra delay from rebooked cargo ──────
        n_rebooked_here = self._rebook_queue.pop(truck.truck_id, 0)
        rebook_delay = n_rebooked_here * self._rebook_delay_per_unit
        if rebook_delay > 0:
            actual_departure += rebook_delay
            self.stats.n_rebooked += n_rebooked_here
            self.stats.total_rebook_delay += rebook_delay

        # Bay release
        self.bay_mgr.truck_departs(truck.truck_id, actual_departure)

        # Transfer check
        connecting_in = self._get_connecting_inbound(truck.truck_id)
        n_success, n_missed = self.cargo_mgr.count_transfers(
            truck, self._actual_arrivals, connecting_in, actual_departure
        )
        self.stats.n_transfers_success += n_success
        self.stats.n_transfers_missed += n_missed

        # ── REBOOKING CASCADE: rebook missed cargo onto later trucks ───────
        # Phase 1 analog: missed PAX get rebooked onto the NEXT flight,
        # NOT spread across all later flights. This concentration creates
        # genuine congestion that impacts OTP.
        #
        # KEY FIX: only rebook onto the NEXT 2 outbound trucks (not all).
        # With n_missed=10 cargo × 2 min/unit = 20 min on each of 2 trucks
        # → pushes them past the 15 min OTP threshold → No Hold OTP drops.
        if n_missed > 0 and self._daily_schedule is not None:
            later_trucks = [
                t for t in self._daily_schedule.outbound_trucks
                if t.scheduled_departure > truck.scheduled_departure
                and t.truck_id != truck.truck_id
            ]
            if later_trucks:
                # Concentrate on NEXT 2 trucks only (like airline rebooking)
                rebook_targets = later_trucks[:2]
                max_rebook_per_truck = 25
                for i in range(n_missed):
                    target = rebook_targets[i % len(rebook_targets)]
                    current = self._rebook_queue.get(target.truck_id, 0)
                    if current < max_rebook_per_truck:
                        self._rebook_queue[target.truck_id] = current + 1

        # Compute actual utilities for global tracking
        actual_cu = self.cargo_mgr.compute_actual_CU(
            truck, self._actual_arrivals, connecting_in, actual_departure
        )
        actual_ou = self.cargo_mgr.compute_actual_OU(
            scheduled_abs, actual_departure
        )
        dep_delay = max(0.0, actual_departure - scheduled_abs)
        self.stats.total_departure_delay += dep_delay
        self.stats.n_total_departures += 1
        if dep_delay <= self.stats.OTP_THRESHOLD:
            self.stats.n_on_time_departures += 1

        # Record outcome for global rolling stats
        self.ctx_engine.record_outcome(
            event_time=actual_departure,
            cargo_utility=actual_cu,
            operator_utility=actual_ou,
            n_success=n_success,
            n_total=n_success + n_missed,
        )

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_connecting_inbound(self, outbound_truck_id: str) -> List[TruckSchedule]:
        """Get inbound trucks that feed into an outbound truck."""
        if self._daily_schedule is None:
            return []
        feeder_ids = self._daily_schedule.feeder_map.get(outbound_truck_id, [])
        inbound_map = {t.truck_id: t for t in self._daily_schedule.inbound_trucks}
        return [inbound_map[fid] for fid in feeder_ids if fid in inbound_map]

    def _find_outbound_truck(self, truck_id: str) -> Optional[TruckSchedule]:
        """Find a TruckSchedule from the daily schedule by truck_id."""
        if self._daily_schedule is None:
            return None
        for t in self._daily_schedule.outbound_trucks:
            if t.truck_id == truck_id:
                return t
        return None

    def reset_day(self):
        """Clear per-day state but PRESERVE episode-level stats.

        Called between days in a multi-day episode.
        """
        self._heap.clear()
        self._actual_arrivals.clear()
        self._inbound_etas.clear()
        self._road_delays.clear()
        self._gb_delays.clear()
        self._departure_delays.clear()
        self._applied_holds.clear()
        self._decided_trucks.clear()
        self._pending_context = None
        self._pending_truck_id = None
        self._rebook_queue.clear()
        self._inter_hub_arrivals.clear()
        self._inter_hub_origins.clear()

    def reset(self):
        """Full reset for a new episode — clears everything including stats."""
        self.reset_day()
        self._current_time = self.cfg.operating_start
        self.stats = EpisodeStats()
