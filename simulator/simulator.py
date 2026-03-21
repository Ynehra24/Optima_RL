"""
Airline Network Microsimulator — main orchestrator.

Implements the discrete-event microsimulator described in Section 6.1:
  • Models flight arrivals, departures, holds
  • Propagates delays *dynamically* through tail plans and PAX connections
  • Tracks PAX movement and missed connections
  • Exposes a Gym-like step(action) → (state, reward, done, info) API

Key design principles from the paper:
  1. Delays are propagated stochastically along the tail plan.
  2. Hold delays propagate through (a) same-tail subsequent flights and
     (b) PAX who miss connections and trigger further holds.
  3. One RL agent makes HNH decisions for ALL flights sequentially.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulator.config import SimConfig
from simulator.context_engine import ContextEngine, FlightContext
from simulator.event_engine import EventEngine
from simulator.generators import (
    DelaySampler,
    generate_airports,
    generate_pax_itineraries,
    generate_tail_plans,
)
from simulator.models import (
    Airport,
    EventType,
    FlightState,
    FlightStatus,
    PaxItinerary,
    ScheduledFlight,
    SimEvent,
    TailPlan,
)


# ======================================================================
# Metrics tracker
# ======================================================================
class MetricsTracker:
    """Accumulates business and validation metrics during simulation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_flights = 0
        self.departed_flights = 0
        self.arrived_flights = 0
        self.ontime_arrivals = 0
        self.arrival_delays: List[float] = []
        self.departure_delays: List[float] = []

        self.total_pax = 0
        self.connecting_pax = 0
        self.missed_connections = 0
        self.successful_connections = 0
        self.pax_delays: List[float] = []

        self.hold_decisions: List[float] = []
        self.rewards: List[float] = []

        # Per-day tracking
        self.daily_missed: Dict[int, int] = defaultdict(int)
        self.daily_connections: Dict[int, int] = defaultdict(int)
        self.daily_ontime: Dict[int, int] = defaultdict(int)
        self.daily_flights: Dict[int, int] = defaultdict(int)

    @property
    def otp(self) -> float:
        if self.arrived_flights == 0:
            return 1.0
        return self.ontime_arrivals / self.arrived_flights

    @property
    def avg_arrival_delay(self) -> float:
        return float(np.mean(self.arrival_delays)) if self.arrival_delays else 0.0

    @property
    def avg_departure_delay(self) -> float:
        return float(np.mean(self.departure_delays)) if self.departure_delays else 0.0

    @property
    def misconnect_rate(self) -> float:
        total = self.missed_connections + self.successful_connections
        return self.missed_connections / total if total else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "total_flights": self.total_flights,
            "departed": self.departed_flights,
            "arrived": self.arrived_flights,
            "OTP": round(self.otp * 100, 2),
            "avg_arrival_delay_min": round(self.avg_arrival_delay, 2),
            "avg_departure_delay_min": round(self.avg_departure_delay, 2),
            "total_connecting_pax": self.connecting_pax,
            "missed_connections": self.missed_connections,
            "successful_connections": self.successful_connections,
            "misconnect_rate_pct": round(self.misconnect_rate * 100, 2),
            "avg_hold_min": (
                round(float(np.mean(self.hold_decisions)), 2)
                if self.hold_decisions
                else 0.0
            ),
        }


# ======================================================================
# Main Simulator
# ======================================================================
class AirlineNetworkSimulator:
    """Discrete-event microsimulator for the airline HNH problem.

    Usage::

        sim = AirlineNetworkSimulator(cfg)
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
        self.delay_sampler = DelaySampler(self.cfg, self.rng)
        self.metrics = MetricsTracker()

        # Data stores
        self.airports: Dict[str, Airport] = {}
        self.flights: Dict[str, FlightState] = {}
        self.tail_plans: Dict[str, TailPlan] = {}
        self.pax: Dict[str, PaxItinerary] = {}

        # Indexes
        self._flights_by_tail: Dict[str, List[str]] = defaultdict(list)
        self._incoming_pax: Dict[str, List[str]] = defaultdict(list)
        self._outgoing_pax: Dict[str, List[str]] = defaultdict(list)
        self._departures_at_airport: Dict[str, List[str]] = defaultdict(list)
        self._next_tail_flight: Dict[str, Optional[str]] = {}
        self._prev_tail_flight: Dict[str, Optional[str]] = {}

        self._done: bool = True
        self._pending_hnh_flight: Optional[str] = None
        self._pending_context: Optional[FlightContext] = None

    # ==================================================================
    # Gym-like API
    # ==================================================================
    def reset(self, seed: int | None = None) -> Tuple[FlightContext, Dict[str, Any]]:
        if seed is not None:
            self.cfg.random_seed = seed
        self.rng = np.random.default_rng(self.cfg.random_seed)

        self.event_engine.clear()
        self.context_engine.reset()
        self.metrics.reset()
        self.flights.clear()
        self.tail_plans.clear()
        self.pax.clear()
        self._flights_by_tail.clear()
        self._incoming_pax.clear()
        self._outgoing_pax.clear()
        self._departures_at_airport.clear()
        self._next_tail_flight.clear()
        self._prev_tail_flight.clear()

        # 1. Airports
        self.airports = generate_airports(self.cfg)

        # 2. Tail plans & flights
        all_scheduled: List[ScheduledFlight] = []
        for day in range(self.cfg.num_days):
            tails, scheds = generate_tail_plans(
                self.cfg, self.airports, day_index=day, rng=self.rng
            )
            for tp in tails:
                self.tail_plans[tp.tail_id] = tp
            all_scheduled.extend(scheds)

        for sf in all_scheduled:
            fs = FlightState(flight=sf)
            self.flights[sf.flight_id] = fs
            self._flights_by_tail[sf.tail_id].append(sf.flight_id)
            self._departures_at_airport[sf.origin].append(sf.flight_id)

        # Sort and build prev/next tail links
        for tail_id in self._flights_by_tail:
            fids = self._flights_by_tail[tail_id]
            fids.sort(key=lambda fid: self.flights[fid].flight.scheduled_departure)
            for i, fid in enumerate(fids):
                self._prev_tail_flight[fid] = fids[i - 1] if i > 0 else None
                self._next_tail_flight[fid] = fids[i + 1] if i < len(fids) - 1 else None

        for apt in self._departures_at_airport:
            self._departures_at_airport[apt].sort(
                key=lambda fid: self.flights[fid].flight.scheduled_departure
            )

        # 3. PAX itineraries
        sched_list = [fs.flight for fs in self.flights.values()]
        pax_list = generate_pax_itineraries(
            self.cfg, sched_list, self.airports, rng=self.rng
        )
        for p in pax_list:
            self.pax[p.pax_id] = p
            if len(p.legs) >= 2:
                self._incoming_pax[p.legs[1]].append(p.pax_id)
            if p.legs:
                self._outgoing_pax[p.legs[0]].append(p.pax_id)

        self.metrics.total_flights = len(self.flights)
        self.metrics.total_pax = sum(p.group_size for p in self.pax.values())
        self.metrics.connecting_pax = sum(
            p.group_size for p in self.pax.values() if len(p.legs) >= 2
        )

        # 4. Sample INTRINSIC delays only (propagated computed dynamically)
        self._sample_intrinsic_delays()

        # 5. Handlers
        self._register_handlers()

        # 6. Schedule events
        self._schedule_initial_events()

        self._done = False
        return self._advance_to_next_hnh()

    def step(self, action: int) -> Tuple[FlightContext, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode is done. Call reset().")

        hold_minutes = self.cfg.hold_actions[action]
        fid = self._pending_hnh_flight
        fs = self.flights[fid]

        # Apply hold
        fs.hold_delay = float(hold_minutes)
        fs.hnh_decided = True
        fs.hnh_action = hold_minutes
        self.metrics.hold_decisions.append(float(hold_minutes))

        # Update propagated delay from previous tail flight's actual arrival
        fs.propagated_departure_delay = self._compute_propagated_delay_dynamic(fid)

        # Compute actual times
        fs.compute_actual_times()

        # Schedule departure
        self.event_engine.schedule(SimEvent(
            time=fs.actual_departure,
            event_type=EventType.FLIGHT_DEPARTURE,
            flight_id=fid,
        ))

        # Compute reward before advancing
        reward = self._compute_basic_reward(fid, hold_minutes)

        # Advance to next HNH
        next_state, info = self._advance_to_next_hnh()
        return next_state, reward, self._done, info

    # ==================================================================
    # Dynamic delay propagation (core mechanism from Section 6.1)
    # ==================================================================
    def _compute_propagated_delay_dynamic(self, flight_id: str) -> float:
        """Compute propagated delay from the previous tail flight.

        Called at HNH decision time.  Uses the ACTUAL arrival of the
        previous leg (if arrived), or its best estimate.

        This is the key mechanism by which hold delays cascade:
        holding flight f → delays f's arrival → next flight on f's
        tail gets a propagated departure delay.
        """
        prev_fid = self._prev_tail_flight.get(flight_id)
        if prev_fid is None:
            return 0.0

        prev_fs = self.flights[prev_fid]
        fs = self.flights[flight_id]

        if prev_fs.actual_arrival is not None:
            prev_arrival = prev_fs.actual_arrival
        else:
            prev_arrival = (
                prev_fs.flight.scheduled_arrival + prev_fs.total_arrival_delay
            )

        prev_arr_delay = prev_arrival - prev_fs.flight.scheduled_arrival

        turnaround_sched = (
            fs.flight.scheduled_departure - prev_fs.flight.scheduled_arrival
        )
        slack = max(0, turnaround_sched - self.cfg.min_turnaround)

        return max(0.0, prev_arr_delay - slack)

    # ==================================================================
    # Intrinsic delays
    # ==================================================================
    def _sample_intrinsic_delays(self) -> None:
        for fid, fs in self.flights.items():
            fs.intrinsic_departure_delay = self.delay_sampler.sample_departure_delay(
                fs.flight
            )
            fs.airtime_delay = self.delay_sampler.sample_airtime_delay(fs.flight)
            fs.ground_departure_delay = self.delay_sampler.sample_ground_delay()
            fs.ground_arrival_delay = self.delay_sampler.sample_ground_delay()
            fs.propagated_departure_delay = 0.0

    # ==================================================================
    # Event scheduling & handlers
    # ==================================================================
    def _schedule_initial_events(self) -> None:
        for fid, fs in self.flights.items():
            hnh_time = fs.flight.scheduled_departure - 10
            self.event_engine.schedule(SimEvent(
                time=max(0, hnh_time),
                event_type=EventType.HNH_DECISION,
                flight_id=fid,
            ))

    def _register_handlers(self) -> None:
        self.event_engine.register_handler(
            EventType.FLIGHT_DEPARTURE, self._handle_departure
        )
        self.event_engine.register_handler(
            EventType.FLIGHT_ARRIVAL, self._handle_arrival
        )
        self.event_engine.register_handler(
            EventType.PAX_CONNECTION_CHECK, self._handle_pax_connection
        )

    def _handle_departure(self, event: SimEvent) -> Optional[List[SimEvent]]:
        fid = event.flight_id
        fs = self.flights[fid]
        if fs.status != FlightStatus.SCHEDULED:
            return None

        fs.status = FlightStatus.DEPARTED
        fs.compute_actual_times()
        self.metrics.departed_flights += 1
        self.metrics.departure_delays.append(fs.departure_delay_D)

        day = int(fs.actual_departure // 1440)
        self.metrics.daily_flights[day] = self.metrics.daily_flights.get(day, 0) + 1

        return [SimEvent(
            time=fs.actual_arrival,
            event_type=EventType.FLIGHT_ARRIVAL,
            flight_id=fid,
        )]

    def _handle_arrival(self, event: SimEvent) -> Optional[List[SimEvent]]:
        fid = event.flight_id
        fs = self.flights[fid]
        fs.status = FlightStatus.ARRIVED
        fs.compute_actual_times()

        arr_delay = fs.arrival_delay_A
        self.metrics.arrived_flights += 1
        self.metrics.arrival_delays.append(arr_delay)

        day = int(fs.actual_arrival // 1440)
        if arr_delay <= self.cfg.ontime_threshold:
            self.metrics.ontime_arrivals += 1
            self.metrics.daily_ontime[day] = self.metrics.daily_ontime.get(day, 0) + 1

        au = 1.0 - min(max(arr_delay, 0), self.cfg.delta_f) / self.cfg.delta_f
        self.context_engine.record_global_au(event.time, au)

        # Trigger PAX connection checks for PAX on this flight with onward legs
        new_events = []
        for pax_id in self._outgoing_pax.get(fid, []):
            pax = self.pax.get(pax_id)
            if pax and len(pax.legs) >= 2 and pax.legs[0] == fid:
                new_events.append(SimEvent(
                    time=event.time + 0.1,
                    event_type=EventType.PAX_CONNECTION_CHECK,
                    flight_id=fid,
                    pax_id=pax_id,
                ))
        return new_events if new_events else None

    def _handle_pax_connection(self, event: SimEvent) -> Optional[List[SimEvent]]:
        pax = self.pax.get(event.pax_id)
        if pax is None or len(pax.legs) < 2:
            return None

        arriving_fid = pax.legs[0]
        connecting_fid = pax.legs[1]
        arr_fs = self.flights.get(arriving_fid)
        conn_fs = self.flights.get(connecting_fid)
        if arr_fs is None or conn_fs is None:
            return None

        actual_arrival = arr_fs.actual_arrival
        if actual_arrival is None:
            actual_arrival = arr_fs.flight.scheduled_arrival + arr_fs.total_arrival_delay

        # Connecting flight's best-known departure time
        if conn_fs.actual_departure is not None:
            conn_departure = conn_fs.actual_departure
        elif conn_fs.hnh_decided:
            conn_departure = (
                conn_fs.flight.scheduled_departure + conn_fs.total_departure_delay
            )
        else:
            conn_departure = (
                conn_fs.flight.scheduled_departure
                + max(conn_fs.intrinsic_departure_delay,
                      conn_fs.propagated_departure_delay)
                + conn_fs.ground_departure_delay
            )

        dest_airport = arr_fs.flight.destination
        apt = self.airports.get(dest_airport)
        mct = apt.mct if apt else self.cfg.mct_default
        connection_window = conn_departure - actual_arrival

        day = int(actual_arrival // 1440)

        if connection_window >= mct:
            self.metrics.successful_connections += pax.group_size
            self.metrics.daily_connections[day] = (
                self.metrics.daily_connections.get(day, 0) + pax.group_size
            )
            pax.missed_connection = False
            pax.delay_to_destination = max(0, arr_fs.arrival_delay_A)
            sigma = self.context_engine._pax_disutility(pax.delay_to_destination)
            self.context_engine.record_global_pu(event.time, 1.0 - sigma)
        else:
            self.metrics.missed_connections += pax.group_size
            self.metrics.daily_missed[day] = (
                self.metrics.daily_missed.get(day, 0) + pax.group_size
            )
            pax.missed_connection = True
            rebook_delay = self._rebook_pax(pax, dest_airport, actual_arrival)
            pax.delay_to_destination = rebook_delay
            self.metrics.pax_delays.append(rebook_delay * pax.group_size)
            sigma = self.context_engine._pax_disutility(rebook_delay)
            self.context_engine.record_global_pu(event.time, 1.0 - sigma)
        return None

    # ==================================================================
    # PAX rebooking
    # ==================================================================
    def _rebook_pax(
        self, pax: PaxItinerary, current_airport: str, current_time: float
    ) -> float:
        final_dest = pax.destination
        candidates = self._departures_at_airport.get(current_airport, [])
        best_delay = self.cfg.delta_p
        for fid in candidates:
            fs = self.flights[fid]
            if fs.flight.destination != final_dest:
                continue
            if fs.flight.scheduled_departure <= current_time:
                continue
            est_arr = fs.flight.scheduled_arrival + fs.total_arrival_delay
            original_arr = self.flights[pax.legs[-1]].flight.scheduled_arrival
            delay = est_arr - original_arr
            if 0 < delay < best_delay:
                best_delay = delay
                pax.rebooked_flight_id = fid
        return max(0, best_delay)

    # ==================================================================
    # Advance to next HNH
    # ==================================================================
    def _advance_to_next_hnh(self) -> Tuple[FlightContext, Dict[str, Any]]:
        hnh_event = self.event_engine.run_until_hnh()
        if hnh_event is None:
            self.event_engine.drain()
            self._done = True
            return FlightContext(flight_id="DONE"), self.metrics.summary()

        fid = hnh_event.flight_id
        fs = self.flights[fid]
        self._pending_hnh_flight = fid

        # Dynamic propagated delay
        fs.propagated_departure_delay = self._compute_propagated_delay_dynamic(fid)

        connecting_pax_ids = self._incoming_pax.get(fid, [])
        connecting_pax = [self.pax[pid] for pid in connecting_pax_ids if pid in self.pax]

        ctx = self.context_engine.build_context(
            flight_state=fs,
            connecting_pax=connecting_pax,
            hold_actions=self.cfg.hold_actions,
            airports=self.airports,
            flight_map=self.flights,
            current_time=hnh_event.time,
        )
        self._pending_context = ctx

        info = {
            "flight_id": fid,
            "origin": fs.flight.origin,
            "destination": fs.flight.destination,
            "connecting_pax_count": len(connecting_pax),
            "scheduled_departure": fs.flight.scheduled_departure,
            "intrinsic_delay": fs.intrinsic_departure_delay,
            "propagated_delay": fs.propagated_departure_delay,
            "tau_star": ctx.tau_star,
        }
        return ctx, info

    # ==================================================================
    # Basic reward placeholder
    # ==================================================================
    def _compute_basic_reward(self, flight_id: str, hold_minutes: float) -> float:
        ctx = self._pending_context
        if ctx is None:
            return 0.0
        action_idx = self.cfg.hold_actions.index(int(hold_minutes))
        local_pu = ctx.PL[action_idx] if action_idx < len(ctx.PL) else 0.5
        local_au = ctx.AL[action_idx] if action_idx < len(ctx.AL) else 0.5
        reward = self.cfg.alpha * local_pu + (1 - self.cfg.alpha) * local_au
        self.metrics.rewards.append(reward)
        return reward

    # ==================================================================
    # Baseline policies
    # ==================================================================
    def run_episode(
        self,
        policy: str = "no_hold",
        max_hold: int = 15,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        ctx, info = self.reset(seed=seed)
        done = False
        while not done:
            action = self._select_baseline_action(policy, max_hold)
            ctx, reward, done, info = self.step(action)
        return self.metrics.summary()

    def _select_baseline_action(self, policy: str, max_hold: int = 15) -> int:
        """Baseline action selection.

        no_hold:     always τ=0
        heuristic:   hold only when connecting PAX are at risk of missing
                     this flight.  Find the minimum hold ≤ max_hold that
                     allows the most-delayed at-risk inbound to connect.
        random:      uniform random.
        """
        if policy == "no_hold":
            return 0

        if policy.startswith("heuristic"):
            fid = self._pending_hnh_flight
            fs = self.flights[fid]

            # PAX whose second leg is this flight (trying to connect IN)
            connecting_pax_ids = self._incoming_pax.get(fid, [])
            if not connecting_pax_ids:
                return 0  # No connecting PAX → no reason to hold

            # Current estimated departure (without any hold)
            est_dep = (
                fs.flight.scheduled_departure
                + max(fs.intrinsic_departure_delay, fs.propagated_departure_delay)
                + fs.ground_departure_delay
            )

            connection_airport = fs.flight.origin
            apt = self.airports.get(connection_airport)
            mct = apt.mct if apt else self.cfg.mct_default

            max_needed_hold = 0.0

            for pax_id in connecting_pax_ids:
                pax = self.pax.get(pax_id)
                if pax is None or len(pax.legs) < 2:
                    continue

                # Inbound (feeder) flight
                inbound_fid = pax.legs[0]
                inbound_fs = self.flights.get(inbound_fid)
                if inbound_fs is None:
                    continue

                # Best estimate of inbound arrival
                if inbound_fs.actual_arrival is not None:
                    inbound_arr = inbound_fs.actual_arrival
                else:
                    inbound_arr = (
                        inbound_fs.flight.scheduled_arrival
                        + inbound_fs.total_arrival_delay
                    )

                # Connection window = outbound departure − inbound arrival
                window = est_dep - inbound_arr
                if window >= mct:
                    continue  # PAX will make it without a hold

                # PAX will miss — compute extra hold needed
                needed = mct - window
                if needed <= max_hold:
                    max_needed_hold = max(max_needed_hold, needed)

            if max_needed_hold <= 0:
                return 0  # All PAX will make it (or need > max_hold)

            # Pick the smallest hold action that covers the need
            for idx, tau in enumerate(self.cfg.hold_actions):
                if tau >= max_needed_hold:
                    return idx
            # Fallback: largest action ≤ max_hold
            for idx in range(len(self.cfg.hold_actions) - 1, -1, -1):
                if self.cfg.hold_actions[idx] <= max_hold:
                    return idx
            return 0

        if policy == "random":
            return int(self.rng.integers(0, len(self.cfg.hold_actions)))
        return 0

    # ==================================================================
    # Data access for delay tree (Raghav's reward engine interface)
    # ==================================================================
    def get_flight_state(self, flight_id: str) -> Optional[FlightState]:
        return self.flights.get(flight_id)

    def get_tail_flights(self, tail_id: str) -> List[str]:
        return self._flights_by_tail.get(tail_id, [])

    def get_incoming_pax_flights(self, flight_id: str) -> List[str]:
        pax_ids = self._incoming_pax.get(flight_id, [])
        feeder = set()
        for pid in pax_ids:
            pax = self.pax.get(pid)
            if pax and len(pax.legs) >= 2:
                feeder.add(pax.legs[0])
        return list(feeder)

    def get_previous_tail_flight(self, flight_id: str) -> Optional[str]:
        return self._prev_tail_flight.get(flight_id)

    def get_next_tail_flight(self, flight_id: str) -> Optional[str]:
        return self._next_tail_flight.get(flight_id)
