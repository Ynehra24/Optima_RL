"""
event_engine.py — Discrete event priority queue.

A direct port of the Phase 1 EventEngine pattern. Events are stored in
a heap, ordered by simulation time (with a sequence number tiebreaker
for stable ordering), and dispatched to registered handlers.

The simulator drives time forward by repeatedly popping the earliest
event and calling its handler. Handlers may return a list of new events
to schedule, allowing event chains (PACKET_ARRIVAL -> HNH_DECISION ->
PACKET_FORWARD -> PACKET_DELIVERED -> PACKET_ARRIVAL at next hop, ...).

`run_until_hnh()` is the key method: it runs the event loop until the
next HNH_DECISION event surfaces, then returns control to the caller
(the RL agent or simulator step()) for the action choice.
"""

from __future__ import annotations
import heapq
from typing import Callable, Dict, List, Optional

from simulator.models import EventType, SimEvent


class EventEngine:
    """Priority queue + handler dispatch.

    Same interface as phase-1/simulator/event_engine.EventEngine:
        schedule(event)         — push event onto the heap
        register_handler(...)   — register callback for an EventType
        run_until_hnh()         — pop events, dispatch handlers, return on HNH
        drain()                 — process all remaining events (episode end)
        clear()                 — reset internal state
    """

    def __init__(self):
        self._heap: List[SimEvent] = []
        self._handlers: Dict[EventType, Callable[[SimEvent], Optional[List[SimEvent]]]] = {}
        self._seq_counter: int = 0
        self._current_time: float = 0.0
        self._processed_count: int = 0

    # ---------------- Lifecycle ----------------
    def clear(self) -> None:
        self._heap.clear()
        self._handlers.clear()
        self._seq_counter = 0
        self._current_time = 0.0
        self._processed_count = 0

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def queue_size(self) -> int:
        return len(self._heap)

    # ---------------- Scheduling ----------------
    def schedule(self, event: SimEvent) -> None:
        """Push an event onto the heap.

        We assign a monotonically increasing seq number so that events
        with identical times are dispatched in FIFO order — this avoids
        nondeterministic ordering across runs.
        """
        if event.time < self._current_time:
            # Don't allow events scheduled in the past — clamp to "now".
            # This can happen with floating-point rounding.
            event.time = self._current_time
        event.seq = self._seq_counter
        self._seq_counter += 1
        heapq.heappush(self._heap, event)

    def schedule_many(self, events: List[SimEvent]) -> None:
        for e in events:
            self.schedule(e)

    # ---------------- Handler registration ----------------
    def register_handler(
        self,
        event_type: EventType,
        handler: Callable[[SimEvent], Optional[List[SimEvent]]],
    ) -> None:
        self._handlers[event_type] = handler

    # ---------------- Main loop ----------------
    def run_until_hnh(self) -> Optional[SimEvent]:
        """Run events until the next HNH_DECISION surfaces or the queue empties.

        Returns the HNH_DECISION event for the caller to act on.
        Returns None if the queue is exhausted with no HNH events.
        """
        while self._heap:
            event = heapq.heappop(self._heap)
            self._current_time = event.time

            if event.event_type == EventType.HNH_DECISION:
                # Hand back to the simulator/agent without dispatching.
                # The simulator will set the action and explicitly schedule
                # the resulting PACKET_FORWARD event.
                return event

            handler = self._handlers.get(event.event_type)
            if handler is not None:
                new_events = handler(event)
                self._processed_count += 1
                if new_events:
                    for ne in new_events:
                        self.schedule(ne)
            else:
                # No handler registered — skip silently. (Phase 1 does
                # the same; missing handlers are usually a sign of a
                # work-in-progress event type, not a bug.)
                self._processed_count += 1

        return None

    def drain(self) -> int:
        """Process all remaining events, no early exit.

        Used at episode end to flush any pending arrivals/deliveries
        so final metrics are accurate.
        """
        drained = 0
        while self._heap:
            event = heapq.heappop(self._heap)
            self._current_time = event.time
            handler = self._handlers.get(event.event_type)
            if handler is not None:
                new_events = handler(event)
                drained += 1
                self._processed_count += 1
                if new_events:
                    for ne in new_events:
                        # Only schedule events at or after current time
                        # to prevent infinite loops at episode end.
                        self.schedule(ne)
            else:
                drained += 1
                self._processed_count += 1
        return drained

    # ---------------- Inspection ----------------
    def peek_next_time(self) -> Optional[float]:
        """Time of the next event without removing it."""
        if not self._heap:
            return None
        return self._heap[0].time

    def peek_next_type(self) -> Optional[EventType]:
        if not self._heap:
            return None
        return self._heap[0].event_type
