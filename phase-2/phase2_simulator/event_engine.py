"""
Discrete Event Engine for the logistics cross-docking microsimulator.

Near-identical copy of simulator/event_engine.py.
The only changes are:
  1. Import path: phase2_simulator.models instead of simulator.models
  2. run_until_hnh()  →  run_until_hold_decision()  (renamed for clarity)
  3. Docstring updated to logistics domain

The min-heap mechanics, handler registration, step(), run_until(),
drain(), and clear() are byte-for-byte the same as the original.

Reference: mirrors simulator/event_engine.py  (Section 6.1 of the
Malladi et al. paper — the discrete-event engine is domain-agnostic).
"""

from __future__ import annotations

import heapq
from typing import Callable, Dict, List, Optional

from phase2_simulator.models import SimEvent, EventType


class EventEngine:
    """Min-heap based discrete event engine.

    Identical to simulator/event_engine.py EventEngine.
    Drives the simulation clock for the cross-docking hub.
    """

    def __init__(self) -> None:
        self._queue: List[SimEvent] = []
        self._handlers: Dict[EventType, Callable[[SimEvent], Optional[List[SimEvent]]]] = {}
        self.current_time: float = 0.0
        self._event_count: int = 0

    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------
    def register_handler(
        self,
        event_type: EventType,
        handler: Callable[[SimEvent], Optional[List[SimEvent]]],
    ) -> None:
        """Register a callback for a specific event type.

        The handler receives the event and may return a list of new events
        to be scheduled.
        """
        self._handlers[event_type] = handler

    # -----------------------------------------------------------------
    # Scheduling
    # -----------------------------------------------------------------
    def schedule(self, event: SimEvent) -> None:
        """Push an event onto the priority queue."""
        heapq.heappush(self._queue, event)

    def schedule_many(self, events: List[SimEvent]) -> None:
        for e in events:
            self.schedule(e)

    # -----------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------
    def has_events(self) -> bool:
        return len(self._queue) > 0

    def peek_time(self) -> Optional[float]:
        if self._queue:
            return self._queue[0].time
        return None

    def step(self) -> Optional[SimEvent]:
        """Process the next event. Returns the processed event or None."""
        if not self._queue:
            return None

        event = heapq.heappop(self._queue)
        self.current_time = event.time
        self._event_count += 1

        handler = self._handlers.get(event.event_type)
        if handler is not None:
            new_events = handler(event)
            if new_events:
                self.schedule_many(new_events)

        return event

    def run_until(self, end_time: float) -> int:
        """Process all events up to (and including) *end_time*.

        Returns the number of events processed.
        """
        processed = 0
        while self._queue and self._queue[0].time <= end_time:
            self.step()
            processed += 1
        return processed

    def run_until_hold_decision(self) -> Optional[SimEvent]:
        """Advance the simulation until the next HOLD_DECISION event.

        All intermediate events (docks, departures, cargo checks) are
        processed automatically.  Returns the HOLD_DECISION event so the
        external RL agent can provide an action.  Returns None if no more
        HOLD_DECISION events remain.

        Renamed from run_until_hnh() in the aviation simulator.
        """
        while self._queue:
            if self._queue[0].event_type == EventType.HOLD_DECISION:
                return heapq.heappop(self._queue)
            self.step()
        return None

    def drain(self) -> int:
        """Process all remaining events. Returns count processed."""
        n = 0
        while self._queue:
            self.step()
            n += 1
        return n

    def clear(self) -> None:
        self._queue.clear()
        self.current_time = 0.0
        self._event_count = 0

    @property
    def total_events_processed(self) -> int:
        return self._event_count
