"""
bay_manager.py — Docking bay assignment and GB (bay-blockage) delay tracking.

This is the core Phase 2 novelty with no Phase 1 analog.

In aviation, gates are plentiful and gate-blocking rarely cascades.
In logistics, docking bays are scarce (15 bays for 80 trucks/day).
A truck held for τ extra minutes blocks its bay for τ extra minutes,
causing the next queued truck to wait — the GB delay.

Key outputs:
  - gb_delay (GB_k): how long a truck waited to get a bay
  - bay_utilization (B_G): fraction of bays occupied right now
  - blocked_trucks: {truck_id: gb_delay} — for bay_congestion attribution

This feeds directly into:
  1. The Delay Tree (GB_k node)
  2. The state vector (B_G, Z_G dimensions)
  3. The bay-congestion reward attribution
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BayAssignment:
    """Tracks one bay's current occupant."""
    bay_id: int
    truck_id: str
    assigned_at: float      # simulation time bay was assigned
    occupied_until: float   # simulation time bay will be freed
    hold_applied: float = 0.0  # extra hold added to this bay's occupation


@dataclass
class QueuedTruck:
    """A truck waiting in queue for a free bay."""
    arrival_time: float
    truck_id: str
    requested_at: float

    def __lt__(self, other: "QueuedTruck") -> bool:
        return self.arrival_time < other.arrival_time


class BayManager:
    """Manages docking bay allocation for the cross-docking hub.

    Responsibilities:
      1. Assign arriving trucks to free bays (or queue them)
      2. Track hold-induced bay occupation extensions
      3. Compute GB delay (time a truck waited for a free bay)
      4. Expose bay utilization metrics for the state vector

    Thread-safety: single-threaded simulation only.
    """

    def __init__(self, n_bays: int, operating_start: float = 360.0):
        """
        Args:
            n_bays: Number of docking bays at the hub
            operating_start: Hub opening time in minutes from midnight
        """
        self.n_bays = n_bays
        self.operating_start = operating_start

        # Bay state: bay_id → BayAssignment or None
        self._bays: Dict[int, Optional[BayAssignment]] = {
            i: None for i in range(n_bays)
        }

        # FIFO queue of waiting trucks
        self._queue: List[QueuedTruck] = []

        # Completed assignments: {truck_id: gb_delay_minutes}
        self._completed_gb: Dict[str, float] = {}

        # Trucks blocked because held_truck blocked their bay
        # {blocking_truck_id: {blocked_truck_id: gb_delay}}
        self._congestion_log: Dict[str, Dict[str, float]] = {}

        # Time tracking for utilization
        self._last_event_time: float = operating_start
        self._accumulated_bay_minutes: float = 0.0   # sum of (bays_occupied × dt)
        self._total_minutes: float = 0.0

    # ── Public API ─────────────────────────────────────────────────────

    def truck_arrives(self, truck_id: str, arrival_time: float,
                      processing_time: float = 30.0) -> Tuple[float, float]:
        """A truck arrives and requests a docking bay.

        Args:
            truck_id: Unique truck identifier
            arrival_time: Simulation time of arrival (minutes)
            processing_time: Base time to unload cargo (minutes, default 30)

        Returns:
            (bay_assigned_time, gb_delay) where:
              - bay_assigned_time: when truck actually got a bay
              - gb_delay: how long it waited (GB_k node value)
        """
        self._update_utilization(arrival_time)

        free_bay = self._find_free_bay(arrival_time)

        if free_bay is not None:
            # Bay available immediately
            gb_delay = 0.0
            occupied_until = arrival_time + processing_time
            self._bays[free_bay] = BayAssignment(
                bay_id=free_bay,
                truck_id=truck_id,
                assigned_at=arrival_time,
                occupied_until=occupied_until,
            )
            self._completed_gb[truck_id] = 0.0
            return arrival_time, 0.0
        else:
            # Must queue: GB delay = time until earliest bay is free
            earliest_free_time = self._earliest_free_time()
            gb_delay = max(0.0, earliest_free_time - arrival_time)
            bay_assigned_time = earliest_free_time

            # Register the queue entry (will be resolved when bay freed)
            heapq.heappush(
                self._queue,
                QueuedTruck(
                    arrival_time=arrival_time,
                    truck_id=truck_id,
                    requested_at=arrival_time,
                )
            )
            self._completed_gb[truck_id] = gb_delay
            return bay_assigned_time, gb_delay

    def apply_hold(self, truck_id: str, hold_minutes: float,
                   current_time: float) -> Dict[str, float]:
        """Apply a hold decision to a truck currently occupying a bay.

        Extending the truck's bay occupation causes downstream trucks
        in the queue to wait longer — this is the bay-congestion cascade.

        Args:
            truck_id: The truck being held
            hold_minutes: Hold duration decided by RL agent
            current_time: Current simulation time

        Returns:
            {blocked_truck_id: additional_gb_delay} for trucks newly
            delayed by this hold. Empty if no trucks are queued.
        """
        if hold_minutes <= 0:
            return {}

        self._update_utilization(current_time)

        # Find the bay this truck occupies
        bay_id = self._find_truck_bay(truck_id)
        if bay_id is None:
            return {}

        assignment = self._bays[bay_id]
        old_until = assignment.occupied_until
        assignment.occupied_until += hold_minutes
        assignment.hold_applied += hold_minutes

        # Track which queued trucks are now additionally delayed
        newly_delayed: Dict[str, float] = {}
        for qt in self._queue:
            # Trucks that were waiting for this specific bay get extra delay
            # (simplified: trucks waiting for any bay may be affected)
            if self._completed_gb.get(qt.truck_id, 0.0) > 0:
                extra_delay = hold_minutes
                self._completed_gb[qt.truck_id] = (
                    self._completed_gb.get(qt.truck_id, 0.0) + extra_delay
                )
                newly_delayed[qt.truck_id] = extra_delay

        # Log congestion for reward attribution
        if newly_delayed:
            self._congestion_log[truck_id] = newly_delayed

        return newly_delayed

    def truck_departs(self, truck_id: str, departure_time: float) -> float:
        """Release the bay when a truck departs.

        Also processes the queue: the next waiting truck gets the freed bay.

        Args:
            truck_id: The departing truck
            departure_time: Actual departure time

        Returns:
            The actual departure time (clamped to bay availability)
        """
        self._update_utilization(departure_time)
        bay_id = self._find_truck_bay(truck_id)

        if bay_id is not None:
            self._bays[bay_id] = None  # Free the bay

        # Process queue: assign the next waiting truck
        if self._queue:
            next_truck = heapq.heappop(self._queue)
            processing_time = 30.0  # default
            occupied_until = departure_time + processing_time
            free_bay_id = bay_id if bay_id is not None else self._find_free_bay(departure_time)
            if free_bay_id is not None:
                self._bays[free_bay_id] = BayAssignment(
                    bay_id=free_bay_id,
                    truck_id=next_truck.truck_id,
                    assigned_at=departure_time,
                    occupied_until=occupied_until,
                )

        return departure_time

    def get_gb_delay(self, truck_id: str) -> float:
        """Get the recorded GB delay for a truck. Returns 0 if not found."""
        return self._completed_gb.get(truck_id, 0.0)

    def get_blocked_trucks(self, holding_truck_id: str) -> Dict[str, float]:
        """Get trucks blocked by a specific holding truck's bay occupation."""
        return self._congestion_log.get(holding_truck_id, {})

    def get_bay_utilization(self, current_time: float) -> float:
        """Return current bay utilization rate B_G ∈ [0, 1].

        B_G = fraction of bays currently occupied.
        """
        self._update_utilization(current_time)
        occupied = sum(
            1 for assignment in self._bays.values()
            if assignment is not None and assignment.occupied_until >= current_time
        )
        return occupied / self.n_bays

    def get_queue_depth(self) -> int:
        """Return number of trucks currently waiting for a bay (Z_G input)."""
        return len(self._queue)

    def get_rolling_utilization(self) -> float:
        """Return time-averaged bay utilization (for global state)."""
        if self._total_minutes < 1.0:
            return 0.0
        return self._accumulated_bay_minutes / (self._total_minutes * self.n_bays)

    def reset(self):
        """Reset bay manager for a new episode."""
        self._bays = {i: None for i in range(self.n_bays)}
        self._queue.clear()
        self._completed_gb.clear()
        self._congestion_log.clear()
        self._accumulated_bay_minutes = 0.0
        self._total_minutes = 0.0
        self._last_event_time = self.operating_start

    # ── Private helpers ────────────────────────────────────────────────

    def _find_free_bay(self, current_time: float) -> Optional[int]:
        """Find any free bay at current_time. Returns None if all occupied."""
        for bay_id, assignment in self._bays.items():
            if assignment is None or assignment.occupied_until <= current_time:
                if assignment is not None and assignment.occupied_until <= current_time:
                    self._bays[bay_id] = None  # mark as freed
                return bay_id
        return None

    def _earliest_free_time(self) -> float:
        """Return the earliest time any bay will be free."""
        times = [
            a.occupied_until for a in self._bays.values() if a is not None
        ]
        return min(times) if times else self._last_event_time

    def _find_truck_bay(self, truck_id: str) -> Optional[int]:
        """Find which bay a truck is occupying."""
        for bay_id, assignment in self._bays.items():
            if assignment is not None and assignment.truck_id == truck_id:
                return bay_id
        return None

    def _update_utilization(self, current_time: float):
        """Update time-accumulated bay utilization metrics."""
        dt = max(0.0, current_time - self._last_event_time)
        occupied = sum(
            1 for a in self._bays.values()
            if a is not None and a.occupied_until >= self._last_event_time
        )
        self._accumulated_bay_minutes += occupied * dt
        self._total_minutes += dt
        self._last_event_time = current_time
