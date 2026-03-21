"""
Core data models for the airline network simulator.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict


# ======================================================================
# Airport
# ======================================================================
@dataclass
class Airport:
    """An airport in the network."""
    code: str                       # e.g. "HUB1", "SPK023"
    is_hub: bool = False
    mct: int = 45                   # minimum connection time (minutes)


# ======================================================================
# Flight & Schedule
# ======================================================================
class FlightStatus(Enum):
    SCHEDULED = auto()
    DEPARTED = auto()
    ARRIVED = auto()
    CANCELLED = auto()


@dataclass
class ScheduledFlight:
    """The *planned* (static) properties of a flight leg."""
    flight_id: str                  # unique ID  e.g. "F-0042-D3" (flight 42, day 3)
    flight_number: str              # repeating number e.g. "F-0042"
    origin: str                     # airport code
    destination: str                # airport code
    scheduled_departure: float      # absolute sim-minutes from t=0
    scheduled_arrival: float        # absolute sim-minutes from t=0
    tail_id: str                    # aircraft tail this leg belongs to
    leg_index: int                  # position in the tail plan for this day
    haul_type: str = "medium"       # "short", "medium", "long"
    seat_capacity: int = 160


@dataclass
class FlightState:
    """Mutable runtime state of a flight during simulation."""
    flight: ScheduledFlight

    # --- Delays (all in minutes, positive = late) ---
    intrinsic_departure_delay: float = 0.0   # sampled from delay distribution
    propagated_departure_delay: float = 0.0  # from previous leg arrival delay
    hold_delay: float = 0.0                  # hold decision by RL agent
    ground_departure_delay: float = 0.0      # turnaround variability
    airtime_delay: float = 0.0               # in-air variability
    ground_arrival_delay: float = 0.0        # taxi-in variability

    # --- Computed actual times ---
    actual_departure: Optional[float] = None
    actual_arrival: Optional[float] = None

    status: FlightStatus = FlightStatus.SCHEDULED

    # --- PAX lists (populated by pax generator) ---
    boarded_pax: List[str] = field(default_factory=list)        # pax IDs on board
    connecting_incoming_pax: List[str] = field(default_factory=list)  # pax connecting IN

    # --- Hold-or-Not-Hold decision tracking ---
    hnh_decided: bool = False
    hnh_action: int = 0             # hold minutes chosen

    # -----------------------------------------------------------------
    # Computed properties
    # -----------------------------------------------------------------
    @property
    def total_departure_delay(self) -> float:
        """Total departure delay including all components."""
        return max(
            self.intrinsic_departure_delay,
            self.propagated_departure_delay,
        ) + self.hold_delay + self.ground_departure_delay

    @property
    def total_arrival_delay(self) -> float:
        """Total arrival delay."""
        dep_delay = self.total_departure_delay
        return dep_delay + self.airtime_delay + self.ground_arrival_delay

    @property
    def departure_delay_D(self) -> float:
        """D_i in the paper: departure delay used for DT."""
        if self.actual_departure is None:
            return self.total_departure_delay
        return self.actual_departure - self.flight.scheduled_departure

    @property
    def arrival_delay_A(self) -> float:
        """A_i in the paper: arrival delay used for DT."""
        if self.actual_arrival is None:
            return self.total_arrival_delay
        return self.actual_arrival - self.flight.scheduled_arrival

    def compute_actual_times(self) -> None:
        """Set actual departure / arrival based on delays."""
        self.actual_departure = (
            self.flight.scheduled_departure + self.total_departure_delay
        )
        self.actual_arrival = (
            self.flight.scheduled_arrival + self.total_arrival_delay
        )


# ======================================================================
# PAX (Passenger itinerary)
# ======================================================================
@dataclass
class PaxItinerary:
    """A single passenger (or group) itinerary with one or more legs."""
    pax_id: str
    group_size: int = 1             # number of PAX in this booking
    legs: List[str] = field(default_factory=list)  # ordered flight_ids
    origin: str = ""                # ultimate origin airport
    destination: str = ""           # ultimate destination airport

    # --- Runtime state ---
    current_leg_idx: int = 0
    missed_connection: bool = False
    rebooked_flight_id: Optional[str] = None
    delay_to_destination: float = 0.0   # minutes of delay experienced


# ======================================================================
# Tail plan (aircraft routing)
# ======================================================================
@dataclass
class TailPlan:
    """Daily route for a single aircraft (tail)."""
    tail_id: str
    legs: List[str] = field(default_factory=list)  # ordered flight_ids for the day
    base_airport: str = ""          # starting airport


# ======================================================================
# Event types for the discrete event engine
# ======================================================================
class EventType(Enum):
    FLIGHT_DEPARTURE = auto()
    FLIGHT_ARRIVAL = auto()
    HNH_DECISION = auto()           # hold-or-not-hold decision point
    PAX_CONNECTION_CHECK = auto()    # check if connecting PAX make it
    DAY_START = auto()
    DAY_END = auto()


@dataclass
class SimEvent:
    """A single event in the discrete-event queue."""
    time: float                     # absolute sim-minutes
    event_type: EventType
    flight_id: str = ""
    pax_id: str = ""
    data: Dict = field(default_factory=dict)

    def __lt__(self, other: "SimEvent") -> bool:
        return self.time < other.time
