"""
Core data models for the logistics cross-docking simulator.

Direct counterpart of simulator/models.py.
Aviation → Logistics mapping:
  Airport          → Hub
  ScheduledFlight  → ScheduledTruck
  FlightState      → TruckState
  PaxItinerary     → CargoUnit
  TailPlan         → TruckPlan
  EventType        → EventType  (renamed events)
  SimEvent         → SimEvent   (identical structure)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


# ======================================================================
# Hub  (replaces Airport)
# ======================================================================
@dataclass
class Hub:
    """A cross-docking hub in the logistics network."""
    hub_id: str                     # e.g. "MAIN0", "SPOKE023"
    is_main: bool = False           # main hub = better infrastructure, lower min-transfer-time
    min_transfer_time: int = 35     # minimum time (min) for cargo to transfer between trucks
    num_bays: int = 20              # total docking bays available at this hub


# ======================================================================
# Scheduled Truck & State  (replaces ScheduledFlight / FlightState)
# ======================================================================
class TruckStatus(Enum):
    SCHEDULED = auto()
    DOCKED = auto()       # truck arrived at hub, being unloaded
    DEPARTED = auto()
    DELIVERED = auto()    # reached final depot
    CANCELLED = auto()


@dataclass
class ScheduledTruck:
    """The *planned* (static) properties of one truck leg.

    Mirrors ScheduledFlight field-for-field.
    """
    truck_id: str               # unique ID e.g. "T-0042-D3" (truck 42, day 3)
    truck_number: str           # repeating number e.g. "T-0042"
    origin_hub: str             # hub code
    dest_hub: str               # hub code
    scheduled_dock: float       # absolute sim-minutes — when truck docks at origin hub
    scheduled_departure: float  # absolute sim-minutes — when truck leaves origin hub
    scheduled_arrival: float    # absolute sim-minutes — when truck arrives at dest hub
    route_id: str               # route this leg belongs to (≈ tail_id)
    leg_index: int              # position in route for this day
    route_type: str = "medium"  # "short", "medium", "long" (by km)
    cargo_capacity: int = 120   # total cargo unit slots on this truck


@dataclass
class TruckState:
    """Mutable runtime state of a truck during simulation.

    Mirrors FlightState field-for-field, with logistics-specific additions
    from the PDF state-space reference.

    Delay components (all in minutes, positive = late):
      intrinsic_departure_delay  ← intrinsic_departure_delay (same)
      propagated_departure_delay ← propagated_departure_delay (same)
      hold_delay                 ← hold_delay (same)
      bay_dwell_delay            ← ground_departure_delay / GA (gate congestion)
      road_delay                 ← airtime_delay / T_i (en-route variance)
      bay_arrival_delay          ← ground_arrival_delay (taxi-in)
    """
    truck: ScheduledTruck

    # --- Core delays (mirroring FlightState) ---
    intrinsic_departure_delay: float = 0.0   # sampled from delay distribution
    propagated_departure_delay: float = 0.0  # cascaded from previous leg
    hold_delay: float = 0.0                  # hold decision by RL agent
    bay_dwell_delay: float = 0.0             # waiting for free docking bay (G_bay_k)
    road_delay: float = 0.0                  # en-route traffic / weather (G_road_k)
    bay_arrival_delay: float = 0.0           # arrival dwell at destination hub

    # --- Actual times ---
    actual_dock: Optional[float] = None       # actual dock time at origin hub
    actual_departure: Optional[float] = None
    actual_arrival: Optional[float] = None

    status: TruckStatus = TruckStatus.SCHEDULED

    # --- Cargo lists ---
    loaded_cargo: List[str] = field(default_factory=list)       # cargo_ids on board
    connecting_inbound_cargo: List[str] = field(default_factory=list)  # cargo connecting IN

    # --- Hold decision tracking ---
    hold_decided: bool = False
    hold_action: int = 0    # hold minutes chosen by agent

    # -----------------------------------------------------------------------
    # New logistics-only state fields (PDF §2 local state)
    # -----------------------------------------------------------------------
    # V_k: cargo value score of connecting inbound cargo (normalised [0,1])
    cargo_value_score: float = 0.5
    # Q_k: cargo volume fraction (connecting units / truck capacity)
    cargo_volume_fraction: float = 0.0
    # X_k: SLA urgency {0=standard, 1=next-day, 2=same-day express}
    sla_urgency: int = 0
    # E_k: cargo perishability fraction (fraction that is temp-sensitive)
    perishability_fraction: float = 0.0
    # L_k: driver hours remaining (minutes) — hard cap on hold duration
    driver_hours_remaining: float = 270.0   # 4.5 h = 270 min (EU HGV default)
    # F_k: downstream deadline pressure (normalised [0,1]; 0=time-critical)
    deadline_pressure: float = 1.0
    # N_in: number of inbound trucks feeding cargo into this outbound truck
    n_inbound_trucks: int = 0
    # BG at decision time (bay utilisation rate at hub)
    bay_utilisation_at_decision: float = 0.0

    # -----------------------------------------------------------------
    # Computed properties  (mirrors FlightState)
    # -----------------------------------------------------------------
    @property
    def total_departure_delay(self) -> float:
        """Total departure delay including all components."""
        return (
            max(self.intrinsic_departure_delay, self.propagated_departure_delay)
            + self.hold_delay
            + self.bay_dwell_delay
        )

    @property
    def total_arrival_delay(self) -> float:
        """Total arrival delay at destination hub / depot."""
        return self.total_departure_delay + self.road_delay + self.bay_arrival_delay

    @property
    def departure_delay_D(self) -> float:
        """D_k: departure delay (minutes beyond scheduled departure)."""
        if self.actual_departure is None:
            return self.total_departure_delay
        return self.actual_departure - self.truck.scheduled_departure

    @property
    def arrival_delay_A(self) -> float:
        """A_k: arrival delay at depot (minutes beyond scheduled arrival)."""
        if self.actual_arrival is None:
            return self.total_arrival_delay
        return self.actual_arrival - self.truck.scheduled_arrival

    def compute_actual_times(self) -> None:
        """Set actual departure / arrival based on accumulated delays."""
        self.actual_departure = (
            self.truck.scheduled_departure + self.total_departure_delay
        )
        self.actual_arrival = (
            self.truck.scheduled_arrival + self.total_arrival_delay
        )


# ======================================================================
# Cargo Unit  (replaces PaxItinerary)
# ======================================================================
@dataclass
class CargoUnit:
    """A single cargo item (or batch of identical items) with a transfer itinerary.

    Mirrors PaxItinerary field-for-field.
    New logistics fields from PDF §2 (V_k, X_k, E_k).
    """
    cargo_id: str
    unit_count: int = 1             # number of physical parcels in this batch
    legs: List[str] = field(default_factory=list)  # ordered truck_ids
    origin_hub: str = ""
    destination_hub: str = ""

    # --- Runtime state ---
    current_leg_idx: int = 0
    missed_transfer: bool = False
    rebooked_truck_id: Optional[str] = None
    delay_to_destination: float = 0.0  # total delay experienced (minutes)

    # --- Logistics-specific attributes (PDF §2) ---
    # V_k,i: per-item value weight ∈ [0,1]
    value_score: float = 0.5
    # X_k: SLA urgency {0, 1, 2}
    sla_urgency: int = 0
    # E_k: perishability flag (True = temperature-sensitive)
    is_perishable: bool = False


# ======================================================================
# Truck Plan  (replaces TailPlan)
# ======================================================================
@dataclass
class TruckPlan:
    """Daily route for a single truck (one route_id = one truck chain).

    Mirrors TailPlan field-for-field.
    """
    route_id: str
    legs: List[str] = field(default_factory=list)  # ordered truck_ids for the day
    base_hub: str = ""                               # starting hub


# ======================================================================
# Event Types & SimEvent  (mirrors models.py exactly)
# ======================================================================
class EventType(Enum):
    TRUCK_DOCK = auto()              # truck arrives and docks at hub
    TRUCK_DEPARTURE = auto()         # truck departs from hub
    HOLD_DECISION = auto()           # RL agent decision point (replaces HNH_DECISION)
    CARGO_TRANSFER_CHECK = auto()    # check if connecting cargo makes the outbound truck
    EPOCH_START = auto()
    EPOCH_END = auto()


@dataclass
class SimEvent:
    """A single event in the discrete-event queue.

    Identical structure to simulator/models.py SimEvent.
    """
    time: float                     # absolute sim-minutes from t=0
    event_type: EventType
    truck_id: str = ""
    cargo_id: str = ""
    data: Dict = field(default_factory=dict)

    def __lt__(self, other: "SimEvent") -> bool:
        return self.time < other.time
