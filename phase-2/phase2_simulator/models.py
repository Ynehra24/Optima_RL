"""
Core data models for the logistics cross-docking simulator.

Aviation → Logistics mapping (every class maps 1-to-1):
  Airport         → Hub
  ScheduledFlight → ScheduledTruck
  FlightState     → TruckState    (+ logistics-specific fields from PDF §2)
  PaxItinerary    → CargoUnit     (+ value_score, sla_urgency, is_perishable)
  TailPlan        → TruckPlan
  EventType       → EventType     (events renamed for logistics domain)
  SimEvent        → SimEvent      (identical structure)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional


# ====================================================================
# Hub  (replaces Airport)
# ====================================================================
@dataclass
class Hub:
    """One cross-docking hub in the logistics network.

    Maps to Airport (simulator/models.py):
      hub_id  ← airport_code
      is_main ← is_hub
      min_transfer_time ← mct (minimum connection time)
      num_bays ← NEW (no aviation analog)
    """
    hub_id: str             # e.g. "MAIN0", "SPOKE0023"
    is_main: bool = False   # True = main hub (faster ops, shorter min_transfer_time)
    min_transfer_time: int = 35   # minutes: minimum dwell to complete cargo transfer
    num_bays: int = 20            # total docking bays (no aviation analog)


# ====================================================================
# TruckStatus  (replaces FlightStatus)
# ====================================================================
class TruckStatus(Enum):
    SCHEDULED = auto()   # not yet arrived
    DOCKED = auto()      # arrived at hub, cargo being unloaded
    DEPARTED = auto()    # left the hub
    DELIVERED = auto()   # reached final depot
    CANCELLED = auto()


# ====================================================================
# ScheduledTruck  (replaces ScheduledFlight)
# ====================================================================
@dataclass
class ScheduledTruck:
    """The PLANNED (static) properties of one truck leg.

    Mirrors ScheduledFlight field-for-field.
    New fields: scheduled_dock (no aviation analog), cargo_capacity.

    CSV columns that seed these fields at generation time:
      eta_variation_hours   → influences scheduled_arrival
      loading_unloading_time → influences dock_lead → scheduled_dock
      shipping_costs        → influences cargo_capacity pricing proxy
    """
    truck_id: str            # unique, e.g. "T00042-D3" (route 42, day 3)
    truck_number: str        # repeating identifier e.g. "T00042" (= tail_number)
    origin_hub: str          # hub code
    dest_hub: str            # hub code
    scheduled_dock: float    # sim-minutes: when truck backs into bay at origin hub
    scheduled_departure: float   # sim-minutes: when truck leaves origin hub
    scheduled_arrival: float     # sim-minutes: when truck arrives at dest hub
    route_id: str            # route this leg belongs to (= tail_id)
    leg_index: int           # position in the route chain for this day
    route_type: str = "medium"   # "short" (<2h), "medium" (2-6h), "long" (>6h)
    cargo_capacity: int = 120    # total cargo unit slots on this truck


# ====================================================================
# TruckState  (replaces FlightState)
# ====================================================================
@dataclass
class TruckState:
    """Mutable runtime state a truck accumulates during simulation.

    Delay component breakdown (mirrors FlightState exactly):
      intrinsic_departure_delay   ← intrinsic_departure_delay (same)
      propagated_departure_delay  ← propagated_departure_delay (same)
      hold_delay                  ← hold_delay (same)
      bay_dwell_delay             ← ground_departure_delay (gate congestion)
      road_delay                  ← airtime_delay (en-route variance)
      bay_arrival_delay           ← ground_arrival_delay (taxi-in)

    New logistics-only fields (PDF §2 Local State):
      cargo_value_score       = V_k
      cargo_volume_fraction   = Q_k
      sla_urgency             = X_k
      perishability_fraction  = E_k
      driver_hours_remaining  = L_k
      deadline_pressure       = F_k
      n_inbound_trucks        = N_in
      bay_utilisation_at_decision = BG snapshot at decision time
      bay_queue_wait          = time spent waiting for a free bay (NEW)
    """
    truck: ScheduledTruck

    # ── Core delay components (all in minutes, positive = late) ───────
    intrinsic_departure_delay: float = 0.0    # sampled from log-normal distribution
    propagated_departure_delay: float = 0.0   # cascaded from previous leg on same route
    hold_delay: float = 0.0                   # added by RL agent's hold decision
    bay_dwell_delay: float = 0.0              # G_bay_k: waiting for a free docking bay
    road_delay: float = 0.0                   # G_road_k: en-route traffic / weather
    bay_arrival_delay: float = 0.0            # small dwell delay at destination hub

    # ── Actual times (set by compute_actual_times) ────────────────────
    actual_dock: Optional[float] = None
    actual_departure: Optional[float] = None
    actual_arrival: Optional[float] = None

    status: TruckStatus = TruckStatus.SCHEDULED

    # ── Cargo lists ───────────────────────────────────────────────────
    loaded_cargo: List[str] = field(default_factory=list)
    connecting_inbound_cargo: List[str] = field(default_factory=list)

    # ── Hold decision tracking ────────────────────────────────────────
    hold_decided: bool = False
    hold_action: int = 0

    # ── Bay queue tracking (NEW — no aviation analog) ─────────────────
    bay_queue_wait: float = 0.0    # minutes spent waiting for a free bay at arrival

    # ── PDF §2 Local State fields ─────────────────────────────────────
    # V_k: average value score of connecting inbound cargo ∈ [0,1]
    # CSV: shipping_costs (normalised by day min/max)
    cargo_value_score: float = 0.5

    # Q_k: cargo volume fraction = connecting_units / truck_capacity ∈ [0,1]
    # CSV: warehouse_inventory_level (proxy for fill level)
    cargo_volume_fraction: float = 0.0

    # X_k: SLA urgency {0=standard, 1=next-day, 2=same-day express}
    # CSV: risk_classification → Low Risk=0, Moderate Risk=1, High Risk=2
    sla_urgency: int = 0

    # E_k: perishability fraction = perishable_units / total_connecting_units ∈ [0,1]
    # CSV: iot_temperature (abs < perishable_temp_max) AND cargo_condition_status < 0.2
    perishability_fraction: float = 0.0

    # L_k: driver hours remaining (minutes) — hard cap on hold duration
    # EU HGV regulation: max 4.5h = 270 min before mandatory rest
    # CSV: fatigue_monitoring_score (inverted — high fatigue = low hours remaining)
    driver_hours_remaining: float = 270.0

    # F_k: downstream deadline pressure ∈ [0,1] (0 = very urgent)
    # CSV: delay_probability, lead_time_days (inverse: short lead = high pressure)
    deadline_pressure: float = 1.0

    # N_in: number of distinct inbound trucks feeding cargo into this outbound truck
    # CSV: port_congestion_level (proxy for how many inbound sources)
    n_inbound_trucks: int = 0

    # BG snapshot at the moment of hold decision
    bay_utilisation_at_decision: float = 0.0

    # ── Computed properties (identical logic to FlightState) ──────────

    @property
    def total_departure_delay(self) -> float:
        return (
            max(self.intrinsic_departure_delay, self.propagated_departure_delay)
            + self.hold_delay
            + self.bay_dwell_delay
        )

    @property
    def total_arrival_delay(self) -> float:
        return self.total_departure_delay + self.road_delay + self.bay_arrival_delay

    @property
    def departure_delay_D(self) -> float:
        """D_k: actual departure delay beyond scheduled_departure."""
        if self.actual_departure is not None:
            return self.actual_departure - self.truck.scheduled_departure
        return self.total_departure_delay

    @property
    def arrival_delay_A(self) -> float:
        """A_k: actual arrival delay beyond scheduled_arrival."""
        if self.actual_arrival is not None:
            return self.actual_arrival - self.truck.scheduled_arrival
        return self.total_arrival_delay

    def compute_actual_times(self) -> None:
        """Set actual_dock, actual_departure, actual_arrival from delay components."""
        dep = self.truck.scheduled_departure + self.total_departure_delay
        arr = self.truck.scheduled_arrival + self.total_arrival_delay
        self.actual_departure = dep
        self.actual_arrival = arr
        # Dock happens dock_lead_minutes before actual departure (not scheduled departure)
        self.actual_dock = self.truck.scheduled_dock


# ====================================================================
# CargoUnit  (replaces PaxItinerary)
# ====================================================================
@dataclass
class CargoUnit:
    """A single cargo item (or identical-item batch) with a transfer itinerary.

    Mirrors PaxItinerary field-for-field.
    New logistics fields from PDF §2 (V_k, X_k, E_k):

    CSV columns:
      shipping_costs         → value_score (normalised [0,1])
      risk_classification    → sla_urgency  {Low=0, Moderate=1, High=2}
      iot_temperature        → is_perishable (flag: |temp| < perishable_temp_max)
      cargo_condition_status → is_perishable (secondary: value < 0.2 → flagged)
      historical_demand      → unit_count (scaled)
    """
    cargo_id: str
    unit_count: int = 1             # number of physical parcels in this batch
    legs: List[str] = field(default_factory=list)  # ordered truck_ids
    origin_hub: str = ""
    destination_hub: str = ""

    # ── Runtime state (mirrors PaxItinerary) ─────────────────────────
    current_leg_idx: int = 0
    missed_transfer: bool = False
    rebooked_truck_id: Optional[str] = None
    delay_to_destination: float = 0.0    # total extra delay (minutes)

    # ── Logistics-specific attributes (PDF §2) ────────────────────────
    value_score: float = 0.5   # V_k,i ∈ [0,1]: higher = more valuable cargo
    sla_urgency: int = 0       # X_k: {0,1,2} — drives σ_i(τ) multiplier
    is_perishable: bool = False  # E_k flag: True → exponential disutility


# ====================================================================
# TruckPlan  (replaces TailPlan)
# ====================================================================
@dataclass
class TruckPlan:
    """Daily route chain for one truck (one route_id = one TruckPlan).
    Mirrors TailPlan field-for-field.
    """
    route_id: str
    legs: List[str] = field(default_factory=list)   # ordered truck_ids for the day
    base_hub: str = ""                               # starting hub (= home_base in aviation)


# ====================================================================
# EventType  (replaces EventType in aviation, renamed events)
# ====================================================================
class EventType(Enum):
    TRUCK_DOCK = auto()             # truck arrives and backs into a docking bay at hub
    TRUCK_DEPARTURE = auto()        # truck leaves the hub (loaded or not)
    HOLD_DECISION = auto()          # RL agent decision point (was HNH_DECISION)
    CARGO_TRANSFER_CHECK = auto()   # triggered on dock: can inbound cargo make outbound?
    EPOCH_START = auto()
    EPOCH_END = auto()


# ====================================================================
# SimEvent  (identical structure to simulator/models.py SimEvent)
# ====================================================================
@dataclass
class SimEvent:
    """A discrete event in the priority queue.
    Identical to simulator/models.py SimEvent — no domain-specific content.
    """
    time: float
    event_type: EventType
    truck_id: str = ""
    cargo_id: str = ""
    data: Dict = field(default_factory=dict)

    def __lt__(self, other: "SimEvent") -> bool:
        return self.time < other.time
