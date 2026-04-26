"""
Phase 3 — Network HNH Simulator package.

Exports the main entry points for the discrete-event microsimulator.
"""

from simulator.config import SimConfig
from simulator.event_engine import EventEngine
from simulator.generators import (
    PacketGenerator,
    PropagationSampler,
    ProcessingSampler,
    generate_fragment_groups,
    generate_topology,
)
from simulator.models import (
    DropCause,
    EventType,
    FlowState,
    FragmentGroup,
    Packet,
    PacketState,
    PacketStatus,
    ProtocolClass,
    Router,
    SimEvent,
    TCPFlagClass,
)
from simulator.simulator import MetricsTracker, NetworkSimulator

__all__ = [
    "SimConfig",
    "EventEngine",
    "EventType", "SimEvent",
    "Packet", "PacketState", "PacketStatus",
    "Router", "FlowState", "FragmentGroup",
    "ProtocolClass", "TCPFlagClass", "DropCause",
    "PacketGenerator", "PropagationSampler", "ProcessingSampler",
    "generate_topology", "generate_fragment_groups",
    "NetworkSimulator", "MetricsTracker",
]
