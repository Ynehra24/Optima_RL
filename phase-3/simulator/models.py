"""
models.py — Dataclasses for the Phase 3 network microsimulator.

Mirrors phase-1/simulator/models.py:
   ScheduledFlight  -> Packet              (the static scheduled entity)
   FlightState      -> PacketState         (its dynamic running state)
   TailPlan         -> FlowState           (the chain of packets in a TCP flow)
   Airport          -> Router              (the node where decisions happen)
   PaxItinerary     -> FragmentGroup       (multi-part deliveries that must
                                            reassemble — the "passengers
                                            waiting at the gate" analog)
   SimEvent / EventType — same shape as Phase 1, just with new event names.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ======================================================================
# Enums
# ======================================================================
class EventType(Enum):
    """Discrete event types for the priority queue.

    Naming mirrors Phase 1 (FLIGHT_DEPARTURE, FLIGHT_ARRIVAL, ...) so the
    EventEngine pattern from Phase 1 carries over with minimal changes.
    """
    PACKET_ARRIVAL = auto()           # packet enters a router queue
    HNH_DECISION = auto()             # agent must choose a hold τ for a packet
    PACKET_FORWARD = auto()           # packet leaves queue, transmission starts
    PACKET_DELIVERED = auto()         # packet arrives at next hop
    PACKET_DROPPED = auto()           # buffer overflow, TTL expiry, or timeout
    REASSEMBLY_CHECK = auto()         # check if all fragments / in-order TCP bytes are present
    TCP_TIMEOUT = auto()              # RTO fires, source retransmits
    EPISODE_END = auto()              # clean shutdown


class PacketStatus(Enum):
    SCHEDULED = auto()                # generated at source, not yet at first router
    QUEUED = auto()                   # in a router buffer awaiting forward
    HELD = auto()                     # buffer + agent has explicitly held it
    IN_FLIGHT = auto()                # currently transmitting on a link
    DELIVERED = auto()                # reached final destination
    DROPPED = auto()                  # lost (buffer overflow / TTL=0 / timeout)


class DropCause(Enum):
    NONE = auto()
    BUFFER_OVERFLOW = auto()
    TTL_EXPIRED = auto()
    TIMEOUT = auto()


class TCPFlagClass(Enum):
    """F_p — TCP connection lifecycle class (state vector dim 19).
    Values match the integer encoding in the Phase 3 state space PDF.
    """
    SYN = 0                 # new connection attempt
    PSH_ACK = 1             # active data
    FIN_RST = 2             # termination
    ACK_ONLY = 3            # control / pure ACK
    NON_TCP = 4             # ICMP, UDP, other


class ProtocolClass(Enum):
    """C_p — protocol class (state vector dim 24)."""
    ICMP = 0
    DNS = 1
    TCP_BULK = 2
    TCP_INTERACTIVE = 3
    UDP_OTHER = 4


# ======================================================================
# Static schedule entities (analog of ScheduledFlight)
# ======================================================================
@dataclass
class Packet:
    """The static, scheduled identity of a packet.

    Created by the generator and immutable thereafter. Per-router
    runtime state (queue wait, hold decisions, current router) lives in
    PacketState.
    """
    packet_id: str
    flow_id: str                              # 5-tuple flow this packet belongs to

    # Path through the network (list of router_ids, in order)
    path: List[str]

    # Source-assigned timestamps (ms since episode start)
    creation_time: float                      # when the source emitted it

    # Header fields (mirror MAWI pcap fields)
    size_bytes: int
    ttl: int                                  # initial IP.ttl at source
    dscp: int                                 # priority class (0-63)
    df_flag: bool                             # Don't Fragment

    # Protocol identification
    protocol_class: ProtocolClass             # C_p — ICMP/TCP-bulk/etc
    tcp_flag_class: TCPFlagClass              # F_p — SYN/PSH+ACK/etc

    # ECN bits (lower 2 bits of IP.tos): 0=not capable, 1/2=ECT, 3=CE
    ecn_bits: int = 0

    # Fragmentation grouping. If part of a fragmented datagram, all
    # fragments share the same `ip_id` and `frag_total`. `frag_offset`
    # is the byte offset of THIS fragment within the original datagram.
    ip_id: int = 0
    frag_offset: int = 0
    frag_total: int = 1                       # 1 = not fragmented

    # TCP-specific (zero/None for non-TCP)
    tcp_seq: int = 0
    tcp_ack: int = 0
    tcp_window: int = 0                       # raw, unnormalised
    src_port: int = 0
    dst_port: int = 0

    # Priority score derived from DSCP at generation time, in [0, 1]
    priority_score: float = 0.0


# ======================================================================
# Runtime per-packet state (analog of FlightState)
# ======================================================================
@dataclass
class PacketState:
    """Dynamic state of one packet as it moves through the network.

    Created when the packet is generated and updated as it traverses
    routers. Each router visit records its own arrival/forward timestamps
    in `hop_history` so the Delay Tree can reconstruct attribution.
    """
    packet: Packet
    status: PacketStatus = PacketStatus.SCHEDULED

    # Current location: index into packet.path. 0 = at first router.
    current_hop_idx: int = 0

    # Current TTL (decrements on each forward, like real IP).
    current_ttl: int = 0                       # initialised in simulator

    # Number of times the source has retransmitted this packet
    retrans_count: int = 0

    # Drop tracking
    drop_cause: DropCause = DropCause.NONE
    drop_router_id: Optional[str] = None
    drop_time: Optional[float] = None

    # Delivery tracking
    delivery_time: Optional[float] = None      # when it reached the final hop
    e2e_delay_ms: float = 0.0                  # cumulative delay budget consumed

    # Per-hop history. One entry per router visited. Each entry is a dict
    # with keys: router_id, arrival_time, queue_wait_ms, hold_ms,
    # forward_time, propagation_ms, hnh_decided (bool).
    # Used by the Delay Tree for reward attribution and by the context
    # engine for Hop_success, E2E_delay, etc.
    hop_history: List[dict] = field(default_factory=list)

    # ----- Per-router state (transient — overwritten on each router) -----
    # The "current decision context". Reset when the packet moves to the
    # next router. Read by the context engine when building the state
    # vector for the agent.
    queue_arrival_time: Optional[float] = None
    queue_wait_ms: float = 0.0                 # Q_delay_p — time waiting at THIS router
    hold_ms: float = 0.0                       # τ chosen by agent at THIS router
    hnh_decided: bool = False                  # has the agent acted on this packet here?
    hnh_action_idx: Optional[int] = None       # which index into hold_actions

    # ----- Convenience properties -----
    @property
    def hops_remaining(self) -> int:
        return max(0, len(self.packet.path) - self.current_hop_idx - 1)

    @property
    def hops_completed(self) -> int:
        # Number of routers this packet has fully traversed (forwarded out).
        return self.current_hop_idx

    @property
    def current_router_id(self) -> Optional[str]:
        if 0 <= self.current_hop_idx < len(self.packet.path):
            return self.packet.path[self.current_hop_idx]
        return None

    @property
    def next_router_id(self) -> Optional[str]:
        nxt = self.current_hop_idx + 1
        if 0 <= nxt < len(self.packet.path):
            return self.packet.path[nxt]
        return None

    @property
    def is_terminal_hop(self) -> bool:
        return self.current_hop_idx >= len(self.packet.path) - 1


# ======================================================================
# Flow (analog of TailPlan: groups packets that share state / dependencies)
# ======================================================================
@dataclass
class FlowState:
    """A 5-tuple TCP flow (or pseudo-flow for UDP/ICMP).

    Tracks aggregate per-flow stats used in the state vector:
    Flow_age, Flow_success, Pkt_inflight, RTT_est, Seq_gap, ACK_gap.

    For TCP flows, the FlowState also tracks expected next sequence
    number to detect out-of-order delivery (Seq_gap).
    """
    flow_id: str                                  # canonical 5-tuple string
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol_class: ProtocolClass

    first_seen_time: Optional[float] = None       # when the first packet of this flow appeared
    packet_ids: List[str] = field(default_factory=list)

    # Per-flow counters (for Flow_success)
    packets_arrived: int = 0
    packets_forwarded: int = 0
    packets_dropped: int = 0
    packets_delivered: int = 0
    packets_inflight: int = 0                     # currently between routers

    # TCP sequence tracking (zero for non-TCP)
    expected_next_seq: int = 0
    max_received_seq: int = 0
    ack_gap: int = 0

    # RTT estimation (rolling)
    rtt_samples_ms: List[float] = field(default_factory=list)
    rtt_est_ms: float = 0.0

    # Last RTO timeout firing time (None if never)
    last_timeout_time: Optional[float] = None


# ======================================================================
# Fragment group (analog of PaxItinerary — the "connecting passengers")
# ======================================================================
@dataclass
class FragmentGroup:
    """All fragments of one fragmented IP datagram.

    The fragment-level analog of PaxItinerary in Phase 1: a group of
    sub-units that must arrive together for the whole "message" to be
    successfully delivered.

    `Msg_complete` in the state vector is computed over these groups.
    """
    ip_id: int
    src_ip: str
    dst_ip: str
    expected_count: int                           # frag_total
    arrived_packet_ids: List[str] = field(default_factory=list)
    completion_time: Optional[float] = None
    abandoned: bool = False                       # true if any sibling was dropped

    @property
    def is_complete(self) -> bool:
        return len(self.arrived_packet_ids) >= self.expected_count

    @property
    def missing_count(self) -> int:
        return max(0, self.expected_count - len(self.arrived_packet_ids))


# ======================================================================
# Router (analog of Airport)
# ======================================================================
@dataclass
class Router:
    """A network node that queues, holds, drops, and forwards packets.

    The router is the *location* where HNH decisions happen — exactly
    like an airport in Phase 1 where flights are held at the gate.
    """
    router_id: str                                # e.g. "H1"
    ip_address: str = ""                          # for cosmetic / logging only
    is_silent: bool = False                       # asterisk hops in traceroute

    buffer_capacity_bytes: int = 256 * 1024
    buffer_used_bytes: int = 0

    # The active queue: ordered list of packet_ids currently buffered here.
    # FIFO order by default; the simulator may re-order when explicitly
    # holding a specific packet.
    queue: List[str] = field(default_factory=list)

    # Per-router rolling counters (for state-vector dims like Q_local_drop,
    # Hold_count, Drain_rate, Local_arr_rate, T_last_fwd, Jitter).
    # The simulator updates these on each event; the context engine reads.
    drops_in_window: List[Tuple[float, str]] = field(default_factory=list)  # (time, packet_id)
    forwards_in_window: List[Tuple[float, str]] = field(default_factory=list)
    arrivals_in_window: List[Tuple[float, str]] = field(default_factory=list)
    queue_length_samples: List[Tuple[float, int]] = field(default_factory=list)
    rtt_samples_in_window: List[Tuple[float, float]] = field(default_factory=list)
    last_forward_time: float = 0.0

    # Active holds — set of packet_ids in `queue` that the agent has
    # explicitly chosen to hold (vs. those merely waiting their turn).
    holds: Dict[str, float] = field(default_factory=dict)        # packet_id -> tau_ms

    # Next time the outgoing link is free for transmission. Used to
    # serialise packets through the link — without this, every packet
    # gets scheduled independently and the queue is meaningless.
    # Updated each time a packet is scheduled to forward.
    next_available_slot: float = 0.0

    # ----- Convenience -----
    @property
    def buffer_utilization(self) -> float:
        if self.buffer_capacity_bytes <= 0:
            return 0.0
        return self.buffer_used_bytes / self.buffer_capacity_bytes

    @property
    def queue_length(self) -> int:
        return len(self.queue)

    @property
    def is_full(self) -> bool:
        return self.buffer_used_bytes >= self.buffer_capacity_bytes


# ======================================================================
# Discrete event (mirrors phase-1/simulator/models.SimEvent)
# ======================================================================
@dataclass(order=True)
class SimEvent:
    time: float
    seq: int = field(default=0, compare=True)         # tiebreaker for stability
    event_type: EventType = field(default=EventType.PACKET_ARRIVAL, compare=False)
    packet_id: Optional[str] = field(default=None, compare=False)
    router_id: Optional[str] = field(default=None, compare=False)
    flow_id: Optional[str] = field(default=None, compare=False)
    extras: dict = field(default_factory=dict, compare=False)
