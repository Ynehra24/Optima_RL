"""
simulator.py — Network microsimulator for the Phase 3 HNH problem.

Discrete-event simulator that models packet flow through a multi-hop
router topology with HNH (hold-or-no-hold) decisions. Mirrors
phase-1/simulator/simulator.AirlineNetworkSimulator:

    Phase 1 entity        Phase 3 entity
    --------------        --------------
    Flight                Packet
    Tail (aircraft)       Flow (5-tuple)
    Airport               Router
    Passenger             (the packet itself; fragments / TCP seq gaps
                           play the "connecting passengers" role)
    HNH per flight        HNH per packet per router

Core mechanism preserved from Phase 1:
  - Holds propagate. Holding packet P at router R delays its arrival at
    R+1, which can trigger fragment-reassembly delays or TCP
    out-of-order events at R+1 — exactly the analog of Phase 1's
    "missed PAX trigger another hold downstream".
  - Dynamic propagated delay is computed at decision time from the
    actual arrival of the predecessor packet, not pre-computed.

API:
    sim = NetworkSimulator(cfg)
    state, info = sim.reset()
    done = False
    while not done:
        action_idx = agent.act(state)
        state, reward, done, info = sim.step(action_idx)
"""

from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


# ======================================================================
# Metrics tracker
# ======================================================================
class MetricsTracker:
    """Accumulates network-level KPIs and validation metrics.

    Direct analog of phase-1/simulator/simulator.MetricsTracker.
    The four "core metrics" exposed in summary() match the Phase 1/2
    style:
        - drop_rate_pct  (analog of misconnect_rate)
        - delivery_rate_pct (analog of OTP)
        - avg_latency_ms (analog of avg arrival delay)
        - avg_hold_ms (analog of avg hold)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_packets = 0
        self.delivered_packets = 0
        self.dropped_packets = 0
        self.dropped_buffer_overflow = 0
        self.dropped_ttl_expired = 0
        self.dropped_timeout = 0
        self.retransmissions = 0

        self.latencies_ms: List[float] = []
        self.hold_decisions_ms: List[float] = []
        self.queue_waits_ms: List[float] = []

        # Fragmentation
        self.total_fragment_groups = 0
        self.completed_fragment_groups = 0
        self.abandoned_fragment_groups = 0

        # Per-hop hop success counter
        self.hop_forwards = 0
        self.hop_drops = 0

        # Reward aggregation
        self.rewards: List[float] = []

    @property
    def delivery_rate(self) -> float:
        finalised = self.delivered_packets + self.dropped_packets
        return self.delivered_packets / finalised if finalised else 1.0

    @property
    def drop_rate(self) -> float:
        finalised = self.delivered_packets + self.dropped_packets
        return self.dropped_packets / finalised if finalised else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def avg_hold_ms(self) -> float:
        return float(np.mean(self.hold_decisions_ms)) if self.hold_decisions_ms else 0.0

    @property
    def fragment_success_rate(self) -> float:
        if self.total_fragment_groups == 0:
            return 1.0
        return self.completed_fragment_groups / self.total_fragment_groups

    def summary(self) -> Dict[str, Any]:
        return {
            "total_packets": self.total_packets,
            "delivered": self.delivered_packets,
            "dropped": self.dropped_packets,
            "drops_by_cause": {
                "buffer_overflow": self.dropped_buffer_overflow,
                "ttl_expired": self.dropped_ttl_expired,
                "timeout": self.dropped_timeout,
            },
            "delivery_rate_pct": round(self.delivery_rate * 100, 2),
            "drop_rate_pct": round(self.drop_rate * 100, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "avg_hold_ms": round(self.avg_hold_ms, 3),
            "retransmissions": self.retransmissions,
            "fragment_groups": self.total_fragment_groups,
            "fragment_success_pct": round(self.fragment_success_rate * 100, 2),
            "hnh_decisions": len(self.hold_decisions_ms),
            "hop_forwards": self.hop_forwards,
            "hop_drops": self.hop_drops,
        }


# ======================================================================
# Main simulator
# ======================================================================
class NetworkSimulator:
    """Discrete-event microsimulator for Phase 3 (Network HNH).

    Usage:
        sim = NetworkSimulator(cfg)
        state, info = sim.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = sim.step(action)
    """

    def __init__(self, cfg: Optional[SimConfig] = None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.random_seed)

        self.event_engine = EventEngine()
        self.prop_sampler = PropagationSampler(self.cfg, self.rng)
        self.proc_sampler = ProcessingSampler(self.cfg, self.rng)
        self.metrics = MetricsTracker()

        # Topology
        self.routers: Dict[str, Router] = {}
        self.default_path: List[str] = []

        # Schedule
        self.packets: Dict[str, Packet] = {}                  # static
        self.packet_states: Dict[str, PacketState] = {}       # dynamic
        self.flows: Dict[str, FlowState] = {}
        self.fragment_groups: Dict[int, FragmentGroup] = {}

        # Indexes
        self._packets_by_flow: Dict[str, List[str]] = defaultdict(list)
        self._packets_by_ip_id: Dict[int, List[str]] = defaultdict(list)
        self._next_in_flow: Dict[str, Optional[str]] = {}
        self._prev_in_flow: Dict[str, Optional[str]] = {}

        # Pending HNH decision state — set when run_until_hnh surfaces a
        # decision and consumed by step(). Mirrors Phase 1 pattern.
        self._pending_hnh_packet: Optional[str] = None
        self._pending_hnh_router: Optional[str] = None
        self._pending_hnh_event_time: float = 0.0
        self._done: bool = True

    # ==================================================================
    # Gym-like API
    # ==================================================================
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the simulator and run forward to the first HNH decision.

        Returns (state_dict, info_dict). The state_dict is the raw
        per-packet/per-router state needed by the context engine to
        compute the 65-dim state vector. The context engine (Yatharth's
        module) consumes this and produces the actual numpy array for
        the agent.

        For now we return the raw dict and let downstream consumers
        (context engine, A2C trainer) shape it as needed.
        """
        if seed is not None:
            self.cfg.random_seed = seed
        self.rng = np.random.default_rng(self.cfg.random_seed)

        # Re-build helpers with the new RNG so re-seeding is honored.
        self.prop_sampler = PropagationSampler(self.cfg, self.rng)
        self.proc_sampler = ProcessingSampler(self.cfg, self.rng)

        # Clear all data
        self.event_engine.clear()
        self.metrics.reset()
        self.packets.clear()
        self.packet_states.clear()
        self.flows.clear()
        self.fragment_groups.clear()
        self._packets_by_flow.clear()
        self._packets_by_ip_id.clear()
        self._next_in_flow.clear()
        self._prev_in_flow.clear()

        # 1. Build topology
        self.routers, self.default_path = generate_topology(self.cfg)

        # 2. Generate the packet schedule
        gen = PacketGenerator(self.cfg, self.rng, self.default_path)
        all_packets = gen.generate()
        for p in all_packets:
            self.packets[p.packet_id] = p
            self._packets_by_flow[p.flow_id].append(p.packet_id)
            if p.frag_total > 1:
                self._packets_by_ip_id[p.ip_id].append(p.packet_id)

            # Initialise per-packet runtime state
            ps = PacketState(packet=p)
            ps.current_ttl = p.ttl
            self.packet_states[p.packet_id] = ps

            # Build / update FlowState
            if p.flow_id not in self.flows:
                self.flows[p.flow_id] = FlowState(
                    flow_id=p.flow_id,
                    src_ip="",  # synthetic; not used in v1
                    dst_ip="",
                    src_port=p.src_port,
                    dst_port=p.dst_port,
                    protocol_class=p.protocol_class,
                    first_seen_time=p.creation_time,
                )
            self.flows[p.flow_id].packet_ids.append(p.packet_id)

        # 3. Build flow-level prev/next links (in creation-time order)
        for fid, pids in self._packets_by_flow.items():
            pids.sort(key=lambda pid: self.packets[pid].creation_time)
            for i, pid in enumerate(pids):
                self._prev_in_flow[pid] = pids[i - 1] if i > 0 else None
                self._next_in_flow[pid] = pids[i + 1] if i < len(pids) - 1 else None

        # 4. Build fragment groups
        self.fragment_groups = generate_fragment_groups(all_packets)
        self.metrics.total_fragment_groups = len(self.fragment_groups)
        self.metrics.total_packets = len(all_packets)

        # 5. Register handlers
        self._register_handlers()

        # 6. Schedule initial PACKET_ARRIVAL events at the source router
        for pid, p in self.packets.items():
            self.event_engine.schedule(SimEvent(
                time=p.creation_time,
                event_type=EventType.PACKET_ARRIVAL,
                packet_id=pid,
                router_id=p.path[0] if p.path else None,
            ))

        # 7. Schedule the EPISODE_END marker
        self.event_engine.schedule(SimEvent(
            time=self.cfg.episode_duration_ms,
            event_type=EventType.EPISODE_END,
        ))

        self._done = False
        return self._advance_to_next_hnh()

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Apply the agent's chosen hold action and advance to the next HNH."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start again.")
        if self._pending_hnh_packet is None:
            raise RuntimeError("No pending HNH decision. Did you call reset()?")

        # Bound the action index
        action_idx = int(np.clip(action_idx, 0, len(self.cfg.hold_actions) - 1))
        hold_ms = float(self.cfg.hold_actions[action_idx])

        pid = self._pending_hnh_packet
        rid = self._pending_hnh_router
        ps = self.packet_states[pid]
        router = self.routers[rid]

        # Enforce TTL / slack hard cap. If holding would exceed slack,
        # silently clip to feasible range. (The state vector also
        # exposes delta_slack so the agent can learn this.)
        slack_ms = self._compute_slack_ms(ps)
        if hold_ms > slack_ms:
            hold_ms = max(0.0, slack_ms)

        # Apply the hold decision.
        ps.hold_ms = hold_ms
        ps.hnh_decided = True
        ps.hnh_action_idx = action_idx
        if hold_ms > 0:
            router.holds[pid] = hold_ms

        self.metrics.hold_decisions_ms.append(hold_ms)

        # Compute reward placeholder (the real reward calculator is the
        # responsibility of the reward-engineering module; we expose a
        # simple proxy here so the API still works during development).
        reward = self._compute_proxy_reward(ps, hold_ms, router)
        self.metrics.rewards.append(reward)

        # Schedule PACKET_FORWARD: serialised through the outgoing link.
        # The earliest the packet can begin transmission is the later
        # of (a) when the link frees up from previous traffic, (b) when
        # the hold expires.
        tx_time = self._transmission_time_ms(ps.packet, router)
        forward_start = max(
            self._pending_hnh_event_time + hold_ms,
            router.next_available_slot,
        )
        forward_end = forward_start + tx_time
        router.next_available_slot = forward_end
        self.event_engine.schedule(SimEvent(
            time=forward_end,
            event_type=EventType.PACKET_FORWARD,
            packet_id=pid,
            router_id=rid,
        ))

        # Clear pending decision
        self._pending_hnh_packet = None
        self._pending_hnh_router = None

        # Advance to next HNH (or end of episode)
        next_state, info = self._advance_to_next_hnh()
        return next_state, reward, self._done, info

    # ==================================================================
    # Event handlers
    # ==================================================================
    def _register_handlers(self) -> None:
        ee = self.event_engine
        ee.register_handler(EventType.PACKET_ARRIVAL, self._handle_arrival)
        ee.register_handler(EventType.PACKET_FORWARD, self._handle_forward)
        ee.register_handler(EventType.PACKET_DELIVERED, self._handle_delivered)
        ee.register_handler(EventType.PACKET_DROPPED, self._handle_dropped)
        ee.register_handler(EventType.REASSEMBLY_CHECK, self._handle_reassembly)
        ee.register_handler(EventType.EPISODE_END, self._handle_episode_end)

    def _handle_arrival(self, event: SimEvent) -> Optional[List[SimEvent]]:
        """Packet enters a router's queue.

        Two outcomes:
          1. Buffer overflow -> immediate PACKET_DROPPED.
          2. Buffer has room -> enqueue, then either an HNH_DECISION
             event (if conditions trigger one) or just a PACKET_FORWARD
             event scheduled at the natural drain time.
        """
        pid = event.packet_id
        rid = event.router_id
        ps = self.packet_states.get(pid)
        router = self.routers.get(rid)
        if ps is None or router is None:
            return None

        # TTL check first — packets with TTL=0 must be dropped.
        if ps.current_ttl <= 0:
            return [SimEvent(
                time=event.time,
                event_type=EventType.PACKET_DROPPED,
                packet_id=pid,
                router_id=rid,
                extras={"cause": "ttl_expired"},
            )]

        # Buffer overflow check
        if router.buffer_used_bytes + ps.packet.size_bytes > router.buffer_capacity_bytes:
            return [SimEvent(
                time=event.time,
                event_type=EventType.PACKET_DROPPED,
                packet_id=pid,
                router_id=rid,
                extras={"cause": "buffer_overflow"},
            )]

        # Enqueue
        router.queue.append(pid)
        router.buffer_used_bytes += ps.packet.size_bytes
        router.arrivals_in_window.append((event.time, pid))
        ps.status = PacketStatus.QUEUED
        ps.queue_arrival_time = event.time

        # Update flow stats
        flow = self.flows.get(ps.packet.flow_id)
        if flow is not None:
            flow.packets_arrived += 1

        # Append to hop history (we'll fill in the rest at forward time)
        ps.hop_history.append({
            "router_id": rid,
            "arrival_time": event.time,
            "queue_wait_ms": 0.0,
            "hold_ms": 0.0,
            "forward_time": None,
            "propagation_ms": None,
            "hnh_decided": False,
        })

        # Decide whether to trigger an HNH event
        if self._should_trigger_hnh(ps, router):
            return [SimEvent(
                time=event.time,
                event_type=EventType.HNH_DECISION,
                packet_id=pid,
                router_id=rid,
            )]

        # Otherwise just schedule the forward at the natural service time.
        # Service time = max(now, link_free_time) + transmission_time.
        # This serialises packets through the outgoing link, so when
        # arrivals burst in faster than the link can clock them out, the
        # queue genuinely fills (and may overflow on subsequent arrivals).
        tx_time = self._transmission_time_ms(ps.packet, router)
        forward_start = max(event.time, router.next_available_slot)
        forward_end = forward_start + tx_time
        router.next_available_slot = forward_end
        return [SimEvent(
            time=forward_end,
            event_type=EventType.PACKET_FORWARD,
            packet_id=pid,
            router_id=rid,
        )]

    def _handle_forward(self, event: SimEvent) -> Optional[List[SimEvent]]:
        """Packet leaves the router and starts traversing the link to next hop."""
        pid = event.packet_id
        rid = event.router_id
        ps = self.packet_states.get(pid)
        router = self.routers.get(rid)
        if ps is None or router is None:
            return None
        if ps.status == PacketStatus.DROPPED or ps.status == PacketStatus.DELIVERED:
            return None

        # Remove from queue
        if pid in router.queue:
            router.queue.remove(pid)
        router.buffer_used_bytes = max(0, router.buffer_used_bytes - ps.packet.size_bytes)
        router.holds.pop(pid, None)
        router.last_forward_time = event.time
        router.forwards_in_window.append((event.time, pid))

        ps.status = PacketStatus.IN_FLIGHT
        ps.queue_wait_ms = max(0.0, event.time - (ps.queue_arrival_time or event.time))
        self.metrics.queue_waits_ms.append(ps.queue_wait_ms)
        self.metrics.hop_forwards += 1

        # Fill in the most recent hop history entry
        if ps.hop_history:
            last = ps.hop_history[-1]
            last["queue_wait_ms"] = ps.queue_wait_ms
            last["hold_ms"] = ps.hold_ms
            last["forward_time"] = event.time
            last["hnh_decided"] = ps.hnh_decided

        # Update flow stats
        flow = self.flows.get(ps.packet.flow_id)
        if flow is not None:
            flow.packets_forwarded += 1
            flow.packets_inflight += 1

        # Decrement TTL
        ps.current_ttl -= 1

        # Decide next: are we at the last hop?
        if ps.is_terminal_hop:
            # Terminal hop = packet is delivered to its final destination
            # (the destination host itself, not another router).
            propagation = self.prop_sampler.sample()
            return [SimEvent(
                time=event.time + propagation,
                event_type=EventType.PACKET_DELIVERED,
                packet_id=pid,
                router_id=rid,
            )]

        # Otherwise propagate to next hop
        propagation = self.prop_sampler.sample()
        if ps.hop_history:
            ps.hop_history[-1]["propagation_ms"] = propagation

        next_router_id = ps.next_router_id
        ps.current_hop_idx += 1
        # Reset transient per-router state for the new router
        ps.hold_ms = 0.0
        ps.hnh_decided = False
        ps.hnh_action_idx = None
        ps.queue_arrival_time = None

        return [SimEvent(
            time=event.time + propagation,
            event_type=EventType.PACKET_ARRIVAL,
            packet_id=pid,
            router_id=next_router_id,
        )]

    def _handle_delivered(self, event: SimEvent) -> Optional[List[SimEvent]]:
        pid = event.packet_id
        ps = self.packet_states.get(pid)
        if ps is None:
            return None
        if ps.status == PacketStatus.DELIVERED:
            return None

        ps.status = PacketStatus.DELIVERED
        ps.delivery_time = event.time
        ps.e2e_delay_ms = event.time - ps.packet.creation_time

        self.metrics.delivered_packets += 1
        self.metrics.latencies_ms.append(ps.e2e_delay_ms)

        # Update flow
        flow = self.flows.get(ps.packet.flow_id)
        if flow is not None:
            flow.packets_delivered += 1
            flow.packets_inflight = max(0, flow.packets_inflight - 1)

        # Trigger reassembly check for fragmented datagrams
        if ps.packet.frag_total > 1:
            return [SimEvent(
                time=event.time + 0.001,  # 1 microsecond later
                event_type=EventType.REASSEMBLY_CHECK,
                packet_id=pid,
                extras={"ip_id": ps.packet.ip_id},
            )]
        return None

    def _handle_dropped(self, event: SimEvent) -> Optional[List[SimEvent]]:
        pid = event.packet_id
        rid = event.router_id
        ps = self.packet_states.get(pid)
        router = self.routers.get(rid) if rid else None
        if ps is None or ps.status == PacketStatus.DROPPED:
            return None

        cause_str = event.extras.get("cause", "buffer_overflow")
        cause = {
            "buffer_overflow": DropCause.BUFFER_OVERFLOW,
            "ttl_expired": DropCause.TTL_EXPIRED,
            "timeout": DropCause.TIMEOUT,
        }.get(cause_str, DropCause.BUFFER_OVERFLOW)

        ps.status = PacketStatus.DROPPED
        ps.drop_cause = cause
        ps.drop_router_id = rid
        ps.drop_time = event.time

        if router is not None:
            # If the packet was already enqueued, free its bytes
            if pid in router.queue:
                router.queue.remove(pid)
                router.buffer_used_bytes = max(0,
                    router.buffer_used_bytes - ps.packet.size_bytes)
            router.holds.pop(pid, None)
            router.drops_in_window.append((event.time, pid))

        self.metrics.dropped_packets += 1
        self.metrics.hop_drops += 1
        if cause == DropCause.BUFFER_OVERFLOW:
            self.metrics.dropped_buffer_overflow += 1
        elif cause == DropCause.TTL_EXPIRED:
            self.metrics.dropped_ttl_expired += 1
        elif cause == DropCause.TIMEOUT:
            self.metrics.dropped_timeout += 1

        # Mark fragment group as abandoned if applicable
        if ps.packet.frag_total > 1:
            grp = self.fragment_groups.get(ps.packet.ip_id)
            if grp is not None and not grp.is_complete:
                grp.abandoned = True

        # Update flow
        flow = self.flows.get(ps.packet.flow_id)
        if flow is not None:
            flow.packets_dropped += 1

        return None

    def _handle_reassembly(self, event: SimEvent) -> Optional[List[SimEvent]]:
        """Check if all fragments of a datagram have arrived.

        Direct analog of phase-1 _handle_pax_connection: when a fragment
        is delivered, see whether its siblings are all present. If yes,
        the message is complete (= "PAX made the connection"). If a
        sibling has been dropped, the entire datagram is abandoned
        (= "missed connection").
        """
        ip_id = event.extras.get("ip_id")
        grp = self.fragment_groups.get(ip_id)
        if grp is None or grp.completion_time is not None:
            return None

        pid = event.packet_id
        if pid is not None and pid not in grp.arrived_packet_ids:
            grp.arrived_packet_ids.append(pid)

        if grp.is_complete and not grp.abandoned:
            grp.completion_time = event.time
            self.metrics.completed_fragment_groups += 1
        elif grp.abandoned:
            self.metrics.abandoned_fragment_groups += 1

        return None

    def _handle_episode_end(self, event: SimEvent) -> Optional[List[SimEvent]]:
        # Marker only — actual termination happens when run_until_hnh
        # exhausts its queue.
        return None

    # ==================================================================
    # Decision triggering
    # ==================================================================
    def _should_trigger_hnh(self, ps: PacketState, router: Router) -> bool:
        """Decide whether this packet's arrival should pause for an HNH decision.

        At ~168K pkt/s a per-packet decision is intractable, so we sub-sample.
        Triggers (any of):
          - decide_every_packet flag is on
          - router buffer utilisation exceeds threshold
          - this packet has waiting predecessors (N_in > 0)
          - TTL is critically low (would otherwise expire soon)
        """
        cfg = self.cfg
        if cfg.decide_every_packet:
            return True

        if router.buffer_utilization >= cfg.decision_trigger_buf_util:
            return True

        if cfg.decision_trigger_on_predecessors:
            if self._count_waiting_predecessors(ps) > 0:
                return True

        if ps.current_ttl <= cfg.decision_trigger_ttl_threshold:
            return True

        return False

    def _count_waiting_predecessors(self, ps: PacketState) -> int:
        """N_in: count of dependent predecessors not yet at this router.

        Two kinds of predecessor:
          1. Sibling fragments: same ip_id, not yet arrived at this router.
          2. TCP in-sequence predecessors: prior packets in same flow with
             lower seq number that haven't arrived/forwarded yet.
        """
        count = 0
        rid = ps.current_router_id
        if rid is None:
            return 0

        # Fragment siblings
        if ps.packet.frag_total > 1:
            siblings = self._packets_by_ip_id.get(ps.packet.ip_id, [])
            for sid in siblings:
                if sid == ps.packet.packet_id:
                    continue
                sps = self.packet_states.get(sid)
                if sps is None:
                    continue
                # Predecessor "not yet arrived" at this router if it's
                # still upstream (current_hop_idx < ps.current_hop_idx)
                # or yet to start (status == SCHEDULED).
                if sps.status == PacketStatus.DROPPED:
                    continue
                if sps.status == PacketStatus.DELIVERED:
                    continue
                if sps.current_hop_idx < ps.current_hop_idx:
                    count += 1

        # TCP in-sequence predecessors (within same flow)
        if ps.packet.protocol_class in (ProtocolClass.TCP_BULK, ProtocolClass.TCP_INTERACTIVE):
            prev_pid = self._prev_in_flow.get(ps.packet.packet_id)
            if prev_pid is not None:
                prev_ps = self.packet_states.get(prev_pid)
                if prev_ps is not None and prev_ps.status not in (
                    PacketStatus.DROPPED, PacketStatus.DELIVERED
                ):
                    if prev_ps.current_hop_idx < ps.current_hop_idx:
                        count += 1

        return count

    # ==================================================================
    # Helpers — link timing & feasibility
    # ==================================================================
    def _transmission_time_ms(self, packet: Packet, router: Router) -> float:
        """Time to clock the packet onto the outgoing link, in ms.

        At 1 Gbps a 1500-byte packet takes 12 microseconds. This is
        deterministic given the packet size — no jitter.
        """
        bits = packet.size_bytes * 8
        link_bps = self.cfg.link_bandwidth_mbps * 1_000_000
        if link_bps <= 0:
            return 0.0
        seconds = bits / link_bps
        return seconds * 1000.0 + self.cfg.processing_delay_ms

    def _compute_slack_ms(self, ps: PacketState) -> float:
        """delta_slack = TTL headroom beyond minimum hops needed (in ms-equiv).

        TTL is a hop count, so we convert to a time budget by multiplying
        by the typical per-hop delay (propagation + processing). This
        gives the agent a meaningful upper bound on how long it can hold
        without risking TTL expiry.
        """
        hops_remaining = ps.hops_remaining
        ttl_headroom = ps.current_ttl - hops_remaining
        if ttl_headroom <= 0:
            return 0.0
        # Conservative per-hop budget (use 2x the mean to be safe)
        per_hop_ms = 2.0 * (self.cfg.propagation_delay_mean_ms + self.cfg.processing_delay_ms)
        return float(ttl_headroom * per_hop_ms)

    # ==================================================================
    # Reward (placeholder — real reward calc lives in rewardEngineering/)
    # ==================================================================
    def _compute_proxy_reward(
        self, ps: PacketState, hold_ms: float, router: Router
    ) -> float:
        """Simple proxy reward for development / testing.

        The real reward calculator (Phase 3 reward engineering, mirroring
        Phase 1's RewardCalculator and Phase 2's LogisticsRewardCalculator)
        will replace this. For now we use:

            R = 1.0 if delivery prospects look good after the hold
              - 0.01 * hold_ms                  (cost of holding)
              - 1.0 if buffer is critically full (about to drop something)
        """
        cfg = self.cfg
        # Local delivery prospect: a hold helps if there are predecessors.
        n_pred = self._count_waiting_predecessors(ps)
        delivery_term = 1.0 if (n_pred > 0 and hold_ms > 0) else 0.5

        hold_cost = 0.01 * hold_ms

        congestion_penalty = 0.0
        if router.buffer_utilization > cfg.BG_threshold:
            congestion_penalty = cfg.lambda_congestion * (
                router.buffer_utilization - cfg.BG_threshold
            ) * (hold_ms / cfg.delta_R_ms)

        reward = (
            cfg.alpha * delivery_term
            + (1 - cfg.alpha) * (1.0 - hold_cost)
            - congestion_penalty
        )
        return float(reward)

    # ==================================================================
    # Loop control
    # ==================================================================
    def _advance_to_next_hnh(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run the event loop until the next HNH_DECISION or queue exhausts."""
        hnh_event = self.event_engine.run_until_hnh()
        if hnh_event is None:
            # Queue exhausted — episode ends. Drain any remaining events
            # so final metrics are accurate.
            self.event_engine.drain()
            self._done = True
            return self._build_state_dict(None), self.metrics.summary()

        if hnh_event.event_type == EventType.EPISODE_END:
            self._done = True
            return self._build_state_dict(None), self.metrics.summary()

        # We have a real HNH decision pending
        self._pending_hnh_packet = hnh_event.packet_id
        self._pending_hnh_router = hnh_event.router_id
        self._pending_hnh_event_time = hnh_event.time

        ps = self.packet_states[hnh_event.packet_id]
        router = self.routers[hnh_event.router_id]
        info = {
            "packet_id": hnh_event.packet_id,
            "router_id": hnh_event.router_id,
            "current_hop_idx": ps.current_hop_idx,
            "hops_remaining": ps.hops_remaining,
            "current_ttl": ps.current_ttl,
            "size_bytes": ps.packet.size_bytes,
            "protocol_class": ps.packet.protocol_class.name,
            "tcp_flag_class": ps.packet.tcp_flag_class.name,
            "buffer_utilization": router.buffer_utilization,
            "queue_length": router.queue_length,
            "n_predecessors": self._count_waiting_predecessors(ps),
            "slack_ms": self._compute_slack_ms(ps),
            "event_time": hnh_event.time,
        }
        return self._build_state_dict(hnh_event), info

    def _build_state_dict(self, hnh_event: Optional[SimEvent]) -> Dict[str, Any]:
        """Return raw state for the context engine to consume.

        The context engine (Yatharth's module) will turn this into the
        65-dim state vector. We expose pointers, not pre-computed
        features, so the context engine has full access to whatever it
        needs.
        """
        if hnh_event is None:
            return {"done": True}

        return {
            "done": False,
            "packet_id": hnh_event.packet_id,
            "router_id": hnh_event.router_id,
            "event_time": hnh_event.time,
            "packet_state_ref": self.packet_states[hnh_event.packet_id],
            "router_ref": self.routers[hnh_event.router_id],
            # Full context handles available for the context engine:
            "all_routers": self.routers,
            "all_packet_states": self.packet_states,
            "all_flows": self.flows,
            "fragment_groups": self.fragment_groups,
            "default_path": self.default_path,
            "hold_actions": self.cfg.hold_actions,
        }

    # ==================================================================
    # Baseline policies (for sanity checks / paper comparison)
    # ==================================================================
    def run_episode(
        self,
        policy: str = "no_hold",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a complete episode under a fixed baseline policy.

        Mirrors phase-1 run_episode. Available policies:
          'no_hold':  always τ=0 (always pick action index 0)
          'random':   uniform over the 7 actions
          'heuristic': hold the smallest τ that gives waiting predecessors
                      time to catch up (clipped to slack)
        """
        state, _info = self.reset(seed=seed)
        done = state.get("done", False)
        while not done:
            action = self._select_baseline_action(policy, state)
            state, _reward, done, _info = self.step(action)
        return self.metrics.summary()

    def _select_baseline_action(self, policy: str, state: Dict[str, Any]) -> int:
        if state.get("done", False):
            return 0
        if policy == "no_hold":
            return 0
        if policy == "random":
            return int(self.rng.integers(0, len(self.cfg.hold_actions)))
        if policy == "heuristic":
            ps: PacketState = state["packet_state_ref"]
            n_pred = self._count_waiting_predecessors(ps)
            if n_pred == 0:
                return 0
            # Hold the smallest non-zero action that fits the slack budget.
            slack = self._compute_slack_ms(ps)
            for idx, tau in enumerate(self.cfg.hold_actions):
                if 0 < tau <= slack:
                    return idx
            return 0
        return 0

    # ==================================================================
    # Read-only accessors (used by reward engine / delay tree)
    # ==================================================================
    def get_packet_state(self, packet_id: str) -> Optional[PacketState]:
        return self.packet_states.get(packet_id)

    def get_router(self, router_id: str) -> Optional[Router]:
        return self.routers.get(router_id)

    def get_flow(self, flow_id: str) -> Optional[FlowState]:
        return self.flows.get(flow_id)

    def get_flow_packets(self, flow_id: str) -> List[str]:
        return list(self._packets_by_flow.get(flow_id, []))

    def get_fragment_group(self, ip_id: int) -> Optional[FragmentGroup]:
        return self.fragment_groups.get(ip_id)

    def get_previous_in_flow(self, packet_id: str) -> Optional[str]:
        return self._prev_in_flow.get(packet_id)

    def get_next_in_flow(self, packet_id: str) -> Optional[str]:
        return self._next_in_flow.get(packet_id)
