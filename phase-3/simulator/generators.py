"""
generators.py — Synthetic generators for topology, packets, and flows.

Calibrated against MAWI 202601011400.pcap statistics:
  - Packet count:        150,998,364
  - Duration:            900.01 s
  - Average rate:        811.52 Mbps  (~167,774 pkt/s)
  - Mean packet size:    604.68 B
  - Protocol breakdown:  ICMP 26.00%, TCP 33.76%, UDP 23.31%, IPSec 8.33%
  - Per-protocol size:   TCP 955.80B, ICMP 66.46B, HTTPS 1623.55B
  - Fragmentation rate:  0.67% (1,009,311 fragmented packets)

For v1 we synthesise packets from these statistics rather than replay
the raw pcap. A future replay-mode generator can ingest the actual
pcap if needed.

Mirrors phase-1/simulator/generators.py (which has DelaySampler,
generate_airports, generate_tail_plans, generate_pax_itineraries):

    generate_topology()         -> Dict[str, Router]            (was generate_airports)
    PacketGenerator             -> generates Packet schedule    (was generate_tail_plans)
    generate_fragment_groups()  -> from emitted Packet stream   (was generate_pax_itineraries)

Also provides DelaySampler equivalents: PropagationSampler (per-hop
propagation jitter) and ProcessingSampler (per-router processing jitter).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator.config import SimConfig, PROTOCOL_MIX_TO_CLASS
from simulator.models import (
    FragmentGroup,
    Packet,
    ProtocolClass,
    Router,
    TCPFlagClass,
)


# ======================================================================
# Topology
# ======================================================================
def generate_topology(cfg: SimConfig) -> Tuple[Dict[str, Router], List[str]]:
    """Build the network graph.

    Returns (routers_by_id, default_path).

    The default_path is a single end-to-end route through all routers
    in order — used by the packet generator as the "path" attribute on
    every emitted packet for the v1 linear topology.

    For the 12-hop trace from the RIPE measurement, H4/H5/H6 are silent
    (asterisks). They still carry traffic; they just don't reply to
    ICMP. We mark them with `is_silent=True` for cosmetic purposes.
    """
    routers: Dict[str, Router] = {}
    silent_hops = {"H4", "H5", "H6"}

    # IP addresses from the actual RIPE Atlas measurement shown in the topology image
    ip_map = {
        "H1": "192.168.1.1",
        "H2": "80.15.225.253",
        "H3": "193.249.214.138",
        "H4": "*",
        "H5": "*",
        "H6": "*",
        "H7": "193.251.240.2",
        "H8": "193.251.255.140",
        "H9": "103.246.249.62",
        "H10": "103.246.249.2",
        "H11": "103.246.249.175",
        "H12": "193.0.14.129",
    }

    for i in range(1, cfg.num_routers + 1):
        rid = f"H{i}"
        routers[rid] = Router(
            router_id=rid,
            ip_address=ip_map.get(rid, ""),
            is_silent=(rid in silent_hops),
            buffer_capacity_bytes=cfg.buffer_capacity_bytes,
        )

    if cfg.linear_topology:
        path = [f"H{i}" for i in range(1, cfg.num_routers + 1)]
    else:
        # Future: branched DAG topology. For now, fall back to linear.
        path = [f"H{i}" for i in range(1, cfg.num_routers + 1)]

    return routers, path


# ======================================================================
# Per-protocol packet size sampling
# ======================================================================
# MAWI per-protocol size means (bytes); std-devs are conservative
# estimates since the dataset doesn't publish them directly.
_SIZE_PARAMS_BY_CLASS: Dict[ProtocolClass, Tuple[float, float]] = {
    ProtocolClass.ICMP:            (66.46,  20.0),
    ProtocolClass.DNS:             (90.0,   30.0),
    ProtocolClass.TCP_BULK:        (955.80, 500.0),
    ProtocolClass.TCP_INTERACTIVE: (300.0,  150.0),
    ProtocolClass.UDP_OTHER:       (500.0,  300.0),
}

# Min/max packet size (Ethernet frame minimum, Internet MTU).
_MIN_PACKET_BYTES = 40
_MAX_PACKET_BYTES = 1500


def _sample_packet_size(klass: ProtocolClass, rng: np.random.Generator) -> int:
    mean, std = _SIZE_PARAMS_BY_CLASS[klass]
    sz = int(rng.normal(mean, std))
    return int(np.clip(sz, _MIN_PACKET_BYTES, _MAX_PACKET_BYTES))


def _sample_protocol_class(cfg: SimConfig, rng: np.random.Generator) -> ProtocolClass:
    """Sample a protocol class according to the configured MAWI mix."""
    probs = np.array(cfg.protocol_mix, dtype=np.float64)
    probs = probs / probs.sum()
    idx = int(rng.choice(len(probs), p=probs))
    return ProtocolClass(PROTOCOL_MIX_TO_CLASS[idx])


def _sample_initial_ttl(cfg: SimConfig, rng: np.random.Generator) -> int:
    ttls, probs = zip(*cfg.ttl_distribution)
    probs = np.array(probs, dtype=np.float64)
    probs = probs / probs.sum()
    return int(rng.choice(ttls, p=probs))


def _sample_dscp(cfg: SimConfig, rng: np.random.Generator) -> int:
    """Most traffic is best-effort (DSCP=0); a small fraction is priority."""
    if rng.random() < cfg.dscp_priority_fraction:
        return int(rng.choice([10, 18, 26, 34, 46]))   # AF11..AF41 + EF
    return cfg.dscp_default


def _dscp_to_priority_score(dscp: int) -> float:
    """Map DSCP class (0-63) to a normalised priority score in [0, 1].

    Phase 3 PDF V_p definition: based on IP.tos DSCP.
    DSCP 0 (best-effort) -> ~0.1, DSCP 46 (EF) -> ~1.0.
    """
    return float(0.1 + 0.9 * min(dscp, 46) / 46.0)


def _tcp_flag_class_for_lifecycle(
    is_syn: bool, is_fin: bool, has_data: bool, is_ack_only: bool
) -> TCPFlagClass:
    if is_syn:
        return TCPFlagClass.SYN
    if is_fin:
        return TCPFlagClass.FIN_RST
    if is_ack_only:
        return TCPFlagClass.ACK_ONLY
    if has_data:
        return TCPFlagClass.PSH_ACK
    return TCPFlagClass.ACK_ONLY


# ======================================================================
# Packet generator
# ======================================================================
class PacketGenerator:
    """Generates the schedule of packets for one episode.

    Roughly analogous to Phase 1's `generate_tail_plans`: produces a list
    of Packet objects with creation times spread across the episode.

    For v1, all packets enter at H1 (the first router in the path) and
    travel the full linear path. Multi-ingress topologies are a TODO.
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator,
                 default_path: List[str]):
        self.cfg = cfg
        self.rng = rng
        self.default_path = default_path
        self._next_packet_seq: int = 0
        self._next_ip_id: int = 1
        self._next_flow_seq: int = 0
        # Re-used flow_ids so multiple packets can belong to the same flow.
        self._flow_pool: List[Tuple[str, str, str, int, int, ProtocolClass]] = []
        # TCP per-flow seq counters
        self._flow_tcp_seq: Dict[str, int] = {}

    # ---------- helpers ----------
    def _new_packet_id(self) -> str:
        pid = f"P{self._next_packet_seq:08d}"
        self._next_packet_seq += 1
        return pid

    def _new_ip_id(self) -> int:
        ip_id = self._next_ip_id
        self._next_ip_id = (self._next_ip_id + 1) % 65536
        return ip_id

    def _get_or_make_flow(
        self, klass: ProtocolClass
    ) -> Tuple[str, str, str, int, int, ProtocolClass]:
        """Reuse existing flows for ~80% of packets so we get realistic
        flow lengths (multiple packets per flow), and create new flows
        for the remaining 20%.
        """
        if self._flow_pool and self.rng.random() < 0.8:
            return self._flow_pool[int(self.rng.integers(0, len(self._flow_pool)))]

        # New flow
        seq = self._next_flow_seq
        self._next_flow_seq += 1
        # Synthetic addressing — just for identification, not realistic
        src_ip = f"10.0.0.{seq % 250 + 1}"
        dst_ip = f"203.0.113.{seq % 250 + 1}"
        sport = int(self.rng.integers(1024, 65535))
        dport_choices = {
            ProtocolClass.TCP_BULK: [80, 443, 21, 8080, 8153],
            ProtocolClass.TCP_INTERACTIVE: [22, 179, 25],
            ProtocolClass.UDP_OTHER: [53, 123, 5004, 5060],
            ProtocolClass.DNS: [53],
            ProtocolClass.ICMP: [0],
        }
        dport = int(self.rng.choice(dport_choices.get(klass, [80])))
        flow_id = f"flow_{seq}_{klass.name}_{sport}_{dport}"

        entry = (flow_id, src_ip, dst_ip, sport, dport, klass)
        if klass in (ProtocolClass.TCP_BULK, ProtocolClass.TCP_INTERACTIVE):
            self._flow_tcp_seq[flow_id] = 0
        self._flow_pool.append(entry)
        return entry

    # ---------- main entry ----------
    def generate(self) -> List[Packet]:
        """Generate packets for one episode.

        Inter-arrival times are exponential (Poisson process) with the
        configured mean rate. Future work: shape with the MAWI burstiness
        envelope (CoV=44.4%).
        """
        cfg = self.cfg
        packets: List[Packet] = []
        t = 0.0
        mean_interarrival_ms = 1.0 / cfg.arrival_rate_pkt_per_ms

        # For v1, cap the number of packets to keep tests/demos fast.
        # Real episode would have ~168K packets per second.
        max_packets = 5000
        emitted = 0

        while t < cfg.episode_duration_ms and emitted < max_packets:
            # Inter-arrival
            dt = self.rng.exponential(mean_interarrival_ms)
            t += dt
            if t >= cfg.episode_duration_ms:
                break

            klass = _sample_protocol_class(cfg, self.rng)
            flow_id, src_ip, dst_ip, sport, dport, _ = self._get_or_make_flow(klass)

            # Decide whether this packet is fragmented.
            is_fragmented = self.rng.random() < cfg.fragment_rate
            if is_fragmented:
                # Emit a small fragment burst sharing the same ip_id
                ip_id = self._new_ip_id()
                frag_total = max(2, int(self.rng.poisson(cfg.fragments_per_datagram_mean)))
                base_size = _sample_packet_size(klass, self.rng)
                # Split base_size across frag_total fragments (rounding).
                frag_size = max(_MIN_PACKET_BYTES, base_size // frag_total)
                for fi in range(frag_total):
                    p = self._make_packet(
                        creation_time=t + fi * 0.001,  # 1 microsecond apart
                        klass=klass, flow_id=flow_id,
                        src_ip=src_ip, dst_ip=dst_ip,
                        sport=sport, dport=dport,
                        size_bytes=frag_size,
                        ip_id=ip_id,
                        frag_offset=fi * frag_size,
                        frag_total=frag_total,
                    )
                    packets.append(p)
                    emitted += 1
                    if emitted >= max_packets:
                        break
            else:
                p = self._make_packet(
                    creation_time=t,
                    klass=klass, flow_id=flow_id,
                    src_ip=src_ip, dst_ip=dst_ip,
                    sport=sport, dport=dport,
                    size_bytes=_sample_packet_size(klass, self.rng),
                    ip_id=self._new_ip_id(),
                    frag_offset=0,
                    frag_total=1,
                )
                packets.append(p)
                emitted += 1

        # Sort by creation time
        packets.sort(key=lambda p: p.creation_time)
        return packets

    def _make_packet(
        self,
        creation_time: float,
        klass: ProtocolClass,
        flow_id: str,
        src_ip: str, dst_ip: str,
        sport: int, dport: int,
        size_bytes: int,
        ip_id: int,
        frag_offset: int,
        frag_total: int,
    ) -> Packet:
        cfg = self.cfg
        ttl = _sample_initial_ttl(cfg, self.rng)
        dscp = _sample_dscp(cfg, self.rng)
        priority = _dscp_to_priority_score(dscp)

        # TCP-specific
        tcp_seq = 0
        tcp_ack = 0
        tcp_window = 0
        tcp_flag_cls = TCPFlagClass.NON_TCP

        if klass in (ProtocolClass.TCP_BULK, ProtocolClass.TCP_INTERACTIVE):
            cur_seq = self._flow_tcp_seq.get(flow_id, 0)
            tcp_seq = cur_seq
            self._flow_tcp_seq[flow_id] = cur_seq + max(1, size_bytes - 40)
            tcp_window = cfg.tcp_default_window
            # Decide lifecycle role: 5% SYN, 1% FIN, the rest are PSH+ACK
            r = self.rng.random()
            if r < 0.05 and cur_seq == 0:
                tcp_flag_cls = TCPFlagClass.SYN
            elif r < 0.06:
                tcp_flag_cls = TCPFlagClass.FIN_RST
            elif size_bytes <= 80:
                tcp_flag_cls = TCPFlagClass.ACK_ONLY
            else:
                tcp_flag_cls = TCPFlagClass.PSH_ACK

        # ECN bits — most are 00 (not capable). 3% have CE flagged (upstream
        # congestion seen). The simulator can also set CE dynamically when a
        # router experiences congestion but chooses not to drop.
        ecn = 0
        if self.rng.random() < 0.03:
            ecn = 3   # CE

        # Don't Fragment: TCP usually sets it (PMTUD), UDP usually doesn't
        df = klass in (ProtocolClass.TCP_BULK, ProtocolClass.TCP_INTERACTIVE)

        return Packet(
            packet_id=self._new_packet_id(),
            flow_id=flow_id,
            path=list(self.default_path),
            creation_time=creation_time,
            size_bytes=size_bytes,
            ttl=ttl,
            dscp=dscp,
            df_flag=df,
            protocol_class=klass,
            tcp_flag_class=tcp_flag_cls,
            ecn_bits=ecn,
            ip_id=ip_id,
            frag_offset=frag_offset,
            frag_total=frag_total,
            tcp_seq=tcp_seq,
            tcp_ack=tcp_ack,
            tcp_window=tcp_window,
            src_port=sport,
            dst_port=dport,
            priority_score=priority,
        )


# ======================================================================
# Fragment groups (analog of generate_pax_itineraries)
# ======================================================================
def generate_fragment_groups(packets: List[Packet]) -> Dict[int, FragmentGroup]:
    """Group fragmented packets by IP.id.

    Returns a dict: ip_id -> FragmentGroup. Only contains entries for
    truly fragmented datagrams (frag_total > 1). The simulator uses
    these to compute Msg_complete and to fire REASSEMBLY_CHECK events.
    """
    groups: Dict[int, FragmentGroup] = {}
    for p in packets:
        if p.frag_total <= 1:
            continue
        if p.ip_id not in groups:
            groups[p.ip_id] = FragmentGroup(
                ip_id=p.ip_id,
                src_ip="",   # filled from first packet
                dst_ip="",
                expected_count=p.frag_total,
            )
    return groups


# ======================================================================
# DelaySampler equivalents (analog of phase-1 DelaySampler)
# ======================================================================
class PropagationSampler:
    """Samples per-link propagation delays.

    Phase 1 has DelaySampler.sample_airtime_delay() — same role here:
    real propagation has small jitter due to physical-layer effects.
    """
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def sample(self) -> float:
        prop = self.rng.normal(
            self.cfg.propagation_delay_mean_ms,
            self.cfg.propagation_delay_stddev_ms,
        )
        return float(max(0.05, prop))   # cap at 50us minimum


class ProcessingSampler:
    """Samples per-router processing delay.

    Smaller and more deterministic than propagation — modern routers
    forward in microseconds.
    """
    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def sample(self) -> float:
        # Tight Gaussian around the configured mean
        proc = self.rng.normal(self.cfg.processing_delay_ms,
                               self.cfg.processing_delay_ms * 0.2)
        return float(max(0.001, proc))
