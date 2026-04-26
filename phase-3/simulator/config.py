"""
config.py — All simulation hyperparameters for Phase 3 (Network HNH).

Mirrors phase-1/simulator/config.py and phase-2/simulator/config.py, but
with knobs calibrated to network timescales (milliseconds, not minutes)
and dataset values from MAWI 202601011400.pcap and RIPE Atlas
measurements (msm_id 5001, 1001).

All time values are in MILLISECONDS unless noted. All sizes are in BYTES.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SimConfig:
    # -----------------------------------------------------------------
    # Episode / clock
    # -----------------------------------------------------------------
    # An episode is one simulated "second" of network traffic. At MAWI's
    # 167,774 pkt/s baseline this means ~167K packets per episode if every
    # packet were tracked, but in practice we sub-sample (see
    # `decision_trigger_*` knobs below) so the agent only acts on a
    # tractable fraction.
    episode_duration_ms: float = 1000.0          # 1 simulated second
    random_seed: int = 42

    # -----------------------------------------------------------------
    # Topology — default is the 12-hop linear chain from the RIPE trace
    # -----------------------------------------------------------------
    num_routers: int = 12
    # If True, build a single linear path H1 -> H2 -> ... -> H_num_routers.
    # If False, the simulator expects an explicit topology dict (DAG).
    linear_topology: bool = True

    # Per-router buffer capacity in BYTES. Real Cisco/Juniper routers
    # typically size buffers at ~50ms x line-rate. At 1Gbps that's
    # ~6.25 MB. We use a much smaller value so congestion happens
    # frequently enough to learn from in a 1-second episode. This is
    # the network analog of the "small hub" Phase-2 calibration —
    # synthetic stress so the agent sees decisions that matter.
    # Tuned so baseline `no_hold` lands in the 3-15% drop range, which
    # mirrors Phase 1's ~3-5.5% baseline missed-connection rate.
    buffer_capacity_bytes: int = 128 * 1024      # 128 KB per router

    # Router processing delay (fixed, hardware-bound). Excluded from the
    # state vector per the PDF (it's a constant, not an observable),
    # but the simulator still adds it to per-hop latency.
    processing_delay_ms: float = 0.05

    # -----------------------------------------------------------------
    # Link characteristics — calibrated to MAWI backbone trace
    # -----------------------------------------------------------------
    # MAWI 202601011400 reports 811.52 Mbps avg. We round up to 1 Gbps
    # link capacity, which means transmission time for a 1500-byte
    # packet is ~12 microseconds.
    link_bandwidth_mbps: float = 1000.0
    # Per-hop propagation delay (ms). RIPE traceroute shows hop-1 RTT
    # ~0.58 ms, hop-2 ~1.45 ms, hop-3 ~3.32 ms — so per-hop one-way
    # propagation is ~0.5-1.5 ms. We use 1.0 ms with some jitter.
    propagation_delay_mean_ms: float = 1.0
    propagation_delay_stddev_ms: float = 0.3

    # -----------------------------------------------------------------
    # Packet generation — MAWI calibration
    # -----------------------------------------------------------------
    # Mean packet size (bytes). MAWI reports 604.68 bytes overall;
    # per-protocol means are TCP=955.80, ICMP=66.46, HTTPS=1623.55.
    # The generator samples per-protocol sizes; this is a fallback.
    packet_size_mean_bytes: float = 604.68
    packet_size_stddev_bytes: float = 400.0

    # Protocol mix from MAWI (rounded to sum to 1.0).
    # Order: TCP-bulk, TCP-interactive, UDP, ICMP, IPSec/other.
    # Mapped onto the Phase 3 C_p classes (0=ICMP, 1=DNS, 2=TCP-bulk,
    # 3=TCP-interactive, 4=UDP/other) via `protocol_class_map`.
    protocol_mix: Tuple[float, ...] = (
        0.30,    # TCP-bulk (HTTP/HTTPS/FTP)  -> C_p=2
        0.04,    # TCP-interactive (SSH/BGP)  -> C_p=3
        0.23,    # UDP/other                  -> C_p=4
        0.26,    # ICMP                       -> C_p=0
        0.17,    # other (IPSec, DNS, ...)    -> C_p=1 or 4
    )

    # DSCP distribution. Most internet traffic is best-effort (DSCP=0).
    # A small fraction is EF (DSCP=46, voice) or AF (DSCP=10/18/26/34).
    dscp_default: int = 0
    dscp_priority_fraction: float = 0.05         # 5% of packets are priority

    # TTL initial distribution. Common defaults are 64 (Linux), 128
    # (Windows), 255 (some routers). RIPE sample showed ttl=253.
    ttl_default: int = 64
    ttl_distribution: Tuple[Tuple[int, float], ...] = (
        (64, 0.55),
        (128, 0.30),
        (255, 0.15),
    )

    # Fragmentation rate from MAWI (~0.67% of packets fragmented).
    fragment_rate: float = 0.0067
    fragments_per_datagram_mean: int = 3         # avg fragments when fragmenting

    # Packet arrival rate (packets per ms). MAWI's full backbone trace
    # baseline is ~168 pkt/ms across a 1 Gbps link — that's effectively
    # 100% line-rate, which makes every queue spike turn into a flood
    # of drops. For a learnable simulator we run well below saturation
    # by default, which still produces meaningful queue dynamics and
    # occasional drops while leaving slack the agent can exploit.
    #
    # 22 pkt/ms was chosen empirically: under `no_hold` baseline this
    # gives a ~7% drop rate, which mirrors Phase 1's 3-5.5% baseline
    # missed-connection rate. `random` policy collapses to ~70% drop
    # showing that holds carry real cost. Set to ~168 to recreate the
    # MAWI worst-case load.
    arrival_rate_pkt_per_ms: float = 22.0

    # Burstiness: MAWI CoV is 44.4%. We use a Poisson process for v1
    # (CoV=1) but allow a future shaper. The Burst_flag in the state
    # vector still fires whenever instantaneous rate exceeds mean+1*sigma.
    burst_stddev_factor: float = 0.444

    # -----------------------------------------------------------------
    # HNH action space
    # -----------------------------------------------------------------
    # Hold durations in milliseconds. Cardinality 7, matching Phase 1/2
    # so the A2C network architecture (input: state_dim -> output: 7)
    # is reusable.
    hold_actions: Tuple[int, ...] = (0, 1, 2, 5, 10, 20, 50)

    # -----------------------------------------------------------------
    # Decision triggering (sub-sampling)
    # -----------------------------------------------------------------
    # At ~168K pkt/s, making an HNH decision per packet is intractable.
    # We trigger HNH decisions only when one of these conditions holds.
    # Set `decide_every_packet=True` to override (useful for unit tests).
    decide_every_packet: bool = False

    # Trigger if router buffer utilization fraction exceeds this.
    decision_trigger_buf_util: float = 0.50

    # Trigger if the packet has waiting predecessors (N_in > 0).
    decision_trigger_on_predecessors: bool = True

    # Trigger if packet TTL is critically low.
    decision_trigger_ttl_threshold: int = 4

    # -----------------------------------------------------------------
    # Utility / disutility constants  (mirror of Phase 1's delta_p / delta_f)
    # -----------------------------------------------------------------
    # Per-protocol latency budget L_proto (ms). A packet with E2E_delay
    # exceeding L_proto starts incurring full disutility.
    # Indexed by C_p class: 0=ICMP, 1=DNS, 2=TCP-bulk, 3=TCP-interactive, 4=UDP
    latency_budget_per_class_ms: Tuple[float, ...] = (
        100.0,   # ICMP — tolerant
        200.0,   # DNS  — somewhat tolerant
        500.0,   # TCP-bulk (HTTP downloads) — most tolerant
         50.0,   # TCP-interactive (SSH/BGP) — strict
        150.0,   # UDP/other (often streaming) — moderate
    )

    # Delta_R: normalising constant for link utility (= max acceptable
    # extra latency a hold can introduce before LL drops to 0). Phase 1
    # uses 30 minutes; here we use 50 ms to match the maximum hold action.
    delta_R_ms: float = 50.0

    # Delta_P: normalising constant for delivery utility (= max latency
    # before delivery utility hits zero floor).
    delta_P_ms: float = 200.0

    # On-time threshold (analog of Phase 1's 15-minute threshold).
    # A packet with E2E_delay <= this is considered "on-time" and
    # contributes 1.0 to delivery utility regardless of small overruns.
    ontime_threshold_ms: float = 10.0

    # Congestion penalty in LL(τ): when BG > BG_threshold,
    # LL(τ) -= lambda_congestion * (BG - BG_threshold) * τ / 50
    BG_threshold: float = 0.80
    lambda_congestion: float = 0.5

    # -----------------------------------------------------------------
    # Reward weights (passed through to the reward calculator)
    # -----------------------------------------------------------------
    # alpha: trade-off between Delivery Utility (DU/CG) and Link Utility
    # (LU/OG). Higher alpha = prioritise delivery over throughput.
    alpha: float = 0.75
    # beta: trade-off between local and global rewards.
    beta: float = 0.75

    # -----------------------------------------------------------------
    # Global window for rolling stats — DRAMATICALLY shorter than P1/P2
    # -----------------------------------------------------------------
    # Phase 1 uses W=24 hours. Network timescales are ~5 orders of
    # magnitude faster, so W=1000 ms = 1 second.
    global_window_ms: float = 1000.0

    # -----------------------------------------------------------------
    # TCP flow modelling
    # -----------------------------------------------------------------
    # Default RTO (retransmission timeout) for TCP flows in ms. Real TCP
    # adapts RTO from RTT samples; we use a simple fixed value for v1.
    tcp_rto_ms: float = 200.0
    tcp_default_mss: int = 1460                  # MSS from MAWI sample

    # TCP receive window default (bytes). 65535 is the un-scaled max.
    tcp_default_window: int = 65535

    # -----------------------------------------------------------------
    # Head-of-Line blocking detection
    # -----------------------------------------------------------------
    # HOL_flag fires if front-of-queue packet has been waiting > this
    # multiple of its protocol's latency budget AND queue length > 1.
    hol_threshold_factor: float = 2.0

    # -----------------------------------------------------------------
    # Sliding-window helpers
    # -----------------------------------------------------------------
    # Width of the very-short rolling window used for dQ_dt and
    # Drain_rate / Local_arr_rate. Short enough to track instantaneous
    # bursts, long enough to smooth single-packet noise.
    fast_window_ms: float = 10.0


# Convenience: protocol class names for debugging output
PROTOCOL_CLASS_NAMES = ("ICMP", "DNS", "TCP-bulk", "TCP-interactive", "UDP-other")

# Mapping from the protocol_mix tuple index -> C_p class id
PROTOCOL_MIX_TO_CLASS = (2, 3, 4, 0, 1)
