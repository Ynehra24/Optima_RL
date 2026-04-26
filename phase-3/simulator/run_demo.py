"""
run_demo.py — Smoke test for the Phase 3 Network HNH simulator.

Runs the simulator under three baseline policies (no_hold, random,
heuristic) and prints the summary metrics for each. Mirrors
phase-1/simulator/run_demo.py and phase-2/simulator/run_demo.py.

Usage:
    cd phase-3/simulator
    python run_demo.py

Expected output (rough — exact numbers depend on random seed):
    no_hold   : drop_rate ~ 5-15%, avg_latency 30-80 ms, avg_hold 0
    random    : drop_rate similar, avg_latency higher, avg_hold ~14-15 ms
    heuristic : drop_rate slightly lower (preserves predecessors),
                avg_hold non-zero only when N_in > 0
"""

from __future__ import annotations
import sys
import os

# Make `simulator` importable when running this file directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from simulator.config import SimConfig
from simulator.simulator import NetworkSimulator


def banner(text: str) -> str:
    return "\n" + "=" * 64 + f"\n  {text}\n" + "=" * 64


def run_one(policy: str, cfg: SimConfig, seed: int = 42) -> dict:
    print(banner(f"Policy: {policy}"))
    sim = NetworkSimulator(cfg)
    summary = sim.run_episode(policy=policy, seed=seed)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv}")
        else:
            print(f"  {k}: {v}")
    return summary


def main():
    print(banner("PHASE 3 — NETWORK HNH SIMULATOR — SMOKE TEST"))

    cfg = SimConfig()
    print(f"Config: {cfg.num_routers} routers, "
          f"{cfg.episode_duration_ms} ms episode, "
          f"buffer={cfg.buffer_capacity_bytes // 1024} KB/router, "
          f"link={cfg.link_bandwidth_mbps:.0f} Mbps")
    print(f"Decision triggers: buf_util>={cfg.decision_trigger_buf_util}, "
          f"on_predecessors={cfg.decision_trigger_on_predecessors}, "
          f"ttl<={cfg.decision_trigger_ttl_threshold}")

    results = {}
    for policy in ("no_hold", "random", "heuristic"):
        results[policy] = run_one(policy, cfg, seed=42)

    # Side-by-side comparison of the four core metrics
    print(banner("SIDE-BY-SIDE — CORE METRICS"))
    print(f"{'Policy':<12} {'Drop %':>10} {'Deliv %':>10} {'AvgLat ms':>12} {'AvgHold ms':>12} {'HNH#':>8}")
    for p, s in results.items():
        print(f"{p:<12} {s['drop_rate_pct']:>10.2f} "
              f"{s['delivery_rate_pct']:>10.2f} "
              f"{s['avg_latency_ms']:>12.3f} "
              f"{s['avg_hold_ms']:>12.3f} "
              f"{s['hnh_decisions']:>8}")

    print(banner("DONE"))


if __name__ == "__main__":
    main()
