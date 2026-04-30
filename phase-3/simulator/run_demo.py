"""
run_demo.py — Baseline policy comparison for the DAG HNH simulator.

Run from the phase-3/ folder:
    python simulator/run_demo.py

Demo uses 1 simulated hour + 500-decision cap — finishes in ~5 seconds.
For real A2C training, remove those caps (see comments in run_baseline).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.config import SimConfig
from simulator.simulator import DAGSchedulingSimulator


def run_baseline(policy: str, seed: int = 42) -> dict:
    cfg = SimConfig()
    # --- DEMO CAPS (remove for training) ---
    cfg.episode_duration_s = 3_600.0   # 1 simulated hour (training = 7 * 86400)
    cfg.max_hnh_decisions  = 500       # cap decisions   (training = 50_000)
    # ----------------------------------------
    sim = DAGSchedulingSimulator(cfg)
    return sim.run_episode(policy=policy, seed=seed)


def main():
    print("=" * 70)
    print("  PHASE 3 — DAG HNH SIMULATOR: BASELINE POLICY COMPARISON")
    print("=" * 70)
    print("  (1h simulated time, 500-decision cap — completes in ~5 seconds)\n")

    results = {}
    for policy in ["no_hold", "heuristic", "random"]:
        print(f"  Running {policy} ...", end="", flush=True)
        results[policy] = run_baseline(policy, seed=42)
        s = results[policy]
        print(f"  done  (jobs={s['total_jobs']}, tasks={s['total_tasks']})")

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    hdr = (f"{'Policy':<14} {'Tasks':>8} {'Done%':>8} {'Evict%':>8}"
           f" {'Stalls':>8} {'Hold%':>8} {'Reward':>10}")
    print(hdr)
    print("-" * 70)
    for pol, s in results.items():
        print(f"{pol:<14} {s['total_tasks']:>8} {s['completed_pct']:>8.1f}"
              f" {s['evicted_pct']:>8.1f} {s['pipeline_stalls']:>8}"
              f" {s['hold_rate_pct']:>8.1f} {s['total_reward']:>10.3f}")

    print("\n" + "=" * 70)
    print("  KEY OBSERVATIONS")
    print("  no_hold  : zero holds — every delay event becomes a pipeline stall")
    print("  heuristic: holds when delay detected AND cluster not congested")
    print("  random   : holds indiscriminately — wastes GPU/CPU resources")
    print("  A2C goal : outperform heuristic (~50% stall reduction, per paper)")
    print("=" * 70)
    print("\n  State: 88 dims  |  Actions: {0, 15, 30, 60, 120} seconds")


if __name__ == "__main__":
    main()