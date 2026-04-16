"""
Demo / test script for the Logistics Cross-Docking Microsimulator.

Mirrors simulator/run_demo.py exactly — same 4 demo functions,
same output structure, same runtime flags.

Usage:
    python -m phase2_simulator.run_demo
    # OR from repo root:
    python phase-2/phase2_simulator/run_demo.py
"""

import time
import sys
import os

# Ensure the repo root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from phase2_simulator.config import SimConfig, HUB_SMALL, HUB_LARGE
from phase2_simulator.simulator import CrossDockSimulator
from phase2_simulator.validation import run_full_validation


def demo_gym_api():
    """Demonstrate the Gym-like API with a small-scale config.

    Mirrors demo_gym_api() from simulator/run_demo.py.
    """
    print("\n" + "=" * 70)
    print("  DEMO: Gym-like API (small scale)")
    print("=" * 70)

    cfg = SimConfig(
        hub=HUB_SMALL,
        num_days=1,
        random_seed=123,
    )

    sim = CrossDockSimulator(cfg)
    state, info = sim.reset()

    print(f"\nInitial state shape: {state.to_array().shape}")
    print(f"State dim:           {state.state_dim}")
    print(f"First HOLD_DECISION for truck: {info['truck_id']}")
    print(f"  Origin hub → Dest hub:     {info['origin_hub']} → {info['dest_hub']}")
    print(f"  Connecting cargo count:    {info['connecting_cargo_count']}")
    print(f"  Intrinsic departure delay: {info['intrinsic_delay']:.1f} min")
    print(f"  SLA urgency:               {info['sla_urgency']}")
    print(f"  Driver hours remaining:    {info['driver_hours_remaining']:.0f} min")
    print(f"  Bay utilisation:           {info['bay_utilisation']:.1%}")
    print(f"  Suggested τ*:              {info['tau_star']:.0f} min")

    # Run a few steps with random actions
    steps = 0
    total_reward = 0.0
    done = False
    while not done and steps < 20:
        action = int(sim.rng.integers(0, len(cfg.hold_actions)))
        state, reward, done, info = sim.step(action)
        total_reward += reward
        steps += 1

    print(f"\nRan {steps} steps, total reward: {total_reward:.3f}")
    if not done:
        print("(stopped early for demo — episode not finished)")
    else:
        print("Episode finished!")
        print(f"Final metrics: {info}")


def demo_baselines():
    """Run baseline policies and compare.

    Mirrors demo_baselines() from simulator/run_demo.py.
    """
    print("\n" + "=" * 70)
    print("  DEMO: Baseline Policy Comparison")
    print("=" * 70)

    cfg = SimConfig(hub=HUB_SMALL, num_days=3, random_seed=42)

    policies = [
        ("no_hold", 0),
        ("heuristic", 15),
        ("heuristic", 30),
    ]

    results = []
    for policy, max_hold in policies:
        label = policy if max_hold == 0 else f"{policy}_{max_hold}"
        print(f"\nRunning {label}...", end=" ", flush=True)
        t0 = time.time()

        sim = CrossDockSimulator(cfg)
        summary = sim.run_episode(policy=policy, max_hold=max_hold, seed=42)
        elapsed = time.time() - t0

        summary["policy"] = label
        summary["elapsed_s"] = round(elapsed, 1)
        results.append(summary)
        print(f"done in {elapsed:.1f}s")

    # Print comparison table
    print(f"\n{'Policy':<18} {'OTP%':>8} {'Failed':>10} {'Avg Dep Delay':>15} {'Avg Hold':>10}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['policy']:<18} {r['schedule_OTP']:>7.1f}% {r['failed_transfers']:>10} "
            f"{r['avg_departure_delay_min']:>14.1f}m {r['avg_hold_min']:>9.1f}m"
        )


def demo_validation():
    """Full validation for HUB_SMALL (7 days, no_hold policy).

    Mirrors demo_validation() from simulator/run_demo.py.
    """
    print("\n")
    cfg_small = SimConfig(hub=HUB_SMALL, num_days=7, random_seed=42)
    run_full_validation(cfg=cfg_small, policy="no_hold", seed=42, verbose=True)


def demo_hub_large():
    """Quick validation for HUB_LARGE (3 days).

    Mirrors demo_air_west_small() from simulator/run_demo.py.
    """
    print("\n")
    cfg_large = SimConfig(hub=HUB_LARGE, num_days=3, random_seed=99)
    run_full_validation(cfg=cfg_large, policy="no_hold", seed=99, verbose=True)


if __name__ == "__main__":
    print("=" * 70)
    print("  Logistics Cross-Docking Microsimulator — Demo")
    print("  Adapted from Malladi et al. (AAMAS 2021) for Phase 2")
    print("=" * 70)

    demo_gym_api()
    demo_baselines()
    demo_validation()
    demo_hub_large()

    print("\n✓ All demos completed successfully.")
