"""
Demo / test script for the Airline Network Microsimulator.

Runs validation for both Air-East and Air-West profiles with
multiple baseline policies and prints a comprehensive report.

Usage:
    python run_demo.py
"""

import time
import sys
import os

# Ensure the parent directory (project root) is on the path
# This allows running the script directly from the simulator folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import SimConfig, AIR_EAST, AIR_WEST
from .simulator import AirlineNetworkSimulator
from .validation import run_full_validation


def demo_gym_api():
    """Demonstrate the Gym-like API with a small config."""
    print("\n" + "=" * 70)
    print("  DEMO: Gym-like API (small scale)")
    print("=" * 70)

    # Use a small-scale config for quick demo
    small_airline = AIR_EAST
    # Override for faster demo
    cfg = SimConfig(
        airline=small_airline,
        num_days=1,
        random_seed=123,
    )

    sim = AirlineNetworkSimulator(cfg)
    state, info = sim.reset()

    print(f"\nInitial state shape: {state.to_array().shape}")
    print(f"State dim: {state.state_dim}")
    print(f"First HNH decision for flight: {info['flight_id']}")
    print(f"  Origin: {info['origin']} -> Destination: {info['destination']}")
    print(f"  Connecting PAX: {info['connecting_pax_count']}")
    print(f"  Intrinsic delay: {info['intrinsic_delay']:.1f} min")
    print(f"  Suggested τ*: {info['tau_star']:.0f} min")

    # Run a few steps with random actions
    steps = 0
    total_reward = 0
    done = False
    while not done and steps < 20:
        # Random action
        action = sim.rng.integers(0, len(cfg.hold_actions))
        state, reward, done, info = sim.step(int(action))
        total_reward += reward
        steps += 1

    print(f"\nRan {steps} steps, total reward: {total_reward:.3f}")
    if not done:
        print("(stopped early for demo — episode not finished)")
    else:
        print("Episode finished!")
        print(f"Final metrics: {info}")


def demo_baselines():
    """Run baseline policies and compare."""
    print("\n" + "=" * 70)
    print("  DEMO: Baseline Policy Comparison")
    print("=" * 70)

    cfg = SimConfig(airline=AIR_EAST, num_days=3, random_seed=42)

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

        sim = AirlineNetworkSimulator(cfg)
        summary = sim.run_episode(policy=policy, max_hold=max_hold, seed=42)
        elapsed = time.time() - t0

        summary["policy"] = label
        summary["elapsed_s"] = round(elapsed, 1)
        results.append(summary)
        print(f"done in {elapsed:.1f}s")

    # Print comparison table
    print(f"\n{'Policy':<18} {'OTP%':>8} {'Missed':>10} {'Avg Arr Delay':>15} {'Avg Hold':>10}")
    print("-" * 65)
    for r in results:
        print(
            f"{r['policy']:<18} {r['OTP']:>7.1f}% {r['missed_connections']:>10} "
            f"{r['avg_arrival_delay_min']:>14.1f}m {r['avg_hold_min']:>9.1f}m"
        )


def demo_validation():
    """Full validation for Air-East."""
    print("\n")
    cfg_east = SimConfig(airline=AIR_EAST, num_days=7, random_seed=42)
    run_full_validation(cfg=cfg_east, policy="no_hold", seed=42, verbose=True)


def demo_air_west_small():
    """Quick validation for Air-West (3 days)."""
    print("\n")
    cfg_west = SimConfig(airline=AIR_WEST, num_days=3, random_seed=99)
    run_full_validation(cfg=cfg_west, policy="no_hold", seed=99, verbose=True)


if __name__ == "__main__":
    print("=" * 70)
    print("  Airline Network Microsimulator — Demo")
    print("  Based on Malladi et al. (AAMAS 2021)")
    print("=" * 70)

    demo_gym_api()
    demo_baselines()
    demo_validation()
    demo_air_west_small()

    print("\n✓ All demos completed successfully.")
