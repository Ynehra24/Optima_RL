"""
Demo / entry-point for the Logistics Cross-Docking Microsimulator.
Mirrors simulator/run_demo.py — same 4 functions, same structure.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2_simulator.config import SimConfig, HUB_SMALL, HUB_LARGE
from phase2_simulator.simulator import CrossDockSimulator
from phase2_simulator.validation import run_full_validation


def demo_gym_api():
    print("\n" + "=" * 70)
    print("  DEMO 1: Gym-like API  (1 day, HUB_SMALL)")
    print("=" * 70)
    cfg = SimConfig(hub=HUB_SMALL, num_days=1, random_seed=123)
    sim = CrossDockSimulator(cfg)
    state, info = sim.reset()

    print(f"\n  State shape       : {state.to_array().shape}")
    print(f"  State dim         : {state.state_dim}")
    print(f"  First decision    : truck {info['truck_id']}")
    print(f"  Route             : {info['origin_hub']} → {info['dest_hub']}")
    print(f"  Connecting cargo  : {info['connecting_cargo_count']}")
    print(f"  Intrinsic delay   : {info['intrinsic_delay']:.1f} min")
    print(f"  Propagated delay  : {info['propagated_delay']:.1f} min")
    print(f"  SLA urgency (X_k) : {info['sla_urgency']}")
    print(f"  Driver hours (L_k): {info['driver_hours_remaining']:.0f} min")
    print(f"  Bay util (BG)     : {info['bay_utilisation']:.1%}")
    print(f"  Perishable frac   : {info['perishability_fraction']:.1%}")
    print(f"  τ* suggestion     : {info['tau_star']:.0f} min")

    steps = 0; total_r = 0.0; done = False
    while not done and steps < 20:
        action = int(sim.rng.integers(0, len(cfg.hold_actions)))
        state, reward, done, info = sim.step(action)
        total_r += reward; steps += 1
    print(f"\n  Ran {steps} steps  |  total reward: {total_r:.3f}")


def demo_baselines():
    print("\n" + "=" * 70)
    print("  DEMO 2: Baseline Policy Comparison  (3 days)")
    print("=" * 70)
    cfg = SimConfig(hub=HUB_SMALL, num_days=3, random_seed=42)
    policies = [("no_hold", 0), ("heuristic", 15), ("heuristic", 30)]
    rows = []
    for policy, max_hold in policies:
        label = policy if max_hold == 0 else f"{policy}_{max_hold}"
        print(f"  Running {label}...", end=" ", flush=True)
        t0 = time.time()
        sim = CrossDockSimulator(cfg)
        s = sim.run_episode(policy=policy, max_hold=max_hold, seed=42)
        s["policy"] = label; s["t"] = round(time.time() - t0, 1)
        rows.append(s)
        print(f"done ({s['t']}s)")

    print(f"\n  {'Policy':<18} {'OTP%':>7} {'Failed%':>8} {'AvgDel':>8} {'BayCong%':>9} {'AvgHold':>8}")
    print("  " + "-" * 62)
    for r in rows:
        print(f"  {r['policy']:<18} "
              f"{r['schedule_OTP_%']:>6.1f}% "
              f"{r['failed_transfer_%']:>7.1f}% "
              f"{r['avg_delivery_delay_min']:>7.1f}m "
              f"{r['avg_bay_utilisation_%']:>8.1f}% "
              f"{r['avg_hold_min']:>7.1f}m")


def demo_validation():
    print()
    cfg = SimConfig(hub=HUB_SMALL, num_days=7, random_seed=42)
    run_full_validation(cfg=cfg, policy="no_hold", seed=42, verbose=True)


def demo_hub_large():
    print()
    cfg = SimConfig(hub=HUB_LARGE, num_days=3, random_seed=99)
    run_full_validation(cfg=cfg, policy="no_hold", seed=99, verbose=True)


if __name__ == "__main__":
    print("=" * 70)
    print("  Logistics Cross-Docking Microsimulator — Demo")
    print("=" * 70)
    demo_gym_api()
    demo_baselines()
    demo_validation()
    demo_hub_large()
    print("\n✓ All demos complete.")
