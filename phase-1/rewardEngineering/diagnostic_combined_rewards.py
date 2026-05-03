"""
diagnostic_full_episode.py
==========================
Run a FULL episode to see when delays propagate and r_g becomes non-zero.
"""

import sys
import os
sys.path.insert(0, '.')

import numpy as np
from simulator.simulator import AirlineNetworkSimulator
from simulator.config import SimConfig

def diagnostic_full_episode():
    """Run one FULL episode and check reward signals."""
    
    # print("\n" + "="*80)
    # print("DIAGNOSTIC: Full Episode Reward Pipeline")
    # print("="*80)

    cfg = SimConfig()
    env = AirlineNetworkSimulator(cfg)
    ctx, _ = env.reset()

    # Run FULL episode
    step_count = 0
    trajectory_data = []

    # print(f"\nRunning full episode...")
    while True:
        # Take greedy action
        action = int(np.argmax(ctx.PL))

        # Step
        ctx, r_l, done, info = env.step(action)

        trajectory_data.append({
            'step': step_count,
            'action': action,
            'r_l': r_l,
            'flight_id': info.get('flight_id', ''),
        })

        step_count += 1
        # if step_count % 100 == 0:
        #     print(f"  ... step {step_count}")

        if done:
            # print(f"\n✓ Episode complete: {step_count} steps")
            break

    # print(f"\nTrajectory collected: {len(trajectory_data)} steps")

    # Analyze local rewards
    # print(f"\nLOCAL REWARDS (r_l):")
    r_l_vals = [t['r_l'] for t in trajectory_data]
    # print(f"  Mean: {np.mean(r_l_vals):.6f}")
    # print(f"  Std:  {np.std(r_l_vals):.6f}")
    # print(f"  Min:  {np.min(r_l_vals):.6f}")
    # print(f"  Max:  {np.max(r_l_vals):.6f}")

    # Query global rewards
    # print(f"\nQUERYING GLOBAL REWARDS (r_g)...")
    global_rewards = {}
    r_g_vals = []

    for flight_id in set(t['flight_id'] for t in trajectory_data if t['flight_id']):
        r_g = env.reward_calculator.get_global_reward(flight_id)
        global_rewards[flight_id] = r_g
        if r_g != 0:
            r_g_vals.append(r_g)

    if not r_g_vals:
        # print(f"  ⚠ No non-zero global rewards found")
        # print(f"  → All attributable outcomes: {len([x for x in global_rewards.values() if x != 0])}")
        pass
    else:
        all_r_g = list(global_rewards.values())
        # print(f"  Found {len(r_g_vals)} flights with NON-ZERO global rewards")
        # print(f"  Non-zero Mean: {np.mean(r_g_vals):.6f}")
        # print(f"  All flights Mean: {np.mean(all_r_g):.6f}")
        # print(f"  Max: {np.max(all_r_g):.6f}")

    # Combine
    # print(f"\nCOMBINING REWARDS (beta=0.75):")
    combined_rewards = []

    for t in trajectory_data:
        r_l = t['r_l']
        r_g = global_rewards.get(t['flight_id'], 0.0)
        beta = env.reward_calculator.beta
        r_total = beta * r_l + (1 - beta) * r_g
        combined_rewards.append(r_total)

    # print(f"  Mean: {np.mean(combined_rewards):.6f}")
    # print(f"  Std:  {np.std(combined_rewards):.6f}")
    # print(f"  Min:  {np.min(combined_rewards):.6f}")
    # print(f"  Max:  {np.max(combined_rewards):.6f}")

    # Analysis
    # print(f"\n" + "="*80)
    # print("ANALYSIS:")
    # print("="*80)

    nonzero_r_g = sum(1 for x in global_rewards.values() if x != 0)

    if nonzero_r_g == 0:
        # print("❌ No non-zero global rewards in entire episode")
        # print("   → Delay tree either not working OR all flights on-time")
        # print(f"   → Simulator metrics: {env.metrics.summary()}")
        pass
    elif nonzero_r_g < len(global_rewards) * 0.1:
        # print(f"⚠ Only {nonzero_r_g}/{len(global_rewards)} flights have non-zero r_g ({100*nonzero_r_g/len(global_rewards):.1f}%)")
        # print("   → Most flights arriving on-time, limited feedback signal")
        pass
    else:
        # print(f"✅ {nonzero_r_g}/{len(global_rewards)} flights have non-zero r_g")
        # print("   → Delay tree is working properly")
        # print(f"   → Combined reward should improve training")
        pass

    # print(f"\n  Episode metrics:")
    metrics = env.metrics.summary()
    # for key in ['OTP', 'missed_connections', 'avg_arrival_delay_min', 'avg_departure_delay_min']:
    #     if key in metrics:
    #         print(f"    {key}: {metrics[key]}")

    # print("\n" + "="*80)

if __name__ == "__main__":
    diagnostic_full_episode()