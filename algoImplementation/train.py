"""
train.py
--------
Main training script for the Hold-No-Hold RL project.

Trains all 4 algorithms and produces plots matching the paper:
  - Figure 6: Business metrics (missed connections, OTP)
  - Figure 7: RL metrics (reward curve, value/Q curve, loss curve)
  - Figure 8: Tunability (alpha/beta sweep)

Usage:
    python train.py                    # train all 4 agents, full run
    python train.py --algo a2c         # train A2C only
    python train.py --episodes 3       # quick smoke test (3 episodes)
    python train.py --algo a2c --episodes 3 --no-plots
"""

import argparse
import numpy as np
import os
import time
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt

from environment import AirlineEnv, EPISODE_LENGTH, ALPHA, BETA
from agents.dqn  import DQNAgent
from agents.a2c  import A2CAgent
from agents.ac   import ACAgent
from agents.ddpg import DDPGAgent


# ── Config (paper §6.2) ────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "n_train_episodes": 25,     # paper: 25 episodes
    "n_test_episodes":  5,      # paper: 5 episodes
    "lr":               0.001,
    "gamma":            0.8,
    "batch_size":       32,
    "alpha":            ALPHA,  # 0.75
    "beta":             BETA,   # 0.75
    "log_every":        500,    # print progress every N steps
    "seed":             42,
}

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Baseline policies (for comparison, like paper §6.2) ───────────────────────

def no_hold_policy(state: np.ndarray) -> int:
    """Always action 0 = hold 0 minutes."""
    return 0

def heuristic_15_policy(state: np.ndarray) -> int:
    """
    Hold up to 15 min if tau* suggests it (mirrors industry heuristic).
    tau* is state[16] normalised; 15 min = index 3 out of 6.
    """
    tau_star_idx = round(state[16] * 6)   # denormalise
    return min(tau_star_idx, 3)           # cap at index 3 (15 min)

def heuristic_30_policy(state: np.ndarray) -> int:
    """Hold up to 30 min based on tau*."""
    return round(state[16] * 6)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_agent(agent, env: AirlineEnv, n_episodes: int,
                algo_name: str, config: dict) -> dict:
    """
    Generic training loop that works for DQN, A2C, AC, and DDPG.
    Returns dict of logged metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Training {algo_name.upper()}  ({n_episodes} episodes)")
    print(f"{'='*60}")

    all_rewards    = []    # per-step rewards
    episode_rewards = []   # mean reward per episode
    t_start        = time.time()
    global_step    = 0

    for ep in range(n_episodes):
        state  = env.reset()
        ep_reward = 0.0
        ep_steps  = 0

        while True:
            # ── Select action ──────────────────────────────────────────────────
            if algo_name == "dqn":
                action = agent.select_action(state)
                value  = None
            elif algo_name in ("a2c", "ac"):
                action, value = agent.select_action(state)
            elif algo_name == "ddpg":
                action = agent.discrete_action(state)
                value  = None

            # ── Step environment ───────────────────────────────────────────────
            next_state, reward, done, info = env.step(action)
            ep_reward   += reward
            ep_steps    += 1
            global_step += 1

            # ── Store and learn ────────────────────────────────────────────────
            if algo_name == "dqn":
                agent.push(state, action, reward, next_state, done)
                agent.update()

            elif algo_name in ("a2c", "ac"):
                agent.store(state, action, reward, value, done)
                if ep_steps % config["batch_size"] == 0 or done:
                    _, last_val = agent.select_action(next_state)
                    last_val = last_val if not done else 0.0
                    agent.update(last_value=last_val)

            elif algo_name == "ddpg":
                agent.push(state, action, reward, next_state, done)
                agent.update()

            all_rewards.append(reward)
            state = next_state

            # ── Logging ────────────────────────────────────────────────────────
            if global_step % config["log_every"] == 0:
                metrics = agent.get_metrics()
                elapsed = time.time() - t_start
                print(f"  Ep {ep+1:3d} | Step {global_step:7d} | "
                      f"AvgR(1k): {metrics['avg_reward_1k']:.4f} | "
                      f"Loss: {metrics['avg_loss_1k']:.4f} | "
                      f"Time: {elapsed:.0f}s")

            if done:
                break

        ep_mean = ep_reward / max(ep_steps, 1)
        episode_rewards.append(ep_mean)
        print(f"  Episode {ep+1:3d} done | Steps: {ep_steps} | "
              f"Mean reward: {ep_mean:.4f}")

    print(f"\n  {algo_name.upper()} training complete in "
          f"{time.time()-t_start:.1f}s")

    return {
        "all_rewards":     np.array(all_rewards),
        "episode_rewards": np.array(episode_rewards),
        "losses":          np.array(agent.losses)     if agent.losses  else np.array([]),
        "q_values":        np.array(agent.q_values)   if agent.q_values else np.array([]),
    }


def evaluate_agent(agent, env: AirlineEnv, n_episodes: int,
                   algo_name: str) -> dict:
    """Run agent in greedy mode for n_episodes. Returns business metrics."""
    print(f"\n  Evaluating {algo_name.upper()} ({n_episodes} test episodes)...")

    total_connections    = 0
    missed_connections   = 0
    on_time_arrivals     = 0
    total_rewards        = []
    total_holds_applied  = 0
    total_steps          = 0

    for ep in range(n_episodes):
        state = env.reset()
        while True:
            if algo_name == "dqn":
                action = agent.greedy_action(state)
            elif algo_name in ("a2c", "ac"):
                action = agent.greedy_action(state)
            elif algo_name == "ddpg":
                action = agent.discrete_action(state)

            next_state, reward, done, info = env.step(action)

            total_rewards.append(reward)
            total_steps += 1

            # Simulate business metrics from state/reward signals
            # (In real env, Member 1 provides these directly)
            otp_proxy   = info["a_local"]    # airline utility ≈ on-time proxy
            pax_proxy   = info["p_local"]    # passenger utility ≈ connection proxy

            total_connections  += 1
            if pax_proxy < 0.5:              # low PAX utility = missed connection
                missed_connections += 1
            if otp_proxy > 0.85:             # high AU = on-time
                on_time_arrivals += 1
            if action > 0:
                total_holds_applied += 1

            state = next_state
            if done:
                break

    otp_pct    = 100.0 * on_time_arrivals / max(total_steps, 1)
    missed_pct = 100.0 * missed_connections / max(total_connections, 1)
    hold_pct   = 100.0 * total_holds_applied / max(total_steps, 1)

    print(f"    OTP: {otp_pct:.1f}%  |  Missed: {missed_pct:.1f}%  |  "
          f"Holds applied: {hold_pct:.1f}%  |  "
          f"Avg reward: {np.mean(total_rewards):.4f}")

    return {
        "otp_pct":    otp_pct,
        "missed_pct": missed_pct,
        "hold_pct":   hold_pct,
        "avg_reward": float(np.mean(total_rewards)),
    }


def evaluate_baseline(policy_fn, env: AirlineEnv,
                       n_episodes: int, name: str) -> dict:
    """Evaluate a rule-based baseline."""
    print(f"  Evaluating baseline: {name}...")
    total_connections = missed = on_time = steps = holds = 0
    rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        while True:
            action = policy_fn(state)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
            steps += 1
            total_connections += 1
            if info["p_local"] < 0.5:
                missed += 1
            if info["a_local"] > 0.85:
                on_time += 1
            if action > 0:
                holds += 1
            if done:
                break

    return {
        "otp_pct":    100.0 * on_time  / max(steps, 1),
        "missed_pct": 100.0 * missed   / max(total_connections, 1),
        "hold_pct":   100.0 * holds    / max(steps, 1),
        "avg_reward": float(np.mean(rewards)),
    }


# ── Plotting (reproduces paper Figures 6, 7, 8) ────────────────────────────────

def smooth(x: np.ndarray, w: int = 1000) -> np.ndarray:
    """Moving average smoothing."""
    if len(x) < w:
        w = max(1, len(x) // 10)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_figure7(train_results: dict, algo_names: list, save_path: str):
    """Reproduce paper Figure 7: RL metrics — reward, value, loss."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"a2c": "#1f77b4", "dqn": "#ff7f0e", "ac": "#2ca02c", "ddpg": "#d62728"}

    for name in algo_names:
        res = train_results[name]
        c   = colors.get(name, "gray")

        # (a) Average reward (smoothed)
        if len(res["all_rewards"]) > 0:
            s = smooth(res["all_rewards"])
            axes[0].plot(s, label=name.upper(), color=c, linewidth=1.5)

        # (b) Q-value / critic value
        if len(res["q_values"]) > 0:
            s = smooth(res["q_values"], w=100)
            axes[1].plot(s, label=name.upper(), color=c, linewidth=1.5)

        # (c) Loss
        if len(res["losses"]) > 0:
            s = smooth(res["losses"], w=100)
            axes[2].plot(s, label=name.upper(), color=c, linewidth=1.5)

    axes[0].set_title("(a) Average reward"); axes[0].set_xlabel("Step"); axes[0].set_ylabel("Avg reward")
    axes[1].set_title("(b) Value");           axes[1].set_xlabel("Step"); axes[1].set_ylabel("Avg value")
    axes[2].set_title("(c) Neural net loss"); axes[2].set_xlabel("Step"); axes[2].set_ylabel("Avg loss")

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Figure 7 — RL metrics (smoothed)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_figure6(eval_results: dict, baseline_results: dict, save_path: str):
    """Reproduce paper Figure 6: business metrics bar chart."""
    all_names   = list(baseline_results.keys()) + list(eval_results.keys())
    missed_vals = ([baseline_results[n]["missed_pct"] for n in baseline_results] +
                   [eval_results[n]["missed_pct"]     for n in eval_results])
    otp_vals    = ([baseline_results[n]["otp_pct"]    for n in baseline_results] +
                   [eval_results[n]["otp_pct"]        for n in eval_results])

    x  = np.arange(len(all_names))
    w  = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w/2, missed_vals, w, label="Missed connections %",
                    color="#d62728", alpha=0.8)
    bars2 = ax2.bar(x + w/2, otp_vals,    w, label="OTP %",
                    color="#1f77b4", alpha=0.8)

    ax1.set_ylabel("Missed connections %", color="#d62728")
    ax2.set_ylabel("On-time performance %", color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.upper() for n in all_names], rotation=15)
    ax1.set_title("Figure 6 — Business metrics: missed connections and OTP")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_figure8(config: dict, save_path: str):
    """
    Reproduce paper Figure 8: tunability — alpha/beta sweep.
    Uses A2C with varying alpha and beta values.
    """
    print("\n  Running alpha/beta sweep for Figure 8 (this takes a while)...")
    alphas = [0.10, 0.25, 0.50, 0.75, 1.00]
    betas  = [0.10, 0.50, 1.00]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    for j, beta in enumerate(betas):
        saved_list = []
        otp_list   = []

        for alpha in alphas:
            env   = AirlineEnv(seed=42, alpha=alpha, beta=beta)
            agent = A2CAgent(lr=config["lr"], gamma=config["gamma"],
                             batch_size=config["batch_size"], seed=42)

            # Short training for sweep (2 episodes)
            for _ in range(2):
                state = env.reset()
                steps = 0
                while True:
                    action, value = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.store(state, action, reward, value, done)
                    if steps % config["batch_size"] == 0 or done:
                        _, lv = agent.select_action(next_state)
                        agent.update(last_value=0.0 if done else lv)
                    state = next_state
                    steps += 1
                    if done:
                        break

            res  = evaluate_agent(agent, AirlineEnv(seed=99, alpha=alpha, beta=beta),
                                  n_episodes=1, algo_name="a2c")
            saved_list.append(100.0 - res["missed_pct"])   # connections saved
            otp_list.append(res["otp_pct"])

        ax1.plot(alphas, saved_list, color=colors[j], marker="o",
                 label=f"β={beta} (saved)")
        ax2.plot(alphas, otp_list,   color=colors[j], marker="s",
                 linestyle="--", label=f"β={beta} (OTP)")

    ax1.set_xlabel("Alpha (α)")
    ax1.set_ylabel("Connections saved %",    color="#2ca02c")
    ax2.set_ylabel("On-time performance %",  color="#1f77b4")
    ax1.set_title("Figure 8 — Tunability: α/β sweep (A2C)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def build_agent(algo: str, config: dict):
    if algo == "dqn":
        return DQNAgent(lr=config["lr"], gamma=config["gamma"],
                        batch_size=config["batch_size"], seed=config["seed"])
    elif algo == "a2c":
        return A2CAgent(lr=config["lr"], gamma=config["gamma"],
                        batch_size=config["batch_size"], seed=config["seed"])
    elif algo == "ac":
        return ACAgent(lr=config["lr"], gamma=config["gamma"],
                       batch_size=config["batch_size"], seed=config["seed"])
    elif algo == "ddpg":
        return DDPGAgent(lr_actor=config["lr"], lr_critic=config["lr"],
                         gamma=config["gamma"], batch_size=config["batch_size"],
                         seed=config["seed"])
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def main():
    parser = argparse.ArgumentParser(description="HNH RL Training")
    parser.add_argument("--algo",     type=str, default="all",
                        choices=["all","a2c","dqn","ac","ddpg"],
                        help="Algorithm to train (default: all)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of training episodes")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip alpha/beta sweep (Figure 8)")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.episodes is not None:
        config["n_train_episodes"] = args.episodes
        config["n_test_episodes"]  = max(1, args.episodes // 5)

    algos = ["a2c", "dqn", "ac", "ddpg"] if args.algo == "all" else [args.algo]

    print("\n" + "="*60)
    print("  Hold-No-Hold RL Training")
    print(f"  Algorithms : {algos}")
    print(f"  Train eps  : {config['n_train_episodes']}")
    print(f"  Test eps   : {config['n_test_episodes']}")
    print(f"  Alpha={config['alpha']}  Beta={config['beta']}")
    print("="*60)

    # ── Train ──────────────────────────────────────────────────────────────────
    train_results = {}
    agents        = {}

    for algo in algos:
        env   = AirlineEnv(seed=config["seed"],
                           alpha=config["alpha"], beta=config["beta"])
        agent = build_agent(algo, config)
        results = train_agent(agent, env,
                              n_episodes=config["n_train_episodes"],
                              algo_name=algo, config=config)
        train_results[algo] = results
        agents[algo]        = agent

    # ── Evaluate baselines ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Evaluating baselines")
    print("="*60)
    test_env = AirlineEnv(seed=config["seed"] + 100,
                          alpha=config["alpha"], beta=config["beta"])

    baseline_results = {
        "no_hold":      evaluate_baseline(no_hold_policy,      test_env, config["n_test_episodes"], "No-Hold"),
        "heuristic_15": evaluate_baseline(heuristic_15_policy, test_env, config["n_test_episodes"], "Heuristic-15"),
        "heuristic_30": evaluate_baseline(heuristic_30_policy, test_env, config["n_test_episodes"], "Heuristic-30"),
    }

    # ── Evaluate RL agents ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Evaluating RL agents")
    print("="*60)
    eval_results = {}
    for algo in algos:
        test_env  = AirlineEnv(seed=config["seed"] + 200,
                               alpha=config["alpha"], beta=config["beta"])
        eval_results[algo] = evaluate_agent(agents[algo], test_env,
                                            config["n_test_episodes"], algo)

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Method':<15} {'Missed%':>8} {'OTP%':>8} {'AvgReward':>10}")
    print("  " + "-"*45)
    for name, res in baseline_results.items():
        print(f"  {name.upper():<15} {res['missed_pct']:>8.1f} "
              f"{res['otp_pct']:>8.1f} {res['avg_reward']:>10.4f}")
    for name, res in eval_results.items():
        print(f"  {name.upper():<15} {res['missed_pct']:>8.1f} "
              f"{res['otp_pct']:>8.1f} {res['avg_reward']:>10.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n  Generating plots...")
        plot_figure7(train_results, algos,
                     os.path.join(RESULTS_DIR, "figure7_rl_metrics.png"))
        plot_figure6(eval_results, baseline_results,
                     os.path.join(RESULTS_DIR, "figure6_business_metrics.png"))
        if not args.no_sweep:
            plot_figure8(config, os.path.join(RESULTS_DIR, "figure8_tunability.png"))

    print("\n  All done. Results saved to ./results/")


if __name__ == "__main__":
    main()
