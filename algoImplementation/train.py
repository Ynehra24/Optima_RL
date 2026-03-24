"""
train.py
--------
Main training script for the Hold-No-Hold RL project.
Integrated with Member 1's real AirlineNetworkSimulator.

Trains all 4 algorithms and produces plots matching the paper:
  - Figure 6: Business metrics (missed connections, OTP)
  - Figure 7: RL metrics (reward curve, value/Q curve, loss curve)
  - Figure 8: Tunability (alpha/beta sweep)

Usage (run from algoImplementation/ folder):
    python train.py                        # train all 4 agents, full run
    python train.py --algo a2c             # train A2C only
    python train.py --episodes 3           # quick smoke test
    python train.py --algo a2c --episodes 3 --no-plots
    python train.py --algo all --episodes 25 --no-sweep
"""

import argparse
import numpy as np
import os
import sys
import time
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import matplotlib.pyplot as plt

# ── Simulator import (Member 1) ───────────────────────────────────────────────
# Go up one level from algoImplementation/ to find simulator/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from simulator.simulator import AirlineNetworkSimulator
from simulator.config    import SimConfig

# ── Agent imports ─────────────────────────────────────────────────────────────
from agents.dqn  import DQNAgent
from agents.a2c  import A2CAgent
from agents.ac   import ACAgent
from agents.ddpg import DDPGAgent


# ── Constants (paper §6.2) ────────────────────────────────────────────────────
ALPHA        = 0.75
BETA         = 0.75
HOLD_ACTIONS = [0, 5, 10, 15, 20, 25, 30]   # minutes
STATE_DIM    = 17
ACTION_DIM   = 7

DEFAULT_CONFIG = {
    "n_train_episodes": 25,
    "n_test_episodes":  5,
    "lr":               0.0001,
    "gamma":            0.8,
    "batch_size":       32,
    "alpha":            ALPHA,
    "beta":             BETA,
    "log_every":        500,
    "seed":             42,
}

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── State conversion (FlightContext → numpy array) ────────────────────────────

def context_to_state(ctx) -> np.ndarray:
    """
    Convert Member 1's FlightContext object → flat numpy array shape (17,).

    FlightContext.to_array() returns:
        [PL(τ0)..PL(τ6), AL(τ0)..AL(τ6), PG, AG, tau_star]
    tau_star comes in as minutes (0–30), we normalise to 0–1.
    """
    # Terminal state when episode is done
    if hasattr(ctx, 'flight_id') and ctx.flight_id == "DONE":
        return np.zeros(STATE_DIM, dtype=np.float32)

    state = ctx.to_array().copy()          # shape (17,) from context_engine.py

    # Normalise tau_star: state[16] is in minutes → divide by 30
    state[16] = float(state[16]) / 30.0

    # Clip everything to [0, 1] for safety
    state = np.clip(state, 0.0, 1.0).astype(np.float32)
    return state


# ── Environment factory ───────────────────────────────────────────────────────

def build_env(alpha: float = ALPHA, seed: int = 42) -> AirlineNetworkSimulator:
    """Create a fresh simulator instance."""
    cfg            = SimConfig()
    cfg.alpha      = alpha
    cfg.random_seed = seed
    return AirlineNetworkSimulator(cfg)


# ── Baseline policies ─────────────────────────────────────────────────────────

def no_hold_policy(state: np.ndarray) -> int:
    """Always hold 0 minutes — action index 0."""
    return 0


def heuristic_15_policy(state: np.ndarray) -> int:
    """
    Hold up to 15 min based on tau*.
    state[16] = tau* normalised to [0,1]. Denormalise → index, cap at 3 (15 min).
    """
    tau_star_idx = round(state[16] * 6)
    return min(tau_star_idx, 3)


def heuristic_30_policy(state: np.ndarray) -> int:
    """Hold up to 30 min based on tau*."""
    return min(round(state[16] * 6), 6)


# ── Extract business metrics from info dict ───────────────────────────────────

def extract_business_metrics(info: dict) -> dict:
    """
    Pull business metrics from Member 1's info dict.
    Falls back gracefully if keys are missing (e.g. mid-episode steps).
    """
    return {
        "missed":  info.get("missed_connections", 0),
        "otp":     info.get("OTP", info.get("otp", 0.0)),
        "arrived": info.get("arrived", 0),
        "ontime":  info.get("ontime_arrivals", 0),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_agent(agent, env: AirlineNetworkSimulator,
                n_episodes: int, algo_name: str, config: dict) -> dict:
    """
    Generic training loop for DQN, A2C, AC, DDPG.
    Works with Member 1's simulator — handles FlightContext states.
    """
    print(f"\n{'='*60}")
    print(f"  Training {algo_name.upper()}  ({n_episodes} episodes)")
    print(f"{'='*60}")

    all_rewards     = []
    episode_rewards = []
    t_start         = time.time()
    global_step     = 0

    for ep in range(n_episodes):

        # ── Reset ──────────────────────────────────────────────────────────────
        ctx, info = env.reset()
        state     = context_to_state(ctx)

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

            # ── Step ──────────────────────────────────────────────────────────
            ctx, reward, done, info = env.step(action)
            next_state = context_to_state(ctx)

            ep_reward   += reward
            ep_steps    += 1
            global_step += 1

            # ── Store & learn ──────────────────────────────────────────────────
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

        # ── End-of-episode summary ─────────────────────────────────────────────
        ep_mean = ep_reward / max(ep_steps, 1)
        episode_rewards.append(ep_mean)

        # Pull real business metrics from final info dict
        bm = info if isinstance(info, dict) else {}
        missed  = bm.get("missed_connections", "?")
        otp     = bm.get("OTP", "?")
        print(f"  Episode {ep+1:3d} done | Steps: {ep_steps:5d} | "
              f"Mean reward: {ep_mean:.4f} | "
              f"Missed: {missed} | OTP: {otp}")

    print(f"\n  {algo_name.upper()} training complete in "
          f"{time.time()-t_start:.1f}s")

    return {
        "all_rewards":     np.array(all_rewards),
        "episode_rewards": np.array(episode_rewards),
        "losses":          np.array(agent.losses)   if agent.losses   else np.array([]),
        "q_values":        np.array(agent.q_values) if agent.q_values else np.array([]),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_agent(agent, env: AirlineNetworkSimulator,
                   n_episodes: int, algo_name: str) -> dict:
    """Run agent greedy for n_episodes. Returns real business metrics."""
    print(f"\n  Evaluating {algo_name.upper()} ({n_episodes} test episodes)...")

    all_missed   = []
    all_otp      = []
    all_rewards  = []
    all_holds    = []

    for ep in range(n_episodes):
        ctx, info   = env.reset()
        state       = context_to_state(ctx)
        ep_rewards  = []
        ep_holds    = 0
        ep_steps    = 0

        while True:
            if algo_name == "dqn":
                action = agent.greedy_action(state)
            elif algo_name in ("a2c", "ac"):
                action = agent.greedy_action(state)
            elif algo_name == "ddpg":
                action = agent.discrete_action(state)

            ctx, reward, done, info = env.step(action)
            next_state = context_to_state(ctx)

            ep_rewards.append(reward)
            ep_steps   += 1
            if action > 0:
                ep_holds += 1

            state = next_state
            if done:
                break

        # Get real metrics from the simulator's final summary
        all_rewards.append(float(np.mean(ep_rewards)))
        all_holds.append(100.0 * ep_holds / max(ep_steps, 1))

        if isinstance(info, dict):
            all_missed.append(info.get("missed_connections", 0))
            all_otp.append(info.get("OTP", 0.0))

    avg_missed  = float(np.mean(all_missed)) if all_missed else 0.0
    avg_otp     = float(np.mean(all_otp))    if all_otp    else 0.0
    avg_reward  = float(np.mean(all_rewards))
    avg_hold_pct = float(np.mean(all_holds))

    print(f"    OTP: {avg_otp:.1f}%  |  Missed: {avg_missed:.0f} PAX  |  "
          f"Holds: {avg_hold_pct:.1f}%  |  Avg reward: {avg_reward:.4f}")

    return {
        "otp_pct":    avg_otp,
        "missed_pct": avg_missed,
        "hold_pct":   avg_hold_pct,
        "avg_reward": avg_reward,
    }


def evaluate_baseline(policy_fn, env: AirlineNetworkSimulator,
                      n_episodes: int, name: str) -> dict:
    """Evaluate a rule-based baseline policy."""
    print(f"  Evaluating baseline: {name}...")

    all_missed  = []
    all_otp     = []
    all_rewards = []
    all_holds   = []

    for _ in range(n_episodes):
        ctx, info  = env.reset()
        state      = context_to_state(ctx)
        ep_rewards = []
        ep_holds   = 0
        ep_steps   = 0

        while True:
            action = policy_fn(state)
            ctx, reward, done, info = env.step(action)
            state = context_to_state(ctx)

            ep_rewards.append(reward)
            ep_steps += 1
            if action > 0:
                ep_holds += 1
            if done:
                break

        all_rewards.append(float(np.mean(ep_rewards)))
        all_holds.append(100.0 * ep_holds / max(ep_steps, 1))

        if isinstance(info, dict):
            all_missed.append(info.get("missed_connections", 0))
            all_otp.append(info.get("OTP", 0.0))

    return {
        "otp_pct":    float(np.mean(all_otp))     if all_otp    else 0.0,
        "missed_pct": float(np.mean(all_missed))  if all_missed else 0.0,
        "hold_pct":   float(np.mean(all_holds)),
        "avg_reward": float(np.mean(all_rewards)),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def smooth(x: np.ndarray, w: int = 1000) -> np.ndarray:
    if len(x) < w:
        w = max(1, len(x) // 10)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_figure7(train_results: dict, algo_names: list, save_path: str):
    """Reproduce paper Figure 7: reward, value, loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"a2c": "#1f77b4", "dqn": "#ff7f0e",
               "ac": "#2ca02c",  "ddpg": "#d62728"}

    for name in algo_names:
        res = train_results[name]
        c   = colors.get(name, "gray")
        if len(res["all_rewards"]) > 0:
            axes[0].plot(smooth(res["all_rewards"]),
                         label=name.upper(), color=c, linewidth=1.5)
        if len(res["q_values"]) > 0:
            axes[1].plot(smooth(res["q_values"], w=100),
                         label=name.upper(), color=c, linewidth=1.5)
        if len(res["losses"]) > 0:
            axes[2].plot(smooth(res["losses"], w=100),
                         label=name.upper(), color=c, linewidth=1.5)

    titles = ["(a) Average reward", "(b) Value", "(c) Neural net loss"]
    ylabels = ["Avg reward", "Avg value", "Avg loss"]
    for ax, t, y in zip(axes, titles, ylabels):
        ax.set_title(t); ax.set_xlabel("Step"); ax.set_ylabel(y)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle("Figure 7 — RL metrics (smoothed)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_figure6(eval_results: dict, baseline_results: dict, save_path: str):
    """Reproduce paper Figure 6: missed connections and OTP bar chart."""
    all_names   = list(baseline_results.keys()) + list(eval_results.keys())
    missed_vals = ([baseline_results[n]["missed_pct"] for n in baseline_results] +
                   [eval_results[n]["missed_pct"]     for n in eval_results])
    otp_vals    = ([baseline_results[n]["otp_pct"]    for n in baseline_results] +
                   [eval_results[n]["otp_pct"]        for n in eval_results])

    x   = np.arange(len(all_names))
    w   = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(x - w/2, missed_vals, w, label="Missed connections",
            color="#d62728", alpha=0.8)
    ax2.bar(x + w/2, otp_vals,    w, label="OTP %",
            color="#1f77b4", alpha=0.8)

    ax1.set_ylabel("Missed connections (PAX)", color="#d62728")
    ax2.set_ylabel("On-time performance %",    color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.upper() for n in all_names], rotation=15)
    ax1.set_title("Figure 6 — Business metrics: missed connections and OTP")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right")
    ax1.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_figure8(config: dict, save_path: str):
    """Reproduce paper Figure 8: alpha/beta tunability sweep using A2C."""
    print("\n  Running alpha/beta sweep for Figure 8...")
    alphas = [0.10, 0.25, 0.50, 0.75, 1.00]
    betas  = [0.10, 0.50, 1.00]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    for j, beta in enumerate(betas):
        saved_list = []
        otp_list   = []

        for alpha in alphas:
            # Fresh env and agent per sweep point
            env   = build_env(alpha=alpha, seed=42)
            agent = A2CAgent(lr=config["lr"], gamma=config["gamma"],
                             batch_size=config["batch_size"], seed=42)

            # Short training — 2 episodes per sweep point
            for _ in range(2):
                ctx, info = env.reset()
                state     = context_to_state(ctx)
                steps     = 0
                while True:
                    action, value = agent.select_action(state)
                    ctx, reward, done, _ = env.step(action)
                    next_state = context_to_state(ctx)
                    agent.store(state, action, reward, value, done)
                    if steps % config["batch_size"] == 0 or done:
                        _, lv = agent.select_action(next_state)
                        agent.update(last_value=0.0 if done else lv)
                    state = next_state
                    steps += 1
                    if done:
                        break

            # Evaluate
            test_env = build_env(alpha=alpha, seed=99)
            res = evaluate_agent(agent, test_env, n_episodes=1, algo_name="a2c")
            saved_list.append(100.0 - res["missed_pct"])
            otp_list.append(res["otp_pct"])

        ax1.plot(alphas, saved_list, color=colors[j], marker="o",
                 label=f"β={beta} (saved)")
        ax2.plot(alphas, otp_list,   color=colors[j], marker="s",
                 linestyle="--", label=f"β={beta} (OTP)")

    ax1.set_xlabel("Alpha (α)")
    ax1.set_ylabel("Connections saved %",   color="#2ca02c")
    ax2.set_ylabel("On-time performance %", color="#1f77b4")
    ax1.set_title("Figure 8 — Tunability: α/β sweep (A2C)")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="center left", fontsize=8)
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── Agent factory ─────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HNH RL Training")
    parser.add_argument("--algo",     type=str, default="all",
                        choices=["all", "a2c", "dqn", "ac", "ddpg"])
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override training episodes (default: 25)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip Figure 8 alpha/beta sweep")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    if args.episodes is not None:
        config["n_train_episodes"] = args.episodes
        config["n_test_episodes"]  = max(1, args.episodes // 5)

    algos = ["a2c", "dqn", "ac", "ddpg"] if args.algo == "all" else [args.algo]

    print("\n" + "="*60)
    print("  Hold-No-Hold RL Training  [Real Simulator]")
    print(f"  Algorithms : {algos}")
    print(f"  Train eps  : {config['n_train_episodes']}")
    print(f"  Test eps   : {config['n_test_episodes']}")
    print(f"  Alpha={config['alpha']}  Beta={config['beta']}")
    print("="*60)

    # ── Train all algorithms ───────────────────────────────────────────────────
    train_results = {}
    agents        = {}

    for algo in algos:
        env    = build_env(alpha=config["alpha"], seed=config["seed"])
        agent  = build_agent(algo, config)
        result = train_agent(agent, env,
                             n_episodes=config["n_train_episodes"],
                             algo_name=algo, config=config)
        train_results[algo] = result
        agents[algo]        = agent

    # ── Evaluate baselines ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Evaluating baselines")
    print("="*60)

    baseline_results = {}
    for bname, bpolicy, bmax in [("no_hold",      no_hold_policy,      0),
                                   ("heuristic_15", heuristic_15_policy, 15),
                                   ("heuristic_30", heuristic_30_policy, 30)]:
        benv = build_env(alpha=config["alpha"], seed=config["seed"] + 100)
        baseline_results[bname] = evaluate_baseline(
            bpolicy, benv, config["n_test_episodes"], bname)

    # ── Evaluate RL agents ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Evaluating RL agents")
    print("="*60)

    eval_results = {}
    for algo in algos:
        tenv = build_env(alpha=config["alpha"], seed=config["seed"] + 200)
        eval_results[algo] = evaluate_agent(
            agents[algo], tenv, config["n_test_episodes"], algo)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Method':<15} {'Missed PAX':>12} {'OTP%':>8} {'AvgReward':>10}")
    print("  " + "-"*50)
    for name, res in baseline_results.items():
        print(f"  {name.upper():<15} {res['missed_pct']:>12.1f} "
              f"{res['otp_pct']:>8.1f} {res['avg_reward']:>10.4f}")
    for name, res in eval_results.items():
        print(f"  {name.upper():<15} {res['missed_pct']:>12.1f} "
              f"{res['otp_pct']:>8.1f} {res['avg_reward']:>10.4f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n  Generating plots...")
        plot_figure7(train_results, algos,
                     os.path.join(RESULTS_DIR, "figure7_rl_metrics.png"))
        plot_figure6(eval_results, baseline_results,
                     os.path.join(RESULTS_DIR, "figure6_business_metrics.png"))
        if not args.no_sweep:
            plot_figure8(config,
                         os.path.join(RESULTS_DIR, "figure8_tunability.png"))

    print("\n  All done. Results saved to ./results/")


if __name__ == "__main__":
    main()