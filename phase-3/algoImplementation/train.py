"""
train.py — Phase 3 DAG Scheduling Hold-or-Not-Hold RL Training.

Trains and evaluates A2C, DQN, AC, and DDPG on DAGSchedulingSimulator.  The
shape is intentionally close to phase-1/2 training scripts so phase comparisons
remain easy:

    python phase-3/algoImplementation/train.py
    python phase-3/algoImplementation/train.py --algo a2c --episodes 2
    python phase-3/algoImplementation/train.py --duration-s 7200 --max-decisions 1000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from typing import Callable, Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_PHASE3 = os.path.abspath(os.path.join(_HERE, ".."))

os.environ.setdefault("MPLCONFIGDIR", os.path.join(_HERE, ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_HERE, ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, _HERE)
sys.path.insert(0, _PHASE3)

from simulator.config import SimConfig
from simulator.simulator import DAGSchedulingSimulator
from agents.a2c import A2CAgent
from agents.dqn import DQNAgent
from agents.ac import ACAgent
from agents.ddpg import DDPGAgent


RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

STATE_DIM = 88
ACTION_DIM = 7
ALL_ALGOS = ["a2c", "dqn", "ac", "ddpg"]

DEFAULT_CONFIG = {
    "n_train_episodes": 8,
    "n_test_episodes": 3,
    "lr": 0.0001,
    "gamma": 0.8,
    "batch_size": 32,
    "alpha": 0.75,
    "beta": 0.75,
    "log_every": 100,
    "seed": 42,
    "episode_duration_s": 3_600.0,
    "max_hnh_decisions": 500,
}

ALGO_COLORS = {
    "a2c": "#1f77b4",
    "dqn": "#ff7f0e",
    "ac": "#2ca02c",
    "ddpg": "#d62728",
    "no_hold": "#7f7f7f",
    "heuristic": "#17becf",
    "gpu_guard": "#9467bd",
}

DISPLAY = {
    "a2c": "A2C",
    "dqn": "DQN",
    "ac": "AC",
    "ddpg": "DDPG",
    "no_hold": "No Hold",
    "heuristic": "Heuristic",
    "gpu_guard": "GPU Guard",
}


def build_env(seed: int, cfg_dict: Dict) -> DAGSchedulingSimulator:
    """Construct a calibrated Phase 3 simulator for training/evaluation."""
    cfg = SimConfig(random_seed=seed)
    cfg.alpha = cfg_dict["alpha"]
    cfg.beta = cfg_dict["beta"]
    cfg.episode_duration_s = float(cfg_dict["episode_duration_s"])
    cfg.max_hnh_decisions = int(cfg_dict["max_hnh_decisions"])
    return DAGSchedulingSimulator(cfg)


def sanitize_state(state: np.ndarray) -> np.ndarray:
    """Keep simulator edge cases from poisoning small NumPy networks."""
    clean = np.nan_to_num(
        np.asarray(state, dtype=np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return np.clip(clean, -5.0, 5.0).astype(np.float32)


def no_hold_policy(state: np.ndarray, info: Dict) -> int:
    return 0


def heuristic_policy(state: np.ndarray, info: Dict) -> int:
    """Mirror simulator.run_episode('heuristic') with train.py-compatible API."""
    upstream_delay = float(info.get("upstream_delay_s", 0.0))
    gpu_util = float(info.get("gpu_util", 0.0))
    if upstream_delay > 0.0 and gpu_util < 0.85:
        return 2
    return 0


def gpu_guard_policy(state: np.ndarray, info: Dict) -> int:
    """Hold longer for delayed GPU-friendly windows, avoid holds under pressure."""
    upstream_delay = float(info.get("upstream_delay_s", 0.0))
    cpu_util = float(info.get("cpu_util", 0.0))
    gpu_util = float(info.get("gpu_util", 0.0))
    if upstream_delay <= 0.0:
        return 0
    if max(cpu_util, gpu_util) > 0.90:
        return 0
    if upstream_delay > 120.0 and gpu_util < 0.70:
        return 4
    if upstream_delay > 30.0:
        return 3
    return 2


def build_agent(algo: str, cfg: Dict):
    kwargs = dict(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        lr=cfg["lr"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
    )
    if algo == "a2c":
        return A2CAgent(**kwargs)
    if algo == "dqn":
        return DQNAgent(**kwargs)
    if algo == "ac":
        return ACAgent(**kwargs)
    if algo == "ddpg":
        return DDPGAgent(
            state_dim=STATE_DIM,
            action_dim=1,
            lr_actor=cfg["lr"],
            lr_critic=cfg["lr"],
            gamma=cfg["gamma"],
            batch_size=cfg["batch_size"],
            seed=cfg["seed"],
        )
    raise ValueError(f"Unknown algo: {algo}")


def train(agent, env: DAGSchedulingSimulator, n_eps: int, algo: str, cfg: Dict) -> Dict:
    print("\n" + "=" * 68)
    print(f"  Training {algo.upper()} on Phase 3 DAG simulator ({n_eps} episodes)")
    print("=" * 68)

    all_rewards: List[float] = []
    ep_rewards: List[float] = []
    ep_lengths: List[int] = []
    t0 = time.time()
    global_step = 0

    for ep in range(n_eps):
        obs, info = env.reset(seed=cfg["seed"] + ep)
        obs = sanitize_state(obs)
        episode_reward = 0.0
        steps = 0

        while True:
            if algo in ("dqn", "ddpg"):
                action = agent.select_action(obs)
                value = None
            else:
                action, value = agent.select_action(obs)

            obs2, reward, done, info2 = env.step(action)
            obs2 = sanitize_state(obs2)

            if algo in ("dqn", "ddpg"):
                agent.push(obs, action, reward, obs2, done)
                agent.update()
            else:
                agent.store(obs, action, reward, value, done)
                if steps % cfg["batch_size"] == 0 or done:
                    _, bootstrap = agent.select_action(obs2)
                    agent.update(last_value=0.0 if done else bootstrap)

            all_rewards.append(float(reward))
            episode_reward += float(reward)
            obs, info = obs2, info2
            steps += 1
            global_step += 1

            if global_step % cfg["log_every"] == 0:
                metrics = agent.get_metrics()
                summary = env.metrics.summary()
                print(
                    f"  Ep {ep + 1:3d} | Step {global_step:6d} | "
                    f"AvgR: {metrics['avg_reward_1k']:.4f} | "
                    f"Loss: {metrics['avg_loss_1k']:.4f} | "
                    f"Hold%: {summary['hold_rate_pct']:5.1f} | "
                    f"Evict%: {summary['evicted_pct']:5.1f} | "
                    f"Done%: {summary['completed_pct']:5.1f} | "
                    f"{time.time() - t0:.0f}s"
                )

            if done:
                break

        mean_reward = episode_reward / max(steps, 1)
        ep_rewards.append(mean_reward)
        ep_lengths.append(steps)
        summary = env.metrics.summary()
        print(
            f"  Episode {ep + 1:3d} done | Steps: {steps:4d} | "
            f"MeanR: {mean_reward:.4f} | Hold%: {summary['hold_rate_pct']:5.1f} | "
            f"Stalls: {summary['pipeline_stalls']:5d} | "
            f"Reward: {summary['total_reward']:8.3f}"
        )

    return {
        "all_rewards": np.array(all_rewards, dtype=np.float32),
        "episode_rewards": np.array(ep_rewards, dtype=np.float32),
        "episode_lengths": np.array(ep_lengths, dtype=np.int32),
        "losses": np.array(agent.losses, dtype=np.float32) if agent.losses else np.array([]),
        "q_values": np.array(agent.q_values, dtype=np.float32) if agent.q_values else np.array([]),
    }


def run_episode(env: DAGSchedulingSimulator, action_fn: Callable[[np.ndarray, Dict], int]) -> Dict:
    obs, info = env.reset()
    obs = sanitize_state(obs)
    rewards: List[float] = []
    holds = 0
    steps = 0

    while True:
        action = int(action_fn(obs, info))
        obs, reward, done, info = env.step(action)
        obs = sanitize_state(obs)
        rewards.append(float(reward))
        holds += 1 if action > 0 else 0
        steps += 1
        if done:
            break

    summary = env.metrics.summary()
    summary["avg_reward"] = float(np.mean(rewards)) if rewards else 0.0
    summary["holds_pct"] = 100.0 * holds / max(steps, 1)
    summary["steps"] = steps
    return summary


def evaluate_agent(agent, cfg: Dict, n_eps: int, algo: str) -> Dict:
    print(f"  Evaluating {DISPLAY.get(algo, algo)} ({n_eps} episodes)...")
    results = []
    for i in range(n_eps):
        env = build_env(seed=cfg["seed"] + 500 + i, cfg_dict=cfg)
        results.append(run_episode(env, lambda state, info: agent.greedy_action(state)))
    return aggregate(results, algo)


def evaluate_baseline(policy_fn: Callable[[np.ndarray, Dict], int], cfg: Dict,
                      n_eps: int, name: str) -> Dict:
    print(f"  Evaluating baseline: {DISPLAY.get(name, name)}...")
    results = []
    for i in range(n_eps):
        env = build_env(seed=cfg["seed"] + 700 + i, cfg_dict=cfg)
        results.append(run_episode(env, policy_fn))
    return aggregate(results, name)


def aggregate(results: List[Dict], name: str) -> Dict:
    out = {}
    keys = sorted({key for row in results for key in row})
    for key in keys:
        vals = [row[key] for row in results if key in row]
        if vals and all(isinstance(v, (int, float, np.floating)) for v in vals):
            out[key] = float(np.mean(vals))
        elif vals:
            out[key] = vals[0]

    print(
        f"    {DISPLAY.get(name, name):<10} | "
        f"Done%: {out.get('completed_pct', 0):5.1f} | "
        f"Evict%: {out.get('evicted_pct', 0):5.1f} | "
        f"Hold%: {out.get('hold_rate_pct', 0):5.1f} | "
        f"Stalls: {out.get('pipeline_stalls', 0):7.1f} | "
        f"Reward: {out.get('total_reward', 0):8.3f}"
    )
    return out


def smooth(values: np.ndarray, width: int = 100) -> np.ndarray:
    if len(values) == 0:
        return values
    width = min(width, max(1, len(values) // 5))
    return np.convolve(values, np.ones(width) / width, mode="valid")


def plot_training_curves(train_results: Dict, algos: List[str], save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for algo in algos:
        res = train_results[algo]
        color = ALGO_COLORS.get(algo, "gray")
        label = DISPLAY.get(algo, algo.upper())
        if len(res["all_rewards"]):
            axes[0].plot(smooth(res["all_rewards"]), color=color, label=label)
        if len(res["episode_rewards"]):
            axes[1].plot(res["episode_rewards"], color=color, marker="o", label=label)
        if len(res["losses"]):
            axes[2].plot(smooth(res["losses"], width=50), color=color, label=label)

    for ax, title, ylabel in zip(
        axes,
        ["Step reward", "Episode mean reward", "Network loss"],
        ["Reward", "Mean reward", "Loss"],
    ):
        ax.set_title(title)
        ax.set_xlabel("Step / episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle("Phase 3 DAG HNH RL Training")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_eval_bars(results: Dict, save_path: str) -> None:
    order = ["no_hold", "heuristic", "gpu_guard", "a2c", "dqn", "ac", "ddpg"]
    methods = [name for name in order if name in results]
    labels = [DISPLAY.get(name, name) for name in methods]
    stalls = [results[name].get("pipeline_stalls", 0.0) for name in methods]
    evicts = [results[name].get("evicted_pct", 0.0) for name in methods]
    colors = [ALGO_COLORS.get(name, "gray") for name in methods]

    x = np.arange(len(methods))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(x - width / 2, stalls, width, color=colors, alpha=0.85)
    ax2.bar(x + width / 2, evicts, width, color=colors, alpha=0.35,
            edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Pipeline stalls")
    ax2.set_ylabel("Evicted tasks (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=10)
    ax1.set_title("Phase 3 DAG: pipeline stalls and evictions")
    ax1.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def print_table(results: Dict) -> None:
    order = ["no_hold", "heuristic", "gpu_guard", "a2c", "dqn", "ac", "ddpg"]
    methods = [name for name in order if name in results]
    print("\n" + "=" * 92)
    print("  RESULTS TABLE — Phase 3 DAG Scheduling HNH")
    print("=" * 92)
    print(
        f"  {'Method':<12} {'Done%':>8} {'Failed%':>9} {'Evict%':>8} "
        f"{'Hold%':>8} {'Stalls':>8} {'AvgR':>9} {'Reward':>10}"
    )
    print("  " + "-" * 82)
    for name in methods:
        row = results[name]
        print(
            f"  {DISPLAY.get(name, name):<12} "
            f"{row.get('completed_pct', 0):>7.1f} "
            f"{row.get('failed_pct', 0):>8.1f} "
            f"{row.get('evicted_pct', 0):>7.1f} "
            f"{row.get('hold_rate_pct', 0):>7.1f} "
            f"{row.get('pipeline_stalls', 0):>8.1f} "
            f"{row.get('avg_reward', 0):>9.4f} "
            f"{row.get('total_reward', 0):>10.3f}"
        )


def save_results(results: Dict, train_results: Dict, cfg: Dict) -> None:
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    payload = {
        "config": cfg,
        "evaluation": {
            name: {
                key: (float(value) if isinstance(value, (int, float, np.floating)) else value)
                for key, value in row.items()
            }
            for name, row in results.items()
        },
        "training": {
            name: {
                "num_rewards": int(len(row["all_rewards"])),
                "episode_rewards": row["episode_rewards"].astype(float).tolist(),
                "episode_lengths": row["episode_lengths"].astype(int).tolist(),
            }
            for name, row in train_results.items()
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 DAG HNH RL")
    parser.add_argument("--algo", default="all", choices=["all"] + ALL_ALGOS)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--max-decisions", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.episodes is not None:
        cfg["n_train_episodes"] = args.episodes
        cfg["n_test_episodes"] = max(1, args.episodes // 2)
    if args.duration_s is not None:
        cfg["episode_duration_s"] = args.duration_s
    if args.max_decisions is not None:
        cfg["max_hnh_decisions"] = args.max_decisions

    algos = ALL_ALGOS if args.algo == "all" else [args.algo]

    print("\n" + "=" * 68)
    print("  Phase 3 DAG Scheduling — Hold-or-Not-Hold RL")
    print(f"  Algorithms       : {algos}")
    print(f"  State dim/actions: {STATE_DIM}/{ACTION_DIM}")
    print(f"  Train/Test eps   : {cfg['n_train_episodes']}/{cfg['n_test_episodes']}")
    print(f"  Episode cap      : {cfg['episode_duration_s']}s, {cfg['max_hnh_decisions']} HNH")
    print(f"  alpha={cfg['alpha']} beta={cfg['beta']} lr={cfg['lr']} gamma={cfg['gamma']}")
    print("=" * 68)

    agents = {}
    train_results = {}
    for algo in algos:
        env = build_env(seed=cfg["seed"], cfg_dict=cfg)
        agent = build_agent(algo, cfg)
        train_results[algo] = train(agent, env, cfg["n_train_episodes"], algo, cfg)
        agents[algo] = agent
        try:
            with open(os.path.join(RESULTS_DIR, f"{algo}_agent.pkl"), "wb") as f:
                pickle.dump(agent, f)
        except Exception as exc:
            print(f"  Warning: could not pickle {algo}: {exc}")

    print("\n" + "=" * 68)
    print("  Evaluating baselines")
    print("=" * 68)
    results = {
        "no_hold": evaluate_baseline(no_hold_policy, cfg, cfg["n_test_episodes"], "no_hold"),
        "heuristic": evaluate_baseline(heuristic_policy, cfg, cfg["n_test_episodes"], "heuristic"),
        "gpu_guard": evaluate_baseline(gpu_guard_policy, cfg, cfg["n_test_episodes"], "gpu_guard"),
    }

    print("\n" + "=" * 68)
    print("  Evaluating RL agents")
    print("=" * 68)
    for algo, agent in agents.items():
        results[algo] = evaluate_agent(agent, cfg, cfg["n_test_episodes"], algo)

    print_table(results)
    save_results(results, train_results, cfg)

    if not args.no_plots:
        plot_training_curves(
            train_results,
            algos,
            os.path.join(RESULTS_DIR, "phase3_training_curves.png"),
        )
        plot_eval_bars(results, os.path.join(RESULTS_DIR, "phase3_eval_bars.png"))

    print("\n  All done. Results are in phase-3/algoImplementation/results/")


if __name__ == "__main__":
    main()
