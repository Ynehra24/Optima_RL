"""
train.py — Phase 2 Logistics Hold-or-Not-Hold RL Training
==========================================================

Trains and evaluates A2C, DQN, and AC on the LogisticsEnv.
Produces logistics-specific metrics and plots analogous to Phase 1 figures.

Usage:
    python train.py                  # train all algorithms
    python train.py --algo a2c       # train A2C only
    python train.py --episodes 10    # quick run
"""

import argparse, json, os, sys, time, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))   # phase-2/

from simulator.config       import SimConfig
from simulator.logistics_env import LogisticsEnv
from simulator.multi_hub_env import MultiHubLogisticsEnv
from agents.a2c import A2CAgent
from agents.dqn import DQNAgent
from agents.ac  import ACAgent

RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Hyperparameters (paper §6.2 values) ───────────────────────────────────────
DEFAULT_CONFIG = {
    "n_train_episodes": 25,
    "n_test_episodes":  5,
    "lr":               0.0001,
    "gamma":            0.8,
    "batch_size":       32,
    "alpha":            0.50,   # 0.5 = equal weight cargo vs punctuality (was 0.75)
    "beta":             0.75,
    "log_every":        200,
    "seed":             42,
}

ALGO_COLORS = {
    "a2c":          "#1f77b4",
    "dqn":          "#ff7f0e",
    "ac":           "#2ca02c",
    "no_hold":      "#7f7f7f",
    "heuristic_15": "#bcbd22",
    "heuristic_30": "#17becf",
}
DISPLAY = {
    "a2c": "A2C", "dqn": "DQN", "ac": "AC",
    "no_hold": "No Hold", "heuristic_15": "Heur.15", "heuristic_30": "Heur.30",
}


# ── Environment factory ──────────────────────────────────────────────────────────────────
def build_env(seed: int = 42, alpha: float = 0.75,
             multi_hub: bool = False):
    cfg = SimConfig(seed=seed, alpha=alpha,
                   n_hubs=10 if multi_hub else 1)
    if multi_hub:
        return MultiHubLogisticsEnv(cfg)
    return LogisticsEnv(cfg)


# ── Baselines ──────────────────────────────────────────────────────────────────
# State dim [22] = delta_slack (normalized).  Positive slack = feeder on time.
# Heuristics: if any feeder is delayed (slack < 0.3), hold.
def no_hold(s):   return 0

def heur15(s):
    """Hold 15 min if there are delayed feeders (delta_slack < threshold)."""
    delta_slack = s[22]   # normalized transfer slack
    return 3 if delta_slack < 0.3 else 0   # index 3 = 15 min

def heur30(s):
    """Hold 30 min if feeders are very late."""
    delta_slack = s[22]
    return 6 if delta_slack < 0.1 else (3 if delta_slack < 0.3 else 0)


# ── Agent factory ──────────────────────────────────────────────────────────────
def build_agent(algo: str, cfg: dict, state_dim: int = 34):
    kw = dict(lr=cfg["lr"], gamma=cfg["gamma"],
              batch_size=cfg["batch_size"], seed=cfg["seed"], state_dim=state_dim)
    if algo == "a2c": return A2CAgent(**kw)
    if algo == "dqn": return DQNAgent(**kw)
    if algo == "ac":  return ACAgent(**kw)
    raise ValueError(f"Unknown algo: {algo}")


# ── Training loop ──────────────────────────────────────────────────────────────
def train(agent, env: LogisticsEnv, n_eps: int, algo: str, cfg: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training {algo.upper()}  ({n_eps} episodes)")
    print(f"{'='*60}")

    all_rewards, ep_rewards = [], []
    t0 = time.time()
    gs = 0   # global step counter

    for ep in range(n_eps):
        obs, info = env.reset()
        epr = eps = 0

        while True:
            # Select action
            if algo == "dqn":
                action = agent.select_action(obs)
                val    = None
            else:
                action, val = agent.select_action(obs)

            # Step
            obs2, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Learn
            if algo == "dqn":
                agent.push(obs, action, reward, obs2, done)
                agent.update()
            else:   # a2c, ac
                agent.store(obs, action, reward, val, done)
                if eps % cfg["batch_size"] == 0 or done:
                    _, lv = agent.select_action(obs2)
                    agent.update(last_value=0.0 if done else lv)

            all_rewards.append(reward)
            obs = obs2
            epr += reward; eps += 1; gs += 1

            if gs % cfg["log_every"] == 0:
                m = agent.get_metrics()
                stats = env.get_episode_stats()
                print(f"  Ep {ep+1:3d} | Step {gs:6d} | "
                      f"AvgR(200): {m['avg_reward_1k']:.4f} | "
                      f"Loss: {m['avg_loss_1k']:.4f} | "
                      f"MissRate: {stats.missed_transfer_rate:.2%} | "
                      f"BayUtil: {stats.mean_bay_utilization:.2%} | "
                      f"Time: {time.time()-t0:.0f}s")

            if done:
                break

        mean = epr / max(eps, 1)
        ep_rewards.append(mean)
        stats = env.get_episode_stats()
        print(f"  Episode {ep+1:3d} done | Steps: {eps:4d} | "
              f"MeanRwd: {mean:.4f} | "
              f"Missed: {stats.n_transfers_missed} | "
              f"MissRate: {stats.missed_transfer_rate:.2%} | "
              f"OTP: {stats.OTP:.1f}%")

    print(f"\n  {algo.upper()} training complete in {time.time()-t0:.1f}s")
    return {
        "all_rewards":     np.array(all_rewards),
        "episode_rewards": np.array(ep_rewards),
        "losses":   np.array(agent.losses)   if agent.losses   else np.array([]),
        "q_values": np.array(agent.q_values) if agent.q_values else np.array([]),
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────
def _run_one_episode(env: LogisticsEnv, action_fn) -> dict:
    """Run one full episode with a fixed policy. Returns episode metrics."""
    obs, _ = env.reset()
    rewards = []
    holds = steps = 0

    while True:
        action = action_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        steps += 1
        if action > 0:
            holds += 1
        if done:
            break

    stats = env.get_episode_stats()
    return {
        "avg_reward":           float(np.mean(rewards)),
        "missed_transfers":     stats.n_transfers_missed,
        "transfer_success":     stats.n_transfers_success,
        "missed_rate":          stats.missed_transfer_rate,
        "OTP":                  stats.OTP,
        "mean_bay_utilization": stats.mean_bay_utilization,
        "mean_departure_delay": stats.total_departure_delay / max(stats.n_total_departures, 1),
        "holds_pct":            100.0 * holds / max(steps, 1),
    }


def _aggregate(results: list, name: str) -> dict:
    """Average metrics across multiple test episodes."""
    out = {}
    for k in results[0]:
        try:    out[k] = float(np.mean([r[k] for r in results]))
        except: out[k] = results[0][k]
    print(f"  {DISPLAY.get(name, name.upper()):<14} | "
          f"MissRate: {out.get('missed_rate', 0):5.2%}  | "
          f"OTP: {out.get('OTP', 0):5.1f}%  | "
          f"Missed: {out.get('missed_transfers', 0):6.0f}  | "
          f"BayUtil: {out.get('mean_bay_utilization', 0):5.2%}  | "
          f"Holds: {out.get('holds_pct', 0):5.1f}%")
    return out


def evaluate_agent(agent, env: LogisticsEnv, n_eps: int, algo: str) -> dict:
    print(f"\n  Evaluating {DISPLAY.get(algo, algo)} ({n_eps} episodes)...")
    fn = lambda s: agent.greedy_action(s)
    return _aggregate([_run_one_episode(env, fn) for _ in range(n_eps)], algo)


def evaluate_baseline(policy_fn, env: LogisticsEnv, n_eps: int, name: str) -> dict:
    print(f"  Evaluating baseline: {name}...")
    return _aggregate([_run_one_episode(env, policy_fn) for _ in range(n_eps)], name)


# ── Plotting ───────────────────────────────────────────────────────────────────
def smooth(x, w=200):
    if len(x) == 0: return x
    w = min(w, max(1, len(x) // 10))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_fig6(results: dict, save: str):
    """Missed transfers + Transfer success rate (analog of paper Figure 6)."""
    order   = ["a2c", "dqn", "ac", "heuristic_30", "heuristic_15", "no_hold"]
    methods = [m for m in order if m in results]
    missed  = [results[m].get("missed_transfers", 0) for m in methods]
    miss_r  = [results[m].get("missed_rate", 0) * 100 for m in methods]
    labels  = [DISPLAY.get(m, m) for m in methods]
    colors  = [ALGO_COLORS.get(m, "#aaa") for m in methods]

    x = np.arange(len(methods)); w = 0.35
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(x - w/2, missed, w, color=colors, alpha=0.85)
    ax2.bar(x + w/2, miss_r, w, color=colors, alpha=0.40,
            edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Missed cargo transfers (count)", color="#d62728", fontsize=11)
    ax2.set_ylabel("Missed transfer rate (%)",        color="#1f77b4", fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=10, fontsize=10)
    ax1.set_title("Figure 6 — Logistics: Missed transfers and miss rate", fontsize=12)
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(fc="gray", alpha=0.85, label="Missed transfers (left)"),
        Patch(fc="gray", alpha=0.40, ec="black", lw=0.8, label="Miss rate % (right)"),
    ], loc="upper right")
    ax1.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


def plot_fig6b(results: dict, save: str):
    """Bay utilization + departure delay (Phase 2 specific)."""
    order   = ["a2c", "dqn", "ac", "heuristic_30", "heuristic_15", "no_hold"]
    methods = [m for m in order if m in results]
    labels  = [DISPLAY.get(m, m) for m in methods]
    bay_u   = [results[m].get("mean_bay_utilization", 0) * 100 for m in methods]
    dep_d   = [results[m].get("mean_departure_delay", 0)        for m in methods]

    x = np.arange(len(methods)); w = 0.35
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(x - w/2, bay_u, w, color="#1f77b4", alpha=0.85, label="Bay utilization %")
    ax2.bar(x + w/2, dep_d, w, color="#ff7f0e", alpha=0.85, label="Departure delay (min)")
    ax1.set_ylabel("Mean bay utilization (%)", color="#1f77b4", fontsize=11)
    ax2.set_ylabel("Mean departure delay (min)", color="#ff7f0e", fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=10, fontsize=10)
    ax1.set_title("Figure 6b — Bay utilization and departure delay", fontsize=12)
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax1.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


def plot_fig7(train_results: dict, algos: list, save: str):
    """Training metrics: reward, value, loss (analog of paper Figure 7)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name in algos:
        res = train_results[name]
        c   = ALGO_COLORS.get(name, "gray")
        lbl = DISPLAY.get(name, name.upper())
        if len(res["all_rewards"]) > 0:
            axes[0].plot(smooth(res["all_rewards"]), label=lbl, color=c, linewidth=1.5)
        if len(res["q_values"]) > 0:
            axes[1].plot(smooth(res["q_values"], w=50), label=lbl, color=c, linewidth=1.5)
        if len(res["losses"]) > 0:
            axes[2].plot(smooth(res["losses"], w=50), label=lbl, color=c, linewidth=1.5)
    for ax, title, ylabel in zip(
        axes,
        ["(a) Average reward", "(b) Value / Q-estimate", "(c) Network loss"],
        ["Avg reward (smoothed)", "Avg value", "Avg loss"]
    ):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Step"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.suptitle("Figure 7 — Phase 2 RL training metrics (smoothed)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


def plot_fig8(cfg: dict, save: str, multi_hub: bool):
    """Alpha/beta tunability sweep (analog of paper Figure 8)."""
    print("\n  Running Figure 8: α/β sweep (A2C, 2 eps each)...")
    alphas = [0.10, 0.25, 0.50, 0.75, 1.00]
    betas  = [0.10, 0.50, 1.00]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Baseline miss rate
    base_env    = build_env(seed=42, multi_hub=multi_hub)
    base_result = _run_one_episode(base_env, no_hold)
    base_miss   = base_result.get("missed_transfers", 100)

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, beta in enumerate(betas):
        saved_l = []
        for alpha in alphas:
            env = build_env(seed=42, alpha=alpha, multi_hub=multi_hub)
            ag  = A2CAgent(lr=cfg["lr"], gamma=cfg["gamma"],
                           batch_size=cfg["batch_size"], seed=42)
            # Quick 2-episode training
            for _ in range(2):
                obs, _ = env.reset(); steps = 0
                while True:
                    a, v = ag.select_action(obs)
                    obs2, r, term, trunc, _ = env.step(a)
                    done = term or trunc
                    ag.store(obs, a, r, v, done)
                    if steps % cfg["batch_size"] == 0 or done:
                        _, lv = ag.select_action(obs2)
                        ag.update(last_value=0.0 if done else lv)
                    obs = obs2; steps += 1
                    if done: break

            ts    = _run_one_episode(build_env(seed=99, alpha=alpha, multi_hub=multi_hub),
                                     lambda s: ag.greedy_action(s))
            miss  = ts.get("missed_transfers", base_miss)
            saved = (base_miss - miss) / max(base_miss, 1) * 100
            saved_l.append(saved)

        ax.plot(alphas, saved_l, color=colors[j], marker="o",
                linewidth=1.5, label=f"β={beta}")

    ax.set_xlabel("Alpha (α) — Cargo vs Operator weight", fontsize=11)
    ax.set_ylabel("Transfer savings % vs No-Hold", fontsize=11)
    ax.set_title("Figure 8 — Tunability: α/β sweep (A2C, Logistics)", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


# ── Results Table ──────────────────────────────────────────────────────────────
def print_table(all_results: dict):
    order   = ["no_hold", "heuristic_15", "heuristic_30", "a2c", "dqn", "ac"]
    methods = [m for m in order if m in all_results]

    print("\n" + "=" * 88)
    print("  RESULTS TABLE — Phase 2 Logistics HNH")
    print("=" * 88)
    print(f"  {'Method':<14} {'MissRate':>10} {'OTP%':>7} {'Missed':>8} "
          f"{'BayUtil':>9} {'DepDly':>8} {'Holds%':>7}")
    print("  " + "-" * 70)
    for m in methods:
        r = all_results[m]
        print(f"  {DISPLAY.get(m, m):<14} "
              f"{r.get('missed_rate', 0):>9.2%} "
              f"{r.get('OTP', 0):>7.1f}% "
              f"{r.get('missed_transfers', 0):>8.0f} "
              f"{r.get('mean_bay_utilization', 0):>8.2%} "
              f"{r.get('mean_departure_delay', 0):>7.2f}m "
              f"{r.get('holds_pct', 0):>6.1f}%")

    # Delta vs no_hold
    nh = all_results.get("no_hold", {})
    nh_miss = nh.get("missed_transfers", 1)
    nh_otp  = nh.get("OTP", 0)
    print("\n" + "=" * 88)
    print("  SAVINGS vs No-Hold  (Transfer Savings ↑ better | OTP Delta ↑ better)")
    print("=" * 88)
    print(f"  {'Method':<8} {'Transfer Savings':>18} {'OTP Delta':>14}")
    print("  " + "-" * 44)
    for algo in ["a2c", "dqn", "ac"]:
        if algo not in all_results: continue
        m     = all_results[algo].get("missed_transfers", nh_miss)
        otp   = all_results[algo].get("OTP", nh_otp)
        pct   = (nh_miss - m) / max(nh_miss, 1) * 100
        otpd  = otp - nh_otp
        sign  = "+" if pct  > 0 else ""
        osign = "+" if otpd > 0 else ""
        print(f"  {DISPLAY[algo]:<8} {sign}{pct:>15.1f}%    {osign}{otpd:>8.1f} pp")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 2 Logistics HNH RL")
    parser.add_argument("--algo",      default="all",
                        choices=["all", "a2c", "dqn", "ac"])
    parser.add_argument("--episodes",  type=int, default=None)
    parser.add_argument("--no-plots",  action="store_true")
    parser.add_argument("--no-sweep",  action="store_true")
    parser.add_argument("--multi-hub", action="store_true",
                        help="Use 3-hub cascading network (true Phase 1 extension)")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.episodes:
        cfg["n_train_episodes"] = args.episodes
        cfg["n_test_episodes"]  = max(1, args.episodes // 5)

    algos = ["a2c", "dqn", "ac"] if args.algo == "all" else [args.algo]
    multi_hub = args.multi_hub
    state_dim = 42 if multi_hub else 34

    print("\n" + "=" * 60)
    print("  Phase 2 Logistics — Hold-or-Not-Hold RL")
    print(f"  Mode        : {'10-Hub FAF5 Mesh CASCADE ✓' if multi_hub else 'Single-Hub'}")
    print(f"  Algorithms  : {algos}")
    print(f"  State dim   : {state_dim}  |  Actions: 7  |  "
          f"Env: {'MultiHubLogisticsEnv' if multi_hub else 'LogisticsEnv'}")
    print(f"  Train eps   : {cfg['n_train_episodes']}  |  "
          f"Test eps: {cfg['n_test_episodes']}")
    print(f"  α={cfg['alpha']}  β={cfg['beta']}  "
          f"lr={cfg['lr']}  γ={cfg['gamma']}")
    print("=" * 60)

    # ── Train ──────────────────────────────────────────────────────────────────
    train_results = {}; agents = {}
    for algo in algos:
        env   = build_env(seed=cfg["seed"], alpha=cfg["alpha"], multi_hub=multi_hub)
        agent = build_agent(algo, cfg, state_dim=state_dim)
        result = train(agent, env, cfg["n_train_episodes"], algo, cfg)
        train_results[algo] = result
        agents[algo]        = agent
        try:
            with open(os.path.join(RESULTS_DIR, f"{algo}_agent.pkl"), "wb") as f:
                pickle.dump(agent, f)
        except Exception:
            pass

    # ── Evaluate baselines ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Evaluating baselines")
    print("=" * 60)
    bl_results = {}
    for name, fn in [("no_hold", no_hold), ("heuristic_15", heur15),
                     ("heuristic_30", heur30)]:
        env = build_env(seed=cfg["seed"] + 100, alpha=cfg["alpha"], multi_hub=multi_hub)
        bl_results[name] = evaluate_baseline(fn, env, cfg["n_test_episodes"], name)

    # ── Evaluate RL agents ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Evaluating RL agents")
    print("=" * 60)
    rl_results = {}
    for algo in algos:
        env = build_env(seed=cfg["seed"] + 100, alpha=cfg["alpha"], multi_hub=multi_hub)
        rl_results[algo] = evaluate_agent(agents[algo], env,
                                          cfg["n_test_episodes"], algo)

    all_results = {**bl_results, **rl_results}
    print_table(all_results)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    try:
        out = {}
        for k, v in all_results.items():
            out[k] = {kk: (float(vv) if isinstance(vv, (int, float, np.floating))
                           else vv)
                      for kk, vv in v.items()}
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Summary saved → {RESULTS_DIR}/summary.json")
    except Exception as e:
        print(f"  Warning: could not save JSON: {e}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n  Generating plots...")
        plot_fig6(all_results,
                  os.path.join(RESULTS_DIR, "figure6_missed_transfers.png"))
        plot_fig6b(all_results,
                   os.path.join(RESULTS_DIR, "figure6b_bay_delay.png"))
        plot_fig7(train_results, algos,
                  os.path.join(RESULTS_DIR, "figure7_rl_metrics.png"))
        if not args.no_sweep:
            plot_fig8(cfg, os.path.join(RESULTS_DIR, "figure8_tunability.png"))

    print("\n  All done. Results in ./results/")


if __name__ == "__main__":
    main()
