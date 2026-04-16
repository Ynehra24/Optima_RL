"""
train.py — Hold-No-Hold RL (Final Version)
==========================================
"To hold or not to hold?" — Malladi et al., AAMAS 2021

Based on the working version that produced:
  no_hold:     2670 missed, 88.6% OTP
  heuristic_15: 2720 missed, 88.4% OTP
  a2c:         2791 missed, 88.3% OTP
  dqn:         2916 missed, 87.7% OTP
  ac:          2729 missed, 88.4% OTP

KEY: Uses sim_reward directly from env.step() — this is the local reward
computed by the simulator's own RewardCalculator at decision time.
DO NOT recompute reward after step() — the context has moved on.

Figures reproduced:
  Figure 6  — Missed PAX + OTP bar chart
  Figure 6c — Arrival/departure delays
  Figure 7  — RL training metrics (reward, value, loss)
  Figure 8  — Alpha/beta tunability sweep (A2C)
"""

import argparse, json, os, sys, time, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from simulator.simulator import AirlineNetworkSimulator
from simulator.config    import SimConfig
from agents.dqn  import DQNAgent
from agents.a2c  import A2CAgent
from agents.ac   import ACAgent
from agents.ddpg import DDPGAgent

# ── Constants (paper §6.2) ─────────────────────────────────────────────────────
ALPHA        = 0.75
BETA         = 0.75
STATE_DIM    = 17
HOLD_ACTIONS = [0, 5, 10, 15, 20, 25, 30]

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

ALGO_COLORS = {
    "a2c":          "#1f77b4",
    "dqn":          "#ff7f0e",
    "ac":           "#2ca02c",
    "ddpg":         "#d62728",
    "no_hold":      "#7f7f7f",
    "heuristic_15": "#bcbd22",
    "heuristic_30": "#17becf",
}
DISPLAY = {
    "a2c":"A2C", "dqn":"DQN", "ac":"AC", "ddpg":"DDPG",
    "no_hold":"No Hold", "heuristic_15":"Heur.15", "heuristic_30":"Heur.30",
}


# ── State ──────────────────────────────────────────────────────────────────────
def ctx2state(ctx):
    if hasattr(ctx, "flight_id") and ctx.flight_id == "DONE":
        return np.zeros(STATE_DIM, dtype=np.float32)
    s = ctx.to_array().copy()
    s[16] = float(s[16]) / 30.0   # tau* normalised to [0,1]
    return np.clip(s, 0.0, 1.0).astype(np.float32)


# ── Environment ────────────────────────────────────────────────────────────────
def build_env(seed=42, alpha=ALPHA):
    cfg = SimConfig()
    cfg.alpha       = alpha
    cfg.random_seed = seed
    return AirlineNetworkSimulator(cfg)


# ── Baselines (paper §6.2) ─────────────────────────────────────────────────────
def no_hold(s):  return 0
def heur15(s):   return min(round(s[16] * 6), 3)   # cap at 15 min = index 3
def heur30(s):   return min(round(s[16] * 6), 6)   # cap at 30 min = index 6


# ── Agent factory ──────────────────────────────────────────────────────────────
def build_agent(algo, cfg):
    kw = dict(lr=cfg["lr"], gamma=cfg["gamma"],
              batch_size=cfg["batch_size"], seed=cfg["seed"])
    if algo == "dqn":  return DQNAgent(**kw)
    if algo == "a2c":  return A2CAgent(**kw)
    if algo == "ac":   return ACAgent(**kw)
    if algo == "ddpg": return DDPGAgent(lr_actor=kw["lr"], lr_critic=kw["lr"],
                                        gamma=kw["gamma"], batch_size=kw["batch_size"],
                                        seed=kw["seed"])
    raise ValueError(algo)


# ── Training ───────────────────────────────────────────────────────────────────
def train(agent, env, n_eps, algo, cfg):
    print(f"\n{'='*60}")
    print(f"  Training {algo.upper()}  ({n_eps} episodes)")
    print(f"{'='*60}")

    all_rewards = []
    ep_rewards  = []
    t0 = time.time()
    gs = 0

    for ep in range(n_eps):
        ctx, _ = env.reset()
        state  = ctx2state(ctx)
        epr = eps = 0

        while True:
            # ── Select action ────────────────────────────────────────────────
            if algo == "dqn":
                action = agent.select_action(state); val = None
            elif algo in ("a2c", "ac"):
                action, val = agent.select_action(state)
            else:
                action = agent.discrete_action(state); val = None

            # ── Step — use sim_reward directly (computed at decision time) ───
            ctx, reward, done, info = env.step(action)
            next_state = ctx2state(ctx)

            epr += reward; eps += 1; gs += 1

            # ── Learn ────────────────────────────────────────────────────────
            if algo == "dqn":
                agent.push(state, action, reward, next_state, done)
                agent.update()
            elif algo in ("a2c", "ac"):
                agent.store(state, action, reward, val, done)
                if eps % cfg["batch_size"] == 0 or done:
                    _, lv = agent.select_action(next_state)
                    agent.update(last_value=0.0 if done else lv)
            else:
                agent.push(state, action, reward, next_state, done)
                agent.update()

            all_rewards.append(reward)
            state = next_state

            # ── Log ──────────────────────────────────────────────────────────
            if gs % cfg["log_every"] == 0:
                m = agent.get_metrics()
                print(f"  Ep {ep+1:3d} | Step {gs:7d} | "
                      f"AvgR(1k): {m['avg_reward_1k']:.4f} | "
                      f"Loss: {m['avg_loss_1k']:.4f} | "
                      f"Time: {time.time()-t0:.0f}s")
            if done:
                break

        mean = epr / eps
        ep_rewards.append(mean)
        print(f"  Episode {ep+1:3d} done | Steps: {eps:5d} | "
              f"Mean reward: {mean:.4f} | "
              f"Missed: {info.get('missed_connections','?')} | "
              f"OTP: {info.get('OTP','?')}")

    print(f"\n  {algo.upper()} training complete in {time.time()-t0:.1f}s")
    return {
        "all_rewards":     np.array(all_rewards),
        "episode_rewards": np.array(ep_rewards),
        "losses":   np.array(agent.losses)   if agent.losses   else np.array([]),
        "q_values": np.array(agent.q_values) if agent.q_values else np.array([]),
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────
def _run_one(env, action_fn):
    """Run one episode, return simulator summary + extras."""
    ctx, _ = env.reset()
    state  = ctx2state(ctx)
    rewards = []
    holds = steps = 0

    while True:
        action = action_fn(state)
        ctx, reward, done, info = env.step(action)
        state = ctx2state(ctx)
        rewards.append(reward)
        steps += 1
        if action > 0:
            holds += 1
        if done:
            break

    summary = env.metrics.summary()
    summary["avg_reward"] = float(np.mean(rewards))
    summary["holds_pct"]  = 100.0 * holds / max(steps, 1)
    # Normalise OTP if returned as fraction
    if summary.get("OTP", 0) <= 1.0:
        summary["OTP"] = summary["OTP"] * 100.0
    return summary


def _aggregate(summaries, name):
    out = {}
    for k in summaries[0]:
        try:    out[k] = float(np.mean([s[k] for s in summaries]))
        except: out[k] = summaries[0][k]
    print(f"  {DISPLAY.get(name,name.upper()):<12} | "
          f"OTP: {out.get('OTP',0):5.1f}%  |  "
          f"Missed: {out.get('missed_connections',0):6.0f}  |  "
          f"Arr: {out.get('avg_arrival_delay_min',0):5.2f}m  |  "
          f"Dep: {out.get('avg_departure_delay_min',0):5.2f}m  |  "
          f"Holds: {out.get('holds_pct',0):5.1f}%")
    return out


def evaluate_agent(agent, env, n_eps, algo):
    print(f"\n  Evaluating {DISPLAY.get(algo,algo)} ({n_eps} episodes)...")
    fn = (lambda s: agent.discrete_action(s)) if algo == "ddpg" \
         else (lambda s: agent.greedy_action(s))
    return _aggregate([_run_one(env, fn) for _ in range(n_eps)], algo)


def evaluate_baseline(policy_fn, env, n_eps, name):
    print(f"  Evaluating baseline: {name}...")
    return _aggregate([_run_one(env, policy_fn) for _ in range(n_eps)], name)


# ── Delta computation ──────────────────────────────────────────────────────────
def compute_deltas(rl_results, bl_results):
    out = {}
    for algo, res in rl_results.items():
        out[algo] = {}
        for base, bres in bl_results.items():
            bm  = bres.get("missed_connections", 1)
            am  = res.get("missed_connections", 0)
            red = (bm - am) / max(bm, 1) * 100
            out[algo][base] = {
                "missed_reduction_%": round(red, 1),
                "otp_delta_%":        round(res.get("OTP",0) - bres.get("OTP",0), 1),
            }
    return out


# ── Summary table ──────────────────────────────────────────────────────────────
def print_table(all_results, dlt):
    order   = ["no_hold","heuristic_15","heuristic_30","a2c","dqn","ac","ddpg"]
    methods = [m for m in order if m in all_results]

    print("\n" + "="*78)
    print("  RESULTS TABLE (paper §7.2)")
    print("="*78)
    print(f"  {'Method':<14} {'Missed PAX':>11} {'OTP%':>7} "
          f"{'Arr Dly':>9} {'Dep Dly':>9} {'Holds%':>7}")
    print("  " + "-"*65)
    for m in methods:
        r = all_results[m]
        print(f"  {DISPLAY.get(m,m):<14} "
              f"{r.get('missed_connections',0):>11.0f} "
              f"{r.get('OTP',0):>7.1f}% "
              f"{r.get('avg_arrival_delay_min',0):>8.2f}m "
              f"{r.get('avg_departure_delay_min',0):>8.2f}m "
              f"{r.get('holds_pct',0):>6.1f}%")

    print("\n" + "="*78)
    print("  MISSED CONNECTION REDUCTION vs BASELINES")
    print("  Paper target: A2C ~50% reduction vs Heuristic-15 (§7.2)")
    print("="*78)
    print(f"  {'Method':<10} {'vs No-Hold':>12} {'vs Heur-15':>12} {'vs Heur-30':>12}")
    print("  " + "-"*50)
    for algo in [m for m in ["a2c","dqn","ac","ddpg"] if m in dlt]:
        d   = dlt[algo]
        nh  = d.get("no_hold",      {}).get("missed_reduction_%", 0)
        h15 = d.get("heuristic_15", {}).get("missed_reduction_%", 0)
        h30 = d.get("heuristic_30", {}).get("missed_reduction_%", 0)
        print(f"  {DISPLAY.get(algo,algo):<10} "
              f"{nh:>10.1f}%  {h15:>10.1f}%  {h30:>10.1f}%")
    print("\n  Positive = RL reduces more missed connections than baseline")


# ── Plotting ───────────────────────────────────────────────────────────────────
def smooth(x, w=1000):
    if len(x) == 0: return x
    w = min(w, max(1, len(x) // 10))
    return np.convolve(x, np.ones(w)/w, mode="valid")


def plot_fig6(results, save):
    """Figure 6a: Missed PAX + OTP per method."""
    order   = ["a2c","dqn","ac","ddpg","heuristic_30","heuristic_15","no_hold"]
    methods = [m for m in order if m in results]
    missed  = [results[m].get("missed_connections", 0) for m in methods]
    otp     = [results[m].get("OTP", 0)                for m in methods]
    labels  = [DISPLAY.get(m, m) for m in methods]
    colors  = [ALGO_COLORS.get(m, "#aaa") for m in methods]

    x = np.arange(len(methods)); w = 0.35
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(x - w/2, missed, w, color=colors, alpha=0.85)
    ax2.bar(x + w/2, otp,    w, color=colors, alpha=0.40,
            edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Missed connections (PAX)", color="#d62728", fontsize=11)
    ax2.set_ylabel("On-time performance (%)",  color="#1f77b4", fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=10, fontsize=10)
    ax1.set_title("Figure 6 — Business metrics: missed PAX and OTP", fontsize=12)
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(fc="gray", alpha=0.85, label="Missed PAX (left axis)"),
        Patch(fc="gray", alpha=0.40, ec="black", lw=0.8, label="OTP % (right axis)"),
    ], loc="upper right")
    ax1.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


def plot_fig6c(results, save):
    """Figure 6c: Arrival and departure delays."""
    order   = ["a2c","dqn","ac","ddpg","heuristic_30","heuristic_15","no_hold"]
    methods = [m for m in order if m in results]
    labels  = [DISPLAY.get(m, m) for m in methods]
    arr = [results[m].get("avg_arrival_delay_min",   0) for m in methods]
    dep = [results[m].get("avg_departure_delay_min", 0) for m in methods]

    x = np.arange(len(methods)); w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, arr, w, color="#1f77b4", alpha=0.85, label="Arrival delay")
    ax.bar(x + w/2, dep, w, color="#ff7f0e", alpha=0.85, label="Departure delay")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Average delay (minutes)", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10, fontsize=10)
    ax.set_title("Figure 6c — Average arrival and departure delays", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


def plot_fig7(train_results, algos, save):
    """Figure 7: RL training metrics — reward, value, loss."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name in algos:
        res = train_results[name]
        c   = ALGO_COLORS.get(name, "gray")
        lbl = DISPLAY.get(name, name.upper())
        if len(res["all_rewards"]) > 0:
            axes[0].plot(smooth(res["all_rewards"]),
                         label=lbl, color=c, linewidth=1.5)
        if len(res["q_values"]) > 0:
            axes[1].plot(smooth(res["q_values"], w=100),
                         label=lbl, color=c, linewidth=1.5)
        if len(res["losses"]) > 0:
            axes[2].plot(smooth(res["losses"], w=100),
                         label=lbl, color=c, linewidth=1.5)
    for ax, t, y in zip(axes,
                         ["(a) Average reward", "(b) Value / Q", "(c) Neural net loss"],
                         ["Avg reward (smoothed)", "Avg value", "Avg loss"]):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel(y)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Figure 7 — RL metrics (smoothed over 1000 steps)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


def plot_fig8(cfg, save):
    """Figure 8: Alpha/beta tunability sweep using A2C."""
    print("\n  Running Figure 8: α/β sweep (A2C, 2 eps each)...")
    alphas = [0.10, 0.25, 0.50, 0.75, 1.00]
    betas  = [0.10, 0.50, 1.00]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Baseline missed connections for computing savings %
    base_env = build_env(seed=42)
    base_sum = _run_one(base_env, no_hold)
    base_m   = base_sum.get("missed_connections", 2700)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    for j, beta in enumerate(betas):
        saved_l = []; otp_l = []
        for alpha in alphas:
            env = build_env(seed=42, alpha=alpha)
            ag  = A2CAgent(lr=cfg["lr"], gamma=cfg["gamma"],
                           batch_size=cfg["batch_size"], seed=42)
            # Short 2-episode training
            for _ in range(2):
                ctx, _ = env.reset(); st = ctx2state(ctx); steps = 0
                while True:
                    a, v = ag.select_action(st)
                    ctx, r, done, _ = env.step(a); ns = ctx2state(ctx)
                    ag.store(st, a, r, v, done)
                    if steps % cfg["batch_size"] == 0 or done:
                        _, lv = ag.select_action(ns)
                        ag.update(last_value=0.0 if done else lv)
                    st = ns; steps += 1
                    if done: break

            ts = _run_one(build_env(seed=99, alpha=alpha),
                          lambda s: ag.greedy_action(s))
            m = ts.get("missed_connections", base_m)
            saved_l.append((base_m - m) / max(base_m, 1) * 100)
            otp_l.append(ts.get("OTP", 0))

        ax1.plot(alphas, saved_l, color=colors[j], marker="o",
                 label=f"β={beta} (saved)")
        ax2.plot(alphas, otp_l,   color=colors[j], marker="s",
                 linestyle="--", label=f"β={beta} (OTP)")

    ax1.set_xlabel("Alpha (α)", fontsize=11)
    ax1.set_ylabel("Connections saved % vs No-Hold", fontsize=11)
    ax2.set_ylabel("On-time performance %",           fontsize=11)
    ax1.set_title("Figure 8 — Tunability: α/β sweep (A2C)", fontsize=12)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1+l2, lb1+lb2, loc="center left", fontsize=8)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HNH RL Training")
    parser.add_argument("--algo",     default="all",
                        choices=["all","a2c","dqn","ac","ddpg"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.episodes:
        cfg["n_train_episodes"] = args.episodes
        cfg["n_test_episodes"]  = max(1, args.episodes // 5)

    algos = ["a2c","dqn","ac","ddpg"] if args.algo == "all" else [args.algo]

    print("\n" + "="*60)
    print("  Hold-No-Hold RL  [Real Simulator + Delay Tree Reward]")
    print(f"  Algorithms  : {algos}")
    print(f"  Train eps   : {cfg['n_train_episodes']}  |  "
          f"Test eps: {cfg['n_test_episodes']}")
    print(f"  α={cfg['alpha']}  β={cfg['beta']}  "
          f"lr={cfg['lr']}  γ={cfg['gamma']}")
    print("="*60)

    # ── Train ──────────────────────────────────────────────────────────────────
    train_results = {}; agents = {}
    for algo in algos:
        env    = build_env(seed=cfg["seed"], alpha=cfg["alpha"])
        agent  = build_agent(algo, cfg)
        result = train(agent, env, cfg["n_train_episodes"], algo, cfg)
        train_results[algo] = result
        agents[algo]        = agent
        try:
            with open(f"{RESULTS_DIR}/{algo}_agent.pkl","wb") as f:
                pickle.dump(agent, f)
        except Exception:
            pass

    # ── Evaluate baselines ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Evaluating baselines  (paper §6.2)")
    print("="*60)
    bl_results = {}
    for name, fn in [("no_hold", no_hold),
                     ("heuristic_15", heur15),
                     ("heuristic_30", heur30)]:
        env = build_env(seed=cfg["seed"] + 100, alpha=cfg["alpha"])
        bl_results[name] = evaluate_baseline(fn, env, cfg["n_test_episodes"], name)

    # ── Evaluate RL agents ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Evaluating RL agents")
    print("="*60)
    rl_results = {}
    for algo in algos:
        env = build_env(seed=cfg["seed"] + 100, alpha=cfg["alpha"])
        rl_results[algo] = evaluate_agent(agents[algo], env,
                                          cfg["n_test_episodes"], algo)

    all_results = {**bl_results, **rl_results}
    dlt         = compute_deltas(rl_results, bl_results)
    print_table(all_results, dlt)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    try:
        out = {}
        for k, v in all_results.items():
            out[k] = {kk: (float(vv) if isinstance(vv,(int,float,np.floating))
                           else vv)
                      for kk, vv in v.items()}
        out["_deltas"] = dlt
        with open(f"{RESULTS_DIR}/summary.json","w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Summary → {RESULTS_DIR}/summary.json")
    except Exception as e:
        print(f"  Warning: could not save JSON: {e}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\n  Generating plots...")
        plot_fig6( all_results,
                   f"{RESULTS_DIR}/figure6_missed_otp.png")
        plot_fig6c(all_results,
                   f"{RESULTS_DIR}/figure6c_delays.png")
        plot_fig7( train_results, algos,
                   f"{RESULTS_DIR}/figure7_rl_metrics.png")
        if not args.no_sweep:
            plot_fig8(cfg, f"{RESULTS_DIR}/figure8_tunability.png")

    print("\n  All done. Results in ./results/")


if __name__ == "__main__":
    main()