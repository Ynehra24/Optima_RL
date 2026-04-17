"""
train.py — Phase 2 Logistics HNH
==================================
"To hold or not to hold?" — Cross-Docking logistics extension.

Trains A2C on the 34-dim logistics state vector.
Mirrors phase-1/algoImplementation/train.py in structure.

Produces:
  Figure 6 equivalent — Failed transfers + Schedule OTP per method
  Figure 7 equivalent — RL training metrics (reward, value, loss)
  Figure 8 equivalent — α/β tunability sweep (A2C)

Usage:
  python train.py                            # full run, stub env
  python train.py --algo a2c --episodes 5   # quick A2C test
  python train.py --no-plots --no-sweep     # headless / CI
  python train.py --use-real-sim            # real CrossDockSimulator
"""

import argparse, json, os, sys, time, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from environment import (
    build_env, ctx2state,
    STATE_DIM, N_ACTIONS, HOLD_DURATIONS,
    ALPHA, BETA, EPISODE_LENGTH,
)
from agents.a2c import A2CAgent

# ── Constants (paper §6.2 — logistics adaptation) ──────────────────────────────
DEFAULT_CONFIG = {
    "n_train_episodes": 25,
    "n_test_episodes":  5,
    "lr":               3e-4,
    "gamma":            0.8,
    "batch_size":       32,
    "alpha":            ALPHA,
    "beta":             BETA,
    "log_every":        200,
    "seed":             42,
}

RESULTS_DIR = os.path.join(_here, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGO_COLORS = {
    "a2c":          "#1f77b4",
    "no_hold":      "#7f7f7f",
    "heuristic_15": "#bcbd22",
    "heuristic_30": "#17becf",
}
DISPLAY = {
    "a2c":          "A2C",
    "no_hold":      "No Hold",
    "heuristic_15": "Heur.15",
    "heuristic_30": "Heur.30",
}


# ── Baseline policies ──────────────────────────────────────────────────────────

def no_hold(state: np.ndarray) -> int:
    return 0

def heur15(state: np.ndarray) -> int:
    """Hold up to 15 min based on τ* hint (state[16])."""
    return min(round(float(state[16]) * 6), 3)   # index 3 = 15 min

def heur30(state: np.ndarray) -> int:
    """Hold up to 30 min based on τ* hint (state[16])."""
    return min(round(float(state[16]) * 6), 6)   # index 6 = 30 min


# ── Training loop ──────────────────────────────────────────────────────────────

def train(agent: A2CAgent, env, n_eps: int, cfg: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training A2C  ({n_eps} episodes)")
    print(f"  State dim: {STATE_DIM}  |  Actions: {N_ACTIONS}")
    print(f"{'='*60}")

    all_rewards = []
    ep_rewards  = []
    t0 = time.time()
    gs = 0   # global step counter

    for ep in range(n_eps):
        state = env.reset()
        # Handle real simulator returning (ctx, info) tuple from reset()
        if isinstance(state, tuple):
            state = ctx2state(state[0])

        epr = eps = 0

        while True:
            action, val = agent.select_action(state)

            result = env.step(action)
            next_raw, reward, done, info = result
            # Handle real simulator returning ctx objects
            if hasattr(next_raw, "to_array"):
                next_state = ctx2state(next_raw)
            elif isinstance(next_raw, np.ndarray):
                next_state = next_raw
            else:
                next_state = np.zeros(STATE_DIM, dtype=np.float32)

            epr += reward; eps += 1; gs += 1

            agent.store(state, action, reward, val, done)
            if eps % cfg["batch_size"] == 0 or done:
                _, lv = agent.select_action(next_state)
                agent.update(last_value=0.0 if done else lv)

            all_rewards.append(reward)
            state = next_state

            if gs % cfg["log_every"] == 0:
                m = agent.get_metrics()
                print(f"  Ep {ep+1:3d} | Step {gs:7d} | "
                      f"AvgR(1k): {m['avg_reward_1k']:.4f} | "
                      f"Loss: {m['avg_loss_1k']:.4f} | "
                      f"Entropy: {m['avg_entropy_1k']:.4f} | "
                      f"Time: {time.time()-t0:.0f}s")
            if done:
                break

        mean = epr / max(eps, 1)
        ep_rewards.append(mean)

        # Show logistics-specific metrics if available
        failed = info.get("failed_transfers",
                  info.get("missed_connections", 0))
        otp    = info.get("schedule_OTP_%",
                  info.get("OTP", 0.0))
        print(f"  Episode {ep+1:3d} done | Steps: {eps:5d} | "
              f"Mean reward: {mean:.4f} | "
              f"Failed transfers: {failed} | "
              f"OTP: {otp}")

    print(f"\n  A2C training complete in {time.time()-t0:.1f}s")
    return {
        "all_rewards":     np.array(all_rewards),
        "episode_rewards": np.array(ep_rewards),
        "losses":   np.array(agent.losses)   if agent.losses   else np.array([]),
        "q_values": np.array(agent.q_values) if agent.q_values else np.array([]),
    }


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _run_one(env, action_fn) -> dict:
    """Run one full episode and return metrics summary."""
    state = env.reset()
    if isinstance(state, tuple):
        state = ctx2state(state[0])

    rewards = []
    holds = steps = 0

    while True:
        action = action_fn(state)
        result = env.step(action)
        next_raw, reward, done, info = result

        if hasattr(next_raw, "to_array"):
            state = ctx2state(next_raw)
        elif isinstance(next_raw, np.ndarray):
            state = next_raw
        else:
            state = np.zeros(STATE_DIM, dtype=np.float32)

        rewards.append(reward)
        steps += 1
        if action > 0:
            holds += 1
        if done:
            break

    # Use simulator summary if available, else derive from info
    if hasattr(env, "metrics_summary"):
        summary = env.metrics_summary()
    else:
        summary = {}

    # Overlay live info values, defaulting to 0 for all missing numeric keys
    for k, v in info.items():
        if k not in summary:
            summary[k] = v

    summary["avg_reward"]  = float(np.mean(rewards))
    summary["holds_pct"]   = 100.0 * holds / max(steps, 1)

    # Normalise OTP to percentage if needed
    otp = summary.get("schedule_OTP_%", summary.get("OTP", 0))
    if otp <= 1.0:
        otp *= 100.0
    summary["OTP"] = otp
    return summary


def _aggregate(summaries: list, name: str) -> dict:
    """Average a list of episode summaries and print the result."""
    out = {}
    for k in summaries[0]:
        try:    out[k] = float(np.mean([s.get(k, 0) for s in summaries]))
        except: out[k] = summaries[0][k]

    failed   = out.get("failed_transfers", out.get("missed_connections", 0))
    fail_pct = out.get("failed_transfer_%", 0.0)
    print(f"  {DISPLAY.get(name, name.upper()):<12} | "
          f"OTP: {out.get('OTP', out.get('schedule_OTP_%', 0)):5.1f}%  |  "
          f"Failed: {failed:6.0f} ({fail_pct:.1f}%)  |  "
          f"Arr: {out.get('avg_delivery_delay_min', 0):5.2f}m  |  "
          f"BayUtil: {out.get('avg_bay_utilisation_%', 0):4.1f}%  |  "
          f"Holds: {out.get('holds_pct', 0):5.1f}%")
    return out


def evaluate_agent(agent: A2CAgent, env, n_eps: int) -> dict:
    print(f"\n  Evaluating A2C ({n_eps} episodes)...")
    fn = lambda s: agent.greedy_action(s)
    return _aggregate([_run_one(env, fn) for _ in range(n_eps)], "a2c")


def evaluate_baseline(policy_fn, env, n_eps: int, name: str) -> dict:
    print(f"  Evaluating baseline: {name}...")
    return _aggregate([_run_one(env, policy_fn) for _ in range(n_eps)], name)


# ── Plotting ───────────────────────────────────────────────────────────────────

def smooth(x: np.ndarray, w: int = 500) -> np.ndarray:
    if len(x) == 0: return x
    w = min(w, max(1, len(x) // 10))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_fig6(results: dict, save: str):
    """Failed transfers + Schedule OTP per method."""
    order   = ["a2c", "heuristic_30", "heuristic_15", "no_hold"]
    methods = [m for m in order if m in results]
    failed  = [results[m].get("failed_transfers",
                results[m].get("missed_connections", 0)) for m in methods]
    otp     = [results[m].get("OTP", 0) for m in methods]
    labels  = [DISPLAY.get(m, m) for m in methods]
    colors  = [ALGO_COLORS.get(m, "#aaa") for m in methods]

    x = np.arange(len(methods)); w = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.bar(x - w/2, failed, w, color=colors, alpha=0.85)
    ax2.bar(x + w/2, otp,    w, color=colors, alpha=0.40,
            edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Failed cargo transfers", color="#d62728", fontsize=11)
    ax2.set_ylabel("Schedule OTP (%)",       color="#1f77b4", fontsize=11)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=10, fontsize=10)
    ax1.set_title("Figure 6 — Logistics metrics: failed transfers & OTP", fontsize=12)
    from matplotlib.patches import Patch
    ax1.legend(handles=[
        Patch(fc="gray", alpha=0.85, label="Failed transfers (left)"),
        Patch(fc="gray", alpha=0.40, ec="black", lw=0.8, label="OTP % (right)"),
    ], loc="upper right")
    ax1.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


def plot_fig7(train_result: dict, save: str):
    """RL training metrics — reward, value, loss."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    c = ALGO_COLORS["a2c"]

    if len(train_result["all_rewards"]) > 0:
        axes[0].plot(smooth(train_result["all_rewards"]),
                     color=c, linewidth=1.5, label="A2C")
    if len(train_result["q_values"]) > 0:
        axes[1].plot(smooth(train_result["q_values"], w=50),
                     color=c, linewidth=1.5, label="A2C")
    if len(train_result["losses"]) > 0:
        axes[2].plot(smooth(train_result["losses"], w=50),
                     color=c, linewidth=1.5, label="A2C")

    for ax, t, y in zip(
        axes,
        ["(a) Average reward", "(b) Value baseline", "(c) Net loss"],
        ["Avg reward (smoothed)", "Avg value", "Avg loss"],
    ):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel(y)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Figure 7 — A2C training metrics (Phase 2 Logistics)", fontsize=11)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


def plot_fig8(cfg: dict, save: str, use_real: bool = False):
    """α/β tunability sweep — mirrors paper Figure 8."""
    print("\n  Running Figure 8: α/β sweep (A2C, 2 eps each)...")
    alphas = [0.10, 0.25, 0.50, 0.75, 1.00]
    betas  = [0.10, 0.50, 1.00]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    base_env = build_env(seed=42, use_real=use_real)
    base_sum = _run_one(base_env, no_hold)
    base_f   = base_sum.get("failed_transfers",
                base_sum.get("missed_connections", 100))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    for j, beta in enumerate(betas):
        saved_l = []; otp_l = []
        for alpha in alphas:
            env = build_env(seed=42, alpha=alpha, use_real=use_real)
            ag  = A2CAgent(state_dim=STATE_DIM, lr=cfg["lr"],
                           gamma=cfg["gamma"], batch_size=cfg["batch_size"],
                           seed=42)
            # Short 2-episode training
            for _ in range(2):
                state = env.reset()
                if isinstance(state, tuple):
                    state = ctx2state(state[0])
                steps = 0
                while True:
                    a, v = ag.select_action(state)
                    result = env.step(a)
                    next_raw, r, done, _ = result
                    if hasattr(next_raw, "to_array"):
                        ns = ctx2state(next_raw)
                    else:
                        ns = next_raw if isinstance(next_raw, np.ndarray) \
                             else np.zeros(STATE_DIM, dtype=np.float32)
                    ag.store(state, a, r, v, done)
                    if steps % cfg["batch_size"] == 0 or done:
                        _, lv = ag.select_action(ns)
                        ag.update(last_value=0.0 if done else lv)
                    state = ns; steps += 1
                    if done: break

            ts = _run_one(build_env(seed=99, alpha=alpha, use_real=use_real),
                          lambda s: ag.greedy_action(s))
            f = ts.get("failed_transfers", ts.get("missed_connections", base_f))
            saved_l.append((base_f - f) / max(base_f, 1) * 100)
            otp_l.append(ts.get("OTP", 0))

        ax1.plot(alphas, saved_l, color=colors[j], marker="o",
                 label=f"β={beta} (saved)")
        ax2.plot(alphas, otp_l,   color=colors[j], marker="s",
                 linestyle="--", label=f"β={beta} (OTP)")

    ax1.set_xlabel("Alpha (α)", fontsize=11)
    ax1.set_ylabel("Failed transfers saved % vs No-Hold", fontsize=11)
    ax2.set_ylabel("Schedule OTP %", fontsize=11)
    ax1.set_title("Figure 8 — Tunability: α/β sweep (A2C, Phase 2)", fontsize=12)
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="center left", fontsize=8)
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


# ── Summary table ──────────────────────────────────────────────────────────────

def print_table(all_results: dict, rl_results: dict, bl_results: dict):
    order   = ["no_hold", "heuristic_15", "heuristic_30", "a2c"]
    methods = [m for m in order if m in all_results]

    print("\n" + "="*80)
    print("  RESULTS TABLE — Phase 2 Logistics HNH")
    print("="*80)
    print(f"  {'Method':<14} {'Failed':<10} {'OTP%':>7} "
          f"{'Arr Dly':>9} {'BayUtil':>9} {'Holds%':>7}")
    print("  " + "-"*65)
    for m in methods:
        r = all_results[m]
        failed   = r.get("failed_transfers", r.get("missed_connections", 0))
        fail_pct = r.get("failed_transfer_%", 0.0)
        print(f"  {DISPLAY.get(m, m):<14} "
              f"{failed:<8.0f}({fail_pct:4.1f}%) "
              f"{r.get('OTP', 0):>7.1f}% "
              f"{r.get('avg_delivery_delay_min', 0):>8.2f}m "
              f"{r.get('avg_bay_utilisation_%', 0):>8.1f}% "
              f"{r.get('holds_pct', 0):>6.1f}%")

    print("\n" + "="*80)
    print("  FAILED TRANSFER REDUCTION vs BASELINES")
    print("="*80)
    print(f"  {'Method':<8} {'vs No-Hold':>12} {'vs Heur-15':>12} {'vs Heur-30':>12}")
    print("  " + "-"*47)
    if "a2c" in rl_results:
        a  = rl_results["a2c"]
        af = a.get("failed_transfers", a.get("missed_connections", 0))
        row = "  A2C     "
        for bname in ["no_hold", "heuristic_15", "heuristic_30"]:
            if bname in bl_results:
                b   = bl_results[bname]
                bf  = b.get("failed_transfers", b.get("missed_connections", 0))
                if bf > 0:
                    red = (bf - af) / bf * 100
                    row += f"{red:>10.1f}%  "
                else:
                    row += f"{'N/A':>10}   "
        print(row)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 2 HNH Logistics RL Training")
    parser.add_argument("--algo",         default="a2c",
                        choices=["a2c"],
                        help="Algorithm to train (only a2c for Phase 2)")
    parser.add_argument("--episodes",     type=int,  default=None)
    parser.add_argument("--no-plots",     action="store_true")
    parser.add_argument("--no-sweep",     action="store_true")
    parser.add_argument("--use-real-sim", action="store_true",
                        help="Use CrossDockSimulator (requires dataset CSV)")
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.episodes:
        cfg["n_train_episodes"] = args.episodes
        cfg["n_test_episodes"]  = max(1, args.episodes // 5)

    mode = "real simulator" if args.use_real_sim else "stub environment"
    print("\n" + "="*60)
    print("  Hold-No-Hold RL  [Phase 2 — Logistics Cross-Docking]")
    print(f"  Mode        : {mode}")
    print(f"  State dim   : {STATE_DIM}  |  Actions: {N_ACTIONS}")
    print(f"  Train eps   : {cfg['n_train_episodes']}  |  "
          f"Test eps: {cfg['n_test_episodes']}")
    print(f"  α={cfg['alpha']}  β={cfg['beta']}  "
          f"lr={cfg['lr']}  γ={cfg['gamma']}")
    print("="*60)

    # ── Train ──────────────────────────────────────────────────────────────────
    env   = build_env(seed=cfg["seed"], alpha=cfg["alpha"],
                      use_real=args.use_real_sim)
    agent = A2CAgent(
        state_dim    = STATE_DIM,
        action_dim   = N_ACTIONS,
        lr           = cfg["lr"],
        gamma        = cfg["gamma"],
        batch_size   = cfg["batch_size"],
        seed         = cfg["seed"],
    )
    train_result = train(agent, env, cfg["n_train_episodes"], cfg)

    try:
        with open(f"{RESULTS_DIR}/a2c_agent.pkl", "wb") as f:
            pickle.dump(agent, f)
        print(f"  Agent saved → {RESULTS_DIR}/a2c_agent.pkl")
    except Exception as e:
        print(f"  Warning: could not save agent: {e}")

    # ── Baselines ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Evaluating baselines")
    print("="*60)
    bl_results = {}
    for name, fn in [("no_hold",      no_hold),
                     ("heuristic_15", heur15),
                     ("heuristic_30", heur30)]:
        env = build_env(seed=cfg["seed"] + 100, use_real=args.use_real_sim)
        bl_results[name] = evaluate_baseline(fn, env, cfg["n_test_episodes"], name)

    # ── Evaluate A2C ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Evaluating A2C agent")
    print("="*60)
    env = build_env(seed=cfg["seed"] + 100, use_real=args.use_real_sim)
    rl_results = {"a2c": evaluate_agent(agent, env, cfg["n_test_episodes"])}

    all_results = {**bl_results, **rl_results}
    print_table(all_results, rl_results, bl_results)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    try:
        out = {}
        for k, v in all_results.items():
            out[k] = {kk: (float(vv) if isinstance(vv, (int, float, np.floating))
                           else vv)
                      for kk, vv in v.items()}
        with open(f"{RESULTS_DIR}/summary.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Summary → {RESULTS_DIR}/summary.json")
    except Exception as e:
        print(f"  Warning: could not save JSON: {e}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not args.no_plots:
        try:
            print("\n  Generating plots...")
            plot_fig6(all_results,
                      f"{RESULTS_DIR}/figure6_failed_transfers.png")
            plot_fig7(train_result,
                      f"{RESULTS_DIR}/figure7_training_metrics.png")
            if not args.no_sweep:
                plot_fig8(cfg, f"{RESULTS_DIR}/figure8_tunability.png",
                          use_real=args.use_real_sim)
        except Exception as e:
            print(f"  Warning: plotting failed: {e}")

    print("\n  All done. Results in ./results/")


if __name__ == "__main__":
    main()
