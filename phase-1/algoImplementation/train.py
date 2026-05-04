"""
train.py — FIXED VERSION
========================
"To hold or not to hold?" — Malladi et al., AAMAS 2021

KEY FIX: Compute rewards at EPISODE END using delay tree attribution.
  - Trajectory collects (state, action, r_l, flight_id) during episode
  - At episode end: Query r_g from delay tree for each flight
  - Combine: r_total = beta * r_l + (1 - beta) * r_g
  - Backfill agent memory with actual combined rewards
"""

import argparse, json, os, sys, time, pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
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
    "lr":               0.001,
    "gamma":            0.8,
    "batch_size":       32,
    "alpha":            ALPHA,
    "beta":             BETA,
    "log_every":        500,
    "seed":             42,
}

RESULTS_DIR = os.path.join(_HERE, "results")
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
def build_env(seed=42, alpha=ALPHA, beta=BETA):
    cfg = SimConfig()
    cfg.alpha       = alpha
    cfg.beta        = beta
    cfg.random_seed = seed
    return AirlineNetworkSimulator(cfg)


# ── Baselines (paper §6.2) ─────────────────────────────────────────────────────
def no_hold(s):
    return 0


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


# ── TRAINING — FIXED VERSION ──────────────────────────────────────────────────
def train(agent, env, n_eps, algo, cfg):
    print(f"\n{'='*80}")
    print(f"  Training {algo.upper()}  ({n_eps} episodes)")
    print(f"  [FIXED] Using episode-end global reward computation")
    print(f"{'='*80}")

    all_rewards = []
    ep_rewards  = []
    all_r_l = []
    all_r_g = []
    t0 = time.time()
    gs = 0

    for ep in range(n_eps):
        ctx, _ = env.reset()
        state  = ctx2state(ctx)
        
        # ← TRAJECTORY BUFFER (stores steps until episode end)
        trajectory = []
        
        epr = eps = 0

        while True:
            # ── Select action ────────────────────────────────────────────────
            if algo == "dqn":
                action = agent.select_action(state); val = None
            elif algo in ("a2c", "ac"):
                action, val = agent.select_action(state)
            else:
                action = agent.discrete_action(state); val = None

            # ── Step — get IMMEDIATE local reward (r_l) ───────────────────
            ctx, r_l, done, info = env.step(action)
            next_state = ctx2state(ctx)

            # ← STORE step in trajectory
            trajectory.append({
                'state': state,
                'action': action,
                'r_l': float(r_l),
                'val': val,
                'flight_id': info.get('flight_id', ''),
                'next_state': next_state,
                'done': done,
            })

            epr += r_l
            eps += 1
            gs += 1
            all_r_l.append(r_l)
            state = next_state

            # ── Log every log_every steps ────────────────────────────────────
            if gs % cfg["log_every"] == 0:
                m = agent.get_metrics()
                print(f"  Ep {ep+1:3d} | Step {gs:7d} | "
                      f"AvgR(1k): {m['avg_reward_1k']:.4f} | "
                      f"Loss: {m['avg_loss_1k']:.4f} | "
                      f"Time: {time.time()-t0:.0f}s")

            if done:
                break

        # ────────────────────────────────────────────────────────────────────
        # ← EPISODE END: Compute global rewards from direct PAX outcomes
        # FIX: delay tree only covers ~7% of flights (H→D→A chain rarely
        # completes in one episode).  Instead, for held flights, r_g =
        # fraction of resolved connecting PAX who made their connection.
        # ────────────────────────────────────────────────────────────────────
        global_rewards = {}
        r_g_values = []

        for flight_id in set(t['flight_id'] for t in trajectory if t['flight_id']):
            fs = env.flights.get(flight_id)

            # No-hold: r_g = 0 (no global consequence from our action)
            if fs is None or fs.hnh_action == 0:
                global_rewards[flight_id] = 0.0
                r_g_values.append(0.0)
                continue

            # Held flight: compute connection success rate for PAX who
            # were waiting to connect onto this flight (its inbound PAX)
            incoming_pax_ids = env._incoming_pax.get(flight_id, [])
            resolved_made  = 0
            resolved_missed = 0
            for pid in incoming_pax_ids:
                pax = env.pax.get(pid)
                if pax is None or len(pax.legs) < 2:
                    continue
                inbound_fs = env.flights.get(pax.legs[0])
                # Only count PAX whose inbound flight actually arrived
                if inbound_fs is None or inbound_fs.actual_arrival is None:
                    continue
                if pax.missed_connection:
                    resolved_missed += pax.group_size
                else:
                    resolved_made += pax.group_size

            total_resolved = resolved_made + resolved_missed
            if total_resolved > 0:
                r_g = float(resolved_made) / total_resolved
            else:
                # No resolved PAX: fall back to delay tree (will be ~0 but harmless)
                r_g = env.reward_calculator.get_global_reward(flight_id)

            global_rewards[flight_id] = float(r_g)
            r_g_values.append(r_g)
        
        #if not r_g_values:
            #print(f"    ⚠ WARNING: No global rewards computed (delay tree empty)")
        #else:
            '''print(f"    Global rewards: mean={np.mean(r_g_values):.6f}, "
                  f"max={np.max(r_g_values):.6f}, "
                  f"nonzero={sum(1 for x in r_g_values if x != 0)}/{len(r_g_values)}")'''

        # ────────────────────────────────────────────────────────────────────
        # ← BACKFILL: Combine r_l + r_g and train agent
        # ────────────────────────────────────────────────────────────────────
        beta = env.reward_calculator.beta
        #print(f"    Combining rewards: R = {beta:.2f}*r_l + {1-beta:.2f}*r_g")
        
        for t_idx, t in enumerate(trajectory):
            flight_id = t['flight_id']
            r_l = t['r_l']
            r_g = global_rewards.get(flight_id, 0.0)
            r_total = beta * r_l + (1 - beta) * r_g
            
            all_rewards.append(r_total)
            all_r_g.append(r_g)

            # ── Learn ────────────────────────────────────────────────────────
            if algo == "dqn":
                agent.push(t['state'], t['action'], r_total, t['next_state'], t['done'])
                agent.update()
            elif algo in ("a2c", "ac"):
                agent.store(t['state'], t['action'], r_total, t['val'], t['done'])
                # Update every batch_size or at episode end
                if (t_idx + 1) % cfg["batch_size"] == 0 or t['done']:
                    if t['done']:
                        last_value = 0.0
                    else:
                        _, last_value = agent.select_action(t['next_state'])
                    agent.update(last_value=last_value)
            else:
                agent.push(t['state'], t['action'], r_total, t['next_state'], t['done'])
                agent.update()


        mean_ep = np.mean([t['r_l'] + global_rewards.get(t['flight_id'], 0.0) 
                          for t in trajectory if t['flight_id']])
        ep_rewards.append(mean_ep)
        print(f"  Episode {ep+1:3d} done | Steps: {eps:5d} | "
              f"Mean reward: {mean_ep:.4f} | "
              f"Missed: {info.get('missed_connections','?')} | "
              f"OTP: {info.get('OTP','?')}\n")

    print(f"\n  {algo.upper()} training complete in {time.time()-t0:.1f}s")
    
    # Final diagnostic
    #print(f"\n  TRAINING DIAGNOSTIC:")
    #print(f"    Avg r_l across all steps: {np.mean(all_r_l):.6f}")
    #print(f"    Avg r_g across all steps: {np.mean(all_r_g):.6f}")
    #print(f"    Nonzero r_g steps: {sum(1 for x in all_r_g if x != 0)}/{len(all_r_g)}")
    
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
    print(f"  Evaluating {DISPLAY.get(algo, algo)} ({n_eps} episodes)...")
    # DDPG.greedy_action returns a continuous float — must discretise for env.step()
    if hasattr(agent, 'discrete_action'):
        action_fn = lambda s: agent.discrete_action(s)
    else:
        action_fn = lambda s: agent.greedy_action(s)
    summaries = [_run_one(env, action_fn) for _ in range(n_eps)]
    return _aggregate(summaries, algo)


def evaluate_baseline(action_fn, env, n_eps, name):
    print(f"  Evaluating baseline: {name}...")
    summaries = [_run_one(env, action_fn) for _ in range(n_eps)]
    return _aggregate(summaries, name)


def evaluate_heuristic_proper(env, n_eps, name, max_hold=15):
    """Evaluate heuristic baseline using simulator's policy."""
    print(f"  Evaluating baseline: {name}...")
    summaries = []
    for _ in range(n_eps):
        ctx, _ = env.reset()
        state = ctx2state(ctx)
        rewards = []
        holds = steps = 0
        while True:
            action = env._select_baseline_action("heuristic", max_hold=max_hold)
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
        if summary.get("OTP", 0) <= 1.0:
            summary["OTP"] = summary["OTP"] * 100.0
        summaries.append(summary)
    return _aggregate(summaries, name)


def compute_deltas(rl_results, bl_results):
    """Compute improvement vs no-hold baseline."""
    dlt = {}
    base_m = bl_results["no_hold"].get("missed_connections", 2700)
    for algo in rl_results:
        m = rl_results[algo].get("missed_connections", base_m)
        dlt[algo] = {
            "missed_saved_pct": (base_m - m) / max(base_m, 1) * 100,
            "otp_delta": rl_results[algo].get("OTP", 0) - bl_results["no_hold"].get("OTP", 0),
        }
    return dlt


def print_table(all_results, deltas):
    order = ["no_hold", "heuristic_15", "heuristic_30", "a2c", "dqn", "ac", "ddpg"]
    methods = [m for m in order if m in all_results]

    print("\n" + "="*78)
    print("  RESULTS TABLE (paper §7.2)")
    print("="*78)
    print(f"  {'Method':<14} {'Missed PAX':>10} {'OTP%':>7} {'Arr Dly':>9} {'Dep Dly':>9} {'Holds%':>7}")
    print("  " + "-"*65)
    for name in methods:
        res = all_results[name]
        print(f"  {DISPLAY.get(name,name.upper()):<14} "
              f"{res.get('missed_connections',0):>10.0f} "
              f"{res.get('OTP',0):>7.1f}% "
              f"{res.get('avg_arrival_delay_min',0):>8.2f}m "
              f"{res.get('avg_departure_delay_min',0):>8.2f}m "
              f"{res.get('holds_pct',0):>6.1f}%")

    # Missed connection reduction table
    nh_miss   = all_results.get("no_hold",      {}).get("missed_connections", 1)
    h15_miss  = all_results.get("heuristic_15", {}).get("missed_connections", 1)
    h30_miss  = all_results.get("heuristic_30", {}).get("missed_connections", 1)

    rl_algos = [m for m in ["a2c", "dqn", "ac", "ddpg"] if m in all_results]
    if rl_algos:
        print("\n" + "="*78)
        print("  MISSED CONNECTION REDUCTION vs BASELINES")
        print("  Paper target: A2C ~50% reduction vs Heuristic-15 (§7.2)")
        print("="*78)
        print(f"  {'Method':<12} {'vs No-Hold':>12} {'vs Heur-15':>12} {'vs Heur-30':>12}")
        print("  " + "-"*50)
        for algo in rl_algos:
            m = all_results[algo].get("missed_connections", nh_miss)
            pct_nh  = (nh_miss  - m) / max(nh_miss,  1) * 100
            pct_h15 = (h15_miss - m) / max(h15_miss, 1) * 100
            pct_h30 = (h30_miss - m) / max(h30_miss, 1) * 100
            print(f"  {DISPLAY.get(algo,algo.upper()):<12} "
                  f"{pct_nh:>+11.1f}% "
                  f"{pct_h15:>+11.1f}% "
                  f"{pct_h30:>+11.1f}%")
        print()
        print("  Positive = RL reduces more missed connections than baseline")


def smooth(data, w=1000):
    """Smooth data with moving average."""
    if len(data) < w:
        return data
    return np.convolve(data, np.ones(w)/w, mode='valid')


def plot_fig6(all_results, save):
    """Figure 6: Missed connections vs OTP."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name in ["no_hold", "heuristic_15", "heuristic_30", "a2c", "dqn", "ac", "ddpg"]:
        if name not in all_results:
            continue
        res = all_results[name]
        c = ALGO_COLORS.get(name, "gray")
        lbl = DISPLAY.get(name, name.upper())
        ax1.bar(lbl, res.get("missed_connections", 0), color=c, alpha=0.7)
        ax2.bar(lbl, res.get("OTP", 0), color=c, alpha=0.7)
    ax1.set_ylabel("Missed connections")
    ax2.set_ylabel("OTP %")
    ax1.set_title("(a) Missed connections")
    ax2.set_title("(b) On-time performance")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


def plot_fig6c(all_results, save):
    """Figure 6c: Delays."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name in ["no_hold", "heuristic_15", "heuristic_30", "a2c", "dqn", "ac", "ddpg"]:
        if name not in all_results:
            continue
        res = all_results[name]
        c = ALGO_COLORS.get(name, "gray")
        lbl = DISPLAY.get(name, name.upper())
        ax1.bar(lbl, res.get("avg_arrival_delay_min", 0), color=c, alpha=0.7)
        ax2.bar(lbl, res.get("avg_departure_delay_min", 0), color=c, alpha=0.7)
    ax1.set_ylabel("Arrival delay (min)")
    ax2.set_ylabel("Departure delay (min)")
    ax1.set_title("(a) Arrival delays")
    ax2.set_title("(b) Departure delays")
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved → {save}")


def plot_fig7(train_results, algos, save):
    """Figure 7: RL training metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name in algos:
        res = train_results[name]
        c = ALGO_COLORS.get(name, "gray")
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
    base_sum = base_env.run_episode(policy="no_hold")
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
    parser = argparse.ArgumentParser(description="HNH RL Training [FIXED]")
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

    print("\n" + "="*80)
    print("  Hold-No-Hold RL  [FIXED: Episode-End Global Reward Attribution]")
    print(f"  Algorithms  : {algos}")
    print(f"  Train eps   : {cfg['n_train_episodes']}  |  "
          f"Test eps: {cfg['n_test_episodes']}")
    print(f"  α={cfg['alpha']}  β={cfg['beta']}  "
          f"lr={cfg['lr']}  γ={cfg['gamma']}")
    print("="*80)

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

    # ── Diagnostic: PAX generation ────────────────────────────────────────────
    print("\n" + "="*80)
    #print("  DIAGNOSTIC: Checking PAX generation")
    print("="*80)
    
    env = build_env(seed=cfg["seed"] + 100, alpha=cfg["alpha"])
    ctx, _ = env.reset()
    
    total_pax = len(env.pax)
    total_connecting = sum(1 for p in env.pax.values() if len(p.legs) >= 2)
    flights_with_incoming = sum(1 for fid, pids in env._incoming_pax.items() if pids)
    total_incoming_pax = sum(len(pids) for pids in env._incoming_pax.values())
    
    #print(f"  Total PAX generated: {total_pax}")
   #print(f"  Connecting PAX (2+ legs): {total_connecting}")
    #print(f"  Flights with incoming connections: {flights_with_incoming} / {len(env.flights)}")
    #print(f"  Total entries in _incoming_pax: {total_incoming_pax}")
    #print("="*80 + "\n")

    # ── Evaluate baselines ─────────────────────────────────────────────────────
    print("="*80)
    print("  Evaluating baselines  (paper §6.2)")
    print("="*80)
    bl_results = {}
    
    # No-hold baseline
    env = build_env(seed=cfg["seed"] + 100, alpha=cfg["alpha"])
    bl_results["no_hold"] = evaluate_baseline(no_hold, env, cfg["n_test_episodes"], "no_hold")
    
    # Heuristic policies
    env = build_env(seed=cfg["seed"] + 100, alpha=cfg["alpha"])
    bl_results["heuristic_15"] = evaluate_heuristic_proper(env, cfg["n_test_episodes"], 
                                                            "heuristic_15", max_hold=15)
    
    env = build_env(seed=cfg["seed"] + 100, alpha=cfg["alpha"])
    bl_results["heuristic_30"] = evaluate_heuristic_proper(env, cfg["n_test_episodes"], 
                                                            "heuristic_30", max_hold=30)

    # ── Evaluate RL agents ─────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  Evaluating RL agents")
    print("="*80)
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
