"""
environment.py — Phase 2 Logistics HNH
=======================================
Environment wrapper for the Cross-Docking Hold-or-Not-Hold problem.

Two modes:
  1. STUB (default)  — pure-NumPy synthetic environment. Works without the
     real dataset CSV. Produces realistic 34-dim state vectors that match
     the exact layout of TruckContext.to_array(). Used for smoke-testing and
     algorithm development before the real simulator is available.

  2. REAL  (--use-real-sim flag)  — wraps CrossDockSimulator directly.
     Requires `phase-2/phase2_dataset/dynamic_supply_chain_logistics_dataset.csv`.

State vector layout (34 dims — mirrors TruckContext.to_array()):
  [0:7]   CL(τ) for τ in {0,5,10,15,20,25,30}
  [7:14]  OL(τ) for τ in {0,5,10,15,20,25,30}
  [14]    CG  — global cargo utility (rolling)
  [15]    OG  — global operator utility (rolling)
  [16]    τ*  — locally optimal hold (normalised 0-1)
  [17]    V_k — mean cargo value score
  [18]    Q_k — volume fraction
  [19]    X_k — SLA urgency (normalised 0-1)
  [20]    E_k — perishable fraction
  [21]    Δ_in  — inbound ETA lag (normalised)
  [22]    Δ_slack — transfer slack (normalised)
  [23]    L_k — driver hours remaining (normalised)
  [24]    F_k — downstream deadline pressure
  [25]    N_in — inbound truck count (normalised)
  [26]    B_G — bay utilisation rate
  [27]    W_G — hub throughput rate
  [28]    Y_G — failed transfer rate (rolling)
  [29]    Z_G — inbound queue depth
  [30]    D_k — departure delay (normalised)
  [31]    A_k — arrival delay (normalised)
  [32]    G_bay_k — bay dwell delay (normalised)
  [33]    G_road_k — road time delay (normalised)

Action space: {0,1,2,3,4,5,6} → {0,5,10,15,20,25,30} minutes
"""

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
HOLD_DURATIONS   = [0, 5, 10, 15, 20, 25, 30]
N_ACTIONS        = len(HOLD_DURATIONS)        # 7
STATE_DIM        = 34
ALPHA            = 0.75   # cargo/operator trade-off
BETA             = 0.75   # local/global trade-off
TRUCKS_PER_DAY   = 120    # logistics hub scale
DAYS_PER_EPISODE = 7
EPISODE_LENGTH   = TRUCKS_PER_DAY * DAYS_PER_EPISODE   # ~840 steps per episode


# ── Stub environment ───────────────────────────────────────────────────────────

class LogisticsEnvStub:
    """
    Stub environment — no dataset required.

    Produces synthetic logistics states that mirror TruckContext.to_array()
    in structure. Reward is computed using the paper's formula:
        R_T = β·R_L + (1-β)·R_G   with R_L = α·CL(τ) + (1-α)·OL(τ)

    Global state drifts slowly to simulate network congestion cycles.
    """

    def __init__(self, seed: int = 42, alpha: float = ALPHA, beta: float = BETA):
        self.rng        = np.random.default_rng(seed)
        self.alpha      = alpha
        self.beta       = beta
        self.step_count = 0

        # Slowly-drifting global hub state
        self._CG    = 0.80   # global cargo utility
        self._OG    = 0.75   # global operator utility
        self._BG    = 0.40   # bay utilisation (congestion level)
        self._WG    = 0.85   # throughput rate
        self._YG    = 0.10   # failed transfer rate
        self._ZG    = 0.15   # inbound queue depth

        # Episode-level counters (mirrors MetricsTracker)
        self._ep_total        = 0
        self._ep_holds        = 0
        self._ep_failed       = 0
        self._ep_successful   = 0
        self._ep_ontime       = 0
        self._ep_priority_fail = 0
        self._ep_bay_util_sum = 0.0
        self._ep_delivery_delays: list = []

    # ── Public interface ───────────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        """Start a new episode. Returns initial state (shape: [34])."""
        self.step_count = 0
        self._CG = 0.80; self._OG = 0.75
        self._BG = 0.40; self._WG = 0.85
        self._YG = 0.10; self._ZG = 0.15
        # Reset episode counters
        self._ep_total        = 0
        self._ep_holds        = 0
        self._ep_failed       = 0
        self._ep_successful   = 0
        self._ep_ontime       = 0
        self._ep_priority_fail = 0
        self._ep_bay_util_sum = 0.0
        self._ep_delivery_delays = []
        return self._make_state()

    def step(self, action: int):
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"
        hold_min = HOLD_DURATIONS[action]

        reward, info = self._compute_reward(action, hold_min)
        self._update_episode_metrics(action, info)
        self._update_global_state(hold_min)

        self.step_count += 1
        done = self.step_count >= EPISODE_LENGTH

        # Enrich info with logistics keys so training log shows real numbers
        summary = self.metrics_summary()
        info["failed_transfers"]    = summary["failed_transfers"]
        info["successful_transfers"] = summary["successful_transfers"]
        info["schedule_OTP_%"]      = summary["schedule_OTP_%"]
        info["avg_bay_utilisation_%"] = summary["avg_bay_utilisation_%"]

        next_state = self._make_state()
        return next_state, reward, done, info

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def action_dim(self) -> int:
        return N_ACTIONS

    def metrics_summary(self) -> dict:
        """Mirrors MetricsTracker.summary() key names exactly."""
        total = max(self._ep_total, 1)
        conn  = max(self._ep_failed + self._ep_successful, 1)
        otp   = 100.0 * self._ep_ontime / total
        fail_pct = 100.0 * self._ep_failed / conn
        avg_dly  = (float(np.mean(self._ep_delivery_delays))
                    if self._ep_delivery_delays else 0.0)
        avg_bay  = 100.0 * self._ep_bay_util_sum / total
        return {
            "schedule_OTP_%":          round(otp, 2),
            "failed_transfer_%":       round(fail_pct, 2),
            "avg_delivery_delay_min":  round(avg_dly, 2),
            "avg_bay_utilisation_%":   round(avg_bay, 2),
            "total_trucks":            self._ep_total,
            "departed_trucks":         self._ep_total,
            "ontime_departures":       self._ep_ontime,
            "avg_departure_delay_min": round(float(np.mean([HOLD_DURATIONS[0]] * total)), 2),
            "failed_transfers":        self._ep_failed,
            "successful_transfers":    self._ep_successful,
            "priority_failed":         self._ep_priority_fail,
            "avg_hold_min":            round(5.0 * self._ep_holds / total, 2),
            # OTP as fraction for compatibility
            "OTP":                     round(otp, 2),
        }

    # ── Episode metric tracking ───────────────────────────────────────────────

    def _update_episode_metrics(self, action: int, info: dict):
        """Update per-episode counters after each truck decision."""
        self._ep_total += 1
        if action > 0:
            self._ep_holds += 1

        # OTP: truck departs on time if hold ≤ 10 min (index ≤ 2)
        if action <= 2:
            self._ep_ontime += 1

        # Simulate transfer outcome:
        # Under-holding when there's inbound cargo at risk → failure
        # Uses CL values from info: if CL is high for a higher hold, agent under-held
        tau_star_idx = round(info.get("tau_star", 0) * (N_ACTIONS - 1))
        in_delay     = info.get("in_delay_normalised", 0.0)
        if in_delay > 0.1 and action < tau_star_idx:
            # Under-holding with meaningful incoming delay → simulate failure
            gap         = (tau_star_idx - action) / N_ACTIONS
            fail_prob   = gap * in_delay * 1.5
            is_priority = info.get("SLA_urgency", 0.0) >= 0.8
            if self.rng.random() < fail_prob:
                self._ep_failed += 1
                delay = float((tau_star_idx - action) * 5 + self.rng.uniform(5, 30))
                self._ep_delivery_delays.append(delay)
                if is_priority:
                    self._ep_priority_fail += 1
            else:
                self._ep_successful += 1
        else:
            self._ep_successful += 1

        # Bay utilisation sample
        self._ep_bay_util_sum += float(np.clip(self._BG + self.rng.normal(0, 0.03), 0, 1))

    # ── State builder ──────────────────────────────────────────────────────────

    def _make_state(self) -> np.ndarray:
        """Build a realistic 34-dim logistics state vector."""
        state = np.zeros(STATE_DIM, dtype=np.float32)

        # Cargo properties for this truck decision
        incoming_delay = float(self.rng.integers(0, 31))   # 0-30 min
        V_k = float(np.clip(self.rng.beta(2, 2), 0, 1))   # value score
        X_k = int(self.rng.integers(0, 3))                 # SLA urgency 0/1/2
        E_k = float(self.rng.beta(1, 4))                   # perishable fraction
        Q_k = float(np.clip(self.rng.beta(3, 2), 0, 1))   # volume fraction
        BG  = float(np.clip(self._BG + self.rng.normal(0, 0.05), 0, 1))

        # CL[0:7] — cargo utility per hold: holding helps if cargo is delayed
        sla_amp = 1.0 + X_k   # express cargo penalised harder
        for i, tau in enumerate(HOLD_DURATIONS):
            if incoming_delay == 0:
                # no feeder delay — holding wastes time
                base = 0.95 - 0.04 * i
            else:
                if tau <= incoming_delay:
                    base = 0.60 + 0.04 * i  # hold covers the delay
                else:
                    base = 0.85 - 0.03 * (i - incoming_delay / 5)
            # Perishable cargo decays faster with delay
            perishable_penalty = E_k * 0.02 * i if tau > 0 else 0
            state[i] = float(np.clip(
                V_k * (base - perishable_penalty) * sla_amp / 3.0
                + self.rng.normal(0, 0.02), 0, 1))

        # OL[7:14] — operator utility per hold: decreases as hold increases
        # Also penalised when bay is congested
        congestion_penalty = max(0.0, BG - 0.60) * 0.05
        for i, tau in enumerate(HOLD_DURATIONS):
            base = 0.95 - 0.08 * i - congestion_penalty * i
            state[7 + i] = float(np.clip(base + self.rng.normal(0, 0.02), 0, 1))

        # CG[14], OG[15] — global utilities
        state[14] = float(np.clip(self._CG + self.rng.normal(0, 0.03), 0, 1))
        state[15] = float(np.clip(self._OG + self.rng.normal(0, 0.03), 0, 1))

        # τ*[16] — locally optimal hold (normalised to [0,1])
        local_score  = self.alpha * state[0:7] + (1 - self.alpha) * state[7:14]
        tau_star_idx = int(np.argmax(local_score))
        state[16]    = tau_star_idx / (N_ACTIONS - 1)
        # Store for episode tracking
        self._last_tau_star     = tau_star_idx
        self._last_sla_urgency  = X_k / 2.0
        self._last_in_delay_n   = float(np.clip(incoming_delay / 30.0, 0, 1))

        # Logistics-specific scalars [17:26]
        state[17] = V_k
        state[18] = Q_k
        state[19] = X_k / 2.0   # normalised to [0,1]
        state[20] = E_k
        state[21] = float(np.clip(incoming_delay / 60.0, 0, 1))  # Δ_in
        state[22] = float(np.clip(self.rng.uniform(0, 1), 0, 1))  # Δ_slack
        state[23] = float(np.clip(self.rng.uniform(0.5, 1.0), 0, 1))  # L_k
        state[24] = float(1.0 - X_k / 2.0)   # F_k deadline pressure
        state[25] = float(np.clip(self.rng.integers(1, 8) / 20.0, 0, 1))  # N_in

        # Global hub state [26:30]
        state[26] = BG
        state[27] = float(np.clip(self._WG + self.rng.normal(0, 0.03), 0, 1))
        state[28] = float(np.clip(self._YG + self.rng.normal(0, 0.02), 0, 1))
        state[29] = float(np.clip(self._ZG + self.rng.normal(0, 0.02), 0, 1))

        # Delay tree signals [30:34]
        dep_delay = float(max(0.0, incoming_delay * 0.7 + self.rng.normal(0, 3)))
        state[30] = float(np.clip(dep_delay / 60.0, 0, 1))
        state[31] = float(np.clip((dep_delay + self.rng.uniform(0, 10)) / 480.0, 0, 1))
        state[32] = float(np.clip(BG * 0.3 + self.rng.uniform(0, 0.1), 0, 1))
        state[33] = float(np.clip(self.rng.uniform(0, 0.2), 0, 1))

        return np.clip(state, 0.0, 1.0)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: int, hold_min: int):
        """
        R_T = β·R_L + (1-β)·R_G
        R_L = α·CL(τ) + (1-α)·OL(τ)
        R_G = α·CG + (1-α)·OG − λ_bay·BayCongestion

        Fix: CL range is now [0.50, 0.95] centred on tau* when there
        is incoming delay.  Noise std is 0.015 so the action gradient
        (up to 0.45) dominates.  Without delay, holding just costs OL.
        """
        tau_star    = getattr(self, "_last_tau_star",    0)
        in_delay_n  = getattr(self, "_last_in_delay_n",  0.0)
        sla_urgency = getattr(self, "_last_sla_urgency", 0.0)
        BG          = float(np.clip(self._BG, 0, 1))
        congestion  = float(np.clip(BG - 0.60, 0, 1)) * (action / N_ACTIONS)
        lambda_bay  = 0.30

        # ── CL: Cargo utility ───────────────────────────────────────────
        # SIGNAL: when there's an incoming delay, holding at tau* is good.
        # Range: 0.50 (worst action) → 0.95 (action == tau*).
        # Noise std 0.015 << signal range 0.45 => learnable.
        if in_delay_n > 0.1 and tau_star > 0:
            # Quality = 1 when action == tau*, 0 when action is maximally wrong
            hold_quality = 1.0 - abs(action - tau_star) / max(tau_star + 1, 1)
            CL_base      = 0.50 + 0.45 * hold_quality
        elif in_delay_n > 0.1 and tau_star == 0:
            # Delay exists but optimal is no-hold (e.g. very short delay):
            # holding is wasteful but not catastrophic
            CL_base = 0.90 - 0.06 * (action / N_ACTIONS)
        else:
            # No incoming delay — holding wastes cargo utility
            CL_base = 0.90 - 0.05 * (action / N_ACTIONS)

        # SLA urgency amplifies CL slightly (express cargo cares more)
        CL_val = float(np.clip(
            CL_base * (1.0 + sla_urgency * 0.08)
            + self.rng.normal(0, 0.015), 0, 1))

        # ── OL: Operator utility ───────────────────────────────────────────
        # Holding always costs OTP — clear decreasing ramp.
        OL_val = float(np.clip(
            0.90 - 0.10 * (action / N_ACTIONS)
            - congestion * 0.40
            + self.rng.normal(0, 0.010), 0, 1))

        # ── Global utilities (action-independent in the stub) ─────────────
        CG_val = float(np.clip(self._CG + self.rng.normal(0, 0.015), 0, 1))
        OG_val = float(np.clip(
            self._OG - 0.015 * action + self.rng.normal(0, 0.010), 0, 1))

        R_L    = self.alpha * CL_val + (1 - self.alpha) * OL_val
        R_G    = (self.alpha * CG_val + (1 - self.alpha) * OG_val
                  - lambda_bay * congestion)
        reward = self.beta * R_L + (1 - self.beta) * R_G

        info = {
            "CL": CL_val, "OL": OL_val,
            "CG": CG_val, "OG": OG_val,
            "R_L": R_L, "R_G": R_G,
            "hold_min": hold_min,
            "bay_congestion_penalty": congestion,
            # Keys used by _update_episode_metrics
            "tau_star":           getattr(self, "_last_tau_star",     0) / max(N_ACTIONS - 1, 1),
            "in_delay_normalised": getattr(self, "_last_in_delay_n",  0.0),
            "SLA_urgency":         getattr(self, "_last_sla_urgency", 0.0),
        }
        return float(reward), info

    # ── Global state drift ────────────────────────────────────────────────────

    def _update_global_state(self, hold_min: int):
        """Holding increases bay utilisation and slowly degrades throughput."""
        drift = 0.0003 * hold_min
        self._OG = float(np.clip(self._OG - drift + self.rng.normal(0, 0.005), 0.3, 1.0))
        self._CG = float(np.clip(self._CG + 0.001 - drift * 0.5 + self.rng.normal(0, 0.005), 0.3, 1.0))
        self._BG = float(np.clip(self._BG + drift * 2 - 0.001 + self.rng.normal(0, 0.01), 0.0, 1.0))
        self._WG = float(np.clip(self._WG - drift * 0.5 + self.rng.normal(0, 0.005), 0.3, 1.0))
        self._YG = float(np.clip(self._YG + drift * 0.3 + self.rng.normal(0, 0.003), 0.0, 0.5))
        self._ZG = float(np.clip(self._ZG + drift * 0.2 + self.rng.normal(0, 0.005), 0.0, 0.5))


# ── Real simulator wrapper ─────────────────────────────────────────────────────

class LogisticsEnvReal:
    """
    Thin wrapper around CrossDockSimulator for use in the training loop.

    Requires:
      - phase-2/phase2_dataset/dynamic_supply_chain_logistics_dataset.csv
      - phase2_simulator package alias (junction) to be set up
    """

    def __init__(self, cfg=None):
        import sys, os
        p2 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..")
        sys.path.insert(0, os.path.abspath(p2))
        from phase2_simulator.simulator import CrossDockSimulator
        from phase2_simulator.config    import SimConfig
        self._sim = CrossDockSimulator(cfg or SimConfig())
        self._done = False

    def reset(self):
        ctx, _ = self._sim.reset()
        return ctx2state(ctx)

    def step(self, action: int):
        ctx, reward, done, info = self._sim.step(action)
        self._done = done
        return ctx2state(ctx), reward, done, info

    @property
    def state_dim(self): return STATE_DIM

    @property
    def action_dim(self): return N_ACTIONS

    def metrics_summary(self):
        return self._sim.metrics.summary()


# ── State extraction helper ────────────────────────────────────────────────────

def ctx2state(ctx) -> np.ndarray:
    """
    Convert a TruckContext (or DONE sentinel) to a flat float32 state vector.
    Mirrors Phase 1's ctx2state() but for 34-dim logistics state.
    """
    if hasattr(ctx, "truck_id") and ctx.truck_id == "DONE":
        return np.zeros(STATE_DIM, dtype=np.float32)
    s = ctx.to_array().copy()
    # tau* at index 16 is already normalised in Phase 2 TruckContext
    return np.clip(s, 0.0, 1.0).astype(np.float32)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_env(seed: int = 42, alpha: float = ALPHA, use_real: bool = False, cfg=None):
    """Build either the stub or real simulator environment."""
    if use_real:
        return LogisticsEnvReal(cfg)
    return LogisticsEnvStub(seed=seed, alpha=alpha)
