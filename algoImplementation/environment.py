"""
environment.py
--------------
Stub Airline Environment for Hold-No-Hold (HNH) RL project.

Simulates the interface that Member 1 will eventually provide.
State vector matches the exact spec from the mappings document:
    state[0:7]  = PL(tau) - Local Passenger Utility for tau in {0,5,10,15,20,25,30}
    state[7:14] = AL(tau) - Local Airline Utility for tau in {0,5,10,15,20,25,30}
    state[14]   = PG      - Global Passenger Utility (last 24h)
    state[15]   = AG      - Global Airline Utility (last 24h)
    state[16]   = tau*    - Locally optimal hold index (normalised 0-1)

Action space: {0,1,2,3,4,5,6} mapping to {0,5,10,15,20,25,30} minutes
Reward: scalar float in [0,1] (computed by Member 3's logic, stubbed here)
"""

import numpy as np


# ── Constants (from paper §6.2 and mappings doc) ──────────────────────────────
HOLD_DURATIONS  = [0, 5, 10, 15, 20, 25, 30]   # minutes
N_ACTIONS       = len(HOLD_DURATIONS)            # 7
STATE_DIM       = 17
ALPHA           = 0.75   # PU/AU trade-off knob  (default from paper)
BETA            = 0.75   # local/global trade-off (default from paper)
FLIGHTS_PER_DAY = 460    # Air-East scale
DAYS_PER_EPISODE = 7
EPISODE_LENGTH  = FLIGHTS_PER_DAY * DAYS_PER_EPISODE   # ~3220 steps per episode


class AirlineEnv:
    """
    Stub environment. Produces realistic-looking random states and rewards.
    Drop-in replaceable with Member 1's real simulator — same interface.
    """

    def __init__(self, seed: int = 42, alpha: float = ALPHA, beta: float = BETA):
        self.rng        = np.random.default_rng(seed)
        self.alpha      = alpha
        self.beta       = beta
        self.step_count = 0

        # Simulate a slowly drifting "network health" so global state is not i.i.d.
        self._pg_global = 0.75   # global passenger utility (drifts over time)
        self._ag_global = 0.80   # global airline utility   (drifts over time)

    # ── Public interface (what you call in your training loop) ─────────────────

    def reset(self) -> np.ndarray:
        """Start a new episode. Returns initial state (shape: [17])."""
        self.step_count  = 0
        self._pg_global  = 0.75
        self._ag_global  = 0.80
        return self._make_state()

    def step(self, action: int):
        """
        Apply hold action and advance one flight decision.

        Parameters
        ----------
        action : int in {0..6}  →  hold duration {0,5,10,15,20,25,30} min

        Returns
        -------
        next_state : np.ndarray  shape [17]
        reward     : float
        done       : bool
        info       : dict  (extra diagnostics)
        """
        assert 0 <= action < N_ACTIONS, f"Invalid action {action}"

        hold_min = HOLD_DURATIONS[action]

        # ── Compute reward (stubs Member 3's reward engine) ────────────────────
        reward, info = self._compute_reward(action, hold_min)

        # ── Drift global utilities based on hold decision ──────────────────────
        self._update_global_state(hold_min)

        # ── Advance step counter ───────────────────────────────────────────────
        self.step_count += 1
        done = self.step_count >= EPISODE_LENGTH

        next_state = self._make_state()
        return next_state, reward, done, info

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def action_dim(self) -> int:
        return N_ACTIONS

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _make_state(self) -> np.ndarray:
        """
        Build a realistic state vector for one flight decision.
        Mirrors the structure from mappings.pdf / paper Table 1.
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)

        # Randomly decide: how delayed is the incoming flight? (0-30 min)
        incoming_delay = self.rng.integers(0, 31)

        # PL[0:7] — passenger utility per hold option
        # Holding longer helps if there's an incoming delay; hurts if not
        for i, tau in enumerate(HOLD_DURATIONS):
            if incoming_delay == 0:
                # No delay — holding is pointless, PAX utility flat/slightly decreasing
                state[i] = np.clip(0.95 - 0.03 * i + self.rng.normal(0, 0.02), 0, 1)
            else:
                # Delay exists — holding up to incoming_delay helps PAX
                if tau <= incoming_delay:
                    state[i] = np.clip(0.60 + 0.04 * i + self.rng.normal(0, 0.03), 0, 1)
                else:
                    state[i] = np.clip(0.85 - 0.02 * (i - incoming_delay // 5)
                                       + self.rng.normal(0, 0.02), 0, 1)

        # AL[7:14] — airline utility per hold option (decreases as hold increases)
        for i, tau in enumerate(HOLD_DURATIONS):
            state[7 + i] = np.clip(0.95 - 0.07 * i + self.rng.normal(0, 0.02), 0, 1)

        # PG[14] — global passenger utility (network-wide last 24h)
        state[14] = np.clip(self._pg_global + self.rng.normal(0, 0.03), 0, 1)

        # AG[15] — global airline utility (network-wide last 24h)
        state[15] = np.clip(self._ag_global + self.rng.normal(0, 0.03), 0, 1)

        # tau*[16] — locally optimal hold: argmax of weighted PL + AL, normalised to [0,1]
        local_score  = self.alpha * state[0:7] + (1 - self.alpha) * state[7:14]
        tau_star_idx = int(np.argmax(local_score))
        state[16]    = tau_star_idx / (N_ACTIONS - 1)   # normalise to [0,1]

        return state

    def _compute_reward(self, action: int, hold_min: int):
        """
        Stub reward — mirrors paper §5 reward formula:
            R_T = beta * R_L + (1 - beta) * R_G
            R_L = alpha * P_L + (1 - alpha) * A_L
            R_G = alpha * P_G + (1 - alpha) * A_G
        """
        # Measured local PU: holding helps if network is congested
        network_stress = 1.0 - self._pg_global          # 0=healthy, 1=stressed
        p_local = np.clip(
            0.70 + 0.04 * (action / N_ACTIONS) * network_stress
            + self.rng.normal(0, 0.05), 0, 1)

        # Measured local AU: holding hurts OTP
        a_local = np.clip(
            0.90 - 0.08 * (action / N_ACTIONS)
            + self.rng.normal(0, 0.03), 0, 1)

        # Global PU/AU attributed to this flight's hold (simplified delay tree)
        p_global = np.clip(self._pg_global + self.rng.normal(0, 0.02), 0, 1)
        a_global = np.clip(self._ag_global - 0.01 * action
                           + self.rng.normal(0, 0.02), 0, 1)

        # Paper reward formula
        r_local  = self.alpha * p_local  + (1 - self.alpha) * a_local
        r_global = self.alpha * p_global + (1 - self.alpha) * a_global
        reward   = self.beta * r_local   + (1 - self.beta)  * r_global

        info = {
            "p_local": p_local, "a_local": a_local,
            "p_global": p_global, "a_global": a_global,
            "r_local": r_local, "r_global": r_global,
            "hold_min": hold_min,
        }
        return float(reward), info

    def _update_global_state(self, hold_min: int):
        """Slowly drift global utilities — longer holds degrade network health."""
        drift = 0.0002 * hold_min   # small cost per hold decision
        self._ag_global = np.clip(
            self._ag_global - drift + self.rng.normal(0, 0.005), 0.3, 1.0)
        self._pg_global = np.clip(
            self._pg_global + 0.001 - drift * 0.5
            + self.rng.normal(0, 0.005), 0.3, 1.0)
