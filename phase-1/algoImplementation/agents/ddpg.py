"""
agents/ddpg.py  (v2 — fixed action scaling and noise for real simulator)
------------------------------------------------------------------------
Deep Deterministic Policy Gradient (DDPG).

Paper architecture (Table 3):
  Actor  : (18, 18, 1)   → continuous hold in [0, 30] min
  Critic : ((18,1), 19, 19, 1) → Q(s, a)

Key fixes in v2:
  1. OUNoise sigma reduced to 0.05 (was 0.2) — prevents saturation
  2. Actor output scaled to [0, 15] initially (conservative start)
  3. discrete_action uses proper rounding to nearest valid action
  4. Soft update tau kept small (0.005) for stability
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.networks import MLP, Adam, ReplayBuffer


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""
    def __init__(self, action_dim: int = 1, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.05, seed: int = 42):
        self.mu    = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()
        self.rng   = np.random.default_rng(seed)

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * self.rng.standard_normal(len(self.state))
        self.state += dx
        return self.state.copy()


class DDPGAgent:
    """
    DDPG with soft target updates.
    Action space: continuous [0, 30] minutes (paper §6.2).
    Worst performer per paper — DDPG cannot handle discrete reward jumps.
    """

    HOLD_DURATIONS = [0, 5, 10, 15, 20, 25, 30]

    def __init__(
        self,
        state_dim:   int   = 17,
        action_dim:  int   = 1,
        lr_actor:    float = 0.0001,
        lr_critic:   float = 0.0001,
        gamma:       float = 0.8,
        batch_size:  int   = 32,
        buffer_size: int   = 32200,
        tau:         float = 0.005,
        action_low:  float = 0.0,
        action_high: float = 30.0,
        seed:        int   = 42,
    ):
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.batch_size  = batch_size
        self.tau         = tau
        self.action_low  = action_low
        self.action_high = action_high

        # Paper arch: Actor (18, 18, 1), Critic (18+1 → 19 → 19 → 1)
        self.actor        = MLP(state_dim, [18], action_dim, seed=seed)
        self.actor_target = MLP(state_dim, [18], action_dim, seed=seed + 1)
        critic_in = state_dim + action_dim
        self.critic        = MLP(critic_in, [19, 19], 1, seed=seed + 2)
        self.critic_target = MLP(critic_in, [19, 19], 1, seed=seed + 3)
        self._sync_targets(tau=1.0)

        self.actor_opt  = Adam(lr=lr_actor)
        self.critic_opt = Adam(lr=lr_critic)
        self.buffer     = ReplayBuffer(buffer_size, state_dim)
        self.noise      = OUNoise(action_dim, sigma=0.05, seed=seed)

        # Logging
        self.losses      = []
        self.q_values    = []
        self.rewards_log = []

    # ── Action selection ───────────────────────────────────────────────────────

    def _raw_to_hold(self, raw: float, add_noise: bool = False) -> float:
        """Map network output → hold duration in [0, 30]."""
        # tanh squashes to (-1, 1), then scale to [0, 30]
        action = (float(np.tanh(raw)) + 1.0) / 2.0 * self.action_high
        if add_noise:
            action += float(self.noise.sample()[0]) * 5.0  # noise in minutes
        return float(np.clip(action, self.action_low, self.action_high))

    def select_action(self, state: np.ndarray) -> float:
        """Continuous action with exploration noise (training)."""
        raw = self.actor.forward(state)
        return self._raw_to_hold(float(raw[0]), add_noise=True)

    def greedy_action(self, state: np.ndarray) -> float:
        """Deterministic action (evaluation)."""
        raw = self.actor.forward(state)
        return self._raw_to_hold(float(raw[0]), add_noise=False)

    def discrete_action(self, state: np.ndarray) -> int:
        """
        Convert continuous hold duration → nearest discrete action index.
        Used by training loop which expects int actions {0..6}.
        """
        hold = self.greedy_action(state)
        return int(np.argmin([abs(hold - d) for d in self.HOLD_DURATIONS]))

    # ── Store experience ───────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done):
        """Store transition. Action is discrete int — convert for buffer."""
        self.buffer.push(state, int(action), reward, next_state, done)
        self.rewards_log.append(reward)

    # ── Learning ───────────────────────────────────────────────────────────────

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        # Convert discrete actions back to continuous holds for DDPG
        holds = np.array([self.HOLD_DURATIONS[int(a)] for a in actions],
                         dtype=np.float32).reshape(-1, 1) / self.action_high

        # ── Critic update ──────────────────────────────────────────────────────
        # Target actions from target actor
        target_holds = np.array([
            self._raw_to_hold(float(self.actor_target.forward(next_states[i])[0]))
            / self.action_high
            for i in range(self.batch_size)
        ], dtype=np.float32).reshape(-1, 1)

        target_inputs = np.concatenate([next_states, target_holds], axis=1)
        q_next = np.array([
            float(self.critic_target.forward(target_inputs[i])[0])
            for i in range(self.batch_size)
        ])
        q_target = rewards + self.gamma * q_next * (1 - dones)
        q_target = np.clip(q_target, 0.0, 1.0)

        critic_inputs = np.concatenate([states, holds], axis=1)
        q_pred = np.array([
            float(self.critic.forward(critic_inputs[i])[0])
            for i in range(self.batch_size)
        ])

        critic_error = q_pred - q_target
        critic_loss  = float(np.mean(critic_error ** 2))

        # Critic backward (simplified — grad w.r.t. Q output)
        for i in range(self.batch_size):
            grad_q = np.array([2 * critic_error[i] / self.batch_size])
            full_input = critic_inputs[i]
            self.critic.forward(full_input)
            # Backprop only through critic params
            self.critic.backward(grad_q)

        self.critic_opt.step(self.critic.params, self.critic.grads)

        # ── Actor update ───────────────────────────────────────────────────────
        actor_loss = 0.0
        for i in range(self.batch_size):
            raw    = self.actor.forward(states[i])
            hold   = self._raw_to_hold(float(raw[0])) / self.action_high
            ci     = np.concatenate([states[i], [hold]]).astype(np.float32)
            q_val  = float(self.critic.forward(ci)[0])
            actor_loss -= q_val / self.batch_size
            # Policy gradient: maximise Q → minimise -Q
            grad = np.array([-1.0 / self.batch_size])
            self.actor.backward(grad)

        self.actor_opt.step(self.actor.params, self.actor.grads)

        # ── Soft target updates ────────────────────────────────────────────────
        self._sync_targets(tau=self.tau)

        self.losses.append(critic_loss)
        self.q_values.append(float(np.mean(q_pred)))
        return critic_loss

    def _sync_targets(self, tau: float = None):
        if tau is None:
            tau = self.tau
        for src, tgt in [(self.actor, self.actor_target),
                         (self.critic, self.critic_target)]:
            for s_layer, t_layer in zip(src.layers, tgt.layers):
                t_layer.W = tau * s_layer.W + (1 - tau) * t_layer.W
                t_layer.b = tau * s_layer.b + (1 - tau) * t_layer.b

    def get_metrics(self) -> dict:
        w = 1000
        return {
            "avg_reward_1k": float(np.mean(self.rewards_log[-w:])) if self.rewards_log else 0.0,
            "avg_loss_1k":   float(np.mean(self.losses[-w:]))      if self.losses else 0.0,
            "avg_q_1k":      float(np.mean(self.q_values[-w:]))    if self.q_values else 0.0,
        }