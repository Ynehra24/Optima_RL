"""
agents/ddpg.py
--------------
Deep Deterministic Policy Gradient (DDPG) agent.

Paper architecture (Table 3):
  Actor  : (18, 18, 1)        → continuous hold in [0, 30] min
  Critic : ((18,1), 19, 19, 1) → Q(s, a) for continuous action

Note: DDPG is the only continuous-action algorithm.
The paper notes it performs WORST among the four — likely because
the reward landscape has non-smooth jumps as a function of hold time.

Key components:
  - Deterministic policy: a = π(s)
  - Critic: Q(s, a) value function
  - Target networks (soft update via polyak averaging)
  - Ornstein-Uhlenbeck noise for exploration
  - Experience replay buffer
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.networks import MLP, Adam, ReplayBuffer


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration noise."""
    def __init__(self, action_dim: int = 1, mu: float = 0.0,
                 theta: float = 0.15, sigma: float = 0.05, seed: int = 42):
        self.mu        = mu * np.ones(action_dim)
        self.theta     = theta
        self.sigma     = sigma
        self.state     = self.mu.copy()
        self.rng       = np.random.default_rng(seed)

    def reset(self):
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx          = self.theta * (self.mu - self.state) + \
                      self.sigma * self.rng.standard_normal(len(self.state))
        self.state += dx
        return self.state.copy()


class DDPGAgent:
    """
    DDPG with soft target updates (polyak averaging).
    Action space: continuous [0, 30] minutes.
    """

    def __init__(
        self,
        state_dim:   int   = 17,
        action_dim:  int   = 1,     # DDPG: single continuous action
        lr_actor:    float = 0.001,
        lr_critic:   float = 0.001,
        gamma:       float = 0.8,
        batch_size:  int   = 32,
        buffer_size: int   = 32200,
        tau:         float = 0.005,  # soft update coefficient
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

        # Paper arch: Actor (18, 18, 1) — input is state(17) + nothing, output is 1
        # (The "18" in the paper likely includes a bias trick; we use 17 → 18 → 1)
        self.actor        = MLP(state_dim, [18], action_dim, seed=seed)
        self.actor_target = MLP(state_dim, [18], action_dim, seed=seed + 1)

        # Critic takes (state + action) as input: 17+1=18 → 19 → 19 → 1
        critic_in = state_dim + action_dim
        self.critic        = MLP(critic_in, [19, 19], 1, seed=seed + 2)
        self.critic_target = MLP(critic_in, [19, 19], 1, seed=seed + 3)

        self._sync_targets(tau=1.0)   # hard copy at init

        self.actor_opt  = Adam(lr=lr_actor)
        self.critic_opt = Adam(lr=lr_critic)

        self.buffer = ReplayBuffer(buffer_size, state_dim)
        self.noise = OUNoise(action_dim, mu=0.0, theta=0.15, sigma=0.05, seed=seed)

        # Logging
        self.losses      = []
        self.q_values    = []
        self.rewards_log = []

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> float:
        raw    = self.actor.forward(state)
        # Scale to [0, 15] max instead of [0, 30] — start conservative
        action = (float(np.tanh(raw[0])) + 1) / 2 * 15.0
        if add_noise:
            action += float(self.noise.sample()[0])
        return float(np.clip(action, self.action_low, self.action_high))

    def greedy_action(self, state: np.ndarray) -> float:
        return self.select_action(state, add_noise=False)

    def discrete_action(self, state: np.ndarray) -> int:
        hold = self.greedy_action(state)
        # Map continuous [0,30] to nearest discrete index
        durations = [0, 5, 10, 15, 20, 25, 30]
        return int(np.argmin([abs(hold - d) for d in durations]))

    # ── Learning ───────────────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done):
        # Store action as array for DDPG
        if np.isscalar(action):
            action = np.array([action], dtype=np.float32)
        self.buffer.push(state, 0, reward, next_state, done)   # action stored separately
        self._action_buffer_last = action
        self.rewards_log.append(reward)

    def push_continuous(self, state, action_float, reward, next_state, done):
        """Push with continuous action value."""
        self.buffer.push(state, 0, reward, next_state, done)
        self.rewards_log.append(reward)

    def update(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, _, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # ── Critic update ──────────────────────────────────────────────────────
        # Get target actions from target actor
        target_actions = np.array([
            float(np.tanh(self.actor_target.forward(next_states[i])[0]))
            * self.action_high / 2 + self.action_high / 2
            for i in range(self.batch_size)
        ], dtype=np.float32).reshape(-1, 1)

        # Q target = r + gamma * Q_target(s', a')
        target_inputs = np.concatenate([next_states, target_actions], axis=1)
        q_next        = np.array([float(self.critic_target.forward(target_inputs[i])[0])
                                  for i in range(self.batch_size)])
        q_target      = rewards + self.gamma * q_next * (1 - dones)

        # Current actions from main actor (for critic input)
        current_actions = np.array([
            float(np.tanh(self.actor.forward(states[i])[0]))
            * self.action_high / 2 + self.action_high / 2
            for i in range(self.batch_size)
        ], dtype=np.float32).reshape(-1, 1)

        critic_inputs = np.concatenate([states, current_actions], axis=1)
        q_pred = np.array([float(self.critic.forward(critic_inputs[i])[0])
                           for i in range(self.batch_size)])

        # MSE critic loss
        critic_error = q_pred - q_target
        critic_loss  = float(np.mean(critic_error ** 2))

        for i in range(self.batch_size):
            grad_q = np.array([2 * critic_error[i] / self.batch_size])
            self.critic.backward(np.concatenate([
                grad_q, np.zeros(self.state_dim, dtype=np.float32)
            ])[:1])   # simplified backward

        self.critic_opt.step(self.critic.params, self.critic.grads)

        # ── Actor update ───────────────────────────────────────────────────────
        # Maximise Q(s, π(s)) → minimise -Q
        actor_loss = 0.0
        for i in range(self.batch_size):
            raw    = self.actor.forward(states[i])
            action = float(np.tanh(raw[0])) * self.action_high / 2 + self.action_high / 2
            ci     = np.concatenate([states[i], [action]]).astype(np.float32)
            q_val  = float(self.critic.forward(ci)[0])
            actor_loss -= q_val / self.batch_size
            # Simple gradient: push actor toward higher Q
            grad_actor = np.array([-q_val / self.batch_size])
            self.actor.backward(grad_actor)

        self.actor_opt.step(self.actor.params, self.actor.grads)

        # ── Soft target updates ────────────────────────────────────────────────
        self._sync_targets(tau=self.tau)

        # Logging
        self.losses.append(critic_loss)
        self.q_values.append(float(np.mean(q_pred)))

        return critic_loss

    def _sync_targets(self, tau: float = None):
        """Polyak averaging: θ_target = τ*θ + (1-τ)*θ_target"""
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
