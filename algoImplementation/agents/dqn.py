"""
agents/dqn.py
-------------
Deep Q-Network (DQN) agent.

Paper architecture (Table 3): (17, 17, 7)
  Input  : state_dim = 17
  Hidden : 17 neurons, ReLU
  Output : action_dim = 7  (Q-value per hold duration)

Paper hyperparameters:
  lr = 0.001, gamma = 0.8, batch = 32
  replay buffer = 10% of training epochs
  epsilon-greedy exploration
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.networks import MLP, Adam, ReplayBuffer


class DQNAgent:
    """
    DQN with:
      - Experience replay
      - Target network (hard update every `target_update` steps)
      - Epsilon-greedy exploration with linear decay
    """

    def __init__(
        self,
        state_dim:     int   = 17,
        action_dim:    int   = 7,
        lr:            float = 0.001,
        gamma:         float = 0.8,
        batch_size:    int   = 32,
        buffer_size:   int   = 32200,     # 10% of 25 eps * 3220 steps
        eps_start:     float = 1.0,
        eps_end:       float = 0.05,
        eps_decay:     int   = 50000,
        target_update: int   = 500,
        seed:          int   = 42,
    ):
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.eps          = eps_start
        self.eps_end      = eps_end
        self.eps_decay    = eps_decay
        self.target_update = target_update
        self.step_count   = 0

        # Q-network and target network — paper arch (17, 17, 7)
        self.q_net      = MLP(state_dim, [17], action_dim, seed=seed)
        self.target_net = MLP(state_dim, [17], action_dim, seed=seed + 1)
        self._sync_target()

        self.optimiser = Adam(lr=lr)
        self.buffer    = ReplayBuffer(buffer_size, state_dim)

        # Logging
        self.losses     = []
        self.q_values   = []
        self.rewards_log = []

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy policy."""
        # Decay epsilon
        self.eps = max(
            self.eps_end,
            self.eps - (1.0 - self.eps_end) / self.eps_decay
        )
        if np.random.random() < self.eps:
            return np.random.randint(self.action_dim)
        q = self.q_net.forward(state)
        return int(np.argmax(q))

    def greedy_action(self, state: np.ndarray) -> int:
        """Pure greedy — used during testing."""
        q = self.q_net.forward(state)
        return int(np.argmax(q))

    # ── Learning ───────────────────────────────────────────────────────────────

    def push(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        self.rewards_log.append(reward)

    def update(self) -> float | None:
        """Sample a minibatch and do one gradient step. Returns loss."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        # ── Compute targets: r + gamma * max_a' Q_target(s', a') ──────────────
        q_next   = self.target_net.forward(next_states)           # (B, 7)
        q_target = rewards + self.gamma * np.max(q_next, axis=1) * (1 - dones)

        # ── Current Q-values ──────────────────────────────────────────────────
        q_pred_all = self.q_net.forward(states)                   # (B, 7)
        q_pred     = q_pred_all[np.arange(self.batch_size), actions]  # (B,)

        # ── MSE loss + backward ───────────────────────────────────────────────
        td_error = q_pred - q_target                              # (B,)
        loss     = float(np.mean(td_error ** 2))

        # Gradient of MSE w.r.t. q_pred_all
        grad_out        = np.zeros_like(q_pred_all)
        grad_out[np.arange(self.batch_size), actions] = \
            (2 / self.batch_size) * td_error

        self.q_net.backward(grad_out)
        self.optimiser.step(self.q_net.params, self.q_net.grads)

        # ── Logging ───────────────────────────────────────────────────────────
        self.losses.append(loss)
        self.q_values.append(float(np.mean(np.max(q_pred_all, axis=1))))

        # ── Hard target update ────────────────────────────────────────────────
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self._sync_target()

        return loss

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _sync_target(self):
        """Copy weights from Q-net to target net."""
        for src_layer, tgt_layer in zip(self.q_net.layers, self.target_net.layers):
            tgt_layer.W = src_layer.W.copy()
            tgt_layer.b = src_layer.b.copy()

    def get_metrics(self) -> dict:
        window = 1000
        return {
            "avg_reward_1k": float(np.mean(self.rewards_log[-window:]))
                             if self.rewards_log else 0.0,
            "avg_loss_1k":   float(np.mean(self.losses[-window:]))
                             if self.losses else 0.0,
            "avg_q_1k":      float(np.mean(self.q_values[-window:]))
                             if self.q_values else 0.0,
            "epsilon":       self.eps,
        }
