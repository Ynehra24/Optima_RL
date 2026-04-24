"""
agents/dqn.py — Deep Q-Network for Phase 2 Logistics HNH
==========================================================

Phase 2 changes vs Phase 1:
  - state_dim = 42 (34 local + 8 network context for multi-hub)
  - Q-network: 42 → [64, 64] → 7  (was 17 → [17] → 7)
  - Same ε-greedy exploration, same experience replay
  - Same target network (soft update τ=0.01)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.networks import MLP, Adam, ReplayBuffer, relu

STATE_DIM  = 42
ACTION_DIM = 7
HIDDEN     = [64, 64]


class DQNAgent:
    """Deep Q-Network agent for the 34-dim logistics state.

    Uses experience replay and a target network for stability.
    Epsilon-greedy exploration decays from 1.0 → 0.05 over training.
    """

    def __init__(
        self,
        state_dim:    int   = STATE_DIM,
        action_dim:   int   = ACTION_DIM,
        lr:           float = 0.0001,
        gamma:        float = 0.8,
        batch_size:   int   = 32,
        buffer_size:  int   = 10000,
        eps_start:    float = 1.0,
        eps_end:      float = 0.05,
        eps_decay:    int   = 15000,   # 10-hub: ~200 steps/ep → needs more exploration
        tau:          float = 0.01,    # soft target update
        seed:         int   = 42,
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.gamma      = gamma
        self.batch_size = batch_size
        self.tau        = tau

        # Epsilon-greedy parameters
        self.eps        = eps_start
        self.eps_end    = eps_end
        self.eps_decay  = eps_decay
        self._step      = 0

        # Q-network and target network
        self.q_net     = MLP(state_dim, HIDDEN, action_dim, seed=seed)
        self.tgt_net   = MLP(state_dim, HIDDEN, action_dim, seed=seed + 1)
        self._sync_target()   # hard copy at start

        self.optimiser = Adam(lr=lr)
        self.buffer    = ReplayBuffer(buffer_size, state_dim)

        # Logging
        self.losses      = []
        self.q_values    = []
        self.rewards_log = []

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection during training."""
        self._step += 1
        # Linear epsilon decay
        self.eps = max(
            self.eps_end,
            1.0 - (1.0 - self.eps_end) * self._step / self.eps_decay
        )
        if np.random.random() < self.eps:
            return int(np.random.randint(self.action_dim))
        return self._greedy(state)

    def greedy_action(self, state: np.ndarray) -> int:
        """Greedy action for evaluation (ε=0)."""
        return self._greedy(state)

    def _greedy(self, state: np.ndarray) -> int:
        q = self.q_net.forward(state)
        return int(np.argmax(q))

    def push(self, state, action, reward, next_state, done):
        """Add transition to replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
        self.rewards_log.append(reward)

    def update(self):
        """Sample minibatch and update Q-network."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.batch_size)

        # Current Q-values
        q_vals = np.array([self.q_net.forward(s) for s in states])
        # Target Q-values using target network
        q_next = np.array([self.tgt_net.forward(s) for s in next_states])
        targets = rewards + self.gamma * q_next.max(axis=1) * (1 - dones)
        targets = np.clip(targets, -10.0, 10.0)

        # TD-error and gradient computation
        total_loss = 0.0
        n = self.batch_size
        for i in range(n):
            q      = self.q_net.forward(states[i])
            target = targets[i]
            error  = q[actions[i]] - target
            loss   = error ** 2
            total_loss += loss
            # Gradient: 2 * error on the taken action only
            grad = np.zeros(self.action_dim, dtype=np.float32)
            grad[actions[i]] = 2.0 * error / n
            self.q_net.backward(grad)

        self.optimiser.step(self.q_net.params, self.q_net.grads)
        self._soft_update_target()

        avg_loss = total_loss / n
        self.losses.append(float(avg_loss))
        self.q_values.append(float(q_vals.max(axis=1).mean()))
        return float(avg_loss)

    def _sync_target(self):
        """Hard copy Q-net weights to target network."""
        for tgt, src in zip(self.tgt_net.params, self.q_net.params):
            tgt[:] = src

    def _soft_update_target(self):
        """Polyak soft update: θ_tgt = τ·θ + (1-τ)·θ_tgt"""
        for tgt, src in zip(self.tgt_net.params, self.q_net.params):
            tgt[:] = self.tau * src + (1 - self.tau) * tgt

    def get_metrics(self) -> dict:
        w = 1000
        return {
            "avg_reward_1k":  float(np.mean(self.rewards_log[-w:])) if self.rewards_log else 0.0,
            "avg_loss_1k":    float(np.mean(self.losses[-w:]))      if self.losses      else 0.0,
            "avg_value_1k":   float(np.mean(self.q_values[-w:]))    if self.q_values    else 0.0,
            "avg_entropy_1k": 0.0,  # DQN has no policy entropy
        }
