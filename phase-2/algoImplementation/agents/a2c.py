"""
agents/a2c.py — Phase 2 Logistics HNH
======================================
Advantage Actor-Critic for the Cross-Docking Hold-or-Not-Hold problem.

Identical algorithm to phase-1/algoImplementation/agents/a2c.py.
Only change: default state_dim = 34 (logistics 34-dim state vector
instead of aviation 17-dim).

Architecture:
  shared trunk  : state_dim → [64, 64] → 64
  policy head   : 64 → action_dim      (softmax → discrete distribution)
  value  head   : 64 → 1               (scalar baseline)

Update rule:
  policy loss = -log π(a|s) · A(s,a)  − entropy_coef · H(π)
  value  loss = value_coef  · (V(s) − R)²
  A(s,a)      = R − V(s)   (normalised per batch)
  R           = discounted returns (γ-weighted, bootstrapped)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.networks import MLP, Adam, relu, softmax


class A2CNetwork:
    """Shared-trunk actor-critic network."""

    def __init__(self, state_dim: int = 34, action_dim: int = 7, seed: int = 42):
        # Wider first layer to handle the richer 34-dim logistics input
        self.trunk  = MLP(state_dim, [128, 64], 64, seed=seed)
        self.policy = MLP(64, [], action_dim, seed=seed + 10)
        self.value  = MLP(64, [], 1,          seed=seed + 20)
        self._trunk_out = None

    def forward(self, state: np.ndarray):
        trunk_raw       = self.trunk.forward(state)
        self._trunk_out = relu(trunk_raw)
        policy_logits   = self.policy.forward(self._trunk_out)
        probs           = softmax(policy_logits)
        value_out       = self.value.forward(self._trunk_out)
        value           = float(value_out.flat[0])
        return probs, value

    @property
    def params(self):
        return self.trunk.params + self.policy.params + self.value.params

    @property
    def grads(self):
        return self.trunk.grads + self.policy.grads + self.value.grads


class A2CAgent:
    """
    A2C agent for logistics Hold-or-Not-Hold.

    API (identical to Phase 1):
        action, value = agent.select_action(state)   # training (stochastic)
        action        = agent.greedy_action(state)   # evaluation (deterministic)
        agent.store(state, action, reward, value, done)
        agent.update(last_value=0.0)
    """

    def __init__(
        self,
        state_dim:    int   = 34,
        action_dim:   int   = 7,
        lr:           float = 3e-4,
        gamma:        float = 0.8,
        batch_size:   int   = 32,
        entropy_coef: float = 0.01,
        value_coef:   float = 0.5,
        seed:         int   = 42,
    ):
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.batch_size   = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef

        self.network   = A2CNetwork(state_dim, action_dim, seed)
        self.optimiser = Adam(lr=lr)

        # Rollout buffer
        self._states  = []
        self._actions = []
        self._rewards = []
        self._values  = []
        self._dones   = []

        # Logging
        self.losses        = []
        self.policy_losses = []
        self.value_losses  = []
        self.entropies     = []
        self.q_values      = []
        self.rewards_log   = []

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray):
        """Stochastic — used during training."""
        probs, value = self.network.forward(state)
        probs        = np.clip(probs, 1e-8, 1.0)
        probs       /= probs.sum()
        action       = int(np.random.choice(self.action_dim, p=probs))
        return action, float(value)

    def greedy_action(self, state: np.ndarray) -> int:
        """
        Greedy action for evaluation — returns argmax of policy unless
        a non-zero hold fails a minimum confidence bar (15%).
        Threshold of 15% (vs old 30%) is calibrated to the practical
        max probabilities seen after convergence on a 7-action space.
        """
        probs, _ = self.network.forward(state)
        probs     = np.clip(probs, 1e-8, 1.0)
        probs    /= probs.sum()
        best      = int(np.argmax(probs))
        # Always allow no-hold (action 0)
        if best == 0:
            return 0
        # Allow hold only if agent is meaningfully confident (>15%)
        if probs[best] >= 0.15:
            return best
        return 0

    # ── Buffer ────────────────────────────────────────────────────────────────

    def store(self, state, action, reward, value, done):
        self._states.append(state.copy())
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._dones.append(done)
        self.rewards_log.append(reward)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, last_value: float = 0.0):
        """Run one A2C gradient update over the buffered rollout."""
        if len(self._states) < self.batch_size and not self._dones[-1]:
            return None

        n         = len(self._states)
        states    = np.array(self._states,  dtype=np.float32)
        actions   = np.array(self._actions, dtype=np.int32)
        rewards   = np.array(self._rewards, dtype=np.float32)
        values    = np.array(self._values,  dtype=np.float32)
        dones     = np.array(self._dones,   dtype=np.float32)

        # ── Discounted returns (bootstrapped) ─────────────────────────────────
        returns = np.zeros(n, dtype=np.float32)
        R = last_value
        for t in reversed(range(n)):
            R          = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R
        returns = np.clip(returns, -10.0, 10.0)

        # ── Normalised advantages ─────────────────────────────────────────────
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0

        for i in range(n):
            probs, value_pred = self.network.forward(states[i])
            probs_clipped     = np.clip(probs, 1e-8, 1.0)
            probs_clipped    /= probs_clipped.sum()

            log_prob = np.log(probs_clipped[actions[i]])
            entropy  = -np.sum(probs_clipped * np.log(probs_clipped))
            adv      = float(advantages[i])
            ret      = float(returns[i])
            p_loss   = -log_prob * adv
            v_error  = value_pred - ret
            v_loss   = v_error ** 2

            # ── Policy gradient + entropy bonus ───────────────────────────────
            one_hot = np.zeros(self.action_dim, dtype=np.float32)
            one_hot[actions[i]] = 1.0
            grad_policy  = -adv * (one_hot - probs_clipped) / n
            grad_entropy = self.entropy_coef * (np.log(probs_clipped) + 1) / n
            self.network.policy.backward(grad_policy + grad_entropy)

            # ── Value gradient ────────────────────────────────────────────────
            grad_value = np.array([2 * self.value_coef * v_error / n])
            self.network.value.backward(grad_value)

            total_policy_loss += float(p_loss)
            total_value_loss  += float(v_loss)
            total_entropy     += float(entropy)

        self.optimiser.step(self.network.params, self.network.grads)

        avg_loss = total_policy_loss / n + self.value_coef * total_value_loss / n
        self.losses.append(avg_loss)
        self.policy_losses.append(total_policy_loss / n)
        self.value_losses.append(total_value_loss / n)
        self.entropies.append(total_entropy / n)
        self.q_values.append(float(np.mean(values)))

        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()

        return avg_loss

    # ── Metrics ───────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        w = 1000
        return {
            "avg_reward_1k":  float(np.mean(self.rewards_log[-w:])) if self.rewards_log else 0.0,
            "avg_loss_1k":    float(np.mean(self.losses[-w:]))      if self.losses else 0.0,
            "avg_value_1k":   float(np.mean(self.q_values[-w:]))    if self.q_values else 0.0,
            "avg_entropy_1k": float(np.mean(self.entropies[-w:]))   if self.entropies else 0.0,
        }
