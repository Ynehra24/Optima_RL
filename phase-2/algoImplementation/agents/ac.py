"""
agents/ac.py — Vanilla Actor-Critic (online, no batching) for Phase 2
======================================================================

Unlike A2C (which batches rollouts), AC updates every single step.
This is a faster but higher-variance learner — useful as a comparison.

Phase 2 changes:
  - state_dim = 42 (34 local + 8 network context for multi-hub)
  - Separate actor and critic networks (not shared trunk)
  - actor  : MLP(42 → [64, 64] → 7)  logits
  - critic : MLP(42 → [64, 64] → 1)  value
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.networks import MLP, Adam, relu, softmax

STATE_DIM  = 42
ACTION_DIM = 7
HIDDEN     = [64, 64]


class ACAgent:
    """Online Actor-Critic: updates at every step (no replay, no batching)."""

    def __init__(
        self,
        state_dim:    int   = STATE_DIM,
        action_dim:   int   = ACTION_DIM,
        lr:           float = 0.0001,
        gamma:        float = 0.8,
        batch_size:   int   = 32,    # kept for API compatibility; AC uses 1
        entropy_coef: float = 0.05,
        value_coef:   float = 0.5,
        seed:         int   = 42,
    ):
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef

        # Separate actor and critic (not shared trunk)
        self.actor    = MLP(state_dim, HIDDEN, action_dim, seed=seed)
        self.critic   = MLP(state_dim, HIDDEN, 1,          seed=seed + 5)
        self.opt_actor  = Adam(lr=lr)
        self.opt_critic = Adam(lr=lr)

        # Rollout buffer (used for compatible train loop)
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

    def _get_probs_value(self, state: np.ndarray):
        logits = self.actor.forward(state)
        probs  = softmax(logits)
        value  = float(self.critic.forward(state).flat[0])
        return probs, value

    def select_action(self, state: np.ndarray):
        probs, value = self._get_probs_value(state)
        probs        = np.clip(probs, 1e-8, 1.0)
        probs       /= probs.sum()
        action       = int(np.random.choice(self.action_dim, p=probs))
        return action, float(value)

    def greedy_action(self, state: np.ndarray) -> int:
        probs, _ = self._get_probs_value(state)
        probs     = np.clip(probs, 1e-8, 1.0)
        probs    /= probs.sum()
        return int(np.argmax(probs))

    def store(self, state, action, reward, value, done):
        self._states.append(state.copy())
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._dones.append(done)
        self.rewards_log.append(reward)

    def update(self, last_value: float = 0.0):
        """Update actor and critic using buffered rollout."""
        if not self._states:
            return None

        n         = len(self._states)
        states    = np.array(self._states,  dtype=np.float32)
        actions   = np.array(self._actions, dtype=np.int32)
        rewards   = np.array(self._rewards, dtype=np.float32)
        values    = np.array(self._values,  dtype=np.float32)
        dones     = np.array(self._dones,   dtype=np.float32)

        # Discounted returns
        returns = np.zeros(n, dtype=np.float32)
        R = last_value
        for t in reversed(range(n)):
            R          = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R
        returns = np.clip(returns, -10.0, 10.0)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0

        for i in range(n):
            # Actor update
            probs         = softmax(self.actor.forward(states[i]))
            probs_clipped = np.clip(probs, 1e-8, 1.0)
            probs_clipped /= probs_clipped.sum()
            entropy       = -np.sum(probs_clipped * np.log(probs_clipped))
            adv           = float(advantages[i])

            one_hot      = np.zeros(self.action_dim, dtype=np.float32)
            one_hot[actions[i]] = 1.0
            grad_actor   = -adv * (one_hot - probs_clipped) / n
            grad_entropy = self.entropy_coef * (np.log(probs_clipped) + 1) / n
            self.actor.backward(grad_actor + grad_entropy)

            total_policy_loss += float(-np.log(probs_clipped[actions[i]]) * adv)
            total_entropy     += float(entropy)

            # Critic update
            v_pred  = float(self.critic.forward(states[i]).flat[0])
            v_error = v_pred - float(returns[i])
            self.critic.backward(np.array([2 * self.value_coef * v_error / n]))
            total_value_loss += v_error ** 2

        self.opt_actor.step(self.actor.params,  self.actor.grads)
        self.opt_critic.step(self.critic.params, self.critic.grads)

        avg_loss = total_policy_loss / n + self.value_coef * total_value_loss / n
        self.losses.append(avg_loss)
        self.policy_losses.append(total_policy_loss / n)
        self.value_losses.append(total_value_loss / n)
        self.entropies.append(total_entropy / n)
        self.q_values.append(float(np.mean(values)))

        self._states.clear(); self._actions.clear()
        self._rewards.clear(); self._values.clear(); self._dones.clear()
        return avg_loss

    def get_metrics(self) -> dict:
        w = 1000
        return {
            "avg_reward_1k":  float(np.mean(self.rewards_log[-w:])) if self.rewards_log else 0.0,
            "avg_loss_1k":    float(np.mean(self.losses[-w:]))      if self.losses      else 0.0,
            "avg_value_1k":   float(np.mean(self.q_values[-w:]))    if self.q_values    else 0.0,
            "avg_entropy_1k": float(np.mean(self.entropies[-w:]))   if self.entropies   else 0.0,
        }
