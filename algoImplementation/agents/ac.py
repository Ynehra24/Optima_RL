"""
agents/ac.py
------------
Actor-Critic (AC) agent — separate actor and critic networks.

Paper architecture (Table 3):
  Actor  : (17, 34, 17, 7)   → softmax → action probabilities
  Critic : ((17,7), 24, 17, 1) → state value V(s)
  (We simplify critic to take state only for clarity)

Key difference from A2C: actor and critic are completely separate networks
with their own optimisers.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.networks import MLP, Adam, softmax


class ACAgent:
    """
    Separate Actor-Critic. On-policy like A2C but with independent networks.
    Paper notes AC generalises less well than A2C during testing.
    """

    def __init__(
        self,
        state_dim:    int   = 17,
        action_dim:   int   = 7,
        lr:           float = 0.001,
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

        # Paper arch: Actor (17, 34, 17, 7), Critic (17, 24, 17, 1)
        self.actor  = MLP(state_dim, [34, 17], action_dim, seed=seed)
        self.critic = MLP(state_dim, [24, 17], 1,          seed=seed + 1)

        self.actor_opt  = Adam(lr=lr)
        self.critic_opt = Adam(lr=lr)

        # Trajectory buffer
        self._states  = []
        self._actions = []
        self._rewards = []
        self._values  = []
        self._dones   = []

        # Logging
        self.losses      = []
        self.q_values    = []
        self.rewards_log = []

    def select_action(self, state: np.ndarray):
        logits = self.actor.forward(state)
        probs  = softmax(logits)
        probs  = np.clip(probs, 1e-8, 1.0)
        probs /= probs.sum()
        action = int(np.random.choice(self.action_dim, p=probs))
        value  = float(self.critic.forward(state)[0])
        return action, value

    def greedy_action(self, state: np.ndarray) -> int:
        logits = self.actor.forward(state)
        return int(np.argmax(logits))

    def store(self, state, action, reward, value, done):
        self._states.append(state.copy())
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._dones.append(done)
        self.rewards_log.append(reward)

    def update(self, last_value: float = 0.0):
        if len(self._states) < self.batch_size and not self._dones[-1]:
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
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for i in range(n):
            # ── Actor update ───────────────────────────────────────────────────
            logits = self.actor.forward(states[i])
            probs  = softmax(logits)
            probs  = np.clip(probs, 1e-8, 1.0)

            adv     = float(advantages[i])
            entropy = -np.sum(probs * np.log(probs))

            one_hot  = np.zeros(self.action_dim, dtype=np.float32)
            one_hot[actions[i]] = 1.0
            grad_actor = (-adv * (one_hot - probs)
                          + self.entropy_coef * (np.log(probs) + 1)) / n
            self.actor.backward(grad_actor)

            # ── Critic update ──────────────────────────────────────────────────
            v_pred  = self.critic.forward(states[i])
            v_error = float(v_pred[0]) - float(returns[i])
            grad_v  = np.array([2 * self.value_coef * v_error / n])
            self.critic.backward(grad_v)

            total_loss += abs(adv) + v_error ** 2

        self.actor_opt.step(self.actor.params,   self.actor.grads)
        self.critic_opt.step(self.critic.params, self.critic.grads)

        self.losses.append(total_loss / n)
        self.q_values.append(float(np.mean(values)))

        self._states.clear(); self._actions.clear()
        self._rewards.clear(); self._values.clear(); self._dones.clear()

        return total_loss / n

    def get_metrics(self) -> dict:
        w = 1000
        return {
            "avg_reward_1k": float(np.mean(self.rewards_log[-w:])) if self.rewards_log else 0.0,
            "avg_loss_1k":   float(np.mean(self.losses[-w:]))      if self.losses else 0.0,
            "avg_value_1k":  float(np.mean(self.q_values[-w:]))    if self.q_values else 0.0,
        }
