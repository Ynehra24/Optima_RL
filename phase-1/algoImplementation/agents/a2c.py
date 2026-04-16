"""
agents/a2c.py  (v2 — fixed for real simulator)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.networks import MLP, Adam, relu, softmax


class A2CNetwork:
    def __init__(self, state_dim: int = 17, action_dim: int = 7, seed: int = 42):
        self.trunk  = MLP(state_dim, [17], 17, seed=seed)
        self.policy = MLP(17, [], action_dim, seed=seed + 10)
        self.value  = MLP(17, [], 1, seed=seed + 20)
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

    def __init__(
        self,
        state_dim:    int   = 17,
        action_dim:   int   = 7,
        lr:           float = 0.0001,
        gamma:        float = 0.8,
        batch_size:   int   = 32,
        entropy_coef: float = 0.05,
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

        self._states  = []
        self._actions = []
        self._rewards = []
        self._values  = []
        self._dones   = []

        self.losses        = []
        self.policy_losses = []
        self.value_losses  = []
        self.entropies     = []
        self.q_values      = []
        self.rewards_log   = []

    def select_action(self, state: np.ndarray):
        """Stochastic — used during training."""
        probs, value = self.network.forward(state)
        probs        = np.clip(probs, 1e-8, 1.0)
        probs       /= probs.sum()
        action       = int(np.random.choice(self.action_dim, p=probs))
        return action, float(value)

    def greedy_action(self, state: np.ndarray) -> int:
        """
        Confident greedy — used during evaluation.
        Only hold if agent assigns >30% probability to that hold action.
        Prevents 100%-hold collapse from pure argmax on a weak policy.
        """
        probs, _ = self.network.forward(state)
        probs     = np.clip(probs, 1e-8, 1.0)
        probs    /= probs.sum()
        best      = int(np.argmax(probs))
        if best == 0:
            return 0
        if probs[best] >= 0.30:
            return best
        return 0

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
            R          = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R
        returns = np.clip(returns, -10.0, 10.0)   # prevent explosion

        # Normalised advantages
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

            one_hot = np.zeros(self.action_dim, dtype=np.float32)
            one_hot[actions[i]] = 1.0

            grad_policy  = -adv * (one_hot - probs_clipped) / n
            grad_entropy = self.entropy_coef * (np.log(probs_clipped) + 1) / n
            self.network.policy.backward(grad_policy + grad_entropy)

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

    def get_metrics(self) -> dict:
        w = 1000
        return {
            "avg_reward_1k":  float(np.mean(self.rewards_log[-w:])) if self.rewards_log else 0.0,
            "avg_loss_1k":    float(np.mean(self.losses[-w:]))      if self.losses else 0.0,
            "avg_value_1k":   float(np.mean(self.q_values[-w:]))    if self.q_values else 0.0,
            "avg_entropy_1k": float(np.mean(self.entropies[-w:]))   if self.entropies else 0.0,
        }