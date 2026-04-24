"""
agents/a2c.py — Advantage Actor-Critic for Phase 2 Logistics HNH
=================================================================

Key differences from Phase 1:
  - state_dim = 42 (34 local + 8 network context for multi-hub)
  - Wider trunk: 42 → 64 → 64  (was 17 → 17)
    The extra width is needed to learn from the 17 new logistics features
    (bay utilization, cargo value, SLA urgency, perishability, etc.)
    plus 8 downstream network context features (cascade awareness)
  - Same A2C update math — advantage, policy gradient, entropy bonus
  - Same Adam optimiser, same hyperparameters (paper §6.2)

Network architecture:
  trunk  : MLP(42 → [64, 64] → 64)   shared feature extractor
  policy : MLP(64 → [] → 7)           logits → softmax → action probs
  value  : MLP(64 → [] → 1)           state value V(s)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.networks import MLP, Adam, relu, softmax

# ── Phase 2 network dimensions ─────────────────────────────────────────────────
STATE_DIM  = 42   # 34 local + 8 network context (multi-hub) | set to 34 for single-hub
ACTION_DIM = 7    # hold ∈ {0,5,10,15,20,25,30} min
TRUNK_OUT  = 64   # shared trunk output dim
HIDDEN     = [64, 64]   # two hidden layers for richer representation


class A2CNetwork:
    """Shared-trunk Actor-Critic network for the 42-dim logistics state."""

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 seed: int = 42):
        # Shared trunk: learns logistics-aware features
        self.trunk  = MLP(state_dim, HIDDEN, TRUNK_OUT, seed=seed)
        # Policy head: maps trunk features → action logits
        self.policy = MLP(TRUNK_OUT, [], action_dim, seed=seed + 10)
        # Value head: maps trunk features → scalar state value
        self.value  = MLP(TRUNK_OUT, [], 1, seed=seed + 20)
        self._trunk_out = None

    def forward(self, state: np.ndarray):
        """Forward pass through trunk → policy + value heads.

        Args:
            state: shape (34,) normalized logistics state vector

        Returns:
            (probs, value): action probability vector and scalar value
        """
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
    """Advantage Actor-Critic agent for the Phase 2 logistics HNH problem.

    Identical update math to Phase 1 A2C — only the network is wider.
    This makes the comparison between Phase 1 and Phase 2 directly fair.
    """

    def __init__(
        self,
        state_dim:    int   = STATE_DIM,
        action_dim:   int   = ACTION_DIM,
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

        # Rollout buffer
        self._states  = []
        self._actions = []
        self._rewards = []
        self._values  = []
        self._dones   = []

        # Training history
        self.losses        = []
        self.policy_losses = []
        self.value_losses  = []
        self.entropies     = []
        self.q_values      = []
        self.rewards_log   = []

    def select_action(self, state: np.ndarray):
        """Stochastic action sampling — used during training.

        Returns:
            (action, value): int action index and float state value estimate
        """
        probs, value = self.network.forward(state)
        probs        = np.clip(probs, 1e-8, 1.0)
        probs       /= probs.sum()
        action       = int(np.random.choice(self.action_dim, p=probs))
        return action, float(value)

    def greedy_action(self, state: np.ndarray) -> int:
        """Confident greedy action — used during evaluation.

        Only holds if agent assigns ≥18% probability to that action
        (just above uniform 1/7 ≈ 14.3%).
        Prevents 100%-hold collapse from pure argmax on weak policy.
        """
        probs, _ = self.network.forward(state)
        probs     = np.clip(probs, 1e-8, 1.0)
        probs    /= probs.sum()
        best      = int(np.argmax(probs))
        if best == 0:
            return 0
        return best if probs[best] >= 0.18 else 0

    def store(self, state, action, reward, value, done):
        """Store one transition in the rollout buffer."""
        self._states.append(state.copy())
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._dones.append(done)
        self.rewards_log.append(reward)

    def update(self, last_value: float = 0.0):
        """Update policy and value using accumulated rollout buffer.

        Uses discounted returns and normalized advantages.
        Called every batch_size steps or at episode end.

        Args:
            last_value: Bootstrap value for the last state (0 if terminal)

        Returns:
            Combined loss (float) or None if buffer too small
        """
        if len(self._states) < self.batch_size and not self._dones[-1]:
            return None

        n         = len(self._states)
        states    = np.array(self._states,  dtype=np.float32)
        actions   = np.array(self._actions, dtype=np.int32)
        rewards   = np.array(self._rewards, dtype=np.float32)
        values    = np.array(self._values,  dtype=np.float32)
        dones     = np.array(self._dones,   dtype=np.float32)

        # ── Discounted returns (Monte Carlo) ──────────────────────────
        returns = np.zeros(n, dtype=np.float32)
        R = last_value
        for t in reversed(range(n)):
            R          = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R
        returns = np.clip(returns, -10.0, 10.0)  # prevent gradient explosion

        # ── Normalized advantages ─────────────────────────────────────
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

            # Policy gradient loss: -log_prob × advantage
            p_loss  = -log_prob * adv
            # Value MSE loss
            v_error = value_pred - ret
            v_loss  = v_error ** 2

            # Backprop — policy head
            one_hot      = np.zeros(self.action_dim, dtype=np.float32)
            one_hot[actions[i]] = 1.0
            grad_policy  = -adv * (one_hot - probs_clipped) / n
            grad_entropy = self.entropy_coef * (np.log(probs_clipped) + 1) / n
            self.network.policy.backward(grad_policy + grad_entropy)

            # Backprop — value head
            grad_value = np.array([2 * self.value_coef * v_error / n])
            self.network.value.backward(grad_value)

            total_policy_loss += float(p_loss)
            total_value_loss  += float(v_loss)
            total_entropy     += float(entropy)

        # Adam step on all parameters
        self.optimiser.step(self.network.params, self.network.grads)

        avg_loss = total_policy_loss / n + self.value_coef * total_value_loss / n
        self.losses.append(avg_loss)
        self.policy_losses.append(total_policy_loss / n)
        self.value_losses.append(total_value_loss / n)
        self.entropies.append(total_entropy / n)
        self.q_values.append(float(np.mean(values)))

        # Clear rollout buffer
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
            "avg_loss_1k":    float(np.mean(self.losses[-w:]))      if self.losses      else 0.0,
            "avg_value_1k":   float(np.mean(self.q_values[-w:]))    if self.q_values    else 0.0,
            "avg_entropy_1k": float(np.mean(self.entropies[-w:]))   if self.entropies   else 0.0,
        }
