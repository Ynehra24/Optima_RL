"""
agents/a2c.py
-------------
Advantage Actor-Critic (A2C) agent — BEST PERFORMER in the paper.

Paper architecture (Table 3): (17, 17, (7, 1))
  Shared trunk : 17 → 17 (ReLU)
  Policy head  : 17 → 7  (softmax → action probabilities)
  Value head   : 17 → 1  (linear → state value V(s))

Paper hyperparameters:
  lr = 0.001, gamma = 0.8, batch = 32
  No replay buffer (on-policy algorithm)

A2C update:
  advantage A(s,a) = r + gamma * V(s') - V(s)
  policy loss      = -log π(a|s) * A(s,a)        (REINFORCE with baseline)
  value loss       = MSE(V(s), r + gamma * V(s'))
  total loss       = policy_loss + 0.5 * value_loss - entropy_coef * entropy
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.networks import MLP, Linear, Adam, relu, softmax


class A2CNetwork:
    """
    Shared trunk + separate policy and value heads.
    Paper arch: input(17) → hidden(17) → [policy(7), value(1)]
    """
    def __init__(self, state_dim: int = 17, action_dim: int = 7, seed: int = 42):
        # Shared trunk: 17 → 17
        self.trunk  = MLP(state_dim, [], 17, seed=seed)         # single hidden layer
        # Policy head: 17 → 7
        self.policy = MLP(17, [], action_dim, seed=seed + 10)
        # Value head:  17 → 1
        self.value  = MLP(17, [], 1, seed=seed + 20)

        self._trunk_out = None   # cache for backprop

    def forward(self, state: np.ndarray):
        """
        Returns
        -------
        probs : np.ndarray  shape [action_dim]   — action probabilities
        value : float                            — state value estimate
        """
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from utils.networks import relu

        trunk_raw       = self.trunk.forward(state)
        self._trunk_out = relu(trunk_raw)

        policy_logits = self.policy.forward(self._trunk_out)
        probs         = softmax(policy_logits)

        value_out = self.value.forward(self._trunk_out)
        value     = float(value_out) if value_out.ndim == 0 else float(value_out[0])

        return probs, value

    @property
    def params(self):
        return self.trunk.params + self.policy.params + self.value.params

    @property
    def grads(self):
        return self.trunk.grads + self.policy.grads + self.value.grads


class A2CAgent:
    """
    On-policy A2C. Collects a batch of transitions, then does one update.
    This is the paper's best-performing algorithm — implement this first.
    """

    def __init__(
        self,
        state_dim:    int   = 17,
        action_dim:   int   = 7,
        lr:           float = 0.001,
        gamma:        float = 0.8,
        batch_size:   int   = 32,
        entropy_coef: float = 0.01,    # encourages exploration
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

        # On-policy trajectory buffer (cleared after each update)
        self._states   = []
        self._actions  = []
        self._rewards  = []
        self._values   = []
        self._dones    = []

        # Logging
        self.losses      = []
        self.policy_losses = []
        self.value_losses  = []
        self.entropies     = []
        self.q_values      = []   # we log V(s) as proxy for "critic value"
        self.rewards_log   = []

    # ── Action selection ───────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray):
        """Sample action from policy distribution."""
        probs, value = self.network.forward(state)
        probs        = np.clip(probs, 1e-8, 1.0)
        probs       /= probs.sum()
        action       = int(np.random.choice(self.action_dim, p=probs))
        return action, float(value)

    def greedy_action(self, state: np.ndarray) -> int:
        """Deterministic greedy — used during testing."""
        probs, _ = self.network.forward(state)
        return int(np.argmax(probs))

    # ── Trajectory storage ────────────────────────────────────────────────────

    def store(self, state, action, reward, value, done):
        self._states.append(state.copy())
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._dones.append(done)
        self.rewards_log.append(reward)

    # ── Learning ───────────────────────────────────────────────────────────────

    def update(self, last_value: float = 0.0) -> float | None:
        """
        Compute advantages and update network.
        Call this every `batch_size` steps or at episode end.

        Parameters
        ----------
        last_value : float
            V(s_T) — bootstrap value for the last state (0 if terminal).
        """
        if len(self._states) < self.batch_size and not self._dones[-1]:
            return None

        n         = len(self._states)
        states    = np.array(self._states,  dtype=np.float32)   # (n, 17)
        actions   = np.array(self._actions, dtype=np.int32)      # (n,)
        rewards   = np.array(self._rewards, dtype=np.float32)    # (n,)
        values    = np.array(self._values,  dtype=np.float32)    # (n,)
        dones     = np.array(self._dones,   dtype=np.float32)    # (n,)

        # ── Compute discounted returns (bootstrapped) ──────────────────────────
        returns = np.zeros(n, dtype=np.float32)
        R       = last_value
        for t in reversed(range(n)):
            R          = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R

        # ── Advantages ────────────────────────────────────────────────────────
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Forward pass to get log-probs and entropy ──────────────────────────
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0

        # We do a manual per-sample update (simple but clear)
        for i in range(n):
            probs, value_pred = self.network.forward(states[i])
            probs_clipped     = np.clip(probs, 1e-8, 1.0)

            log_prob  = np.log(probs_clipped[actions[i]])
            entropy   = -np.sum(probs_clipped * np.log(probs_clipped))

            adv       = float(advantages[i])
            ret       = float(returns[i])

            # Policy gradient loss: -log π(a|s) * A
            p_loss = -log_prob * adv

            # Value loss: MSE(V(s), return)
            v_error = value_pred - ret
            v_loss  = v_error ** 2

            total_loss = p_loss + self.value_coef * v_loss - self.entropy_coef * entropy

            # ── Backward through policy head ───────────────────────────────────
            # Gradient of -log π(a) * A w.r.t. logits (before softmax):
            #   = A * (probs - one_hot(a))   [standard softmax cross-entropy grad]
            one_hot    = np.zeros(self.action_dim, dtype=np.float32)
            one_hot[actions[i]] = 1.0
            grad_policy_logits = -adv * (one_hot - probs_clipped) / n

            # Entropy gradient: -(log π + 1) / n
            grad_entropy_logits = self.entropy_coef * (np.log(probs_clipped) + 1) / n

            grad_policy_total = grad_policy_logits + grad_entropy_logits
            self.network.policy.backward(grad_policy_total)

            # ── Backward through value head ────────────────────────────────────
            grad_value = np.array([2 * self.value_coef * v_error / n])
            self.network.value.backward(grad_value)

            total_policy_loss += float(p_loss)
            total_value_loss  += float(v_loss)
            total_entropy     += float(entropy)

        # ── Optimiser step ─────────────────────────────────────────────────────
        self.optimiser.step(self.network.params, self.network.grads)

        # ── Logging ────────────────────────────────────────────────────────────
        avg_loss  = total_policy_loss / n + self.value_coef * total_value_loss / n
        self.losses.append(avg_loss)
        self.policy_losses.append(total_policy_loss / n)
        self.value_losses.append(total_value_loss / n)
        self.entropies.append(total_entropy / n)
        self.q_values.append(float(np.mean(values)))

        # ── Clear trajectory buffer ────────────────────────────────────────────
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._values.clear()
        self._dones.clear()

        return avg_loss

    def get_metrics(self) -> dict:
        window = 1000
        return {
            "avg_reward_1k":      float(np.mean(self.rewards_log[-window:]))
                                  if self.rewards_log else 0.0,
            "avg_loss_1k":        float(np.mean(self.losses[-window:]))
                                  if self.losses else 0.0,
            "avg_value_1k":       float(np.mean(self.q_values[-window:]))
                                  if self.q_values else 0.0,
            "avg_entropy_1k":     float(np.mean(self.entropies[-window:]))
                                  if self.entropies else 0.0,
        }
