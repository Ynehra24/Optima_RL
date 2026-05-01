"""Deep Deterministic Policy Gradient for Phase 3 DAG scheduling HNH.

Phase 1 includes DDPG as the continuous-action comparison model.  The Phase 3
simulator still exposes a discrete hold-action API, so this agent learns a
continuous hold duration in seconds and maps it to the nearest simulator action
index for interaction.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.networks import MLP, Adam, ReplayBuffer


STATE_DIM = 88
ACTION_DIM = 1
HOLD_DURATIONS_S = [0, 15, 30, 60, 120, 180, 300]


class OUNoise:
    """Ornstein-Uhlenbeck process for continuous hold exploration."""

    def __init__(
        self,
        action_dim: int = 1,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.08,
        seed: int = 42,
    ):
        self.mu = mu * np.ones(action_dim, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self.state = self.mu.copy()

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * self.rng.standard_normal(len(self.state))
        self.state += dx
        return self.state.copy()


class DDPGAgent:
    """Continuous-action DDPG adapted to Phase 3's discrete hold API."""

    HOLD_DURATIONS = HOLD_DURATIONS_S

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        lr_actor: float = 0.0001,
        lr_critic: float = 0.0001,
        gamma: float = 0.8,
        batch_size: int = 32,
        buffer_size: int = 50_000,
        tau: float = 0.005,
        action_low: float = 0.0,
        action_high: float = 300.0,
        seed: int = 42,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high

        self.actor = MLP(state_dim, [128, 128], action_dim, seed=seed)
        self.actor_target = MLP(state_dim, [128, 128], action_dim, seed=seed + 1)

        critic_in = state_dim + action_dim
        self.critic = MLP(critic_in, [128, 128], 1, seed=seed + 2)
        self.critic_target = MLP(critic_in, [128, 128], 1, seed=seed + 3)
        self._sync_targets(tau=1.0)

        self.actor_opt = Adam(lr=lr_actor)
        self.critic_opt = Adam(lr=lr_critic)
        self.buffer = ReplayBuffer(buffer_size, state_dim)
        self.noise = OUNoise(action_dim=action_dim, sigma=0.08, seed=seed)

        self.losses = []
        self.q_values = []
        self.rewards_log = []

    def _raw_to_hold(self, raw: float, add_noise: bool = False) -> float:
        action = (float(np.tanh(raw)) + 1.0) * 0.5 * self.action_high
        if add_noise:
            action += float(self.noise.sample()[0]) * 30.0
        return float(np.clip(action, self.action_low, self.action_high))

    def _hold_to_index(self, hold_s: float) -> int:
        return int(np.argmin([abs(hold_s - value) for value in self.HOLD_DURATIONS]))

    def select_action(self, state: np.ndarray) -> int:
        raw = self.actor.forward(state)
        hold_s = self._raw_to_hold(float(raw[0]), add_noise=True)
        return self._hold_to_index(hold_s)

    def greedy_action(self, state: np.ndarray) -> int:
        raw = self.actor.forward(state)
        hold_s = self._raw_to_hold(float(raw[0]), add_noise=False)
        return self._hold_to_index(hold_s)

    def discrete_action(self, state: np.ndarray) -> int:
        return self.greedy_action(state)

    def push(self, state, action, reward, next_state, done) -> None:
        self.buffer.push(state, int(action), reward, next_state, done)
        self.rewards_log.append(float(reward))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        holds = np.array(
            [self.HOLD_DURATIONS[int(action)] / self.action_high for action in actions],
            dtype=np.float32,
        ).reshape(-1, 1)

        target_holds = np.array(
            [
                self._raw_to_hold(float(self.actor_target.forward(next_states[i])[0]))
                / self.action_high
                for i in range(self.batch_size)
            ],
            dtype=np.float32,
        ).reshape(-1, 1)

        target_inputs = np.concatenate([next_states, target_holds], axis=1)
        q_next = np.array(
            [float(self.critic_target.forward(target_inputs[i])[0])
             for i in range(self.batch_size)],
            dtype=np.float32,
        )
        q_target = rewards + self.gamma * q_next * (1.0 - dones)
        q_target = np.clip(q_target, -10.0, 10.0)

        critic_inputs = np.concatenate([states, holds], axis=1)
        q_pred = np.array(
            [float(self.critic.forward(critic_inputs[i])[0])
             for i in range(self.batch_size)],
            dtype=np.float32,
        )
        critic_error = q_pred - q_target
        critic_loss = float(np.mean(critic_error ** 2))

        for i in range(self.batch_size):
            self.critic.forward(critic_inputs[i])
            self.critic.backward(np.array([2.0 * critic_error[i] / self.batch_size]))
        self.critic_opt.step(self.critic.params, self.critic.grads)

        actor_loss = 0.0
        for i in range(self.batch_size):
            raw = self.actor.forward(states[i])
            hold = self._raw_to_hold(float(raw[0])) / self.action_high
            critic_input = np.concatenate([states[i], [hold]]).astype(np.float32)
            q_val = float(self.critic.forward(critic_input)[0])
            actor_loss -= q_val / self.batch_size
            self.actor.backward(np.array([-1.0 / self.batch_size]))
        self.actor_opt.step(self.actor.params, self.actor.grads)

        self._sync_targets(tau=self.tau)
        self.losses.append(critic_loss)
        self.q_values.append(float(np.mean(q_pred)))
        return critic_loss

    def _sync_targets(self, tau: float | None = None) -> None:
        tau = self.tau if tau is None else tau
        for src, tgt in ((self.actor, self.actor_target),
                         (self.critic, self.critic_target)):
            for src_layer, tgt_layer in zip(src.layers, tgt.layers):
                tgt_layer.W = tau * src_layer.W + (1.0 - tau) * tgt_layer.W
                tgt_layer.b = tau * src_layer.b + (1.0 - tau) * tgt_layer.b

    def get_metrics(self) -> dict:
        w = 1000
        return {
            "avg_reward_1k": float(np.mean(self.rewards_log[-w:])) if self.rewards_log else 0.0,
            "avg_loss_1k": float(np.mean(self.losses[-w:])) if self.losses else 0.0,
            "avg_value_1k": float(np.mean(self.q_values[-w:])) if self.q_values else 0.0,
            "avg_entropy_1k": 0.0,
        }
