"""
multi_hub_env.py — Multi-hub cascading logistics Gym environment.

This is the Phase 2 equivalent of Phase 1's full airline network simulator.
It wraps the HubChain to present a single Gym interface to the RL agent
while internally managing 10 FAF5 hubs with cascading truck delay propagation.

State vector: 42-dim = 34 (local hub context) + 8 (network context)
  - Dims  0-33: Same as single-hub env (see context_engine.py)
  - Dims 34-41: HubNetworkState.to_vector() (downstream hub status)

This makes the Phase 2 extension TRUE — holding a truck at Hub X now
propagates a late arrival at Hub Y, which the Hub Y agent observes and
must decide whether to cascade the hold further.
"""

from __future__ import annotations
import os
from typing import Dict, Optional, Tuple
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_VERSION = "gymnasium"
except ImportError:
    import gym
    from gym import spaces
    GYM_VERSION = "gym"

from simulator.config import SimConfig
from simulator.hub_chain import HubChain
from simulator.event_queue import EpisodeStats


class MultiHubLogisticsEnv(gym.Env):
    """Multi-hub cascading Hold-or-Not-Hold environment.

    Observation Space: Box(42,) — 34-dim local + 8-dim network context
    Action Space:      Discrete(7) — hold durations [0,5,10,15,20,25,30] min

    Episode structure:
      - One episode = cfg.episode_days days of operations
      - All 10 FAF5 hubs run in parallel (full mesh network)
      - At each step(), the agent handles the earliest pending HOLD_DECISION
        across all hubs (centralized agent, same as Phase 1)

    Cascade effect (the key feature vs single-hub):
      - Hub X holds truck 15 min → truck departs late
      - Hub Y receives it ~60-240 min later but still late
      - Hub Y's agent sees upstream_inter_hub_delay > 0 in its state
      - Hub Y decides whether to hold its outbound truck → cascade
      - Missed cargo is rebooked onto later trucks → more downstream delay
      - Strategic holding PREVENTS this cascade (exactly like Phase 1 airlines!)
    """

    metadata = {"render_modes": ["human", "ansi"]}
    STATE_DIM = 42   # 34 local + 8 network

    def __init__(self, cfg: Optional[SimConfig] = None, render_mode: str = "ansi"):
        super().__init__()
        self.cfg = cfg or SimConfig()
        self.render_mode = render_mode

        # ── Spaces ────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.cfg.n_actions)

        # ── Hub chain ─────────────────────────────────────────────────
        self.chain = HubChain(self.cfg)

        # ── Episode state ─────────────────────────────────────────────
        self._current_day = 0
        self._current_hub_id: Optional[str] = None
        self._current_truck_id: Optional[str] = None
        self._episode_done = False
        self._total_steps = 0
        self._episode_rewards = []
        self._last_action = 0
        self._last_hold_minutes = 0.0

    # ── Gym API ───────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset for a new episode."""
        if seed is not None:
            self.cfg.seed = seed

        self._current_day = 0
        self._episode_done = False
        self._total_steps = 0
        self._episode_rewards = []
        self._current_hub_id = None
        self._current_truck_id = None

        self.chain.reset()

        # Load first day across all hubs
        obs = self._start_next_day()
        if obs is None:
            obs = np.zeros(self.STATE_DIM, dtype=np.float32)

        info = self._get_info()
        if GYM_VERSION == "gymnasium":
            return obs, info
        return obs

    def step(self, action: int) -> Tuple:
        """Apply hold action to the current hub's truck and advance."""
        if self._episode_done:
            obs = np.zeros(self.STATE_DIM, dtype=np.float32)
            if GYM_VERSION == "gymnasium":
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        hold_minutes = float(self.cfg.hold_actions[action])
        self._last_action = action
        self._last_hold_minutes = hold_minutes

        hub_id = self._current_hub_id
        truck_id = self._current_truck_id

        if hub_id is None or truck_id is None:
            obs = np.zeros(self.STATE_DIM, dtype=np.float32)
            if GYM_VERSION == "gymnasium":
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        # Apply hold — cascade propagation happens inside HubChain
        local_reward = self.chain.apply_hold(hub_id, truck_id, hold_minutes)

        # Compute global reward from that hub's EventQueue
        eq = self.chain.event_queues[hub_id]
        global_reward = eq.get_global_reward(truck_id)
        total_reward = self.cfg.beta * local_reward + (1 - self.cfg.beta) * global_reward

        self._episode_rewards.append(total_reward)
        self._total_steps += 1

        # Advance to next decision across all hubs
        obs = self._advance()

        terminated = self._episode_done
        info = self._get_info()

        if GYM_VERSION == "gymnasium":
            return obs, total_reward, terminated, False, info
        return obs, total_reward, terminated, info

    def render(self):
        stats = self.chain.get_combined_stats()
        print(
            f"Day {self._current_day}/{self.cfg.episode_days} | "
            f"Hub: {self._current_hub_id} | "
            f"Step {self._total_steps} | "
            f"Hold={self._last_hold_minutes}min | "
            f"Network MissRate={stats.missed_transfer_rate:.1%} | "
            f"OTP={stats.OTP:.1f}%"
        )

    # ── Internal helpers ──────────────────────────────────────────────

    def _start_next_day(self) -> Optional[np.ndarray]:
        """Load next day across all hubs and advance to first decision."""
        if self._current_day >= self.cfg.episode_days:
            self._episode_done = True
            return np.zeros(self.STATE_DIM, dtype=np.float32)

        self.chain.reset_day()
        self.chain.load_day(self._current_day)
        self._current_day += 1
        return self._advance()

    def _advance(self) -> np.ndarray:
        """Advance to the next HOLD_DECISION across all hubs."""
        result = self.chain.get_next_decision()

        if result is None:
            # All hubs done for the day — start next day
            obs = self._start_next_day()
            return obs if obs is not None else np.zeros(self.STATE_DIM, dtype=np.float32)

        hub_id, ctx, truck_id = result
        self._current_hub_id = hub_id
        self._current_truck_id = truck_id

        # Build 42-dim state: 34 local + 8 network
        local_state = ctx.to_state_vector()    # 34-dim
        net_state = self.chain.get_network_state(hub_id).to_vector()  # 8-dim
        full_state = np.concatenate([local_state, net_state])  # 42-dim
        return full_state.astype(np.float32)

    def _get_info(self) -> Dict:
        stats = self.chain.get_combined_stats()
        per_hub = {
            hub_id: {
                "missed_rate": self.chain.event_queues[hub_id].stats.missed_transfer_rate,
                "OTP":         self.chain.event_queues[hub_id].stats.OTP,
            }
            for hub_id in self.chain.hub_ids
        }
        return {
            "day":                  self._current_day,
            "step":                 self._total_steps,
            "current_hub":          self._current_hub_id,
            "truck_id":             self._current_truck_id,
            "hold_minutes":         self._last_hold_minutes,
            "missed_transfer_rate": stats.missed_transfer_rate,
            "OTP":                  stats.OTP,
            "n_transfers_missed":   stats.n_transfers_missed,
            "n_transfers_success":  stats.n_transfers_success,
            "mean_bay_utilization": stats.mean_bay_utilization,
            "per_hub":              per_hub,
        }

    def get_episode_stats(self) -> EpisodeStats:
        """Return combined episode stats across all hubs."""
        return self.chain.get_combined_stats()


# ── Quick sanity test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running MultiHubLogisticsEnv sanity test...")
    cfg = SimConfig(n_hubs=3, episode_days=1)
    env = MultiHubLogisticsEnv(cfg)

    if GYM_VERSION == "gymnasium":
        obs, info = env.reset()
    else:
        obs = env.reset()

    assert obs.shape == (42,), f"Expected (42,), got {obs.shape}"
    assert not np.any(np.isnan(obs)), "NaN in initial observation!"
    print(f"Initial obs shape: {obs.shape}")
    print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Local state (0-33): [{obs[:34].min():.3f}, {obs[:34].max():.3f}]")
    print(f"Network state (34-41): {obs[34:].round(3)}")

    total_reward = 0.0; steps = 0; done = False
    while not done:
        result = env.step(env.action_space.sample())
        if GYM_VERSION == "gymnasium":
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        total_reward += reward; steps += 1
        if steps % 20 == 0:
            env.render()

    stats = env.get_episode_stats()
    print(f"\nEpisode done. Steps={steps}, Total reward={total_reward:.3f}")
    print(f"Network missed transfer rate: {stats.missed_transfer_rate:.1%}")
    print(f"Network OTP:                  {stats.OTP:.1f}%")
    print(f"Per-hub: {info['per_hub']}")
    print("PASSED: MultiHubLogisticsEnv sanity test")
