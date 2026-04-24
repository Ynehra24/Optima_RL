"""
logistics_env.py — OpenAI Gym-compatible logistics cross-docking environment.

Drop-in compatible with Phase 1's A2C agent. The agent interface is
identical — same observation_space, same action_space, same step() API.

The richness comes from:
  - 34-dimensional state vector (vs 17 in Phase 1)
  - Bay-congestion dynamics and GB delay attribution
  - Heterogeneous cargo (value, SLA, perishability)
  - Extended Delay Tree with GB node

Usage:
    from simulator.logistics_env import LogisticsEnv
    from simulator.config import SimConfig

    cfg = SimConfig()
    env = LogisticsEnv(cfg)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
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
from simulator.hub_config import load_hub_network
from simulator.delay_sampler import DelaySampler
from simulator.schedule_generator import ScheduleGenerator, load_cargo_profiles
from simulator.bay_manager import BayManager
from simulator.cargo_manager import CargoManager
from simulator.context_engine import ContextEngine, TruckContext
from simulator.event_queue import EventQueue, EpisodeStats
from rewardEngineering.reward_calculator import LogisticsRewardCalculator


class LogisticsEnv(gym.Env):
    """Cross-docking logistics environment for the HNH RL agent.

    Observation Space: Box(34,) — the 34-dimensional state vector
    Action Space:      Discrete(7) — hold_actions = [0,5,10,15,20,25,30] min

    Each step() corresponds to one HOLD_DECISION for one outbound truck.
    The environment internally processes all events (arrivals, departures)
    between decisions, advancing the simulation clock automatically.

    Episode structure:
      - One episode = cfg.episode_days days of truck operations
      - ~80 trucks/day × 1/3 outbound ratio ≈ ~185 decisions/episode
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, cfg: Optional[SimConfig] = None, render_mode: str = "ansi"):
        super().__init__()
        self.cfg = cfg or SimConfig()
        self.render_mode = render_mode

        # ── Spaces ───────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(34,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.cfg.n_actions)

        # ── Build all components ─────────────────────────────────────
        self._rng = np.random.default_rng(self.cfg.seed)
        self._build_components()

        # ── Episode state ────────────────────────────────────────────
        self._current_day: int = 0
        self._current_truck_id: Optional[str] = None
        self._current_context: Optional[TruckContext] = None
        self._episode_done: bool = False
        self._total_steps: int = 0
        self._episode_rewards: list = []

        # Track last action for info dict
        self._last_action: int = 0
        self._last_hold_minutes: float = 0.0

    # ── Gym API ────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment for a new episode.

        Returns:
            (observation, info) for gymnasium, or just observation for gym
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.cfg.seed = seed

        # Reset all components
        self._current_day = 0
        self._episode_done = False
        self._total_steps = 0
        self._episode_rewards = []
        self.bay_manager.reset()
        self.context_engine.reset()
        self.reward_calculator.reset()
        self.event_queue.reset()

        # Load first day
        obs = self._start_next_day()
        if obs is None:
            obs = np.zeros(34, dtype=np.float32)

        info = self._get_info()
        if GYM_VERSION == "gymnasium":
            return obs, info
        return obs

    def step(self, action: int) -> Tuple:
        """Apply hold action and advance to next decision point.

        Args:
            action: Integer in [0, 6] indexing into cfg.hold_actions

        Returns:
            (observation, reward, terminated, truncated, info) for gymnasium
            (observation, reward, done, info) for gym
        """
        if self._episode_done:
            obs = np.zeros(34, dtype=np.float32)
            if GYM_VERSION == "gymnasium":
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        # Convert action to hold minutes
        hold_minutes = float(self.cfg.hold_actions[action])
        self._last_action = action
        self._last_hold_minutes = hold_minutes

        truck_id = self._current_truck_id
        if truck_id is None:
            obs = np.zeros(34, dtype=np.float32)
            if GYM_VERSION == "gymnasium":
                return obs, 0.0, True, False, {}
            return obs, 0.0, True, {}

        # Apply hold — get immediate local reward
        local_reward = self.event_queue.apply_hold(truck_id, hold_minutes)

        # Advance simulation to next decision
        obs, next_truck_id = self._advance()

        # Compute total reward (local + global) for the action we JUST took
        global_reward = self.event_queue.get_global_reward(truck_id)
        total_reward = (
            self.cfg.beta * local_reward + (1 - self.cfg.beta) * global_reward
        )

        self._episode_rewards.append(total_reward)
        self._total_steps += 1

        terminated = self._episode_done
        info = self._get_info()

        if GYM_VERSION == "gymnasium":
            return obs, total_reward, terminated, False, info
        return obs, total_reward, terminated, info

    def render(self):
        """Render current environment state."""
        stats = self.event_queue.stats
        print(
            f"Day {self._current_day}/{self.cfg.episode_days} | "
            f"Step {self._total_steps} | "
            f"Hold={self._last_hold_minutes}min | "
            f"Missed={stats.missed_transfer_rate:.1%} | "
            f"BayUtil={stats.mean_bay_utilization:.1%} | "
            f"MeanRwd={stats.mean_reward:.3f}"
        )

    # ── Internal helpers ───────────────────────────────────────────────

    def _build_components(self):
        """Instantiate all simulator components."""
        calibrated_dir = self.cfg.calibrated_dir

        # Hub network
        self.hub_network = load_hub_network(calibrated_dir)

        # Delay sampler
        self.delay_sampler = DelaySampler(calibrated_dir, rng=self._rng)

        # Cargo profiles
        cargo_profiles = load_cargo_profiles(calibrated_dir)

        # Schedule generator
        self.schedule_generator = ScheduleGenerator(
            cfg=self.cfg,
            hub_network=self.hub_network,
            delay_sampler=self.delay_sampler,
            cargo_profiles=cargo_profiles,
            rng=self._rng,
        )

        # Bay manager
        self.bay_manager = BayManager(
            n_bays=self.cfg.n_bays,
            operating_start=float(self.cfg.operating_start),
        )

        # Cargo manager
        self.cargo_manager = CargoManager(self.cfg)

        # Context engine
        self.context_engine = ContextEngine(
            cfg=self.cfg,
            bay_manager=self.bay_manager,
            cargo_manager=self.cargo_manager,
        )

        # Reward calculator (Phase 2 version from rewardEngineering/)
        self.reward_calculator = LogisticsRewardCalculator(self.cfg)

        # Event queue (ties everything together)
        self.event_queue = EventQueue(
            cfg=self.cfg,
            delay_sampler=self.delay_sampler,
            bay_manager=self.bay_manager,
            cargo_manager=self.cargo_manager,
            context_engine=self.context_engine,
            reward_calculator=self.reward_calculator,
            rng=self._rng,
        )

    def _start_next_day(self) -> Optional[np.ndarray]:
        """Generate and load the next day's schedule, advance to first decision."""
        if self._current_day >= self.cfg.episode_days:
            self._episode_done = True
            return np.zeros(34, dtype=np.float32)

        # Generate synthetic daily schedule
        schedule = self.schedule_generator.generate_day(self._current_day)
        # Use reset_day() to clear per-day state but KEEP episode stats
        self.event_queue.reset_day()
        self.event_queue.load_day(schedule)
        self._current_day += 1

        # Advance to first decision
        obs, truck_id = self._advance()
        return obs

    def _advance(self) -> Tuple[np.ndarray, Optional[str]]:
        """Advance event queue to next decision. Return (obs, truck_id)."""
        result = self.event_queue.advance_to_next_decision()

        if result is None:
            # Day is done — start next day
            obs = self._start_next_day()
            return obs, self._current_truck_id

        ctx, truck_id = result
        self._current_truck_id = truck_id
        self._current_context = ctx
        return ctx.to_state_vector(), truck_id

    def _get_info(self) -> Dict:
        """Build info dict for the current step."""
        stats = self.event_queue.stats
        return {
            "day": self._current_day,
            "step": self._total_steps,
            "truck_id": self._current_truck_id,
            "hold_minutes": self._last_hold_minutes,
            "missed_transfer_rate": stats.missed_transfer_rate,
            "n_transfers_success": stats.n_transfers_success,
            "n_transfers_missed": stats.n_transfers_missed,
            "mean_bay_utilization": stats.mean_bay_utilization,
            "mean_reward": stats.mean_reward,
        }

    def get_episode_stats(self) -> EpisodeStats:
        """Return accumulated episode statistics."""
        return self.event_queue.stats


# ── Quick sanity test ──────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running LogisticsEnv quick sanity test...")
    cfg = SimConfig()
    env = LogisticsEnv(cfg)

    if GYM_VERSION == "gymnasium":
        obs, info = env.reset()
    else:
        obs = env.reset()

    assert obs.shape == (34,), f"Expected (34,), got {obs.shape}"
    assert not np.any(np.isnan(obs)), "NaN in initial observation!"

    print(f"Initial obs shape: {obs.shape}")
    print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")

    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = env.action_space.sample()
        result = env.step(action)
        if GYM_VERSION == "gymnasium":
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        total_reward += reward
        steps += 1
        if steps % 50 == 0:
            env.render()

    print(f"\nEpisode done. Steps={steps}, Total reward={total_reward:.3f}")
    print(f"Missed transfer rate: {info['missed_transfer_rate']:.1%}")
    print("PASSED: LogisticsEnv sanity test")
