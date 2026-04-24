"""
delay_sampler.py — Per-truck delay sampling calibrated from Cargo 2000.

Samples intrinsic delays for trucks and cargo dwell times using the
rescaled distribution parameters extracted from the Cargo 2000 dataset.

Each delay is drawn from a truncated normal distribution with:
  - mean and std calibrated by route type (short/medium/long)
  - shape informed by the percentile structure from Cargo 2000
  - hard clip at [0, MAX_DELAY] to prevent negative or extreme values
"""

from __future__ import annotations
import json
import math
import os
from typing import Dict, Optional
import numpy as np


# Hard upper bounds per delay type (minutes)
MAX_ROAD_DELAY = 180.0      # Road/transit overrun: max 3h
MAX_GROUND_DELAY = 60.0     # Intrinsic hub delay: max 1h
MAX_BAY_DWELL = 90.0        # Bay dwell time: max 1.5h

# Default params if calibrated data not found
_DEFAULT_DELAY_PARAMS = {
    "short":  {"mean": 15.0, "std": 12.0, "on_time_frac": 0.72},
    "medium": {"mean": 30.0, "std": 22.0, "on_time_frac": 0.63},
    "long":   {"mean": 50.0, "std": 38.0, "on_time_frac": 0.55},
}


class DelaySampler:
    """Samples intrinsic delays for trucks using calibrated distributions.

    All delays are in minutes. Uses numpy's truncated normal sampling
    to avoid negative delays and extreme outliers.
    """

    def __init__(self, calibrated_dir: str, rng: Optional[np.random.Generator] = None):
        """
        Args:
            calibrated_dir: Path containing delay_params.json
            rng: numpy random generator (created from SimConfig.seed if None)
        """
        self.rng = rng or np.random.default_rng(42)
        self._params = self._load_params(calibrated_dir)

    def _load_params(self, calibrated_dir: str) -> Dict:
        path = os.path.join(calibrated_dir, "delay_params.json")
        if os.path.exists(path):
            with open(path) as f:
                params = json.load(f)
            print(f"[DelaySampler] Loaded calibrated delay params from {path}")
            return params
        else:
            print("[DelaySampler] WARNING: delay_params.json not found. Using defaults.")
            return _DEFAULT_DELAY_PARAMS

    def _truncated_normal(self, mean: float, std: float,
                          low: float = 0.0, high: float = 180.0) -> float:
        """Sample from a truncated normal distribution.

        Uses rejection sampling — fast because typical rejection rate is low.
        """
        if std == 0:
            return float(np.clip(mean, low, high))

        for _ in range(100):  # max 100 rejections (extremely rare to hit this)
            sample = self.rng.normal(mean, std)
            if low <= sample <= high:
                return float(sample)
        return float(np.clip(mean, low, high))  # fallback

    # ── Public sampling methods ────────────────────────────────────────

    def sample_road_delay(self, route_type: str) -> float:
        """Sample intrinsic road/transit time overrun for a truck.

        This is the T_k node in the Delay Tree (road-time delay).
        Analogous to air-time delay in Phase 1.

        Args:
            route_type: 'short', 'medium', or 'long'

        Returns:
            Delay in minutes (>= 0, capped at MAX_ROAD_DELAY)
        """
        p = self._params.get(route_type, _DEFAULT_DELAY_PARAMS["medium"])
        on_time_frac = p.get("on_time_frac", 0.65)

        # With probability on_time_frac, truck arrives on time (small delay)
        if self.rng.random() < on_time_frac:
            return self._truncated_normal(mean=2.0, std=3.0, low=0.0, high=14.0)
        else:
            return self._truncated_normal(
                mean=p["mean"], std=p["std"],
                low=15.0, high=MAX_ROAD_DELAY
            )

    def sample_ground_departure_delay(self) -> float:
        """Sample intrinsic departure ground delay (GD_k node).

        Captures mechanical/admin delays at the origin depot.
        Typically small; heavy tail for breakdowns.

        Returns:
            Delay in minutes (>= 0, capped at MAX_GROUND_DELAY)
        """
        # ~75% of trucks depart on time, ~25% have some ground delay
        if self.rng.random() < 0.75:
            return self._truncated_normal(mean=1.0, std=2.0, low=0.0, high=5.0)
        else:
            return self._truncated_normal(mean=15.0, std=12.0,
                                          low=5.0, high=MAX_GROUND_DELAY)

    def sample_arrival_bay_delay(self) -> float:
        """Sample intrinsic arrival-side bay dwell delay (GA_k node).

        Time spent on the arrival side before cargo is counted as
        'received'. Distinct from bay-blockage (GB) which is queue-based.

        Returns:
            Delay in minutes (>= 0, capped at 30 min)
        """
        return self._truncated_normal(mean=5.0, std=4.0, low=0.0, high=30.0)

    def sample_cargo_count(self, mean: int = 30) -> int:
        """Sample number of cargo units on a truck (Poisson-distributed).

        Returns:
            Integer cargo count >= 1
        """
        count = int(self.rng.poisson(mean))
        return max(1, count)

    def sample_driver_hours_remaining(self) -> float:
        """Sample remaining legal driving hours for a truck's driver.

        US HOS (Hours of Service) allows 11 hours driving / 14 hours on duty.
        Returns remaining driving time in minutes.

        Returns:
            Minutes remaining in [120, 660] (2h to 11h)
        """
        return float(self.rng.uniform(120.0, 660.0))

    def get_on_time_frac(self, route_type: str) -> float:
        """Return the calibrated on-time fraction for a route type."""
        p = self._params.get(route_type, _DEFAULT_DELAY_PARAMS["medium"])
        return float(p.get("on_time_frac", 0.65))
