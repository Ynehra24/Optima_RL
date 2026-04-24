"""
extract_delay_params.py — Cargo 2000 → per-segment delay distributions

Reads the UCI Cargo 2000 airfreight tracking dataset, computes planned vs
actual delay distributions per hop count (1=short, 2=medium, 3=long), then
rescales from airfreight scale to truck scale using per-segment target means.

Rescaling approach:
  truck_delay = (airfreight_delay / airfreight_mean) * target_mean
  This preserves the distribution SHAPE while adjusting the scale.

Target means (from ATRI trucking industry benchmarks):
  short  (1-hop): mean=15 min, std=12 min
  medium (2-hop): mean=30 min, std=22 min
  long   (3-hop): mean=50 min, std=38 min

Output: phase-2/simulator/calibrated/delay_params.json

Usage:
    python -m simulator.data_preprocessing.extract_delay_params
"""

import json
import os
import numpy as np
import pandas as pd

# Cargo 2000 column patterns
# i1_dep_1_p = planned departure time for item 1, leg 1 (minutes)
# i1_dep_1_e = actual departure time for item 1, leg 1 (minutes)
# i1_hops    = number of hops for item 1

# Target truck delay parameters (ATRI-informed)
TARGET_PARAMS = {
    "short":  {"mean": 15.0, "std": 12.0},
    "medium": {"mean": 30.0, "std": 22.0},
    "long":   {"mean": 50.0, "std": 38.0},
}

HOP_TO_SEGMENT = {1: "short", 2: "medium", 3: "long"}

# Clip to 3x max hold time for realism
MAX_TRUCK_DELAY = 180.0  # minutes


def _compute_leg_delays(df: pd.DataFrame, item_prefix: str) -> pd.Series:
    """Extract delay = actual - planned for a given item prefix (i1, i2, o)."""
    dep_p_col = f"{item_prefix}_dep_1_p"
    dep_e_col = f"{item_prefix}_dep_1_e"
    hops_col = f"{item_prefix}_hops"

    if dep_p_col not in df.columns or dep_e_col not in df.columns:
        return pd.Series(dtype=float)

    p = pd.to_numeric(df[dep_p_col], errors="coerce")
    e = pd.to_numeric(df[dep_e_col], errors="coerce")
    delay = e - p
    return delay


def extract_delay_params(data_dir: str, calibrated_dir: str) -> dict:
    """Read Cargo 2000 and extract rescaled delay parameters per route segment.

    Args:
        data_dir: Path to directory containing cargo2000.csv
        calibrated_dir: Path to write delay_params.json

    Returns:
        delay_params dict with mean, std, shape params per segment
    """
    path = os.path.join(data_dir, "cargo2000.csv")
    print(f"[extract_delay_params] Reading {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"[extract_delay_params] Loaded {len(df):,} rows, {df.shape[1]} columns")

    # Collect delays per segment based on hop count
    segment_delays: dict = {"short": [], "medium": [], "long": []}

    for item_prefix in ["i1", "i2", "i3", "o"]:
        hops_col = f"{item_prefix}_hops"
        if hops_col not in df.columns:
            continue
        hops = pd.to_numeric(df[hops_col], errors="coerce")
        delays = _compute_leg_delays(df, item_prefix)

        for hop_count, segment in HOP_TO_SEGMENT.items():
            mask = (hops == hop_count) & delays.notna()
            seg_delays = delays[mask].values
            seg_delays = seg_delays[np.isfinite(seg_delays)]
            segment_delays[segment].extend(seg_delays.tolist())

    # Compute raw airfreight stats and rescale to truck domain
    delay_params = {}

    for segment, raw_delays in segment_delays.items():
        arr = np.array(raw_delays)
        if len(arr) == 0:
            # Fallback to target params directly
            delay_params[segment] = TARGET_PARAMS[segment].copy()
            delay_params[segment]["n_samples"] = 0
            delay_params[segment]["source"] = "default"
            continue

        raw_mean = float(np.mean(arr))
        raw_std = float(np.std(arr))

        target = TARGET_PARAMS[segment]

        # Shape-preserving rescale: z-score then rescale
        # truck_delay = (airfreight_delay - raw_mean) / raw_std * target_std + target_mean
        # But we store the raw distribution shape for runtime sampling
        delay_params[segment] = {
            "mean": target["mean"],
            "std": target["std"],
            "raw_airfreight_mean": raw_mean,
            "raw_airfreight_std": raw_std,
            "scale_factor": target["mean"] / raw_mean if raw_mean != 0 else 1.0,
            "n_samples": len(arr),
            "source": "cargo2000_rescaled",
            # Percentile shape for non-parametric sampling
            "p10": float(max(0, np.percentile(arr, 10) * target["mean"] / raw_mean if raw_mean != 0 else target["mean"] * 0.3)),
            "p25": float(max(0, np.percentile(arr, 25) * target["mean"] / raw_mean if raw_mean != 0 else target["mean"] * 0.5)),
            "p50": float(max(0, np.percentile(arr, 50) * target["mean"] / raw_mean if raw_mean != 0 else target["mean"])),
            "p75": float(max(0, np.percentile(arr, 75) * target["mean"] / raw_mean if raw_mean != 0 else target["mean"] * 1.5)),
            "p90": float(min(MAX_TRUCK_DELAY, np.percentile(arr, 90) * target["mean"] / raw_mean if raw_mean != 0 else target["mean"] * 2.0)),
            "on_time_frac": float(np.mean(arr <= 0)),  # fraction arriving on time
        }

        print(f"[extract_delay_params] {segment:6s}: "
              f"raw_mean={raw_mean:.1f}min → truck_mean={target['mean']}min "
              f"(n={len(arr):,}, on_time={delay_params[segment]['on_time_frac']:.1%})")

    # Save
    os.makedirs(calibrated_dir, exist_ok=True)
    out_path = os.path.join(calibrated_dir, "delay_params.json")
    with open(out_path, "w") as f:
        json.dump(delay_params, f, indent=2)
    print(f"[extract_delay_params] Saved → {out_path}")

    return delay_params


if __name__ == "__main__":
    data_dir = os.path.join("phase-2", "data")
    calibrated_dir = os.path.join("phase-2", "simulator", "calibrated")
    extract_delay_params(data_dir, calibrated_dir)
