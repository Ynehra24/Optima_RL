"""
extract_cargo_profiles.py — CFS 2022 → cargo type proportions

Reads the BTS Commodity Flow Survey (2022) and extracts:
  1. perishable_frac  : fraction of cargo requiring temperature control
  2. value_percentiles: distribution of shipment values (for value_score)
  3. sla_distribution : SLA urgency distribution (0/1/2) by SCTG commodity code
  4. hazmat_frac      : fraction of hazardous material shipments

SCTG → SLA Urgency Mapping:
  SCTG  1-5  : Live animals / food / perishables  → X_k = 2 (express)
  SCTG  6-10 : Grains / agricultural products    → X_k = 1 (priority)
  SCTG 11-14 : Minerals / stone / ores           → X_k = 0 (standard)
  SCTG 15-19 : Coal / petroleum products         → X_k = 0 (standard)
  SCTG 20-30 : Manufactured goods / machinery    → X_k = 1 (priority)
  SCTG 31-43 : Electronics / instruments / misc  → X_k = 2 (express)

Output: phase-2/simulator/calibrated/cargo_profiles.json

Usage:
    python -m simulator.data_preprocessing.extract_cargo_profiles
"""

import json
import os
import numpy as np
import pandas as pd

# CFS 2022 truck mode codes
TRUCK_MODES = {112, 111, 113}  # parcel, TL, LTL

# SCTG → SLA urgency mapping
def sctg_to_sla(sctg: int) -> int:
    """Map SCTG commodity code to SLA urgency level."""
    if 1 <= sctg <= 5:
        return 2   # express: live animals, fresh food
    elif 6 <= sctg <= 10:
        return 1   # priority: grains, agricultural
    elif 11 <= sctg <= 19:
        return 0   # standard: minerals, coal, petroleum
    elif 20 <= sctg <= 30:
        return 1   # priority: manufactured goods, machinery
    elif 31 <= sctg <= 43:
        return 2   # express: electronics, instruments
    else:
        return 0   # standard (unknown)


def extract_cargo_profiles(data_dir: str, calibrated_dir: str) -> dict:
    """Read CFS 2022 and extract cargo type proportions for simulation.

    Uses chunked reading to handle the 2.8 GB file.

    Args:
        data_dir: Path to directory containing cfs_2022.csv
        calibrated_dir: Path to write cargo_profiles.json

    Returns:
        cargo_profiles dict
    """
    cfs_path = os.path.join(data_dir, "cfs_2022.csv")
    print(f"[extract_cargo_profiles] Reading {cfs_path} in chunks ...")

    usecols = ["MODE", "SCTG", "SHIPMT_VALUE", "SHIPMT_WGHT",
               "TEMP_CNTL_YN", "HAZMAT", "WGT_FACTOR"]

    chunks = []
    chunk_size = 500_000
    total_rows = 0

    for chunk in pd.read_csv(cfs_path, chunksize=chunk_size, usecols=usecols,
                              low_memory=False):
        # Filter to truck modes only
        chunk = chunk[chunk["MODE"].isin(TRUCK_MODES)]
        chunks.append(chunk)
        total_rows += len(chunk)
        print(f"[extract_cargo_profiles]   processed {total_rows:,} truck rows...",
              end="\r")

    print()
    df = pd.concat(chunks, ignore_index=True)
    print(f"[extract_cargo_profiles] Total truck rows: {len(df):,}")

    # ── 1. Perishable fraction ─────────────────────────────────────────
    perishable_frac = float((df["TEMP_CNTL_YN"] == "Y").mean())
    print(f"[extract_cargo_profiles] Perishable fraction: {perishable_frac:.3f}")

    # ── 2. Hazmat fraction ─────────────────────────────────────────────
    hazmat_frac = float((df["HAZMAT"] != "N").mean())
    print(f"[extract_cargo_profiles] Hazmat fraction:     {hazmat_frac:.3f}")

    # ── 3. Value score percentiles ─────────────────────────────────────
    vals = pd.to_numeric(df["SHIPMT_VALUE"], errors="coerce").dropna()
    vals = vals[vals > 0]
    value_percentiles = {
        "p10": float(np.percentile(vals, 10)),
        "p25": float(np.percentile(vals, 25)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "p99": float(np.percentile(vals, 99)),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
    }
    # value_score = 0-1 normalized using log-scale (heavy right skew)
    log_vals = np.log1p(vals)
    log_min, log_max = log_vals.min(), log_vals.max()
    value_percentiles["log_min"] = float(log_min)
    value_percentiles["log_max"] = float(log_max)
    print(f"[extract_cargo_profiles] Value p50=${value_percentiles['p50']:.0f}, "
          f"mean=${value_percentiles['mean']:.0f}")

    # ── 4. SLA urgency distribution by SCTG ───────────────────────────
    sctg = pd.to_numeric(df["SCTG"], errors="coerce").fillna(25).astype(int)
    sla_urgency = sctg.map(sctg_to_sla)

    sla_counts = sla_urgency.value_counts().sort_index()
    sla_total = sla_counts.sum()
    sla_distribution = {
        int(k): float(v / sla_total) for k, v in sla_counts.items()
    }
    print(f"[extract_cargo_profiles] SLA distribution: {sla_distribution}")

    # ── 5. Weight distribution (for truck utilization) ─────────────────
    weights = pd.to_numeric(df["SHIPMT_WGHT"], errors="coerce").dropna()
    weights = weights[weights > 0]
    weight_stats = {
        "mean_kg": float(weights.mean() * 0.453592),   # lbs → kg
        "std_kg":  float(weights.std() * 0.453592),
        "p50_kg":  float(np.percentile(weights, 50) * 0.453592),
        "p90_kg":  float(np.percentile(weights, 90) * 0.453592),
    }

    # ── Assemble output ────────────────────────────────────────────────
    cargo_profiles = {
        "perishable_frac": perishable_frac,
        "hazmat_frac": hazmat_frac,
        "value_percentiles": value_percentiles,
        "sla_distribution": sla_distribution,
        "weight_stats": weight_stats,
        "n_samples": len(df),
        "source": "cfs_2022_truck_modes",
    }

    # Save
    os.makedirs(calibrated_dir, exist_ok=True)
    out_path = os.path.join(calibrated_dir, "cargo_profiles.json")
    with open(out_path, "w") as f:
        json.dump(cargo_profiles, f, indent=2)
    print(f"[extract_cargo_profiles] Saved → {out_path}")

    return cargo_profiles


if __name__ == "__main__":
    data_dir = os.path.join("phase-2", "data")
    calibrated_dir = os.path.join("phase-2", "simulator", "calibrated")
    extract_cargo_profiles(data_dir, calibrated_dir)
