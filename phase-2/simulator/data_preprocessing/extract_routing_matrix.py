"""
extract_routing_matrix.py — FAF5 → cargo routing matrix

Reads the FAF5 O-D flow table and produces a probability routing matrix:
  routing_matrix[origin_zone][dest_zone] = P(cargo goes to dest | at origin)

Also extracts distance bands (short/medium/long) per O-D pair.

Output: phase-2/simulator/calibrated/routing_matrix.json

Usage:
    python -m simulator.data_preprocessing.extract_routing_matrix
"""

import json
import os
import sys
import pandas as pd

# FAF5 dms_mode codes
TRUCK_MODE = 1

# Target: top N hub zones by outbound volume
TOP_N_HUBS = 10

# FAF5 dist_band codes → route type
DIST_BAND_MAP = {
    1: "short",   # < 100 miles
    2: "short",   # 100-249 miles
    3: "medium",  # 250-499 miles
    4: "medium",  # 500-749 miles
    5: "long",    # 750-999 miles
    6: "long",    # 1000-1499 miles
    7: "long",    # >= 1500 miles
}


def extract_routing_matrix(data_dir: str, calibrated_dir: str) -> dict:
    """Read FAF5 and extract truck O-D routing matrix.

    Args:
        data_dir: Path to directory containing faf5_flows.csv
        calibrated_dir: Path to write routing_matrix.json

    Returns:
        routing_matrix dict
    """
    faf5_path = os.path.join(data_dir, "faf5_flows.csv")
    print(f"[extract_routing_matrix] Reading {faf5_path} ...")

    # Read in chunks to handle 1.1 GB file
    chunks = []
    chunk_size = 200_000
    usecols = ["dms_orig", "dms_dest", "dms_mode", "tons_2022", "dist_band"]

    for chunk in pd.read_csv(faf5_path, chunksize=chunk_size, usecols=usecols,
                              low_memory=False):
        # Filter to truck mode only
        chunk = chunk[chunk["dms_mode"] == TRUCK_MODE]
        chunk = chunk[chunk["tons_2022"] > 0]
        chunk = chunk.dropna(subset=["dms_orig", "dms_dest", "tons_2022"])
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"[extract_routing_matrix] Loaded {len(df):,} truck O-D rows")

    # Cast zone IDs to int
    df["dms_orig"] = df["dms_orig"].astype(int)
    df["dms_dest"] = df["dms_dest"].astype(int)
    df["dist_band"] = df["dist_band"].fillna(3).astype(int)

    # Aggregate tons by O-D pair (take modal dist_band per pair)
    od_agg = (
        df.groupby(["dms_orig", "dms_dest"])
        .agg(tons=("tons_2022", "sum"), dist_band=("dist_band", "median"))
        .reset_index()
    )
    od_agg["dist_band"] = od_agg["dist_band"].round().astype(int)
    od_agg["route_type"] = od_agg["dist_band"].map(DIST_BAND_MAP).fillna("medium")

    # Select top N origins by total outbound tonnage
    origin_totals = od_agg.groupby("dms_orig")["tons"].sum().sort_values(ascending=False)
    top_origins = origin_totals.head(TOP_N_HUBS).index.tolist()
    print(f"[extract_routing_matrix] Top {TOP_N_HUBS} origins: {top_origins}")

    # Build routing matrix
    routing_matrix = {}
    od_filtered = od_agg[od_agg["dms_orig"].isin(top_origins)]

    for orig in top_origins:
        orig_rows = od_filtered[od_filtered["dms_orig"] == orig]
        # Only route to destinations that are also in top_origins
        orig_rows = orig_rows[orig_rows["dms_dest"].isin(top_origins)]
        if orig_rows.empty:
            # Fallback: allow routing to all top origins except self
            dests = [z for z in top_origins if z != orig]
            probs = {str(d): 1.0 / len(dests) for d in dests}
            route_types = {str(d): "medium" for d in dests}
        else:
            total = orig_rows["tons"].sum()
            probs = {}
            route_types = {}
            for _, row in orig_rows.iterrows():
                dest = str(int(row["dms_dest"]))
                if str(int(row["dms_orig"])) != dest:  # exclude self-loops
                    probs[dest] = float(row["tons"]) / total
                    route_types[dest] = row["route_type"]

            # Re-normalize after removing self-loops
            total_prob = sum(probs.values())
            if total_prob > 0:
                probs = {k: v / total_prob for k, v in probs.items()}

        routing_matrix[str(orig)] = {
            "probabilities": probs,
            "route_types": route_types,
            "total_tons_2022": float(origin_totals[orig]),
        }

    # Save
    os.makedirs(calibrated_dir, exist_ok=True)
    out_path = os.path.join(calibrated_dir, "routing_matrix.json")
    with open(out_path, "w") as f:
        json.dump(routing_matrix, f, indent=2)
    print(f"[extract_routing_matrix] Saved → {out_path}")

    return routing_matrix


if __name__ == "__main__":
    # Run from project root: python -m simulator.data_preprocessing.extract_routing_matrix
    data_dir = os.path.join("phase-2", "data")
    calibrated_dir = os.path.join("phase-2", "simulator", "calibrated")
    extract_routing_matrix(data_dir, calibrated_dir)
