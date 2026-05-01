"""
extract_alibaba_calibration.py — Alibaba PAI GPU v2020 → Phase 3 simulator calibration

Reads the Alibaba Cluster Trace GPU v2020 tables and extracts all numeric
parameters needed to ground the Phase 3 DAG HNH simulator in real data.

Tables consumed:
  pai_task_table.csv       — job/task metadata, resource plans, timing
  pai_instance_table.csv   — per-instance resource usage and placement
  pai_sensor_table.csv     — real-time CPU/GPU/mem sensor readings
  pai_machine_spec.csv     — per-machine hardware capacity
  pai_machine_metric.csv   — per-machine utilisation time series

Output: simulator/calibrated/alibaba_calibration.json

Kaggle source:
  https://www.kaggle.com/datasets/derrickmwiti/cluster-trace-gpu-v2020

Usage:
    python extract_alibaba_calibration.py --data_dir ./data/alibaba \
                                          --out_dir ./simulator/calibrated

FIX (v2): usecols are now validated against the file's actual header before
loading, so missing columns (workload, scheduling_class, priority, group, etc.)
no longer crash the extractor — they are silently dropped and each extractor
falls back gracefully.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Column name aliases — map common Alibaba variant names to our canonical names
# The actual Kaggle CSV sometimes uses slightly different names than the paper.
# ---------------------------------------------------------------------------

# Maps: canonical_name -> [list of possible column names in the CSV]
COLUMN_ALIASES: dict[str, list[str]] = {
    # pai_task_table
    "job_name":         ["job_name", "job_id", "job"],
    "task_name":        ["task_name", "task_id", "task"],
    "inst_num":         ["inst_num", "instance_num", "num_instances"],
    "status":           ["status", "task_status", "state"],
    "start_time":       ["start_time", "start", "start_ts"],
    "end_time":         ["end_time", "end", "end_ts"],
    "plan_cpu":         ["plan_cpu", "cpu_request", "cpu"],
    "plan_mem":         ["plan_mem", "mem_request", "memory", "memory_request"],
    "plan_gpu":         ["plan_gpu", "gpu_request", "gpu"],
    "gpu_type":         ["gpu_type", "gpu_name", "gpu_model", "gpu_type_spec"],
    "workload":         ["workload", "workload_type", "task_type", "job_type"],
    "group":            ["group", "job_group", "team", "user_group"],
    "scheduling_class": ["scheduling_class", "sched_class", "job_class", "priority_class"],
    "priority":         ["priority", "job_priority", "task_priority"],
    # pai_instance_table
    "inst_id":          ["inst_id", "instance_id", "inst_name"],
    "user":             ["user", "user_id", "username", "owner"],
    "machine":          ["machine", "machine_id", "host", "node"],
    "gpu_type_spec":    ["gpu_type_spec", "gpu_type", "gpu_name"],
    # pai_sensor_table
    "worker_name":      ["worker_name", "worker_id", "worker"],
    "cpu_usage":        ["cpu_usage", "cpu_util", "cpu_used"],
    "gpu_wrk_util":     ["gpu_wrk_util", "gpu_util", "gpu_usage", "gpu_worker_util"],
    "avg_mem":          ["avg_mem", "mem_usage", "avg_mem_usage", "memory_usage"],
    "max_mem":          ["max_mem", "peak_mem", "max_mem_usage"],
    # pai_machine_spec
    "cap_cpu":          ["cap_cpu", "cpu_capacity", "num_cpu", "cpu"],
    "cap_mem":          ["cap_mem", "mem_capacity", "memory_capacity", "memory"],
    "cap_gpu":          ["cap_gpu", "gpu_capacity", "num_gpu", "gpu"],
    # pai_machine_metric
    "machine_cpu_usr":  ["machine_cpu_usr", "cpu_usr", "cpu_user", "cpu_usage"],
    "machine_cpu_kernel": ["machine_cpu_kernel", "cpu_kernel", "cpu_sys"],
    "machine_cpu_iowait": ["machine_cpu_iowait", "cpu_iowait", "iowait"],
    "machine_gpu":      ["machine_gpu", "gpu_util", "gpu_usage"],
    "machine_load_1":   ["machine_load_1", "load_1", "load1", "load_avg_1"],
    "machine_net_receive": ["machine_net_receive", "net_receive", "net_rx",
                            "network_receive", "rx_bytes"],
}


def resolve_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Return the first alias for `canonical` that exists in df, or None."""
    for alias in COLUMN_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
    return None


def get_col(df: pd.DataFrame, canonical: str,
            dtype=None, default=None) -> pd.Series:
    """Fetch a column by canonical name (resolving aliases).

    Returns pd.Series of the resolved column (numeric-coerced if dtype given),
    or a Series of `default` values if the column is missing.
    """
    col = resolve_column(df, canonical)
    if col is None:
        if default is not None:
            return pd.Series([default] * len(df), dtype=float)
        return pd.Series(dtype=float)
    s = df[col]
    if dtype == "numeric":
        s = pd.to_numeric(s, errors="coerce")
    return s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_lognormal_fit(series: pd.Series) -> dict:
    """Fit log-normal mu/sigma to a positive series. Returns dict of params."""
    vals = series.dropna()
    vals = vals[vals > 0]
    if len(vals) < 10:
        return {"mu": 5.7, "sigma": 1.2, "n_samples": 0, "source": "default"}
    log_vals = np.log(vals)
    return {
        "mu": float(log_vals.mean()),
        "sigma": float(log_vals.std()),
        "n_samples": int(len(vals)),
        "source": "alibaba_pai_task_table",
    }


def percentile_dict(series: pd.Series, label: str) -> dict:
    vals = series.dropna()
    if len(vals) == 0:
        return {}
    return {
        f"{label}_mean": float(vals.mean()),
        f"{label}_std":  float(vals.std()),
        f"{label}_p10":  float(np.percentile(vals, 10)),
        f"{label}_p25":  float(np.percentile(vals, 25)),
        f"{label}_p50":  float(np.percentile(vals, 50)),
        f"{label}_p75":  float(np.percentile(vals, 75)),
        f"{label}_p90":  float(np.percentile(vals, 90)),
        f"{label}_p99":  float(np.percentile(vals, 99)),
        f"{label}_n":    int(len(vals)),
    }


def _get_actual_columns(path: str) -> list[str]:
    """Read only the header row of a CSV to get the actual column names."""
    try:
        header = pd.read_csv(path, nrows=0)
        return list(header.columns)
    except Exception:
        return []


def read_csv_safe(path: str, wanted_cols: list[str] | None = None,
                  chunksize: int = 500_000) -> pd.DataFrame:
    """Read a CSV, loading only `wanted_cols` that actually exist.

    FIX: validates wanted_cols against the file's real header before loading,
    preventing ValueError when requested columns are absent.
    """
    if not os.path.exists(path):
        print(f"  [WARNING] File not found: {path} — skipping.")
        return pd.DataFrame()

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Reading {os.path.basename(path)} ({size_mb:.0f} MB) ...")

    # Detect which of the wanted columns actually exist in this file
    actual_cols = _get_actual_columns(path)
    if actual_cols:
        print(f"    Columns found ({len(actual_cols)}): {actual_cols[:12]}"
              f"{'...' if len(actual_cols) > 12 else ''}")

    if wanted_cols:
        usecols = [c for c in wanted_cols if c in actual_cols]
        missing = [c for c in wanted_cols if c not in actual_cols]
        if missing:
            print(f"    [INFO] Columns not in file (will alias/default): {missing}")
        # If nothing matched, load all columns (extractor will use alias resolution)
        usecols = usecols if usecols else None
    else:
        usecols = None

    try:
        if size_mb > 500:
            chunks = []
            for chunk in pd.read_csv(path, chunksize=chunksize,
                                     usecols=usecols, low_memory=False):
                chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True)
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception as e:
        print(f"  [ERROR] Failed to read {path}: {e}")
        print(f"  [FALLBACK] Loading all columns without usecols filter ...")
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception as e2:
            print(f"  [ERROR] Fallback also failed: {e2}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Section extractors
# ---------------------------------------------------------------------------

def extract_task_duration(task_df: pd.DataFrame) -> dict:
    """Task duration = end_time - start_time (seconds). Fit log-normal."""
    print("  [task_duration] computing durations ...")

    start = get_col(task_df, "start_time", dtype="numeric")
    end   = get_col(task_df, "end_time",   dtype="numeric")

    if start.empty or end.empty:
        print("    [WARN] start_time/end_time missing — using defaults")
        return {"lognormal_fit": {"mu": 5.7, "sigma": 1.2,
                                  "n_samples": 0, "source": "default"},
                "stats": {}}

    duration_s = end - start
    duration_s = duration_s[(duration_s > 0) & (duration_s < 7 * 86400)]

    lognorm = safe_lognormal_fit(duration_s)
    stats   = percentile_dict(duration_s, "duration_s")

    print(f"    mu={lognorm['mu']:.3f}  sigma={lognorm['sigma']:.3f}"
          f"  median={np.exp(lognorm['mu']):.0f}s  n={lognorm['n_samples']:,}")
    return {"lognormal_fit": lognorm, "stats": stats}


def extract_delay_probability(task_df: pd.DataFrame) -> dict:
    """Intrinsic delay probability and magnitude distribution."""
    print("  [delay_probability] computing ...")

    status_col = resolve_column(task_df, "status")
    if status_col:
        failed_frac = float(task_df[status_col].isin(
            ["Failed", "FAILED", "Killed", "KILLED",
             "Evicted", "EVICTED", "killed", "failed"]
        ).mean())
    else:
        failed_frac = 0.12

    start = get_col(task_df, "start_time", dtype="numeric")
    end   = get_col(task_df, "end_time",   dtype="numeric")
    duration_s = (end - start).dropna()
    duration_s = duration_s[duration_s > 0]

    if len(duration_s) < 10:
        return {
            "task_delay_probability": 0.12,
            "task_delay_mean_s":      120.0,
            "task_delay_stddev_s":    90.0,
            "failed_task_frac":       failed_frac,
            "overrun_proxy_frac":     0.12,
            "delay_magnitude_stats":  {},
        }

    median_dur = duration_s.median()
    delayed_mask = duration_s > median_dur * 2
    overrun_frac = float(delayed_mask.mean())

    delay_magnitudes = (duration_s[delayed_mask] - median_dur).dropna()
    delay_stats = percentile_dict(delay_magnitudes, "delay_magnitude_s")

    combined_prob = float(np.clip((failed_frac + overrun_frac) / 2, 0.05, 0.35))
    delay_mean = float(delay_magnitudes.mean()) if len(delay_magnitudes) > 0 else 120.0
    delay_std  = float(delay_magnitudes.std())  if len(delay_magnitudes) > 0 else 90.0

    print(f"    delay_probability={combined_prob:.3f}  "
          f"delay_mean={delay_mean:.1f}s  delay_std={delay_std:.1f}s")
    return {
        "task_delay_probability": combined_prob,
        "task_delay_mean_s":      float(np.clip(delay_mean,  10.0, 600.0)),
        "task_delay_stddev_s":    float(np.clip(delay_std,   5.0,  400.0)),
        "failed_task_frac":       failed_frac,
        "overrun_proxy_frac":     overrun_frac,
        "delay_magnitude_stats":  delay_stats,
    }


def extract_resource_demands(task_df: pd.DataFrame) -> dict:
    """plan_cpu, plan_mem, plan_gpu demand distributions."""
    print("  [resource_demands] computing ...")

    cpu = get_col(task_df, "plan_cpu", dtype="numeric").dropna()
    cpu = cpu[(cpu > 0) & (cpu <= 1.0)]

    mem = get_col(task_df, "plan_mem", dtype="numeric").dropna()
    mem = mem[(mem > 0) & (mem <= 1.0)]

    gpu_raw = get_col(task_df, "plan_gpu", dtype="numeric").dropna()
    # If max GPU value is > 1, it's in absolute GPU count — normalise by 8
    if len(gpu_raw) > 0 and gpu_raw.max() > 1.0:
        gpu_raw = gpu_raw / gpu_raw.quantile(0.99).clip(1)
    gpu_request_prob = float((gpu_raw > 0).mean()) if len(gpu_raw) > 0 else 0.65
    gpu = gpu_raw[gpu_raw > 0]

    # If plan_cpu looks like absolute cores (values >> 1), normalise by cap
    # Heuristic: if median > 2, treat as absolute core count / 96 (Alibaba typical)
    if len(cpu) > 0 and cpu.median() > 2.0:
        cpu = cpu / 96.0
        cpu = cpu[(cpu > 0) & (cpu <= 1.0)]
    if len(mem) > 0 and mem.median() > 2.0:
        mem = mem / 512.0
        mem = mem[(mem > 0) & (mem <= 1.0)]

    result = {
        "task_cpu_demand_mean":   float(cpu.mean())   if len(cpu) > 0 else 0.12,
        "task_cpu_demand_stddev": float(cpu.std())    if len(cpu) > 0 else 0.15,
        "task_mem_demand_mean":   float(mem.mean())   if len(mem) > 0 else 0.10,
        "task_mem_demand_stddev": float(mem.std())    if len(mem) > 0 else 0.12,
        "task_gpu_demand_mean":   float(np.clip(gpu.mean(), 0.01, 1.0)) if len(gpu) > 0 else 0.50,
        "task_gpu_demand_stddev": float(np.clip(gpu.std(),  0.01, 1.0)) if len(gpu) > 0 else 0.30,
        "task_gpu_request_probability": float(np.clip(gpu_request_prob, 0.1, 0.95)),
        "cpu_stats": percentile_dict(cpu, "plan_cpu"),
        "mem_stats": percentile_dict(mem, "plan_mem"),
        "gpu_stats": percentile_dict(gpu, "plan_gpu"),
    }
    print(f"    cpu_mean={result['task_cpu_demand_mean']:.3f}  "
          f"mem_mean={result['task_mem_demand_mean']:.3f}  "
          f"gpu_mean={result['task_gpu_demand_mean']:.3f}  "
          f"gpu_request_prob={result['task_gpu_request_probability']:.3f}")
    return result


def extract_job_size_distribution(task_df: pd.DataFrame) -> dict:
    """Count tasks per job, bucket into canonical size classes."""
    print("  [job_size] computing ...")

    job_col = resolve_column(task_df, "job_name")
    if job_col is None:
        print("    [WARN] job_name column missing — using defaults")
        return {"job_size_distribution": [
            [1, 0.25], [2, 0.20], [3, 0.15], [5, 0.15],
            [8, 0.10], [12, 0.08], [20, 0.05], [50, 0.02]
        ], "job_size_stats": {}}

    job_sizes = task_df[job_col].value_counts()
    total = len(job_sizes)

    buckets = [1, 2, 3, 5, 8, 12, 20, 50]
    counts  = []
    for i, b in enumerate(buckets):
        hi = buckets[i + 1] - 1 if i < len(buckets) - 1 else 10_000
        lo = b
        cnt = int(((job_sizes >= lo) & (job_sizes <= hi)).sum())
        counts.append(cnt)

    total_bucketed = max(sum(counts), 1)
    probs = [c / total_bucketed for c in counts]
    dist = [[b, round(p, 4)] for b, p in zip(buckets, probs)]
    size_stats = percentile_dict(job_sizes, "job_size")

    print(f"    total_jobs={total:,}  "
          f"median_size={job_sizes.median():.0f}  max={job_sizes.max()}")
    return {"job_size_distribution": dist, "job_size_stats": size_stats}


def extract_workload_distribution(task_df: pd.DataFrame) -> dict:
    """Workload type distribution — tries multiple column aliases."""
    print("  [workload_type] computing ...")

    # Try canonical workload column first, then group/task_type
    col = resolve_column(task_df, "workload")
    if col is None:
        col = resolve_column(task_df, "group")

    if col is None:
        print("    [WARN] no workload/group column found — using defaults")
        return {"workload_type_distribution": [
            ["training", 0.35], ["inference", 0.20], ["etl", 0.15],
            ["pipeline", 0.12], ["serving", 0.10], ["other", 0.08]
        ]}

    raw_counts = task_df[col].fillna("other").astype(str).str.lower().value_counts()
    total = max(raw_counts.sum(), 1)

    canonical_map = {
        "training":  ["train", "training", "finetune", "pretrain", "fine_tune"],
        "inference": ["infer", "inference", "predict", "serving_infer", "online_infer"],
        "etl":       ["etl", "data", "preprocess", "pipeline_data", "extract"],
        "pipeline":  ["pipeline", "workflow", "dag", "flow"],
        "serving":   ["serv", "serving", "online", "realtime"],
        "other":     [],
    }
    canonical_counts = {k: 0 for k in canonical_map}

    for raw_label, cnt in raw_counts.items():
        matched = False
        for canonical, keywords in canonical_map.items():
            if canonical == "other":
                continue
            if any(kw in str(raw_label) for kw in keywords):
                canonical_counts[canonical] += cnt
                matched = True
                break
        if not matched:
            canonical_counts["other"] += cnt

    total_mapped = max(sum(canonical_counts.values()), 1)
    dist = [[k, round(v / total_mapped, 4)] for k, v in canonical_counts.items()]

    # If training dominates suspiciously (>80%), spread to defaults proportionally
    if canonical_counts["training"] / total_mapped > 0.80:
        print("    [WARN] >80% training — keyword matching sparse; blending with defaults")
        default_probs = [0.35, 0.20, 0.15, 0.12, 0.10, 0.08]
        names = list(canonical_counts.keys())
        dist = [[n, round(0.5 * (p + d), 4)]
                for n, p, d in zip(names, [v / total_mapped for v in canonical_counts.values()],
                                   default_probs)]

    print(f"    " + "  ".join(f"{d[0]}={d[1]:.3f}" for d in dist))
    return {"workload_type_distribution": dist}


def extract_inst_num_distribution(task_df: pd.DataFrame) -> dict:
    """inst_num: number of worker instances per task."""
    print("  [inst_num] computing ...")

    inst_col = resolve_column(task_df, "inst_num")
    if inst_col is None:
        print("    [WARN] inst_num column missing — using defaults")
        return {"inst_num_distribution": [
            [1, 0.50], [2, 0.20], [4, 0.15], [8, 0.10], [16, 0.05]
        ]}

    inst = pd.to_numeric(task_df[inst_col], errors="coerce").dropna()
    inst = inst[inst >= 1].astype(int)
    total = max(len(inst), 1)

    buckets = [1, 2, 4, 8, 16]
    probs   = []
    for i, b in enumerate(buckets):
        hi = buckets[i + 1] if i < len(buckets) - 1 else 9999
        cnt = int(((inst >= b) & (inst < hi)).sum())
        probs.append(cnt / total)

    dist = [[b, round(p, 4)] for b, p in zip(buckets, probs)]
    print(f"    " + "  ".join(f"{b}:{p:.3f}" for b, p in zip(buckets, probs)))
    return {"inst_num_distribution": dist}


def extract_scheduling_class(task_df: pd.DataFrame) -> dict:
    """scheduling_class distribution — tries scheduling_class or priority column."""
    print("  [scheduling_class] computing ...")

    col = resolve_column(task_df, "scheduling_class")
    if col is None:
        col = resolve_column(task_df, "priority")

    if col is None:
        print("    [WARN] no scheduling_class or priority column — using defaults")
        return {"scheduling_class_distribution": [
            [3, 0.15, 0.0,   "production"],
            [2, 0.25, 30.0,  "mid_tier"],
            [1, 0.40, 120.0, "batch"],
            [0, 0.20, 300.0, "best_effort"],
        ]}

    vals = pd.to_numeric(task_df[col], errors="coerce").dropna()
    if len(vals) == 0:
        return {"scheduling_class_distribution": [
            [3, 0.15, 0.0, "production"], [2, 0.25, 30.0, "mid_tier"],
            [1, 0.40, 120.0, "batch"], [0, 0.20, 300.0, "best_effort"],
        ]}

    # Normalise to 0-3 range regardless of source scale
    v_min, v_max = vals.min(), vals.max()
    if v_max > v_min:
        vals = ((vals - v_min) / (v_max - v_min) * 3).round().astype(int)
    else:
        vals = pd.Series([1] * len(vals))  # all same class — treat as batch

    counts = vals.value_counts().sort_index()
    total = max(counts.sum(), 1)

    sla_map  = {3: 0.0, 2: 30.0, 1: 120.0, 0: 300.0}
    name_map = {3: "production", 2: "mid_tier", 1: "batch", 0: "best_effort"}

    dist = []
    for cls in [3, 2, 1, 0]:
        frac = float(counts.get(cls, 0)) / total
        dist.append([cls, round(frac, 4), sla_map[cls], name_map[cls]])

    print(f"    " + "  ".join(f"cls{d[0]}={d[1]:.3f}" for d in dist))
    return {"scheduling_class_distribution": dist}


def extract_gpu_type_distribution(task_df: pd.DataFrame,
                                  instance_df: pd.DataFrame) -> dict:
    """GPU type distribution — searches both task and instance tables."""
    print("  [gpu_type] computing ...")

    col = None
    source_df = None
    for df, name in [(task_df, "task"), (instance_df, "instance")]:
        c = resolve_column(df, "gpu_type")
        if c is None:
            c = resolve_column(df, "gpu_type_spec")
        if c is not None and len(df) > 0:
            col = c
            source_df = df
            print(f"    Using '{col}' from {name} table")
            break

    canonical = ["V100", "A100", "T4", "P100", "A10"]
    if col is None or source_df is None or source_df.empty:
        print("    [WARN] no gpu_type column found — using defaults")
        probs = [0.30, 0.25, 0.20, 0.15, 0.07, 0.03]
        return {"gpu_type_distribution": [[g, p] for g, p in
                                           zip(canonical + ["other"], probs)]}

    raw = source_df[col].fillna("other").astype(str).str.upper()
    total = max(len(raw), 1)
    counts = {g: int(raw.str.contains(g, na=False).sum()) for g in canonical}
    counts["other"] = total - sum(counts.values())

    dist = [[k, round(v / total, 4)] for k, v in counts.items()]
    print(f"    " + "  ".join(f"{k}={v:.3f}" for k, v in counts.items()))
    return {"gpu_type_distribution": dist}


def extract_utilisation_factors(sensor_df: pd.DataFrame) -> dict:
    """Actual vs planned utilisation ratios from pai_sensor_table."""
    print("  [utilisation_factors] computing ...")

    result = {
        "cpu_utilisation_factor_mean":   0.50,
        "cpu_utilisation_factor_stddev": 0.20,
        "gpu_utilisation_factor_mean":   0.55,
        "gpu_utilisation_factor_stddev": 0.25,
        "mem_utilisation_factor_mean":   0.65,
        "mem_utilisation_factor_stddev": 0.15,
    }
    if sensor_df.empty:
        print("    [WARN] sensor table empty — using defaults")
        return result

    # CPU
    cpu_col = resolve_column(sensor_df, "cpu_usage")
    if cpu_col:
        cpu_usage = pd.to_numeric(sensor_df[cpu_col], errors="coerce").dropna()
        # Normalise to [0,1] if in percent form
        if cpu_usage.max() > 1.5:
            cpu_usage = cpu_usage / 100.0
        cpu_usage = cpu_usage[(cpu_usage > 0) & (cpu_usage <= 1.0)]
        if len(cpu_usage) > 100:
            result["cpu_utilisation_factor_mean"]   = float(np.clip(cpu_usage.mean(), 0.1, 1.0))
            result["cpu_utilisation_factor_stddev"] = float(np.clip(cpu_usage.std(),  0.01, 0.5))
            print(f"    cpu_util: mean={result['cpu_utilisation_factor_mean']:.3f}"
                  f"  std={result['cpu_utilisation_factor_stddev']:.3f}  n={len(cpu_usage):,}")

    # GPU
    gpu_col = resolve_column(sensor_df, "gpu_wrk_util")
    if gpu_col:
        gpu_util = pd.to_numeric(sensor_df[gpu_col], errors="coerce").dropna()
        if gpu_util.max() > 1.5:
            gpu_util = gpu_util / 100.0
        gpu_util = gpu_util[(gpu_util > 0) & (gpu_util <= 1.0)]
        if len(gpu_util) > 100:
            result["gpu_utilisation_factor_mean"]   = float(np.clip(gpu_util.mean(), 0.1, 1.0))
            result["gpu_utilisation_factor_stddev"] = float(np.clip(gpu_util.std(),  0.01, 0.5))
            print(f"    gpu_util: mean={result['gpu_utilisation_factor_mean']:.3f}"
                  f"  std={result['gpu_utilisation_factor_stddev']:.3f}  n={len(gpu_util):,}")

    # Memory
    mem_col = resolve_column(sensor_df, "avg_mem")
    if mem_col:
        mem_usage = pd.to_numeric(sensor_df[mem_col], errors="coerce").dropna()
        if mem_usage.max() > 1.5:
            mem_usage = mem_usage / 100.0
        mem_usage = mem_usage[(mem_usage > 0) & (mem_usage <= 1.0)]
        if len(mem_usage) > 100:
            result["mem_utilisation_factor_mean"]   = float(np.clip(mem_usage.mean(), 0.1, 1.0))
            result["mem_utilisation_factor_stddev"] = float(np.clip(mem_usage.std(),  0.01, 0.5))
            print(f"    mem_util: mean={result['mem_utilisation_factor_mean']:.3f}"
                  f"  std={result['mem_utilisation_factor_stddev']:.3f}  n={len(mem_usage):,}")

    return result


def extract_machine_spec(machine_spec_df: pd.DataFrame) -> dict:
    """Per-machine capacity: cap_cpu, cap_gpu, cap_mem."""
    print("  [machine_spec] computing ...")

    if machine_spec_df.empty:
        print("    [WARN] machine spec table empty — using defaults")
        return {
            "machine_cpu_cores":    32.0,
            "machine_mem_gb":       128.0,
            "machine_gpu_count":    4.0,
            "gpu_machine_fraction": 0.30,
        }

    cpu_col = resolve_column(machine_spec_df, "cap_cpu")
    mem_col = resolve_column(machine_spec_df, "cap_mem")
    gpu_col = resolve_column(machine_spec_df, "cap_gpu")

    def safe_median(col):
        if col is None:
            return None
        vals = pd.to_numeric(machine_spec_df[col], errors="coerce").dropna()
        return float(vals.median()) if len(vals) > 0 else None

    cpu_median = safe_median(cpu_col) or 32.0
    mem_median = safe_median(mem_col) or 128.0

    gpu_vals = (pd.to_numeric(machine_spec_df[gpu_col], errors="coerce").dropna()
                if gpu_col else pd.Series(dtype=float))
    gpu_median = float(gpu_vals[gpu_vals > 0].median()) if (gpu_col and len(gpu_vals[gpu_vals > 0]) > 0) else 4.0
    gpu_frac   = float((gpu_vals > 0).mean()) if (gpu_col and len(gpu_vals) > 0) else 0.30

    # Scale check: Alibaba cap_cpu might be in fractional or absolute cores
    # If median cap_cpu < 2, it's already fractional — scale up
    if cpu_median < 2.0:
        cpu_median = cpu_median * 96.0   # Alibaba machines have ~96 cores
    if mem_median < 2.0:
        mem_median = mem_median * 512.0  # and ~512 GB RAM

    print(f"    cpu_cores={cpu_median:.0f}  mem_gb={mem_median:.0f}  "
          f"gpu_count={gpu_median:.0f}  gpu_frac={gpu_frac:.3f}")
    return {
        "machine_cpu_cores":    float(np.clip(cpu_median, 4.0,  512.0)),
        "machine_mem_gb":       float(np.clip(mem_median, 8.0, 4096.0)),
        "machine_gpu_count":    float(np.clip(gpu_median, 0.0,   16.0)),
        "gpu_machine_fraction": float(np.clip(gpu_frac,   0.01,  1.0)),
        "cpu_stats": percentile_dict(
            pd.to_numeric(machine_spec_df[cpu_col], errors="coerce").dropna(),
            "cap_cpu") if cpu_col else {},
    }


def extract_cluster_utilisation(metric_df: pd.DataFrame) -> dict:
    """Rolling cluster CPU/GPU utilisation and network stats."""
    print("  [cluster_utilisation] computing ...")

    if metric_df.empty:
        print("    [WARN] metric table empty — using defaults")
        return {
            "target_cpu_util":     0.60,
            "target_gpu_util":     0.40,
            "network_util_mean":   0.45,
            "network_util_stddev": 0.15,
        }

    cpu_col = resolve_column(metric_df, "machine_cpu_usr")
    gpu_col = resolve_column(metric_df, "machine_gpu")
    net_col = resolve_column(metric_df, "machine_net_receive")

    def safe_util(col, lo=0.0, hi=1.0, default=0.5):
        if col is None:
            return default, 0.1
        vals = pd.to_numeric(metric_df[col], errors="coerce").dropna()
        if vals.max() > 1.5:
            vals = vals / 100.0
        vals = vals[(vals >= lo) & (vals <= hi)]
        if len(vals) == 0:
            return default, 0.1
        return float(vals.mean()), float(vals.std())

    target_cpu, _ = safe_util(cpu_col, default=0.60)
    target_gpu, _ = safe_util(gpu_col, default=0.40)

    # Network: raw bytes/s — normalise by 99th percentile
    net_mean, net_std = 0.45, 0.15
    if net_col:
        net = pd.to_numeric(metric_df[net_col], errors="coerce").dropna()
        net = net[net >= 0]
        if len(net) > 10:
            cap = net.quantile(0.99)
            if cap > 0:
                net = net / cap
            net = net[net <= 1.0]
            net_mean = float(net.mean()) if len(net) > 0 else 0.45
            net_std  = float(net.std())  if len(net) > 0 else 0.15

    print(f"    target_cpu={target_cpu:.3f}  target_gpu={target_gpu:.3f}  "
          f"net_mean={net_mean:.3f}")
    return {
        "target_cpu_util":     float(np.clip(target_cpu, 0.1, 0.95)),
        "target_gpu_util":     float(np.clip(target_gpu, 0.05, 0.95)),
        "network_util_mean":   float(np.clip(net_mean,  0.0, 1.0)),
        "network_util_stddev": float(np.clip(net_std,   0.01, 0.5)),
    }


def extract_dag_topology(task_df: pd.DataFrame) -> dict:
    """Infer DAG topology distribution and job arrival rate."""
    print("  [dag_topology] computing ...")

    task_col = resolve_column(task_df, "task_name")
    if task_col:
        names = task_df[task_col].fillna("").astype(str).str.lower()
        total = max(len(names), 1)
        chain_cnt   = int(names.str.contains(r"chain|seq|step_\d", regex=True, na=False).sum())
        fanout_cnt  = int(names.str.contains(r"fan.?out|broadcast|split|scatter", regex=True, na=False).sum())
        funnel_cnt  = int(names.str.contains(r"funnel|merge|gather|reduce|join", regex=True, na=False).sum())
        diamond_cnt = int(names.str.contains(r"diamond|fork.*merge|split.*join", regex=True, na=False).sum())
        random_cnt  = max(0, total - chain_cnt - fanout_cnt - funnel_cnt - diamond_cnt)

        named_frac = (chain_cnt + fanout_cnt + funnel_cnt + diamond_cnt) / total
        if named_frac < 0.02:
            # Keyword match too sparse — use empirical defaults
            print("    [INFO] topology keywords sparse (<2% match) — using empirical defaults")
            dist = [["chain", 0.30], ["fan_out", 0.20], ["funnel", 0.15],
                    ["diamond", 0.15], ["random", 0.20]]
        else:
            dist = [
                ["chain",   round(chain_cnt  / total, 4)],
                ["fan_out", round(fanout_cnt  / total, 4)],
                ["funnel",  round(funnel_cnt  / total, 4)],
                ["diamond", round(diamond_cnt / total, 4)],
                ["random",  round(random_cnt  / total, 4)],
            ]
    else:
        dist = [["chain", 0.30], ["fan_out", 0.20], ["funnel", 0.15],
                ["diamond", 0.15], ["random", 0.20]]

    # Arrival rate from job submission timestamps
    arrival_rate = 0.5
    job_col = resolve_column(task_df, "job_name")
    ts_col  = resolve_column(task_df, "start_time")

    if job_col and ts_col:
        job_ts = (pd.to_numeric(task_df[ts_col], errors="coerce")
                  .groupby(task_df[job_col]).min().dropna().sort_values())
        job_ts = job_ts[(job_ts > 0) & (job_ts < 1e12)]
        if len(job_ts) > 100:
            duration = float(job_ts.max() - job_ts.min())
            if duration > 0:
                raw_rate = len(job_ts) / duration
                arrival_rate = float(np.clip(raw_rate, 0.01, 10.0))

    print(f"    job_arrival_rate={arrival_rate:.4f}/s")
    return {
        "dag_topology_distribution": dist,
        "job_arrival_rate_per_s": arrival_rate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Canonical wanted columns per table — the loader will filter to what exists
_TASK_COLS = [
    "job_name", "task_name", "inst_num", "status",
    "start_time", "end_time", "plan_cpu", "plan_mem", "plan_gpu",
    "gpu_type", "workload", "group", "scheduling_class", "priority",
]
_INSTANCE_COLS = [
    "job_name", "inst_id", "user", "status",
    "start_time", "end_time", "machine", "gpu_type_spec", "group",
]
_SENSOR_COLS = [
    "task_name", "worker_name", "machine",
    "cpu_usage", "gpu_wrk_util", "avg_mem", "max_mem",
]
_MACHINE_SPEC_COLS = ["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"]
_METRIC_COLS = [
    "machine", "machine_cpu_usr", "machine_cpu_kernel",
    "machine_cpu_iowait", "machine_gpu", "machine_load_1", "machine_net_receive",
]


def extract_all(data_dir: str, out_dir: str) -> dict:
    print("\n" + "=" * 70)
    print("  Phase 3 — Alibaba PAI GPU v2020 Calibration Extractor (v2)")
    print("=" * 70)

    # ── Load tables (robust — skips missing/misnamed columns gracefully) ──
    task_df         = read_csv_safe(os.path.join(data_dir, "pai_task_table.csv"),     _TASK_COLS)
    instance_df     = read_csv_safe(os.path.join(data_dir, "pai_instance_table.csv"), _INSTANCE_COLS)
    sensor_df       = read_csv_safe(os.path.join(data_dir, "pai_sensor_table.csv"),   _SENSOR_COLS)
    machine_spec_df = read_csv_safe(os.path.join(data_dir, "pai_machine_spec.csv"),   _MACHINE_SPEC_COLS)
    metric_df       = read_csv_safe(os.path.join(data_dir, "pai_machine_metric.csv"), _METRIC_COLS)

    if task_df.empty:
        print("\n[FATAL] pai_task_table.csv could not be loaded. "
              "Check --data_dir points to the folder containing pai_*.csv files, "
              "not to a specific CSV file.\n")
        return {}

    # ── Run extractors ───────────────────────────────────────────────────
    print("\n[1/9] Task duration distribution")
    duration_params = extract_task_duration(task_df)

    print("\n[2/9] Delay probability and magnitude")
    delay_params = extract_delay_probability(task_df)

    print("\n[3/9] Resource demand distributions")
    resource_params = extract_resource_demands(task_df)

    print("\n[4/9] Job size distribution")
    job_size_params = extract_job_size_distribution(task_df)

    print("\n[5/9] Workload type distribution")
    workload_params = extract_workload_distribution(task_df)

    print("\n[6/9] Inst_num distribution")
    inst_num_params = extract_inst_num_distribution(task_df)

    print("\n[7/9] Scheduling class distribution")
    sched_class_params = extract_scheduling_class(task_df)

    print("\n[8/9] GPU type distribution")
    gpu_type_params = extract_gpu_type_distribution(task_df, instance_df)

    print("\n[9a/9] Utilisation factors from sensor table")
    util_params = extract_utilisation_factors(sensor_df)

    print("\n[9b/9] Machine spec")
    machine_params = extract_machine_spec(machine_spec_df)

    print("\n[9c/9] Cluster utilisation targets")
    cluster_util = extract_cluster_utilisation(metric_df)

    print("\n[9d/9] DAG topology and arrival rate")
    dag_params = extract_dag_topology(task_df)

    # ── Assemble ─────────────────────────────────────────────────────────
    calibration = {
        "_meta": {
            "source":     "alibaba_pai_gpu_v2020",
            "kaggle":     "https://www.kaggle.com/datasets/derrickmwiti/cluster-trace-gpu-v2020",
            "extractor":  "extract_alibaba_calibration.py v2",
            "tables_used": [
                "pai_task_table.csv", "pai_instance_table.csv",
                "pai_sensor_table.csv", "pai_machine_spec.csv",
                "pai_machine_metric.csv",
            ],
        },

        # Task duration
        "task_duration_lognormal_mu":    duration_params["lognormal_fit"]["mu"],
        "task_duration_lognormal_sigma": duration_params["lognormal_fit"]["sigma"],
        "task_duration_stats":           duration_params["stats"],

        # Delay model
        "task_delay_probability":        delay_params["task_delay_probability"],
        "task_delay_mean_s":             delay_params["task_delay_mean_s"],
        "task_delay_stddev_s":           delay_params["task_delay_stddev_s"],
        "delay_detail":                  delay_params,

        # Resource demands
        "task_cpu_demand_mean":          resource_params["task_cpu_demand_mean"],
        "task_cpu_demand_stddev":        resource_params["task_cpu_demand_stddev"],
        "task_mem_demand_mean":          resource_params["task_mem_demand_mean"],
        "task_mem_demand_stddev":        resource_params["task_mem_demand_stddev"],
        "task_gpu_demand_mean":          resource_params["task_gpu_demand_mean"],
        "task_gpu_demand_stddev":        resource_params["task_gpu_demand_stddev"],
        "task_gpu_request_probability":  resource_params["task_gpu_request_probability"],
        "resource_demand_detail":        resource_params,

        # Job size
        "job_size_distribution":         job_size_params["job_size_distribution"],
        "job_size_stats":                job_size_params.get("job_size_stats", {}),

        # Workload type
        "workload_type_distribution":    workload_params["workload_type_distribution"],

        # Inst num
        "inst_num_distribution":         inst_num_params["inst_num_distribution"],

        # Scheduling class
        "scheduling_class_distribution": sched_class_params["scheduling_class_distribution"],

        # GPU type
        "gpu_type_distribution":         gpu_type_params["gpu_type_distribution"],

        # Utilisation factors
        "cpu_utilisation_factor_mean":   util_params["cpu_utilisation_factor_mean"],
        "cpu_utilisation_factor_stddev": util_params["cpu_utilisation_factor_stddev"],
        "gpu_utilisation_factor_mean":   util_params["gpu_utilisation_factor_mean"],
        "gpu_utilisation_factor_stddev": util_params["gpu_utilisation_factor_stddev"],
        "mem_utilisation_factor_mean":   util_params["mem_utilisation_factor_mean"],
        "mem_utilisation_factor_stddev": util_params["mem_utilisation_factor_stddev"],

        # Machine spec
        "machine_cpu_cores":             machine_params["machine_cpu_cores"],
        "machine_mem_gb":                machine_params["machine_mem_gb"],
        "machine_gpu_count":             machine_params["machine_gpu_count"],
        "gpu_machine_fraction":          machine_params["gpu_machine_fraction"],
        "machine_spec_detail":           machine_params,

        # Cluster utilisation targets
        "target_cpu_util":               cluster_util["target_cpu_util"],
        "target_gpu_util":               cluster_util["target_gpu_util"],
        "network_util_mean":             cluster_util["network_util_mean"],
        "network_util_stddev":           cluster_util["network_util_stddev"],

        # DAG topology and arrival rate
        "dag_topology_distribution":     dag_params["dag_topology_distribution"],
        "job_arrival_rate_per_s":        dag_params["job_arrival_rate_per_s"],
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "alibaba_calibration.json")
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Saved → {out_path}")
    print(f"  Total keys: {len(calibration)}")
    print(f"{'='*70}\n")
    return calibration


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Phase 3 calibration from Alibaba PAI GPU v2020"
    )
    parser.add_argument(
        "--data_dir", default="data/alibaba",
        help="Directory containing pai_*.csv files (NOT a path to a specific CSV)"
    )
    parser.add_argument(
        "--out_dir", default="simulator/calibrated",
        help="Output directory for alibaba_calibration.json"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"\n[ERROR] --data_dir must be a directory, got: {args.data_dir}")
        print("  If you passed a CSV file path by mistake, pass its parent folder instead.")
        print(f"  e.g.  --data_dir {os.path.dirname(args.data_dir)}\n")
        raise SystemExit(1)

    extract_all(args.data_dir, args.out_dir)