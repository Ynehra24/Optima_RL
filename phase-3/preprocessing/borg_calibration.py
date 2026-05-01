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
        f"{label}_mean":  float(vals.mean()),
        f"{label}_std":   float(vals.std()),
        f"{label}_p10":   float(np.percentile(vals, 10)),
        f"{label}_p25":   float(np.percentile(vals, 25)),
        f"{label}_p50":   float(np.percentile(vals, 50)),
        f"{label}_p75":   float(np.percentile(vals, 75)),
        f"{label}_p90":   float(np.percentile(vals, 90)),
        f"{label}_p99":   float(np.percentile(vals, 99)),
        f"{label}_n":     int(len(vals)),
    }


def read_csv_safe(path: str, usecols=None, chunksize: int = 500_000) -> pd.DataFrame:
    """Read a potentially large CSV safely, optionally in chunks."""
    if not os.path.exists(path):
        print(f"  [WARNING] File not found: {path} — skipping.")
        return pd.DataFrame()
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Reading {os.path.basename(path)} ({size_mb:.0f} MB) ...")
    if size_mb > 500:
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize,
                                  usecols=usecols, low_memory=False):
            chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True)
    return pd.read_csv(path, usecols=usecols, low_memory=False)


# ---------------------------------------------------------------------------
# Section extractors
# ---------------------------------------------------------------------------

def extract_task_duration(task_df: pd.DataFrame) -> dict:
    """
    Task duration = end_time - start_time (seconds).
    Fit log-normal for the simulator's task_duration_lognormal_mu/sigma.
    """
    print("  [task_duration] computing durations ...")
    t = task_df.copy()

    # Alibaba times are in seconds since epoch — already in seconds
    t["start_time"] = pd.to_numeric(t.get("start_time", pd.Series()), errors="coerce")
    t["end_time"]   = pd.to_numeric(t.get("end_time",   pd.Series()), errors="coerce")
    t["duration_s"] = t["end_time"] - t["start_time"]

    # Sanity: keep only positive, <7-day durations
    t = t[(t["duration_s"] > 0) & (t["duration_s"] < 7 * 86400)]

    lognorm = safe_lognormal_fit(t["duration_s"])
    stats   = percentile_dict(t["duration_s"], "duration_s")

    print(f"    mu={lognorm['mu']:.3f}  sigma={lognorm['sigma']:.3f}"
          f"  median={np.exp(lognorm['mu']):.0f}s  n={lognorm['n_samples']:,}")
    return {"lognormal_fit": lognorm, "stats": stats}


def extract_delay_probability(task_df: pd.DataFrame) -> dict:
    """
    Intrinsic delay: fraction of tasks where actual start > expected start.
    Uses start_time vs the earliest possible start given parent dependencies.
    Since we cannot reconstruct the full DAG from a single table pass, we
    approximate: a task is 'delayed' if its status is FAILED or if
    actual_duration > 2 * plan duration (runtime overrun proxy).
    Also compute delay magnitude distribution.
    """
    print("  [delay_probability] computing ...")
    t = task_df.copy()

    # Method 1: status-based (FAILED / evicted)
    if "status" in t.columns:
        failed_frac = float((t["status"].isin(["Failed", "FAILED", "Killed",
                                                "KILLED", "Evicted", "EVICTED"])
                             ).mean())
    else:
        failed_frac = 0.12

    # Method 2: runtime overrun proxy (actual >> plan)
    t["start_time"]   = pd.to_numeric(t.get("start_time",   pd.Series(dtype=float)), errors="coerce")
    t["end_time"]     = pd.to_numeric(t.get("end_time",     pd.Series(dtype=float)), errors="coerce")
    t["plan_cpu"]     = pd.to_numeric(t.get("plan_cpu",     pd.Series(dtype=float)), errors="coerce")
    t["duration_s"]   = t["end_time"] - t["start_time"]

    # Delay magnitude among delayed tasks (seconds)
    # Approximate: tasks with duration 2x their plan_cpu-weighted expected time
    # We use the residual beyond the median as a proxy for delay magnitude
    median_dur = t["duration_s"].median()
    delayed_mask = t["duration_s"] > median_dur * 2
    delay_prob = float(delayed_mask.mean()) if not np.isnan(delayed_mask.mean()) else 0.12

    delay_magnitudes = (t.loc[delayed_mask, "duration_s"] - median_dur).dropna()
    delay_stats = percentile_dict(delay_magnitudes, "delay_magnitude_s")

    # Use failed_frac as a lower bound, overrun proxy as upper — average them
    combined_prob = float(np.clip((failed_frac + delay_prob) / 2, 0.05, 0.35))

    delay_mean = float(delay_magnitudes.mean()) if len(delay_magnitudes) > 0 else 120.0
    delay_std  = float(delay_magnitudes.std())  if len(delay_magnitudes) > 0 else 90.0

    print(f"    delay_probability={combined_prob:.3f}  "
          f"delay_mean={delay_mean:.1f}s  delay_std={delay_std:.1f}s")
    return {
        "task_delay_probability":  combined_prob,
        "task_delay_mean_s":       float(np.clip(delay_mean,  10.0, 600.0)),
        "task_delay_stddev_s":     float(np.clip(delay_std,   5.0,  400.0)),
        "failed_task_frac":        failed_frac,
        "overrun_proxy_frac":      delay_prob,
        "delay_magnitude_stats":   delay_stats,
    }


def extract_resource_demands(task_df: pd.DataFrame) -> dict:
    """
    plan_cpu, plan_mem, plan_gpu demand distributions.
    Alibaba normalises these as fractions of machine capacity already.
    """
    print("  [resource_demands] computing ...")
    t = task_df.copy()

    for col in ["plan_cpu", "plan_mem", "plan_gpu"]:
        t[col] = pd.to_numeric(t.get(col, pd.Series(dtype=float)), errors="coerce")

    # plan_cpu: already fraction of machine cores in Alibaba trace
    cpu = t["plan_cpu"].dropna()
    cpu = cpu[(cpu > 0) & (cpu <= 1.0)]

    mem = t["plan_mem"].dropna()
    mem = mem[(mem > 0) & (mem <= 1.0)]

    gpu = t["plan_gpu"].dropna()
    gpu_request_prob = float((gpu > 0).mean()) if len(gpu) > 0 else 0.65
    gpu = gpu[gpu > 0]

    result = {
        "task_cpu_demand_mean":   float(cpu.mean())   if len(cpu) > 0 else 0.12,
        "task_cpu_demand_stddev": float(cpu.std())    if len(cpu) > 0 else 0.15,
        "task_mem_demand_mean":   float(mem.mean())   if len(mem) > 0 else 0.10,
        "task_mem_demand_stddev": float(mem.std())    if len(mem) > 0 else 0.12,
        "task_gpu_demand_mean":   float(gpu.mean())   if len(gpu) > 0 else 0.50,
        "task_gpu_demand_stddev": float(gpu.std())    if len(gpu) > 0 else 0.30,
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
    """
    Count tasks per job (job_name), build empirical distribution,
    then bucket into {1,2,3,5,8,12,20,50} size classes.
    """
    print("  [job_size] computing ...")
    if "job_name" not in task_df.columns:
        print("    job_name column missing — using defaults")
        return {"job_size_distribution": [
            [1, 0.25], [2, 0.20], [3, 0.15], [5, 0.15],
            [8, 0.10], [12, 0.08], [20, 0.05], [50, 0.02]
        ]}

    job_sizes = task_df.groupby("job_name").size()
    total = len(job_sizes)

    buckets = [1, 2, 3, 5, 8, 12, 20, 50]
    counts  = []
    remaining = job_sizes.copy()

    for i, b in enumerate(buckets):
        if i < len(buckets) - 1:
            lo = buckets[i]
            hi = buckets[i + 1] - 1
            mask = (remaining >= lo) & (remaining <= hi) if i > 0 else (remaining == 1)
        else:
            mask = remaining >= b
        cnt = int(mask.sum())
        counts.append(cnt)

    # Normalize
    total_bucketed = sum(counts)
    if total_bucketed == 0:
        total_bucketed = 1
    probs = [c / total_bucketed for c in counts]

    dist = [[b, round(p, 4)] for b, p in zip(buckets, probs)]

    size_stats = percentile_dict(job_sizes, "job_size")
    print(f"    total_jobs={total:,}  median_size={job_sizes.median():.0f}  "
          f"max_size={job_sizes.max()}")
    return {
        "job_size_distribution": dist,
        "job_size_stats": size_stats,
    }


def extract_workload_distribution(task_df: pd.DataFrame) -> dict:
    """
    Extract workload type distribution from the workload/group columns.
    Maps to: training, inference, etl, pipeline, serving, other.
    """
    print("  [workload_type] computing ...")
    col = None
    for c in ["workload", "group", "task_type"]:
        if c in task_df.columns:
            col = c
            break

    if col is None:
        print("    no workload column found — using defaults")
        return {"workload_type_distribution": [
            ["training", 0.35], ["inference", 0.20], ["etl", 0.15],
            ["pipeline", 0.12], ["serving", 0.10], ["other", 0.08]
        ]}

    raw_counts = task_df[col].fillna("other").str.lower().value_counts()
    total = raw_counts.sum()

    # Canonical mapping — Alibaba group names vary; we map by keyword
    canonical_map = {
        "training":  ["train", "training", "finetune", "pretrain"],
        "inference": ["infer", "inference", "predict", "serving_infer"],
        "etl":       ["etl", "data", "preprocess", "pipeline_data"],
        "pipeline":  ["pipeline", "workflow", "dag"],
        "serving":   ["serv", "serving", "online"],
        "other":     [],  # catch-all
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

    total_mapped = sum(canonical_counts.values())
    if total_mapped == 0:
        total_mapped = 1

    dist = [[k, round(v / total_mapped, 4)] for k, v in canonical_counts.items()]
    print(f"    " + "  ".join(f"{k}={v:.2f}" for k, v in canonical_counts.items()
                              if v > 0))
    return {"workload_type_distribution": dist}


def extract_inst_num_distribution(task_df: pd.DataFrame) -> dict:
    """
    inst_num: number of worker instances per task.
    Bucket into {1, 2, 4, 8, 16}.
    """
    print("  [inst_num] computing ...")
    if "inst_num" not in task_df.columns:
        return {"inst_num_distribution": [
            [1, 0.50], [2, 0.20], [4, 0.15], [8, 0.10], [16, 0.05]
        ]}

    inst = pd.to_numeric(task_df["inst_num"], errors="coerce").dropna()
    inst = inst[inst >= 1].astype(int)
    total = len(inst)

    buckets = [1, 2, 4, 8, 16]
    probs   = []
    for i, b in enumerate(buckets):
        hi = buckets[i + 1] if i < len(buckets) - 1 else 9999
        lo = b
        cnt = int(((inst >= lo) & (inst < hi)).sum())
        probs.append(cnt / total if total > 0 else 0.0)

    dist = [[b, round(p, 4)] for b, p in zip(buckets, probs)]
    print(f"    " + "  ".join(f"{b}:{p:.3f}" for b, p in zip(buckets, probs)))
    return {"inst_num_distribution": dist}


def extract_scheduling_class(task_df: pd.DataFrame) -> dict:
    """
    scheduling_class distribution from the trace.
    Alibaba uses priority or scheduling_class columns.
    """
    print("  [scheduling_class] computing ...")
    col = None
    for c in ["scheduling_class", "priority", "job_priority"]:
        if c in task_df.columns:
            col = c
            break

    if col is None:
        return {"scheduling_class_distribution": [
            [3, 0.15, 0.0,   "production"],
            [2, 0.25, 30.0,  "mid_tier"],
            [1, 0.40, 120.0, "batch"],
            [0, 0.20, 300.0, "best_effort"],
        ]}

    vals = pd.to_numeric(task_df[col], errors="coerce").dropna()
    # Normalise to 0-3 range (Alibaba may use 0-7)
    vals = ((vals - vals.min()) / max(vals.max() - vals.min(), 1) * 3).round().astype(int)
    counts = vals.value_counts().sort_index()
    total = counts.sum()

    sla_map = {3: 0.0, 2: 30.0, 1: 120.0, 0: 300.0}
    name_map = {3: "production", 2: "mid_tier", 1: "batch", 0: "best_effort"}

    dist = []
    for cls in [3, 2, 1, 0]:
        frac = float(counts.get(cls, 0)) / total if total > 0 else 0.25
        dist.append([cls, round(frac, 4), sla_map[cls], name_map[cls]])

    print(f"    " + "  ".join(f"cls{d[0]}={d[1]:.3f}" for d in dist))
    return {"scheduling_class_distribution": dist}


def extract_gpu_type_distribution(task_df: pd.DataFrame,
                                  instance_df: pd.DataFrame) -> dict:
    """
    GPU type distribution from gpu_type column in task or instance table.
    Maps to: V100, A100, T4, P100, A10, other.
    """
    print("  [gpu_type] computing ...")
    col = None
    source_df = None
    for df, name in [(task_df, "task"), (instance_df, "instance")]:
        for c in ["gpu_type", "gpu_name", "gpu_type_spec"]:
            if c in df.columns:
                col = c
                source_df = df
                break
        if col:
            break

    canonical = ["V100", "A100", "T4", "P100", "A10"]
    if col is None or source_df is None:
        probs = [0.30, 0.25, 0.20, 0.15, 0.07, 0.03]
        dist  = [[g, p] for g, p in zip(canonical + ["other"], probs)]
        return {"gpu_type_distribution": dist}

    raw = source_df[col].fillna("other").str.upper()
    total = len(raw)
    counts = {}
    for g in canonical:
        counts[g] = int(raw.str.contains(g).sum())
    counts["other"] = total - sum(counts.values())

    dist = [[k, round(v / total, 4)] for k, v in counts.items() if total > 0]
    print(f"    " + "  ".join(f"{k}={v:.3f}" for k, v in counts.items()))
    return {"gpu_type_distribution": dist}


def extract_utilisation_factors(sensor_df: pd.DataFrame,
                                task_df: pd.DataFrame) -> dict:
    """
    Actual vs planned utilisation ratios from pai_sensor_table.
    cpu_usage / plan_cpu, gpu_wrk_util (raw), avg_mem / plan_mem.
    """
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
        print("    sensor table empty — using defaults")
        return result

    s = sensor_df.copy()

    # CPU utilisation factor
    if "cpu_usage" in s.columns:
        cpu_usage = pd.to_numeric(s["cpu_usage"], errors="coerce").dropna()
        cpu_usage = cpu_usage[(cpu_usage > 0) & (cpu_usage <= 1.0)]
        if len(cpu_usage) > 100:
            result["cpu_utilisation_factor_mean"]   = float(np.clip(cpu_usage.mean(), 0.1, 1.0))
            result["cpu_utilisation_factor_stddev"] = float(np.clip(cpu_usage.std(),  0.01, 0.5))
            print(f"    cpu_util_factor: mean={result['cpu_utilisation_factor_mean']:.3f}  "
                  f"std={result['cpu_utilisation_factor_stddev']:.3f}  n={len(cpu_usage):,}")

    # GPU utilisation (direct, not a ratio — it IS the factor)
    if "gpu_wrk_util" in s.columns:
        gpu_util = pd.to_numeric(s["gpu_wrk_util"], errors="coerce").dropna()
        gpu_util = gpu_util[(gpu_util > 0) & (gpu_util <= 1.0)]
        if len(gpu_util) > 100:
            result["gpu_utilisation_factor_mean"]   = float(np.clip(gpu_util.mean(), 0.1, 1.0))
            result["gpu_utilisation_factor_stddev"] = float(np.clip(gpu_util.std(),  0.01, 0.5))
            print(f"    gpu_util_factor: mean={result['gpu_utilisation_factor_mean']:.3f}  "
                  f"std={result['gpu_utilisation_factor_stddev']:.3f}  n={len(gpu_util):,}")

    # Memory utilisation factor (avg_mem / plan_mem)
    if "avg_mem" in s.columns:
        mem_usage = pd.to_numeric(s["avg_mem"], errors="coerce").dropna()
        mem_usage = mem_usage[(mem_usage > 0) & (mem_usage <= 1.0)]
        if len(mem_usage) > 100:
            result["mem_utilisation_factor_mean"]   = float(np.clip(mem_usage.mean(), 0.1, 1.0))
            result["mem_utilisation_factor_stddev"] = float(np.clip(mem_usage.std(),  0.01, 0.5))
            print(f"    mem_util_factor: mean={result['mem_utilisation_factor_mean']:.3f}  "
                  f"std={result['mem_utilisation_factor_stddev']:.3f}  n={len(mem_usage):,}")

    return result


def extract_machine_spec(machine_spec_df: pd.DataFrame) -> dict:
    """
    Per-machine capacity stats: cap_cpu, cap_gpu, cap_mem.
    Used to set machine_cpu_cores, machine_gpu_count, machine_mem_gb.
    Also computes gpu_machine_fraction.
    """
    print("  [machine_spec] computing ...")
    if machine_spec_df.empty:
        return {
            "machine_cpu_cores":    32.0,
            "machine_mem_gb":       128.0,
            "machine_gpu_count":    4.0,
            "gpu_machine_fraction": 0.30,
        }

    m = machine_spec_df.copy()
    for col in ["cap_cpu", "cap_gpu", "cap_mem"]:
        m[col] = pd.to_numeric(m.get(col, pd.Series(dtype=float)), errors="coerce")

    cpu_median = float(m["cap_cpu"].dropna().median()) if "cap_cpu" in m else 32.0
    mem_median = float(m["cap_mem"].dropna().median()) if "cap_mem" in m else 128.0

    gpu_col = m["cap_gpu"].dropna() if "cap_gpu" in m else pd.Series(dtype=float)
    gpu_median   = float(gpu_col[gpu_col > 0].median()) if len(gpu_col[gpu_col > 0]) > 0 else 4.0
    gpu_frac     = float((gpu_col > 0).mean()) if len(gpu_col) > 0 else 0.30

    print(f"    cpu_cores={cpu_median:.0f}  mem_gb={mem_median:.0f}  "
          f"gpu_count={gpu_median:.0f}  gpu_frac={gpu_frac:.3f}")
    return {
        "machine_cpu_cores":    float(np.clip(cpu_median, 4.0,  512.0)),
        "machine_mem_gb":       float(np.clip(mem_median, 8.0,  4096.0)),
        "machine_gpu_count":    float(np.clip(gpu_median, 0.0,  16.0)),
        "gpu_machine_fraction": float(np.clip(gpu_frac,   0.01, 1.0)),
        "cpu_stats":            percentile_dict(m["cap_cpu"].dropna(), "cap_cpu")
                                    if "cap_cpu" in m else {},
    }


def extract_cluster_utilisation(metric_df: pd.DataFrame) -> dict:
    """
    Rolling cluster CPU/GPU utilisation targets and network stats.
    From pai_machine_metric table.
    """
    print("  [cluster_utilisation] computing ...")
    if metric_df.empty:
        return {
            "target_cpu_util":      0.60,
            "target_gpu_util":      0.40,
            "network_util_mean":    0.45,
            "network_util_stddev":  0.15,
        }

    m = metric_df.copy()
    for col in ["machine_cpu_usr", "machine_gpu", "machine_net_receive"]:
        m[col] = pd.to_numeric(m.get(col, pd.Series(dtype=float)), errors="coerce")

    cpu_util = m["machine_cpu_usr"].dropna()
    cpu_util = cpu_util[(cpu_util >= 0) & (cpu_util <= 1.0)]
    target_cpu = float(cpu_util.mean()) if len(cpu_util) > 0 else 0.60

    gpu_util = m["machine_gpu"].dropna()
    gpu_util = gpu_util[(gpu_util >= 0) & (gpu_util <= 1.0)]
    target_gpu = float(gpu_util.mean()) if len(gpu_util) > 0 else 0.40

    net = m["machine_net_receive"].dropna()
    # Normalise to [0,1] if needed (may be bytes/s in raw form)
    if net.max() > 1.0:
        cap = net.quantile(0.99)
        net = net / cap if cap > 0 else net
    net = net[(net >= 0) & (net <= 1.0)]
    net_mean = float(net.mean()) if len(net) > 0 else 0.45
    net_std  = float(net.std())  if len(net) > 0 else 0.15

    print(f"    target_cpu_util={target_cpu:.3f}  target_gpu_util={target_gpu:.3f}  "
          f"net_mean={net_mean:.3f}  net_std={net_std:.3f}")
    return {
        "target_cpu_util":     float(np.clip(target_cpu, 0.1, 0.95)),
        "target_gpu_util":     float(np.clip(target_gpu, 0.05, 0.95)),
        "network_util_mean":   float(np.clip(net_mean,   0.0,  1.0)),
        "network_util_stddev": float(np.clip(net_std,    0.01, 0.5)),
    }


def extract_dag_topology(task_df: pd.DataFrame) -> dict:
    """
    Infer DAG topology type distribution from task naming conventions.
    Alibaba task names often encode their DAG role (e.g. task_0, merge_task,
    fan_out_task). We use simple keyword heuristics.
    Also infer job arrival rate from timestamps.
    """
    print("  [dag_topology] computing ...")

    # Topology inference from task_name keywords
    if "task_name" in task_df.columns:
        names = task_df["task_name"].fillna("").str.lower()
        total = len(names)

        chain_cnt   = int(names.str.contains(r"chain|seq|step_\d").sum())
        fanout_cnt  = int(names.str.contains(r"fan.?out|broadcast|split|scatter").sum())
        funnel_cnt  = int(names.str.contains(r"funnel|merge|gather|reduce|join").sum())
        diamond_cnt = int(names.str.contains(r"diamond|fork.*merge|split.*join").sum())
        random_cnt  = max(0, total - chain_cnt - fanout_cnt - funnel_cnt - diamond_cnt)

        t = max(total, 1)
        dist = [
            ["chain",   round(chain_cnt  / t, 4)],
            ["fan_out", round(fanout_cnt / t, 4)],
            ["funnel",  round(funnel_cnt / t, 4)],
            ["diamond", round(diamond_cnt / t, 4)],
            ["random",  round(random_cnt / t, 4)],
        ]
        # If all zeros (no keywords matched), fall back to defaults
        if all(d[1] == 0 for d in dist[:-1]):
            dist = [
                ["chain",   0.30],
                ["fan_out", 0.20],
                ["funnel",  0.15],
                ["diamond", 0.15],
                ["random",  0.20],
            ]
    else:
        dist = [
            ["chain",   0.30],
            ["fan_out", 0.20],
            ["funnel",  0.15],
            ["diamond", 0.15],
            ["random",  0.20],
        ]

    # Job arrival rate: from submission timestamps
    arrival_rate = 0.5  # default
    for col in ["start_time", "submit_time", "inst_id"]:
        if col in task_df.columns:
            ts = pd.to_numeric(task_df[col], errors="coerce").dropna().sort_values()
            ts = ts[(ts > 0) & (ts < 1e12)]  # filter obviously wrong values
            if len(ts) > 100:
                # Count distinct jobs per second
                if "job_name" in task_df.columns:
                    job_ts = task_df.groupby("job_name")[col].min()
                    job_ts = pd.to_numeric(job_ts, errors="coerce").dropna().sort_values()
                    duration = float(job_ts.max() - job_ts.min())
                    if duration > 0:
                        arrival_rate = float(len(job_ts) / duration)
                        arrival_rate = float(np.clip(arrival_rate, 0.01, 10.0))
                break

    print(f"    job_arrival_rate={arrival_rate:.4f}/s  topology_dist={[d[0]+':'+str(d[1]) for d in dist]}")
    return {
        "dag_topology_distribution": dist,
        "job_arrival_rate_per_s": arrival_rate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def extract_all(data_dir: str, out_dir: str) -> dict:
    print("\n" + "=" * 70)
    print("  Phase 3 — Alibaba PAI GPU v2020 Calibration Extractor")
    print("=" * 70)

    # ── Load tables ──────────────────────────────────────────────────────
    task_df = read_csv_safe(
        os.path.join(data_dir, "pai_task_table.csv"),
        usecols=["job_name", "task_name", "inst_num", "status",
                 "start_time", "end_time", "plan_cpu", "plan_mem",
                 "plan_gpu", "gpu_type", "workload", "group",
                 "scheduling_class", "priority"],
    )

    instance_df = read_csv_safe(
        os.path.join(data_dir, "pai_instance_table.csv"),
        usecols=["job_name", "inst_id", "user", "status",
                 "start_time", "end_time", "machine", "gpu_type_spec",
                 "group"],
    )

    sensor_df = read_csv_safe(
        os.path.join(data_dir, "pai_sensor_table.csv"),
        usecols=["task_name", "worker_name", "machine",
                 "cpu_usage", "gpu_wrk_util", "avg_mem", "max_mem"],
    )

    machine_spec_df = read_csv_safe(
        os.path.join(data_dir, "pai_machine_spec.csv"),
        usecols=["machine", "gpu_type", "cap_cpu", "cap_mem", "cap_gpu"],
    )

    machine_metric_df = read_csv_safe(
        os.path.join(data_dir, "pai_machine_metric.csv"),
        usecols=["machine", "machine_cpu_usr", "machine_cpu_kernel",
                 "machine_cpu_iowait", "machine_gpu",
                 "machine_load_1", "machine_net_receive"],
    )

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
    util_params = extract_utilisation_factors(sensor_df, task_df)

    print("\n[9b/9] Machine spec")
    machine_params = extract_machine_spec(machine_spec_df)

    print("\n[9c/9] Cluster utilisation targets")
    cluster_util = extract_cluster_utilisation(machine_metric_df)

    print("\n[9d/9] DAG topology and arrival rate")
    dag_params = extract_dag_topology(task_df)

    # ── Assemble ─────────────────────────────────────────────────────────
    calibration = {
        "_meta": {
            "source": "alibaba_pai_gpu_v2020",
            "kaggle": "https://www.kaggle.com/datasets/derrickmwiti/cluster-trace-gpu-v2020",
            "tables_used": [
                "pai_task_table.csv", "pai_instance_table.csv",
                "pai_sensor_table.csv", "pai_machine_spec.csv",
                "pai_machine_metric.csv",
            ],
        },

        # §1 — Task duration (→ SimConfig.task_duration_lognormal_*)
        "task_duration_lognormal_mu":    duration_params["lognormal_fit"]["mu"],
        "task_duration_lognormal_sigma": duration_params["lognormal_fit"]["sigma"],
        "task_duration_stats":           duration_params["stats"],

        # §2 — Delay model (→ SimConfig.task_delay_*)
        "task_delay_probability":        delay_params["task_delay_probability"],
        "task_delay_mean_s":             delay_params["task_delay_mean_s"],
        "task_delay_stddev_s":           delay_params["task_delay_stddev_s"],
        "delay_detail":                  delay_params,

        # §3 — Resource demands (→ SimConfig.task_cpu/mem/gpu_demand_*)
        "task_cpu_demand_mean":          resource_params["task_cpu_demand_mean"],
        "task_cpu_demand_stddev":        resource_params["task_cpu_demand_stddev"],
        "task_mem_demand_mean":          resource_params["task_mem_demand_mean"],
        "task_mem_demand_stddev":        resource_params["task_mem_demand_stddev"],
        "task_gpu_demand_mean":          resource_params["task_gpu_demand_mean"],
        "task_gpu_demand_stddev":        resource_params["task_gpu_demand_stddev"],
        "task_gpu_request_probability":  resource_params["task_gpu_request_probability"],
        "resource_demand_detail":        resource_params,

        # §4 — Job size (→ SimConfig.job_size_distribution)
        "job_size_distribution":         job_size_params["job_size_distribution"],
        "job_size_stats":                job_size_params.get("job_size_stats", {}),

        # §5 — Workload type (→ SimConfig.workload_type_distribution)
        "workload_type_distribution":    workload_params["workload_type_distribution"],

        # §6 — Inst num (→ SimConfig.inst_num_distribution)
        "inst_num_distribution":         inst_num_params["inst_num_distribution"],

        # §7 — Scheduling class (→ SimConfig.scheduling_class_distribution)
        "scheduling_class_distribution": sched_class_params["scheduling_class_distribution"],

        # §8 — GPU type (→ SimConfig.gpu_type_distribution)
        "gpu_type_distribution":         gpu_type_params["gpu_type_distribution"],

        # §9 — Utilisation factors (→ SimConfig.*_utilisation_factor_*)
        "cpu_utilisation_factor_mean":   util_params["cpu_utilisation_factor_mean"],
        "cpu_utilisation_factor_stddev": util_params["cpu_utilisation_factor_stddev"],
        "gpu_utilisation_factor_mean":   util_params["gpu_utilisation_factor_mean"],
        "gpu_utilisation_factor_stddev": util_params["gpu_utilisation_factor_stddev"],
        "mem_utilisation_factor_mean":   util_params["mem_utilisation_factor_mean"],
        "mem_utilisation_factor_stddev": util_params["mem_utilisation_factor_stddev"],

        # §10 — Machine spec (→ SimConfig.machine_*)
        "machine_cpu_cores":             machine_params["machine_cpu_cores"],
        "machine_mem_gb":                machine_params["machine_mem_gb"],
        "machine_gpu_count":             machine_params["machine_gpu_count"],
        "gpu_machine_fraction":          machine_params["gpu_machine_fraction"],
        "machine_spec_detail":           machine_params,

        # §11 — Cluster utilisation targets (→ SimConfig.target_*)
        "target_cpu_util":               cluster_util["target_cpu_util"],
        "target_gpu_util":               cluster_util["target_gpu_util"],
        "network_util_mean":             cluster_util["network_util_mean"],
        "network_util_stddev":           cluster_util["network_util_stddev"],

        # §12 — DAG topology and arrival rate (→ SimConfig.dag_topology_distribution)
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
    parser.add_argument("--data_dir", default="data/alibaba",
                        help="Directory containing pai_*.csv files")
    parser.add_argument("--out_dir",  default="simulator/calibrated",
                        help="Output directory for alibaba_calibration.json")
    args = parser.parse_args()
    extract_all(args.data_dir, args.out_dir)