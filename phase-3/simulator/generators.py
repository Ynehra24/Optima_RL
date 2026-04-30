"""
generators.py — Synthetic job/cluster data generation.

Generates Borg+Alibaba-calibrated synthetic data:
  - Machines with realistic CPU/GPU/memory capacities
  - Jobs with DAGs of varying topology and size
  - Tasks with resource demands, durations, and intrinsic delays
  - Cluster-wide utilisation patterns

Everything here is SYNTHETIC — no external files needed.
The numeric parameters are grounded in the Google Borg 2019 and
Alibaba PAI 2020 papers, not in the actual datasets.
"""

from __future__ import annotations

import math
import random
import string
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from simulator.config import SimConfig
from simulator.models import (
    GpuType, Job, Machine, SchedulingClass, Task,
    TaskState, TaskStatus, WorkloadType,
)


# ===========================================================================
# Machine generator
# ===========================================================================

def generate_cluster(cfg: SimConfig, rng: np.random.Generator) -> Dict[str, Machine]:
    """Generate a synthetic cluster of machines.

    GPU machines are allocated first (cfg.gpu_machine_fraction),
    the remainder are CPU-only.
    """
    machines: Dict[str, Machine] = {}

    gpu_types = [GpuType(t) for t, _ in cfg.gpu_type_distribution]
    gpu_probs  = [p for _, p in cfg.gpu_type_distribution]
    # Normalise
    total = sum(gpu_probs)
    gpu_probs = [p / total for p in gpu_probs]

    num_gpu_machines = int(cfg.num_machines * cfg.gpu_machine_fraction)

    for i in range(cfg.num_machines):
        mid = f"M{i:04d}"
        has_gpu = i < num_gpu_machines
        if has_gpu:
            idx = rng.choice(len(gpu_types), p=gpu_probs)
            gpu_type = gpu_types[idx]
            cap_gpu = cfg.machine_gpu_count
        else:
            gpu_type = GpuType.NONE
            cap_gpu = 0.0

        # Small variance in machine specs (heterogeneous cluster)
        cpu_var = float(rng.normal(1.0, 0.1))
        mem_var = float(rng.normal(1.0, 0.1))
        machines[mid] = Machine(
            machine_id=mid,
            cap_cpu=max(8.0, cfg.machine_cpu_cores * cpu_var),
            cap_mem=max(16.0, cfg.machine_mem_gb * mem_var),
            cap_gpu=cap_gpu,
            gpu_type=gpu_type,
        )
    return machines


# ===========================================================================
# DAG topology generators
# ===========================================================================

def _chain_dag(task_ids: List[str]) -> List[Tuple[str, str]]:
    """A→B→C→…  linear chain."""
    return [(task_ids[i], task_ids[i+1]) for i in range(len(task_ids)-1)]


def _fan_out_dag(task_ids: List[str]) -> List[Tuple[str, str]]:
    """First task feeds all others in parallel."""
    if len(task_ids) <= 1:
        return []
    root = task_ids[0]
    return [(root, task_ids[i]) for i in range(1, len(task_ids))]


def _funnel_dag(task_ids: List[str]) -> List[Tuple[str, str]]:
    """All tasks feed into the last task."""
    if len(task_ids) <= 1:
        return []
    sink = task_ids[-1]
    return [(task_ids[i], sink) for i in range(len(task_ids)-1)]


def _diamond_dag(task_ids: List[str]) -> List[Tuple[str, str]]:
    """Source → parallel middle → sink (diamond shape)."""
    n = len(task_ids)
    if n < 4:
        return _chain_dag(task_ids)
    source = task_ids[0]
    sink = task_ids[-1]
    middle = task_ids[1:-1]
    edges = [(source, m) for m in middle] + [(m, sink) for m in middle]
    return edges


def _random_dag(task_ids: List[str], rng: np.random.Generator) -> List[Tuple[str, str]]:
    """Random DAG: each task (except the first) has ≥1 parent from earlier tasks.

    Ensures the graph is a valid DAG (edges only go forward in the ordering)
    and is connected (every task reachable from at least one source).
    """
    if len(task_ids) <= 1:
        return []
    edges: List[Tuple[str, str]] = []
    for i in range(1, len(task_ids)):
        # At least one parent from earlier tasks (ensures connectivity)
        num_parents = int(rng.integers(1, min(i + 1, 4)))
        parents_idx = rng.choice(i, size=num_parents, replace=False)
        for j in parents_idx:
            edges.append((task_ids[j], task_ids[i]))
    return edges


def _build_dag_edges(
    task_ids: List[str],
    topology: str,
    rng: np.random.Generator,
) -> List[Tuple[str, str]]:
    dispatch = {
        "chain":    lambda: _chain_dag(task_ids),
        "fan_out":  lambda: _fan_out_dag(task_ids),
        "funnel":   lambda: _funnel_dag(task_ids),
        "diamond":  lambda: _diamond_dag(task_ids),
        "random":   lambda: _random_dag(task_ids, rng),
    }
    return dispatch.get(topology, lambda: _chain_dag(task_ids))()


# ===========================================================================
# Job generator
# ===========================================================================

_WORKLOAD_ORDER = [wt.value for wt in WorkloadType]
_GPU_TYPE_ORDER = [gt.value for gt in GpuType]


def _sample_weighted(choices, rng: np.random.Generator):
    """Sample from (value, probability) pairs."""
    vals = [v for v, _ in choices]
    probs = [p for _, p in choices]
    total = sum(probs)
    probs = [p / total for p in probs]
    idx = rng.choice(len(vals), p=probs)
    return vals[idx]


def generate_job(
    job_id: str,
    arrival_time_s: float,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> Tuple[Job, Dict[str, TaskState]]:
    """Generate one synthetic job with a DAG of tasks.

    Returns (Job, dict of task_id -> TaskState) so the simulator
    can maintain the static Job structure separately from mutable state.
    """
    # Sample job attributes
    sched_class_raw = _sample_weighted(
        [(sc, p) for sc, p, _, _ in cfg.scheduling_class_distribution], rng
    )
    sched_class = SchedulingClass(sched_class_raw)
    slo_s = next(
        slo for sc, p, slo, _ in cfg.scheduling_class_distribution
        if sc == sched_class_raw
    )

    workload_name = _sample_weighted(cfg.workload_type_distribution, rng)
    workload = WorkloadType(workload_name)
    workload_idx = _WORKLOAD_ORDER.index(workload_name)

    num_tasks = _sample_weighted(cfg.job_size_distribution, rng)
    topology = _sample_weighted(cfg.dag_topology_distribution, rng)

    # Determine whether this job uses GPUs
    is_gpu_job = (rng.random() < cfg.task_gpu_request_probability)
    if is_gpu_job:
        gpu_name = _sample_weighted(cfg.gpu_type_distribution, rng)
        gpu_type = GpuType(gpu_name)
        gpu_type_idx = _GPU_TYPE_ORDER.index(gpu_name)
    else:
        gpu_type = GpuType.NONE
        gpu_type_idx = _GPU_TYPE_ORDER.index("none")

    # Job-level SLO deadline (job must finish within this many seconds)
    # Base: sum of all task durations (worst case serial) + slack
    task_durations = []
    tasks: Dict[str, Task] = {}

    for i in range(num_tasks):
        task_id = f"{job_id}_T{i}"

        # Sample duration (log-normal)
        dur = float(rng.lognormal(
            cfg.task_duration_lognormal_mu,
            cfg.task_duration_lognormal_sigma,
        ))
        dur = max(10.0, min(dur, 7200.0))  # clamp [10s, 2h]
        task_durations.append(dur)

        # Resource demands (clamp to [0.01, 1.0])
        plan_cpu = float(np.clip(
            rng.normal(cfg.task_cpu_demand_mean, cfg.task_cpu_demand_stddev),
            0.01, 1.0
        ))
        plan_mem = float(np.clip(
            rng.normal(cfg.task_mem_demand_mean, cfg.task_mem_demand_stddev),
            0.01, 1.0
        ))
        plan_gpu = (
            float(np.clip(
                rng.normal(cfg.task_gpu_demand_mean, cfg.task_gpu_demand_stddev),
                0.01, 1.0
            )) if is_gpu_job else 0.0
        )

        # Actual utilisation
        cpu_util = float(np.clip(
            rng.normal(cfg.cpu_utilisation_factor_mean, cfg.cpu_utilisation_factor_stddev),
            0.05, 1.0
        ))
        gpu_util = float(np.clip(
            rng.normal(cfg.gpu_utilisation_factor_mean, cfg.gpu_utilisation_factor_stddev),
            0.05, 1.0
        )) if is_gpu_job else 0.0
        mem_util = float(np.clip(
            rng.normal(cfg.mem_utilisation_factor_mean, cfg.mem_utilisation_factor_stddev),
            0.05, 1.0
        ))

        # Borg raw priority: production (sched_class 3) → priority 9-11,
        #                    batch (1) → 2-5, etc.
        priority_map = {3: (9, 11), 2: (6, 8), 1: (2, 5), 0: (0, 2)}
        lo, hi = priority_map.get(sched_class_raw, (0, 5))
        priority = int(rng.integers(lo, hi + 1))

        # Instance count
        inst_num = _sample_weighted(cfg.inst_num_distribution, rng)

        tasks[task_id] = Task(
            task_id=task_id,
            job_id=job_id,
            task_index=i,
            scheduling_class=sched_class,
            workload_type=workload,
            gpu_type=gpu_type,
            priority=priority,
            plan_cpu=plan_cpu,
            plan_mem=plan_mem,
            plan_gpu=plan_gpu,
            cpu_usage=cpu_util,
            gpu_wrk_util=gpu_util,
            avg_mem_usage=mem_util,
            max_mem_usage=min(1.0, mem_util * float(rng.uniform(1.0, 1.4))),
            expected_duration_s=dur,
            inst_num=inst_num,
            slo_deadline_s=slo_s,
            gpu_type_idx=gpu_type_idx,
            workload_type_idx=workload_idx,
        )

    # Build DAG edges
    task_id_list = list(tasks.keys())
    edges = _build_dag_edges(task_id_list, topology, rng)

    # Job deadline: longest serial path × 1.5 + baseline slack
    max_serial = sum(task_durations)
    job_deadline = arrival_time_s + max_serial * 1.5 + max(slo_s * 2, 300.0)

    job = Job(
        job_id=job_id,
        tasks=tasks,
        edges=edges,
        scheduling_class=sched_class,
        workload_type=workload,
        arrival_time_s=arrival_time_s,
        job_deadline_s=job_deadline,
    )

    # Compute scheduled_start_s for each task (forward pass from job arrival)
    task_states: Dict[str, TaskState] = {}
    earliest_starts = _compute_earliest_starts(job)
    for tid, task in tasks.items():
        # Sample intrinsic delay
        has_delay = rng.random() < cfg.task_delay_probability
        if has_delay:
            delay = float(rng.normal(cfg.task_delay_mean_s, cfg.task_delay_stddev_s))
            delay = max(5.0, delay)
        else:
            delay = 0.0

        ts = TaskState(
            task=task,
            status=TaskStatus.PENDING if job.get_parents(tid) else TaskStatus.READY,
            submit_time=arrival_time_s,
            scheduled_start_s=arrival_time_s + earliest_starts.get(tid, 0.0),
            has_intrinsic_delay=has_delay,
            intrinsic_delay_s=delay,
        )
        task_states[tid] = ts

    return job, task_states


def _compute_earliest_starts(job: Job) -> Dict[str, float]:
    """Forward pass: earliest start time for each task relative to job arrival."""
    task_ids = list(job.tasks.keys())
    es: Dict[str, float] = {tid: 0.0 for tid in task_ids}

    # Topological order (BFS Kahn's)
    in_degree = {tid: len(job.get_parents(tid)) for tid in task_ids}
    queue = [tid for tid in task_ids if in_degree[tid] == 0]
    while queue:
        tid = queue.pop(0)
        dur = job.tasks[tid].expected_duration_s
        for child in job.get_children(tid):
            es[child] = max(es[child], es[tid] + dur)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return es


# ===========================================================================
# Cluster utilisation generator
# ===========================================================================

def generate_cluster_load_profile(
    cfg: SimConfig,
    rng: np.random.Generator,
    duration_s: float,
) -> Dict[str, float]:
    """Generate a coarse load profile for the cluster.

    Returns a dict of simulation-second -> load_multiplier.
    The multiplier scales arrival rates to simulate diurnal patterns.

    In real clusters, load peaks during business hours and drops at night.
    We model this as a sinusoid with noise.
    """
    profile: Dict[str, float] = {}
    for t in range(0, int(duration_s), 3600):  # hourly resolution
        # Diurnal cycle: peak at t mod 86400 ≈ 9h-17h UTC
        hour_of_day = (t % 86400) / 3600.0
        base = 0.5 + 0.5 * math.sin(math.pi * (hour_of_day - 6) / 12)
        base = max(0.2, base)
        noise = float(rng.normal(1.0, 0.1))
        profile[str(t)] = max(0.1, base * noise)
    return profile
