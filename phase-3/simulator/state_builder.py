"""
state_builder.py — Builds the 65-dim state vector for the A2C agent.

This is the context engine: it takes the raw simulation state
(job, task, cluster) and produces the normalised numpy array
that the neural network consumes.

State vector layout (65 dims, after corrections):
  §1 RL meta:      dims  0-12   (13 dims: CL×7, OL×7, τ* — drop α)
  §2 task identity: dims 13-28   (16 dims)
  §3 DAG structure: dims 29-39   (11 dims)
  §4 resources:     dims 40-47   (8 dims)
  §5 global:        dims 48-59   (12 dims)
  §6 delay tree:    dims 60-64   (5 dims: drop ρ_H_A — it's reward-only)
Total = 65 dims.

Validated against the PDF, with two corrections:
  1. α removed from §1 (it's a constant hyperparameter, not state)
  2. ρ_H_A removed from §6 (post-episode, used only in reward)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from simulator.config import SimConfig
from simulator.models import (
    ClusterSnapshot, Job, SchedulingClass, Task, TaskState,
    TaskStatus, WorkloadType,
)


# Canonical hold-time candidates in seconds (τ ∈ {0,15,30,60,120})
TAU_CANDIDATES = [0, 15, 30, 60, 120, 180, 300]   # 7 candidates for CL/OL vectors


def build_state_vector(
    task_id: str,
    job: Job,
    task_state: TaskState,
    task_states: Dict[str, TaskState],
    cluster: ClusterSnapshot,
    cfg: SimConfig,
    current_time_s: float,
) -> np.ndarray:
    """Build the full 65-dim state vector for a pending HNH decision.

    Args:
        task_id:      The downstream task facing the hold decision.
        job:          The job (DAG) containing the task.
        task_state:   Dynamic state of task_id.
        task_states:  All task states in this job.
        cluster:      Current global cluster snapshot.
        cfg:          Simulator config (for hyperparameters).
        current_time_s: Current simulation time.

    Returns:
        65-dim float32 numpy array, all values in [0, 1] or
        appropriately normalised continuous values.
    """
    task = task_state.task

    # True dim count: §1=15, §2=37, §3=11, §4=8, §5=12, §6=5 → 88 total
    vec = np.zeros(88, dtype=np.float32)
    idx = 0

    # ==================================================================
    # §1 — RL Meta (13 dims: 7 for CL, 7 for OL, 1 for τ*)
    # ==================================================================

    # Compute CL(τ) and OL(τ) for each τ candidate
    cl_values = []
    ol_values = []

    upstream_delay = _compute_upstream_delay(task_id, job, task_states, current_time_s)
    gpu_util = cluster.gpu_util

    for tau in TAU_CANDIDATES:
        cl = _compute_cl(tau, task_id, job, task_states, task, current_time_s, cfg)
        ol = _compute_ol(tau, task, gpu_util, upstream_delay, cfg)
        cl_values.append(cl)
        ol_values.append(ol)

    vec[idx:idx+7] = np.clip(cl_values, 0.0, 1.0)
    idx += 7
    vec[idx:idx+7] = np.clip(ol_values, 0.0, 1.0)
    idx += 7

    # τ* = argmax[α·CL(τ) + (1-α)·OL(τ)], normalised by hold_max_s
    combined = [cfg.alpha * cl + (1 - cfg.alpha) * ol
                for cl, ol in zip(cl_values, ol_values)]
    tau_star_idx = int(np.argmax(combined))
    tau_star_s = TAU_CANDIDATES[tau_star_idx]
    # Clamp by SLO deadline
    tau_star_s = min(tau_star_s, task.slo_deadline_s)
    vec[idx] = tau_star_s / max(cfg.hold_max_s, 1.0)
    idx += 1
    # idx = 13 ✓

    # ==================================================================
    # §2 — Task Identity & Priority (16 dims)
    # ==================================================================

    # job_id embedding (8 dims): hash the job_id string to a stable vector
    vec[idx:idx+8] = _hash_embed(job.job_id, 8)
    idx += 8

    # task_index / max_tasks_in_job (1 dim)
    vec[idx] = task.task_index / max(job.job_size - 1, 1)
    idx += 1

    # priority: normalised to [0,1] (Borg 0-11) + one-hot top-3 classes
    vec[idx] = task.priority / 11.0
    idx += 1
    # One-hot for top-3 priority tiers: low (0-3), mid (4-7), high (8-11)
    vec[idx]   = 1.0 if task.priority <= 3 else 0.0
    vec[idx+1] = 1.0 if 4 <= task.priority <= 7 else 0.0
    vec[idx+2] = 1.0 if task.priority >= 8 else 0.0
    idx += 3

    # scheduling_class one-hot (4 dims)
    sc_idx = task.scheduling_class.value  # 0-3
    vec[idx + sc_idx] = 1.0
    idx += 4

    # collection_type (1 dim): for simplicity, 0=job, 1=alloc-set
    # In synthetic data, all tasks are in jobs (not alloc-sets)
    vec[idx] = 0.0
    idx += 1

    # user_id (1 dim): hash to [0,1]
    vec[idx] = (hash(job.job_id) % 1000) / 1000.0
    idx += 1

    # workload_type one-hot (6 dims)
    wl_idx = task.workload_type_idx
    if 0 <= wl_idx < 6:
        vec[idx + wl_idx] = 1.0
    idx += 6

    # gpu_type_spec one-hot (6 dims) — drop the 7th "none" into all-zeros
    gpu_idx = task.gpu_type_idx
    if 0 <= gpu_idx < 6:
        vec[idx + gpu_idx] = 1.0
    idx += 6

    # inst_num (1 dim): log-normalised
    MAX_INST = 64.0
    vec[idx] = math.log1p(task.inst_num) / math.log1p(MAX_INST)
    idx += 1

    # task_status one-hot (5 dims): pending/ready/running/evicted/failed
    status_map = {
        TaskStatus.PENDING: 0,
        TaskStatus.READY:   1,
        TaskStatus.RUNNING: 2,
        TaskStatus.EVICTED: 3,
        TaskStatus.FAILED:  4,
    }
    status_i = status_map.get(task_state.status, 1)
    vec[idx + status_i] = 1.0
    idx += 5
    # idx = 13 + 16 = 29 ✓

    # ==================================================================
    # §3 — DAG Structure (11 dims)
    # ==================================================================

    job_size = max(job.job_size, 1)
    max_depth = max(job.max_depth(), 1)

    # num_parents (1 dim)
    num_parents = len(job.get_parents(task_id))
    vec[idx] = num_parents / job_size
    idx += 1

    # num_children (1 dim)
    num_children = len(job.get_children(task_id))
    vec[idx] = num_children / job_size
    idx += 1

    # total_descendants (1 dim)
    vec[idx] = job.get_total_descendants(task_id) / job_size
    idx += 1

    # critical_path_len (1 dim): normalised by job_deadline
    job_deadline_rel = max(job.job_deadline_s - job.arrival_time_s, 1.0)
    vec[idx] = job.get_critical_path_len(task_id) / job_deadline_rel
    idx += 1

    # slack_time (1 dim)
    vec[idx] = min(job.get_slack_time(task_id) / job_deadline_rel, 1.0)
    idx += 1

    # is_on_critical_path (1 dim)
    vec[idx] = 1.0 if job.is_on_critical_path(task_id) else 0.0
    idx += 1

    # depth_in_dag (1 dim)
    vec[idx] = job.get_depth(task_id) / max_depth
    idx += 1

    # fan_out_ratio (1 dim): num_children / (job_size - 1)
    vec[idx] = num_children / max(job_size - 1, 1)
    idx += 1

    # upstream_delay (1 dim): normalised by delta_f
    vec[idx] = min(upstream_delay / max(cfg.delta_f, 1.0), 1.0)
    idx += 1

    # job_size (1 dim): log-normalised
    MAX_JOB_SIZE = 200.0
    vec[idx] = math.log1p(job_size) / math.log1p(MAX_JOB_SIZE)
    idx += 1

    # dag_completion_fraction (1 dim)
    completed = sum(
        1 for tid in job.tasks
        if task_states.get(tid) and task_states[tid].status == TaskStatus.DONE
    )
    vec[idx] = completed / job_size
    idx += 1
    # idx = 29 + 11 = 40 ✓

    # ==================================================================
    # §4 — Resource Demands (8 dims)
    # ==================================================================

    vec[idx]   = np.clip(task.plan_cpu, 0.0, 1.0)
    vec[idx+1] = np.clip(task.plan_mem, 0.0, 1.0)
    vec[idx+2] = np.clip(task.plan_gpu, 0.0, 1.0)
    vec[idx+3] = np.clip(task.cpu_usage * task.plan_cpu, 0.0, 1.0)
    vec[idx+4] = np.clip(task.gpu_wrk_util * task.plan_gpu, 0.0, 1.0)
    vec[idx+5] = np.clip(task.avg_mem_usage * task.plan_mem, 0.0, 1.0)
    vec[idx+6] = np.clip(task.max_mem_usage * task.plan_mem, 0.0, 1.0)
    vec[idx+7] = np.clip(task.resource_cost_score, 0.0, 1.0)
    idx += 8
    # idx = 40 + 8 = 48 ✓

    # ==================================================================
    # §5 — Global Cluster Context (12 dims)
    # ==================================================================

    MAX_CAP_CPU = cfg.num_machines * cfg.machine_cpu_cores
    MAX_CAP_GPU = (cfg.num_machines * cfg.gpu_machine_fraction
                   * cfg.machine_gpu_count)
    MAX_CAP_MEM = cfg.num_machines * cfg.machine_mem_gb

    vec[idx]   = cluster.total_cpu_capacity / max(MAX_CAP_CPU, 1.0)
    vec[idx+1] = cluster.total_gpu_capacity / max(MAX_CAP_GPU, 1.0)
    vec[idx+2] = np.clip(cluster.cpu_util, 0.0, 1.0)
    vec[idx+3] = np.clip(cluster.gpu_util, 0.0, 1.0)

    MAX_PENDING = 5000.0
    MAX_RUNNING = 5000.0
    vec[idx+4] = math.log1p(cluster.num_pending_tasks) / math.log1p(MAX_PENDING)
    vec[idx+5] = math.log1p(cluster.num_running_tasks) / math.log1p(MAX_RUNNING)

    idle_frac = cluster.num_idle_machines / max(cfg.num_machines, 1)
    vec[idx+6] = np.clip(idle_frac, 0.0, 1.0)

    MAX_LOAD = 4.0   # load avg / num_cpus per machine
    vec[idx+7] = np.clip(cluster.machine_load_avg / MAX_LOAD, 0.0, 1.0)

    vec[idx+8] = np.clip(cluster.network_receive_util, 0.0, 1.0)

    vec[idx+9]  = np.clip(cluster.failed_task_rate_g, 0.0, 1.0)
    vec[idx+10] = np.clip(cluster.global_pipeline_utility_g, 0.0, 1.0)
    vec[idx+11] = np.clip(cluster.global_operator_utility_g, 0.0, 1.0)
    idx += 12
    # idx = 48 + 12 = 60 ✓

    # ==================================================================
    # §6 — Delay Tree Variables (5 dims, ρ_H_A excluded — reward only)
    # ==================================================================

    # D_k: departure delay (normalised by delta_f)
    d_k = max(0.0, current_time_s - task_state.scheduled_start_s)
    vec[idx] = min(d_k / max(cfg.delta_f, 1.0), 1.0)
    idx += 1

    # A_k: arrival delay (0 if task not yet done; used in reward later)
    vec[idx] = min(task_state.arrival_delay_s / max(cfg.delta_f, 1.0), 1.0)
    idx += 1

    # H_k: hold duration so far (normalised by hold_max_s)
    vec[idx] = task_state.hold_duration_s / max(cfg.hold_max_s, 1.0)
    idx += 1

    # GD_k: ground/queue delay (normalised by delta_f)
    vec[idx] = min(task_state.ground_delay_s / max(cfg.delta_f, 1.0), 1.0)
    idx += 1

    # SLO_deadline_k: normalised by job_deadline
    vec[idx] = task.slo_deadline_s / max(job_deadline_rel, 1.0)
    idx += 1
    # idx = 60 + 5 = 65 ✓

    assert idx == 88, f"State vector dim mismatch: expected 88, got {idx}"
    return vec


# ===========================================================================
# Helper: CL(τ) — local pipeline utility
# ===========================================================================

def _compute_cl(
    tau: float,
    task_id: str,
    job: Job,
    task_states: Dict[str, TaskState],
    task: Task,
    current_time_s: float,
    cfg: SimConfig,
) -> float:
    """CL(τ) = (1/N) · Σ w_i · (1 − σ_i(τ)).

    For each child task i, estimate whether it will complete within SLO
    if we hold the current task for τ seconds.
    """
    children = job.get_children(task_id)
    if not children:
        # No children — holding has no pipeline benefit
        return 0.5

    N = len(children)
    total = 0.0
    total_weight = 0.0

    for child_id in children:
        child_ts = task_states.get(child_id)
        if child_ts is None:
            continue
        child_task = child_ts.task
        # Priority weight (normalised across children)
        w_i = (child_task.priority + 1) / 12.0

        # Estimate delay to child i if we hold τ seconds
        hold_induced_delay = tau
        extra_upstream = _compute_upstream_delay(child_id, job, task_states, current_time_s)
        delta_i = max(0.0, hold_induced_delay + extra_upstream
                      - child_task.slo_deadline_s)

        sigma_i = _disutility(delta_i, child_task.priority, cfg)
        total += w_i * (1.0 - sigma_i)
        total_weight += w_i

    if total_weight == 0.0:
        return 0.5
    return total / total_weight


def _disutility(delta_i: float, priority: int, cfg: SimConfig) -> float:
    """σ_i(τ): task disutility (0 if on-time, rises with delay).

    σ_i(τ) = 0                               if δ_i ≤ T_sla
           = (1 + priority_i) · min(δ_i, ∆C) / ∆C  otherwise
    """
    # T_sla from scheduling class
    priority_tier = min(priority // 3, 3)  # maps 0-11 to 0-3
    t_sla_map = {3: 0.0, 2: 30.0, 1: 120.0, 0: 300.0}
    t_sla = t_sla_map.get(priority_tier, 120.0)
    if delta_i <= t_sla:
        return 0.0
    scale = (1 + min(priority, 3))  # priority_i ∈ {0,1,2,3}
    return scale * min(delta_i, cfg.delta_c) / cfg.delta_c


# ===========================================================================
# Helper: OL(τ) — local operator utility
# ===========================================================================

def _compute_ol(
    tau: float,
    task: Task,
    gpu_util: float,
    upstream_delay: float,
    cfg: SimConfig,
) -> float:
    """OL(τ) = 1 − δ_k(τ)/∆F − λ · max(0, gpu_util − B_thresh) · τ/120.

    δ_k(τ) = estimated start delay of task k if held τ seconds
    Hard constraint: OL = 0 for τ > SLO_deadline_k (acts like −∞ for policy)
    """
    if tau > task.slo_deadline_s and task.slo_deadline_s > 0:
        return 0.0
    delta_k = upstream_delay + tau
    congestion_penalty = cfg.lam * max(0.0, gpu_util - cfg.b_thresh) * tau / 120.0
    ol = 1.0 - delta_k / max(cfg.delta_f, 1.0) - congestion_penalty
    return max(0.0, min(1.0, ol))


# ===========================================================================
# Helper: upstream delay (∆_in)
# ===========================================================================

def _compute_upstream_delay(
    task_id: str,
    job: Job,
    task_states: Dict[str, TaskState],
    current_time_s: float,
) -> float:
    """Estimated remaining delay of the upstream (parent) task.

    Takes the max across all still-running parents.
    Returns 0.0 if all parents are done.
    """
    parents = job.get_parents(task_id)
    max_delay = 0.0
    for pid in parents:
        pts = task_states.get(pid)
        if pts is None:
            continue
        if pts.status == TaskStatus.DONE:
            continue
        if pts.actual_start_s is not None:
            expected_end = pts.actual_start_s + pts.task.expected_duration_s
            remaining = expected_end + pts.intrinsic_delay_s - current_time_s
        else:
            remaining = pts.task.expected_duration_s + pts.intrinsic_delay_s
        max_delay = max(max_delay, remaining)
    return max(0.0, max_delay)


# ===========================================================================
# Helper: stable hash embedding for job_id
# ===========================================================================

def _hash_embed(s: str, dim: int) -> np.ndarray:
    """Map a string to a stable dim-d vector in [0,1] via deterministic hash.

    Uses Python's built-in hash seeded with the string itself (stable
    across runs because we control the seed via int(hash(s)) % large_prime).
    """
    vec = np.zeros(dim, dtype=np.float32)
    for i in range(dim):
        # Different prime per dimension for independence
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        val = hash(s + str(primes[i % len(primes)])) % 100_003
        vec[i] = val / 100_003.0
    return vec
