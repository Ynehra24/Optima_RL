"""
reward_engine.py — Reward computation via the Delay Tree.

Directly implements §7 of the state space document:
  R_T_k = β · R_L_k + (1−β) · R_G_k
  R_L_k = α · CL_k + (1−α) · OL_k
  R_G_k = α · CG_k + (1−α) · OG_k

The Delay Tree attribution (§6):
  ρ(H_k, A_k) = H_k / max(Σ positive_delay_components, ε)
used to split global rewards across hold decisions.

This is a *step-level* reward function. It runs after the hold duration
expires (not post-episode), using the realised values of A_k, D_k, H_k.
ρ(H_k, A_k) is computed here and is NOT part of the state vector.
"""

from __future__ import annotations

from typing import Dict

from simulator.config import SimConfig
from simulator.models import ClusterSnapshot, Job, TaskState, TaskStatus


def compute_reward(
    task_id: str,
    job: Job,
    task_state: TaskState,
    task_states: Dict[str, TaskState],
    cluster: ClusterSnapshot,
    cfg: SimConfig,
) -> float:
    """Compute R_T_k for the hold decision just taken on task_id.

    Called after the hold window expires (when we know the realised
    departure delay D_k and can estimate A_k).

    Returns a scalar reward in roughly [-1, 1].
    """
    if task_state.status == TaskStatus.FAILED:
        return -1.0
    if task_state.status == TaskStatus.EVICTED:
        return -0.8

    r_local = _compute_local_reward(task_id, job, task_state, task_states, cfg)
    r_global = _compute_global_reward(task_id, task_state, cluster, cfg)

    r_total = cfg.beta * r_local + (1.0 - cfg.beta) * r_global
    r_total += _compute_action_shaping(task_id, job, task_state, cluster, cfg)
    if task_state.restart_count > 0:
        r_total -= min(0.3, 0.1 * task_state.restart_count)
    return float(max(-1.0, min(1.0, r_total)))


def _compute_action_shaping(
    task_id: str,
    job: Job,
    task_state: TaskState,
    cluster: ClusterSnapshot,
    cfg: SimConfig,
) -> float:
    """Dense action-quality signal for learning.

    The base reward is intentionally paper-shaped, but after switching to real
    completion events it is too weakly tied to the immediate HNH choice. This
    shaping rewards holds that cover real intrinsic delay and penalizes excess
    hold time, residual missed delay, and holding under resource pressure.
    """
    hold_s = max(0.0, task_state.hold_duration_s)
    intrinsic_s = max(0.0, task_state.intrinsic_delay_s)
    useful_hold_s = min(hold_s, intrinsic_s)
    wasted_hold_s = max(0.0, hold_s - intrinsic_s)
    residual_s = max(0.0, intrinsic_s - hold_s)

    # Downstream tasks are where HNH has real pipeline value. Leaf tasks still
    # get a tiny benefit for absorbing their own delay, but not enough to make
    # blanket holding attractive.
    downstream_weight = 1.0 if job.get_children(task_id) else 0.35
    priority_weight = 0.5 + 0.5 * (task_state.task.priority / 11.0)
    resource_pressure = max(cluster.cpu_util, cluster.gpu_util)
    resource_cost = task_state.task.resource_cost_score

    benefit = 0.42 * downstream_weight * priority_weight * (
        useful_hold_s / max(cfg.delta_f, 1.0)
    )
    residual_penalty = 0.34 * downstream_weight * priority_weight * (
        residual_s / max(cfg.delta_c, 1.0)
    )
    waste_penalty = 0.30 * (wasted_hold_s / max(cfg.delta_f, 1.0))
    occupancy_penalty = 0.18 * resource_cost * resource_pressure * (
        hold_s / max(cfg.hold_max_s, 1.0)
    )

    # No-hold should not be artificially worse when there is little actionable
    # delay. This small bonus helps agents learn "do nothing" as a valid action.
    no_hold_bonus = 0.04 if hold_s == 0.0 and intrinsic_s <= 5.0 else 0.0

    return benefit + no_hold_bonus - residual_penalty - waste_penalty - occupancy_penalty


# ===========================================================================
# Local reward: R_L_k = α · CL_k + (1−α) · OL_k
# ===========================================================================

def _compute_local_reward(
    task_id: str,
    job: Job,
    task_state: TaskState,
    task_states: Dict[str, TaskState],
    cfg: SimConfig,
) -> float:
    """R_L_k: measured (not forecasted) local utility after the hold.

    CL_k: fraction of child tasks that completed on time.
    OL_k: operator utility — penalise based on actual departure delay D_k.
    """
    cl_k = _measure_pipeline_utility(task_id, job, task_state, task_states, cfg)
    ol_k = _measure_operator_utility(task_state, cfg)
    return cfg.alpha * cl_k + (1.0 - cfg.alpha) * ol_k


def _measure_pipeline_utility(
    task_id: str,
    job: Job,
    task_state: TaskState,
    task_states: Dict[str, TaskState],
    cfg: SimConfig,
) -> float:
    """CL_k: fraction of child tasks that completed within SLO.

    If hold saved the connection → high CL.
    If no children → moderate reward (no pipeline to save).
    """
    children = job.get_children(task_id)
    if not children:
        # No downstream dependency to save; keep this neutral so leaf tasks
        # cannot dominate reward through positive pipeline credit.
        return 0.5

    N = len(children)
    total = 0.0
    total_weight = 0.0

    for cid in children:
        cts = task_states.get(cid)
        if cts is None:
            continue
        child_task = cts.task
        w_i = (child_task.priority + 1) / 12.0

        if cts.status == TaskStatus.DONE:
            # Compute actual delay to child
            delta_i = cts.completion_delay_s
        elif cts.status in (TaskStatus.FAILED, TaskStatus.EVICTED):
            # Pipeline stalled — maximum disutility
            delta_i = cfg.delta_c
        else:
            # Still running — use current delay as estimate
            delta_i = cts.departure_delay_s + cts.hold_duration_s

        from simulator.state_builder import _disutility
        sigma_i = _disutility(delta_i, child_task.priority, cfg)
        total += w_i * (1.0 - sigma_i)
        total_weight += w_i

    if total_weight == 0.0:
        return 0.5
    return total / total_weight


def _measure_operator_utility(task_state: TaskState, cfg: SimConfig) -> float:
    """OL_k: operator utility based on actual departure delay D_k.

    1 - D_k/∆F, clamped to [0,1].
    Hard penalty if τ > SLO_deadline_k (SLO violation).
    """
    # SLO violation
    if task_state.hold_duration_s > task_state.task.slo_deadline_s > 0:
        return 0.0

    d_k = task_state.departure_delay_s
    residual_delay = max(0.0, task_state.intrinsic_delay_s - task_state.hold_duration_s)
    wasted_hold = max(0.0, task_state.hold_duration_s - task_state.intrinsic_delay_s)
    ol = 1.0 - (d_k + 0.5 * residual_delay + 0.75 * wasted_hold) / max(cfg.delta_f, 1.0)
    return max(0.0, min(1.0, ol))


# ===========================================================================
# Global reward: R_G_k = α · CG_k + (1−α) · OG_k (attributed via ρ)
# ===========================================================================

def _compute_global_reward(
    task_id: str,
    task_state: TaskState,
    cluster: ClusterSnapshot,
    cfg: SimConfig,
) -> float:
    """R_G_k: global cluster utility attributed to this hold decision.

    Uses the Delay Tree influence index ρ(H_k, A_k) to scale the
    contribution of this decision to the global reward.

    CG_k = cluster.global_pipeline_utility_g (rolling 24h average)
    OG_k = cluster.global_operator_utility_g
    """
    rho = _compute_rho(task_state)

    cg_k = cluster.global_pipeline_utility_g
    og_k = cluster.global_operator_utility_g

    r_global = cfg.alpha * cg_k + (1.0 - cfg.alpha) * og_k

    # Scale by influence index: this hold's contribution to global delay
    return rho * r_global


def _compute_rho(task_state: TaskState) -> float:
    """ρ(H_k, A_k): influence of hold H_k on arrival delay A_k.

    ρ = H_k / max(Σ positive_delay_components, ε)

    Delay components that contribute to A_k:
      H_k  — hold duration
      GD_k — ground/queue delay (excluding hold)
      noise — residual (task.intrinsic_delay_s not covered above)
    """
    epsilon = 1e-6
    h_k  = max(0.0, task_state.hold_duration_s)
    gd_k = max(0.0, task_state.ground_delay_s)
    intrinsic = max(0.0, task_state.intrinsic_delay_s)

    positive_components = h_k + gd_k + intrinsic
    rho = h_k / max(positive_components, epsilon)

    # Store on task_state so the Delay Tree can walk back to it later
    task_state.rho_h_a = float(rho)
    return rho


# ===========================================================================
# Delay Tree: attribute global reward back to past hold decisions
# ===========================================================================

def attribute_global_reward_delay_tree(
    job: Job,
    task_states: Dict[str, TaskState],
    cfg: SimConfig,
) -> Dict[str, float]:
    """Post-episode Delay Tree attribution.

    For each task in the job that completed with a non-zero arrival delay,
    walk back through the delay tree and apportion the global reward
    contribution to each hold decision that caused (part of) the delay.

    Returns: dict of task_id -> global_reward_contribution.

    This is the exact mechanism from Malladi et al. §5.1 adapted to DAGs.
    """
    try:
        from rewardEngineering.delay_tree import DAGDelayTree

        tree = DAGDelayTree()
        attributed = tree.attribute_job_outcomes(
            job=job,
            task_states=task_states,
            outcome_scale=1.0 / max(cfg.delta_f, 1.0),
        )
        contributions: Dict[str, float] = {tid: 0.0 for tid in job.tasks}
        for task_id, value in attributed.items():
            if task_id in contributions:
                contributions[task_id] = float(value)
        return contributions
    except Exception:
        # Keep the original local fallback available if the standalone reward
        # engineering package is not on sys.path.
        pass

    contributions = {tid: 0.0 for tid in job.tasks}

    for sink_id in job.tasks:
        ts = task_states.get(sink_id)
        if ts is None or ts.arrival_delay_s <= 0:
            continue

        # Walk the delay tree rooted at this sink's A_k
        _attribute_from_sink(
            sink_id=sink_id,
            sink_delay=ts.arrival_delay_s,
            job=job,
            task_states=task_states,
            contributions=contributions,
            influence_so_far=1.0,
            visited=set(),
        )

    return contributions


def _attribute_from_sink(
    sink_id: str,
    sink_delay: float,
    job: Job,
    task_states: Dict[str, TaskState],
    contributions: Dict[str, float],
    influence_so_far: float,
    visited: set,
):
    """Recursive Delay Tree walk from a delayed task back to hold decisions."""
    if sink_id in visited:
        return
    visited.add(sink_id)

    ts = task_states.get(sink_id)
    if ts is None:
        return

    # This task held — attribute a share of the delay here
    if ts.hold_duration_s > 0:
        # ρ(H_k, A_k): fraction of this task's delay attributable to its hold
        rho = _compute_rho(ts)
        contributions[sink_id] += influence_so_far * rho * sink_delay

    # Walk to parents (Rule 2 of the Delay Tree: departure delay propagates)
    parents = job.get_parents(sink_id)
    if not parents:
        return

    # Apportion influence across parents proportional to their delay contribution
    parent_delays = {}
    for pid in parents:
        pts = task_states.get(pid)
        if pts is None:
            continue
        # Parent's arrival delay that flowed into this task's departure delay
        parent_delays[pid] = max(0.0, pts.arrival_delay_s)

    total_parent_delay = sum(parent_delays.values())
    if total_parent_delay <= 0:
        return

    for pid, pd in parent_delays.items():
        parent_rho = pd / total_parent_delay
        _attribute_from_sink(
            sink_id=pid,
            sink_delay=pd,
            job=job,
            task_states=task_states,
            contributions=contributions,
            influence_so_far=influence_so_far * parent_rho,
            visited=visited,
        )
