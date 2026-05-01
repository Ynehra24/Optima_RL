"""
simulator.py — Phase 3 DAG Hold-or-Not-Hold Simulator.

Calibration priority: alibaba_calibration.json > borg_calibration.json > SimConfig defaults.
All generator parameters are patched onto cfg before generate_job/generate_cluster
are called, so calibration values actually reach the synthetic data generation.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulator.config import SimConfig
from simulator.generators import generate_cluster, generate_job
from simulator.models import (
    ClusterSnapshot, Job, Machine, MetricsTracker,
    Task, TaskState, TaskStatus,
)
from simulator.reward_engine import compute_reward
from simulator.state_builder import build_state_vector


# Keys from calibration JSON that map directly onto SimConfig attributes.
# On load, we write these through to cfg so generators.py picks them up.
_CAL_TO_CFG_KEYS = [
    "task_duration_lognormal_mu",
    "task_duration_lognormal_sigma",
    "task_delay_probability",
    "task_delay_mean_s",
    "task_delay_stddev_s",
    "task_cpu_demand_mean",
    "task_cpu_demand_stddev",
    "task_mem_demand_mean",
    "task_mem_demand_stddev",
    "task_gpu_demand_mean",
    "task_gpu_demand_stddev",
    "task_gpu_request_probability",
    "cpu_utilisation_factor_mean",
    "cpu_utilisation_factor_stddev",
    "gpu_utilisation_factor_mean",
    "gpu_utilisation_factor_stddev",
    "mem_utilisation_factor_mean",
    "mem_utilisation_factor_stddev",
    "machine_cpu_cores",
    "machine_mem_gb",
    "machine_gpu_count",
    "gpu_machine_fraction",
    "target_cpu_util",
    "target_gpu_util",
    "network_util_mean",
    "network_util_stddev",
    "job_arrival_rate_per_s",
    "job_size_distribution",
    "dag_topology_distribution",
    "workload_type_distribution",
    "inst_num_distribution",
    "scheduling_class_distribution",
    "gpu_type_distribution",
]

# Alibaba-specific calibration fixes applied before writing to cfg.
# The raw extractor produces some pathological values that need adjustment.
_CALIBRATION_FIXUPS = {
    # plan_cpu in the Alibaba sample happened to be all 1.0 (normalised by
    # machine capacity already). A mean of 1.0 means every task wants a whole
    # machine → pathological. We clamp to a reasonable upper bound.
    "task_cpu_demand_mean":   lambda v: min(v, 0.50),
    "task_cpu_demand_stddev": lambda v: max(v, 0.10),   # enforce non-zero spread
    # plan_mem median was ~0.96, also pathological. Clamp similarly.
    "task_mem_demand_mean":   lambda v: min(v, 0.50),
    # Delay mean of 600s and 30% probability are calibrated from Alibaba
    # but they make the simulator very congested for demo purposes.
    # We keep them as-is for training; they are realistic.
    # hold_max_s must cover the largest action. cfg has 300s but default is 120s.
    # This is fixed by updating hold_max_s after loading actions.
}


class DAGSchedulingSimulator:
    """Phase 3 Hold-or-Not-Hold simulator for cloud DAG scheduling."""

    def __init__(self, cfg: Optional[SimConfig] = None):
        self.cfg = cfg or SimConfig()
        self._calibration: Dict[str, Any] = {}
        self._load_calibration()          # load JSON
        self._patch_cfg_from_calibration()  # write-through to cfg

        self.rng = np.random.default_rng(self.cfg.random_seed)

        self.machines: Dict[str, Machine] = {}
        self.jobs: Dict[str, Job] = {}
        self.task_states: Dict[str, TaskState] = {}

        self._hnh_queue: deque = deque()
        self._pending: Optional[Tuple[float, str, str]] = None

        self.current_time_s: float = 0.0
        self.metrics = MetricsTracker()
        self._done: bool = True
        self._hnh_count: int = 0

        self._cluster_snapshot = ClusterSnapshot()
        self._recent_cl: deque = deque(maxlen=10_000)
        self._recent_ol: deque = deque(maxlen=10_000)
        self._recent_failed: deque = deque(maxlen=10_000)

        self._next_arrival_s: float = 0.0
        self._job_counter: int = 0
        self._state_dim: int = 88

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _load_calibration(self):
        """Load the first available calibration JSON from simulator/calibrated/."""
        base = Path(__file__).parent
        for name in ("alibaba_calibration.json", "borg_calibration.json",
                     "calibration.json"):
            p = base / "calibrated" / name
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        self._calibration = json.load(f)
                    return
                except Exception:
                    pass
        self._calibration = {}

    def _patch_cfg_from_calibration(self):
        """Write calibration values through to cfg so generators pick them up.

        FIX: generators.py reads cfg attributes directly, not _param().
        Without this, generate_job uses SimConfig defaults regardless of
        what is in the calibration JSON.
        """
        for key in _CAL_TO_CFG_KEYS:
            if key not in self._calibration:
                continue
            val = self._calibration[key]
            # Apply any fixups before writing
            if key in _CALIBRATION_FIXUPS:
                val = _CALIBRATION_FIXUPS[key](val)
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, val)

        # FIX: keep hold_max_s consistent with the actual action space.
        # If hold_actions_s has values > hold_max_s, state_builder would
        # normalise τ* to values > 1. Update hold_max_s to cover the space.
        hold_actions = getattr(self.cfg, "hold_actions_s", [0, 120])
        if hold_actions:
            self.cfg.hold_max_s = float(max(hold_actions))

    def _param(self, name: str, default: Any = None) -> Any:
        """Return cfg.name (already patched from calibration) else default."""
        return getattr(self.cfg, name, default)

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the simulator and return (state, info) at the first HNH decision."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng(self.cfg.random_seed)

        self.machines = generate_cluster(self.cfg, self.rng)
        self.jobs.clear()
        self.task_states.clear()
        self._hnh_queue.clear()
        self._pending = None
        self.current_time_s = 0.0
        self.metrics = MetricsTracker()
        self._done = False
        self._hnh_count = 0
        self._recent_cl.clear()
        self._recent_ol.clear()
        self._recent_failed.clear()
        self._job_counter = 0

        self._next_arrival_s = self._sample_inter_arrival()
        self._update_cluster_snapshot()

        state, _, _, info = self._advance_to_next_hnh()
        return state, info

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply hold action and advance to the next HNH decision."""
        if self._done:
            raise RuntimeError("Episode done — call reset() first.")
        if self._pending is None:
            raise RuntimeError("No pending HNH — did you call reset()?")

        decision_time, task_id, job_id = self._pending
        job = self.jobs.get(job_id)
        ts = self.task_states.get(task_id)
        if job is None or ts is None:
            self._pending = None
            return self._advance_to_next_hnh(reward=0.0)

        hold_actions = self.cfg.hold_actions_s
        action_idx = int(np.clip(action_idx, 0, len(hold_actions) - 1))
        hold_s = float(hold_actions[action_idx])
        if ts.task.slo_deadline_s > 0:
            hold_s = min(hold_s, ts.task.slo_deadline_s)

        ts.hold_duration_s = hold_s
        ts.hnh_decided = True
        ts.hnh_action_idx = action_idx

        self.current_time_s = decision_time + hold_s
        ts.actual_start_s = self.current_time_s
        ts.departure_delay_s = max(0.0, self.current_time_s - ts.scheduled_start_s)
        ts.status = TaskStatus.RUNNING

        if hold_s == 0:
            # No-hold: preempt immediately
            ts.status = TaskStatus.EVICTED
            ts.restart_count += 1
            self.metrics.no_hold_decisions += 1
            self.metrics.total_pipeline_stalls += 1
            self.metrics.evicted_tasks += 1
        else:
            self.metrics.hold_decisions += 1
            self._assign_machine(task_id, ts)

        # Snapshot after assignment, before reward (Fix 1)
        self._update_cluster_snapshot()

        reward = compute_reward(
            task_id=task_id, job=job, task_state=ts,
            task_states={t: self.task_states[t] for t in job.tasks
                         if t in self.task_states},
            cluster=self._cluster_snapshot, cfg=self.cfg,
        )

        self.metrics.total_reward += reward
        self.metrics.episode_steps += 1
        self._hnh_count += 1
        self._pending = None

        if ts.status == TaskStatus.RUNNING:
            self._simulate_task_completion(task_id, ts, job)
        elif ts.status == TaskStatus.EVICTED:
            self._requeue_evicted_task(task_id, ts, job)

        return self._advance_to_next_hnh(reward=reward)

    def run_episode(self, policy: str = "no_hold",
                    seed: Optional[int] = None) -> Dict[str, Any]:
        """Run a full episode with a fixed policy. Returns metrics summary."""
        state, info = self.reset(seed=seed)
        done = False
        while not done:
            if policy == "no_hold":
                action = 0
            elif policy == "heuristic":
                info = self._get_current_info()
                if (info.get("upstream_delay_s", 0) > 0
                        and self._cluster_snapshot.gpu_util < self.cfg.b_thresh):
                    # Hold for 30s if available in action space, else midpoint
                    try:
                        action = self.cfg.hold_actions_s.index(30)
                    except ValueError:
                        action = len(self.cfg.hold_actions_s) // 2
                else:
                    action = 0
            elif policy == "random":
                action = int(self.rng.integers(0, len(self.cfg.hold_actions_s)))
            else:
                action = 0
            state, reward, done, info = self.step(action)
        return self.metrics.summary()

    # ------------------------------------------------------------------
    # Internal: event loop
    # ------------------------------------------------------------------

    def _advance_to_next_hnh(self, reward: float = 0.0
                              ) -> Tuple[np.ndarray, float, bool, Dict]:
        while True:
            if self.current_time_s >= self.cfg.episode_duration_s:
                self._done = True
                return self._null_state(), reward, True, self.metrics.summary()
            if self._hnh_count >= self.cfg.max_hnh_decisions:
                self._done = True
                return self._null_state(), reward, True, self.metrics.summary()

            if self._hnh_queue:
                event_time, task_id, job_id = self._hnh_queue.popleft()
                self.current_time_s = max(self.current_time_s, event_time)
                self._process_arrivals_up_to(self.current_time_s)

                ts = self.task_states.get(task_id)
                job = self.jobs.get(job_id)
                if ts is None or job is None:
                    continue
                if ts.status not in (TaskStatus.PENDING, TaskStatus.READY):
                    continue

                self._update_cluster_snapshot()
                self._pending = (event_time, task_id, job_id)
                state = build_state_vector(
                    task_id=task_id, job=job, task_state=ts,
                    task_states={t: self.task_states[t] for t in job.tasks
                                 if t in self.task_states},
                    cluster=self._cluster_snapshot, cfg=self.cfg,
                    current_time_s=self.current_time_s,
                )
                self._state_dim = len(state)
                info = self._get_current_info()
                info["task_id"] = task_id
                info["job_id"] = job_id
                return state, reward, False, info

            # No queued events — advance to next job arrival
            self._process_arrivals_up_to(self._next_arrival_s)
            self.current_time_s = self._next_arrival_s
            self._next_arrival_s += self._sample_inter_arrival()
            if not self._hnh_queue:
                self.current_time_s = min(self._next_arrival_s,
                                          self.cfg.episode_duration_s)

    def _process_arrivals_up_to(self, time_limit_s: float):
        while self._next_arrival_s <= time_limit_s:
            self._spawn_job(self._next_arrival_s)
            self._next_arrival_s += self._sample_inter_arrival()

    def _spawn_job(self, arrival_s: float):
        job_id = f"J{self._job_counter:06d}"
        self._job_counter += 1

        job, new_states = generate_job(job_id, arrival_s, self.cfg, self.rng)
        self.jobs[job_id] = job
        self.metrics.total_jobs += 1

        for tid, ts in new_states.items():
            self.task_states[tid] = ts
            self.metrics.total_tasks += 1
            if ts.has_intrinsic_delay:
                event_time = max(arrival_s,
                                 ts.scheduled_start_s - ts.intrinsic_delay_s * 0.3)
                self._hnh_queue.append((event_time, tid, job_id))

        # Propagate parent delays to child observed_upstream_delay_s (Fix 9)
        for tid, ts in new_states.items():
            parents = job.get_parents(tid)
            if not parents:
                continue
            max_parent_delay = max(
                (new_states[pid].intrinsic_delay_s
                 for pid in parents
                 if pid in new_states and new_states[pid].has_intrinsic_delay),
                default=0.0,
            )
            if max_parent_delay > 0:
                ts.observed_upstream_delay_s = max_parent_delay

        # Keep queue sorted by event time (insertion sort would be O(n) but
        # this is called infrequently relative to step())
        self._hnh_queue = deque(sorted(self._hnh_queue, key=lambda x: x[0]))

    def _sample_inter_arrival(self) -> float:
        rate = self.cfg.job_arrival_rate_per_s
        if rate <= 0:
            return float("inf")
        return float(self.rng.exponential(1.0 / rate))

    # ------------------------------------------------------------------
    # Internal: machine management
    # ------------------------------------------------------------------

    def _assign_machine(self, task_id: str, ts: TaskState):
        task = ts.task
        req_cpu = task.plan_cpu * self.cfg.machine_cpu_cores
        req_mem = task.plan_mem * self.cfg.machine_mem_gb
        req_gpu = task.plan_gpu * self.cfg.machine_gpu_count

        for mid, machine in self.machines.items():
            if task.is_gpu_task and machine.cap_gpu == 0:
                continue
            if (machine.free_cpu >= req_cpu and machine.free_mem >= req_mem
                    and machine.free_gpu >= req_gpu):
                machine.used_cpu += req_cpu
                machine.used_mem += req_mem
                machine.used_gpu += req_gpu
                ts.machine_id = mid
                ts.status = TaskStatus.RUNNING
                return

        # No machine available — evict
        ts.status = TaskStatus.EVICTED
        ts.restart_count += 1
        self.metrics.evicted_tasks += 1

    def _release_machine(self, ts: TaskState):
        if ts.machine_id and ts.machine_id in self.machines:
            m = self.machines[ts.machine_id]
            task = ts.task
            m.used_cpu = max(0.0, m.used_cpu - task.plan_cpu * self.cfg.machine_cpu_cores)
            m.used_mem = max(0.0, m.used_mem - task.plan_mem * self.cfg.machine_mem_gb)
            m.used_gpu = max(0.0, m.used_gpu - task.plan_gpu * self.cfg.machine_gpu_count)
            ts.machine_id = None

    def _simulate_task_completion(self, task_id: str, ts: TaskState, job: Job):
        if ts.actual_start_s is None:
            return
        completion_time = ts.actual_start_s + ts.task.expected_duration_s + ts.intrinsic_delay_s
        ts.actual_end_s = completion_time
        ts.arrival_delay_s = max(0.0, completion_time
                                 - (ts.actual_start_s + ts.task.expected_duration_s))
        ts.status = TaskStatus.DONE
        self.metrics.completed_tasks += 1

        self._release_machine(ts)
        self._unlock_children(task_id, job)

        self._recent_cl.append(
            1.0 if ts.arrival_delay_s == 0
            else max(0.0, 1.0 - ts.arrival_delay_s / self.cfg.delta_c)
        )
        self._recent_ol.append(
            1.0 if ts.hold_duration_s == 0
            else max(0.0, 1.0 - ts.hold_duration_s / self.cfg.delta_f)
        )
        self._update_cluster_snapshot()

    def _requeue_evicted_task(self, task_id: str, ts: TaskState, job: Job):
        """Exponential backoff retry; mark FAILED after max_restarts. (Fix 2)"""
        max_restarts = getattr(self.cfg, "max_restarts", 3)
        if ts.restart_count >= max_restarts:
            ts.status = TaskStatus.FAILED
            self.metrics.failed_tasks += 1
            self._recent_failed.append(1.0)
            # FIX: unlock children even on permanent failure (Fix 3)
            self._unlock_children(task_id, job, parent_failed=True)
            return

        backoff_unit = getattr(self.cfg, "evict_backoff_unit_s", 30.0)
        # restart_count was already incremented; count 1 = first retry = 0s delay
        restart_delay = backoff_unit * (ts.restart_count - 1)
        requeue_time = self.current_time_s + restart_delay
        ts.scheduled_start_s = requeue_time
        ts.status = TaskStatus.READY

        if ts.has_intrinsic_delay:
            self._hnh_queue.append((requeue_time, task_id, job.job_id))
            self._hnh_queue = deque(sorted(self._hnh_queue, key=lambda x: x[0]))

    def _unlock_children(self, task_id: str, job: Job,
                         parent_failed: bool = False):
        """Unlock child tasks when a parent completes or permanently fails. (Fix 3)

        If the parent failed, children with no other path to completion are
        marked FAILED too (cascade). Children with other live parents remain
        PENDING and may still complete.
        """
        for child_id in job.get_children(task_id):
            cts = self.task_states.get(child_id)
            if cts is None or cts.status not in (TaskStatus.PENDING, TaskStatus.READY):
                continue

            parents = job.get_parents(child_id)
            parent_states = [self.task_states.get(p) for p in parents]

            if parent_failed:
                # If any parent is permanently failed and no remaining live path,
                # cascade failure to this child
                any_failed = any(
                    ps is not None and ps.status == TaskStatus.FAILED
                    for ps in parent_states
                )
                any_live = any(
                    ps is not None and ps.status not in (TaskStatus.DONE, TaskStatus.FAILED)
                    for ps in parent_states
                )
                if any_failed and not any_live:
                    cts.status = TaskStatus.FAILED
                    self.metrics.failed_tasks += 1
                    # Recursively cascade
                    self._unlock_children(child_id, job, parent_failed=True)
                continue

            # Normal completion: check if all parents are done
            all_done = all(
                ps is not None and ps.status == TaskStatus.DONE
                for ps in parent_states
            )
            if all_done:
                cts.status = TaskStatus.READY

    # ------------------------------------------------------------------
    # Internal: cluster snapshot
    # ------------------------------------------------------------------

    def _update_cluster_snapshot(self):
        snap = self._cluster_snapshot
        snap.total_machines = len(self.machines)
        snap.total_cpu_capacity = sum(m.cap_cpu for m in self.machines.values())
        snap.total_gpu_capacity = sum(m.cap_gpu for m in self.machines.values())
        snap.total_mem_capacity = sum(m.cap_mem for m in self.machines.values())

        snap.cpu_util = (sum(m.used_cpu for m in self.machines.values())
                         / max(snap.total_cpu_capacity, 1.0))
        snap.gpu_util = (sum(m.used_gpu for m in self.machines.values())
                         / max(snap.total_gpu_capacity, 1.0))
        snap.num_idle_machines = sum(1 for m in self.machines.values() if m.is_idle)

        snap.machine_load_avg = float(np.clip(
            self.cfg.target_cpu_util + self.rng.normal(0, 0.05), 0.0, 2.0))
        snap.network_receive_util = float(np.clip(
            self.cfg.network_util_mean + self.rng.normal(0, 0.08), 0.0, 1.0))

        snap.num_pending_tasks = sum(
            1 for ts in self.task_states.values()
            if ts.status in (TaskStatus.PENDING, TaskStatus.READY))
        snap.num_running_tasks = sum(
            1 for ts in self.task_states.values()
            if ts.status == TaskStatus.RUNNING)

        if self._recent_cl:
            snap.global_pipeline_utility_g = float(np.mean(list(self._recent_cl)))
        if self._recent_ol:
            snap.global_operator_utility_g = float(np.mean(list(self._recent_ol)))
        if self._recent_failed:
            snap.failed_task_rate_g = float(np.mean(list(self._recent_failed)))
        elif self.metrics.total_tasks > 0:
            snap.failed_task_rate_g = (
                (self.metrics.failed_tasks + self.metrics.evicted_tasks)
                / self.metrics.total_tasks)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _null_state(self) -> np.ndarray:
        return np.zeros(self._state_dim, dtype=np.float32)

    def _get_current_info(self) -> Dict[str, Any]:
        snap = self._cluster_snapshot
        return {
            "time_s":               self.current_time_s,
            "hnh_count":            self._hnh_count,
            "cpu_util":             round(snap.cpu_util, 3),
            "gpu_util":             round(snap.gpu_util, 3),
            "pending_tasks":        snap.num_pending_tasks,
            "running_tasks":        snap.num_running_tasks,
            "global_pipeline_util": round(snap.global_pipeline_utility_g, 3),
            "upstream_delay_s": (
                self.task_states[self._pending[1]].intrinsic_delay_s
                if self._pending and self._pending[1] in self.task_states
                else 0.0
            ),
        }