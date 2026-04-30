"""
simulator.py — Phase 3 DAG Hold-or-Not-Hold Simulator.

Gym-like API (reset / step) for the A2C agent to train against.

Domain: Cloud cluster DAG scheduling.
  - Jobs arrive as Poisson process with diurnal load variation.
  - Each job is a DAG of tasks with resource demands.
  - When an upstream task is delayed, the scheduler must decide
    whether to hold the downstream task's resources (Hold) or
    release them (No-Hold / preempt).
  - The A2C agent makes one HNH decision per triggered event.
  - Reward = local pipeline utility + global cluster efficiency,
    attributed via the Delay Tree.

Key analogies to Malladi et al.:
  Job (DAG)      ↔  Airline tail plan
  Task           ↔  Flight
  Upstream delay ↔  Incoming flight delay
  Hold           ↔  Hold departing flight
  Pipeline stall ↔  Missed passenger connection
  SLO_deadline   ↔  OTP buffer (15-min grace period)
  Cluster        ↔  Airline network
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
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


class DAGSchedulingSimulator:
    """Phase 3 Hold-or-Not-Hold simulator for cloud DAG scheduling.

    Usage:
        cfg = SimConfig()
        sim = DAGSchedulingSimulator(cfg)
        state, info = sim.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, info = sim.step(action)
    """

    def __init__(self, cfg: Optional[SimConfig] = None):
        self.cfg = cfg or SimConfig()
        self.rng = np.random.default_rng(self.cfg.random_seed)

        # Cluster
        self.machines: Dict[str, Machine] = {}

        # Active jobs and task states
        self.jobs: Dict[str, Job] = {}
        self.task_states: Dict[str, TaskState] = {}   # task_id -> state

        # HNH decision queue: (time_s, task_id, job_id)
        self._hnh_queue: deque = deque()

        # Pending HNH (surfaced to the agent via reset/step)
        self._pending: Optional[Tuple[float, str, str]] = None

        # Episode tracking
        self.current_time_s: float = 0.0
        self.metrics = MetricsTracker()
        self._done: bool = True
        self._hnh_count: int = 0

        # Cluster snapshot (updated at each step)
        self._cluster_snapshot = ClusterSnapshot()

        # Rolling window for global utilities (24h)
        self._recent_cl: deque = deque(maxlen=10_000)
        self._recent_ol: deque = deque(maxlen=10_000)
        self._recent_failed: deque = deque(maxlen=10_000)

        # Next job arrival time (Poisson process)
        self._next_arrival_s: float = 0.0
        self._job_counter: int = 0

    # ==================================================================
    # Gym-like API
    # ==================================================================

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the simulator and advance to the first HNH decision.

        Returns (state_vector, info_dict).
        """
        if seed is not None:
            self.cfg.random_seed = seed
        self.rng = np.random.default_rng(self.cfg.random_seed)

        # Reset all state
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

        # Schedule first job arrival
        self._next_arrival_s = self._sample_inter_arrival()
        self._update_cluster_snapshot()

        # Advance to first HNH decision
        return self._advance_to_next_hnh()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply hold action and advance to the next HNH decision.

        Args:
            action_idx: Index into cfg.hold_actions_s.
                        0 = no hold; higher = longer hold.

        Returns:
            (state, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")
        if self._pending is None:
            raise RuntimeError("No pending HNH. Did you call reset()?")

        decision_time, task_id, job_id = self._pending

        job = self.jobs.get(job_id)
        ts = self.task_states.get(task_id)
        if job is None or ts is None:
            # Task/job expired — zero reward, advance
            reward = 0.0
            self._pending = None
            return self._advance_to_next_hnh(reward=reward)

        # Resolve action
        action_idx = int(np.clip(action_idx, 0, len(self.cfg.hold_actions_s) - 1))
        hold_s = float(self.cfg.hold_actions_s[action_idx])

        # Clamp to SLO deadline
        if ts.task.slo_deadline_s > 0:
            hold_s = min(hold_s, ts.task.slo_deadline_s)

        # Apply the hold
        ts.hold_duration_s = hold_s
        ts.hnh_decided = True
        ts.hnh_action_idx = action_idx

        # Advance simulation clock past the hold window
        self.current_time_s = decision_time + hold_s
        ts.actual_start_s = self.current_time_s
        ts.departure_delay_s = max(
            0.0, self.current_time_s - ts.scheduled_start_s
        )
        ts.status = TaskStatus.RUNNING

        # Assign machine resources (or evict if no-hold)
        if hold_s == 0:
            # No-hold: release resources, mark as pending eviction
            # (task re-enters scheduler queue — counted as a stall)
            ts.status = TaskStatus.EVICTED
            ts.restart_count += 1
            self.metrics.no_hold_decisions += 1
            self.metrics.total_pipeline_stalls += 1
        else:
            self.metrics.hold_decisions += 1
            self._assign_machine(task_id, ts)

        # Compute reward using realised values
        reward = compute_reward(
            task_id=task_id,
            job=job,
            task_state=ts,
            task_states={
                tid: self.task_states[tid]
                for tid in job.tasks
                if tid in self.task_states
            },
            cluster=self._cluster_snapshot,
            cfg=self.cfg,
        )

        self.metrics.total_reward += reward
        self.metrics.episode_steps += 1
        self._hnh_count += 1
        self._pending = None

        # Simulate task completion if it is running
        if ts.status == TaskStatus.RUNNING:
            self._simulate_task_completion(task_id, ts, job)

        return self._advance_to_next_hnh(reward=reward)

    def run_episode(
        self,
        policy: str = "no_hold",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run a full episode with a fixed policy (for baseline evaluation).

        policy options:
          "no_hold"   — always action 0 (never hold)
          "heuristic" — hold if upstream delay < 30s and cluster not congested
          "random"    — random action
        """
        state, _, done, info = self.reset(seed=seed)
        done = False
        while not done:
            if policy == "no_hold":
                action = 0
            elif policy == "heuristic":
                # Hold for 30s if there is an upstream delay and cluster not overloaded
                info = self._get_current_info()
                if (info.get("upstream_delay_s", 0) > 0
                        and self._cluster_snapshot.gpu_util < self.cfg.b_thresh):
                    action = 2   # 30s hold
                else:
                    action = 0
            elif policy == "random":
                action = int(self.rng.integers(0, len(self.cfg.hold_actions_s)))
            else:
                action = 0
            state, reward, done, info = self.step(action)

        return self.metrics.summary()

    # ==================================================================
    # Internal: advance simulation to the next HNH event
    # ==================================================================

    def _advance_to_next_hnh(
        self, reward: float = 0.0
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """Drive the simulation forward until an HNH decision is needed.

        Interleaves:
          1. Job arrivals (Poisson process)
          2. Task completions
          3. HNH triggering (upstream delay detected)
        """
        while True:
            # Episode end conditions
            if self.current_time_s >= self.cfg.episode_duration_s:
                self._done = True
                return self._null_state(), reward, True, self.metrics.summary()
            if self._hnh_count >= self.cfg.max_hnh_decisions:
                self._done = True
                return self._null_state(), reward, True, self.metrics.summary()

            # Process the next queued HNH event
            if self._hnh_queue:
                event_time, task_id, job_id = self._hnh_queue.popleft()

                # Advance clock
                self.current_time_s = max(self.current_time_s, event_time)

                # Spawn any jobs that arrived before this event
                self._process_arrivals_up_to(self.current_time_s)

                # Check the task still exists and is in a decidable state
                ts = self.task_states.get(task_id)
                job = self.jobs.get(job_id)
                if ts is None or job is None:
                    continue
                if ts.status not in (TaskStatus.PENDING, TaskStatus.READY):
                    continue

                # Update cluster snapshot
                self._update_cluster_snapshot()

                # Build and return state
                self._pending = (event_time, task_id, job_id)
                state = build_state_vector(
                    task_id=task_id,
                    job=job,
                    task_state=ts,
                    task_states={
                        tid: self.task_states[tid]
                        for tid in job.tasks if tid in self.task_states
                    },
                    cluster=self._cluster_snapshot,
                    cfg=self.cfg,
                    current_time_s=self.current_time_s,
                )
                info = self._get_current_info()
                info["task_id"] = task_id
                info["job_id"] = job_id
                return state, reward, False, info

            # No queued events — advance to next job arrival
            self._process_arrivals_up_to(self._next_arrival_s)
            self.current_time_s = self._next_arrival_s
            self._next_arrival_s += self._sample_inter_arrival()

            # If still no HNH events after processing arrivals, skip forward
            if not self._hnh_queue:
                self.current_time_s = min(
                    self._next_arrival_s,
                    self.cfg.episode_duration_s
                )

    def _process_arrivals_up_to(self, time_limit_s: float):
        """Spawn all jobs that should have arrived by time_limit_s."""
        while self._next_arrival_s <= time_limit_s:
            self._spawn_job(self._next_arrival_s)
            self._next_arrival_s += self._sample_inter_arrival()

    def _spawn_job(self, arrival_s: float):
        """Generate a new job and add it to the simulation."""
        job_id = f"J{self._job_counter:06d}"
        self._job_counter += 1

        job, new_task_states = generate_job(job_id, arrival_s, self.cfg, self.rng)
        self.jobs[job_id] = job
        self.metrics.total_jobs += 1

        for tid, ts in new_task_states.items():
            self.task_states[tid] = ts
            self.metrics.total_tasks += 1

            # Queue an HNH event for each task that has an upstream delay
            if ts.has_intrinsic_delay:
                # Event fires when the upstream task would normally complete
                # (that's when the downstream task notices the delay)
                event_time = ts.scheduled_start_s - ts.intrinsic_delay_s * 0.3
                event_time = max(arrival_s, event_time)
                self._hnh_queue.append((event_time, tid, job_id))

        # Sort queue by event time (insertion is usually near the end)
        self._hnh_queue = deque(sorted(self._hnh_queue, key=lambda x: x[0]))

    def _sample_inter_arrival(self) -> float:
        """Sample time until next job arrival (exponential / Poisson process)."""
        rate = self.cfg.job_arrival_rate_per_s
        return float(self.rng.exponential(1.0 / rate))

    # ==================================================================
    # Internal: machine assignment, task completion
    # ==================================================================

    def _assign_machine(self, task_id: str, ts: TaskState):
        """Try to assign the task to a machine. Evict if no capacity."""
        task = ts.task
        for mid, machine in self.machines.items():
            # Skip GPU tasks on CPU-only machines
            if task.is_gpu_task and machine.cap_gpu == 0:
                continue
            # Check capacity
            req_cpu = task.plan_cpu * self.cfg.machine_cpu_cores
            req_mem = task.plan_mem * self.cfg.machine_mem_gb
            req_gpu = task.plan_gpu * self.cfg.machine_gpu_count
            if (machine.free_cpu >= req_cpu
                    and machine.free_mem >= req_mem
                    and machine.free_gpu >= req_gpu):
                # Assign
                machine.used_cpu += req_cpu
                machine.used_mem += req_mem
                machine.used_gpu += req_gpu
                ts.machine_id = mid
                ts.status = TaskStatus.RUNNING
                return

        # No machine available — evict and count as stall
        ts.status = TaskStatus.EVICTED
        ts.restart_count += 1
        self.metrics.evicted_tasks += 1

    def _simulate_task_completion(self, task_id: str, ts: TaskState, job: Job):
        """Advance the clock and mark the task as complete."""
        if ts.actual_start_s is None:
            return
        duration = ts.task.expected_duration_s + ts.intrinsic_delay_s
        completion_time = ts.actual_start_s + duration

        ts.actual_end_s = completion_time
        expected_end = ts.actual_start_s + ts.task.expected_duration_s
        ts.arrival_delay_s = max(0.0, completion_time - expected_end)
        ts.status = TaskStatus.DONE
        self.metrics.completed_tasks += 1

        # Release machine resources
        if ts.machine_id and ts.machine_id in self.machines:
            machine = self.machines[ts.machine_id]
            task = ts.task
            machine.used_cpu = max(
                0.0, machine.used_cpu - task.plan_cpu * self.cfg.machine_cpu_cores
            )
            machine.used_mem = max(
                0.0, machine.used_mem - task.plan_mem * self.cfg.machine_mem_gb
            )
            machine.used_gpu = max(
                0.0, machine.used_gpu - task.plan_gpu * self.cfg.machine_gpu_count
            )
            ts.machine_id = None

        # Unlock children: mark them READY if all parents done
        for child_id in job.get_children(task_id):
            cts = self.task_states.get(child_id)
            if cts is None or cts.status != TaskStatus.PENDING:
                continue
            all_parents_done = all(
                self.task_states.get(pid) is not None
                and self.task_states[pid].status == TaskStatus.DONE
                for pid in job.get_parents(child_id)
            )
            if all_parents_done:
                cts.status = TaskStatus.READY

        # Track pipeline utility for global rolling window
        self._recent_cl.append(1.0 if ts.arrival_delay_s == 0 else
                                max(0.0, 1.0 - ts.arrival_delay_s / self.cfg.delta_c))
        self._recent_ol.append(1.0 if ts.hold_duration_s == 0 else
                                max(0.0, 1.0 - ts.hold_duration_s / self.cfg.delta_f))

    # ==================================================================
    # Internal: cluster snapshot update
    # ==================================================================

    def _update_cluster_snapshot(self):
        """Recompute the global cluster snapshot from machine states."""
        snap = self._cluster_snapshot

        snap.total_machines = len(self.machines)
        snap.total_cpu_capacity = sum(m.cap_cpu for m in self.machines.values())
        snap.total_gpu_capacity = sum(m.cap_gpu for m in self.machines.values())
        snap.total_mem_capacity = sum(m.cap_mem for m in self.machines.values())

        used_cpu = sum(m.used_cpu for m in self.machines.values())
        used_gpu = sum(m.used_gpu for m in self.machines.values())
        snap.cpu_util = used_cpu / max(snap.total_cpu_capacity, 1.0)
        snap.gpu_util = used_gpu / max(snap.total_gpu_capacity, 1.0)

        snap.num_idle_machines = sum(
            1 for m in self.machines.values() if m.is_idle
        )

        # Add synthetic load variance (Gaussian noise around target)
        load_noise = float(self.rng.normal(0, 0.05))
        snap.machine_load_avg = float(np.clip(
            self.cfg.target_cpu_util + load_noise, 0.0, 2.0
        ))
        net_noise = float(self.rng.normal(0, 0.08))
        snap.network_receive_util = float(np.clip(
            self.cfg.network_util_mean + net_noise, 0.0, 1.0
        ))

        # Task queue depths
        snap.num_pending_tasks = sum(
            1 for ts in self.task_states.values()
            if ts.status in (TaskStatus.PENDING, TaskStatus.READY)
        )
        snap.num_running_tasks = sum(
            1 for ts in self.task_states.values()
            if ts.status == TaskStatus.RUNNING
        )

        # Rolling 24h window metrics
        if self._recent_cl:
            snap.global_pipeline_utility_g = float(np.mean(list(self._recent_cl)))
        if self._recent_ol:
            snap.global_operator_utility_g = float(np.mean(list(self._recent_ol)))

        # Failed task rate
        if self._recent_failed:
            snap.failed_task_rate_g = float(np.mean(list(self._recent_failed)))
        elif self.metrics.total_tasks > 0:
            snap.failed_task_rate_g = (
                (self.metrics.failed_tasks + self.metrics.evicted_tasks)
                / self.metrics.total_tasks
            )

    # ==================================================================
    # Internal utilities
    # ==================================================================

    def _null_state(self) -> np.ndarray:
        return np.zeros(88, dtype=np.float32)

    def _get_current_info(self) -> Dict[str, Any]:
        snap = self._cluster_snapshot
        return {
            "time_s":              self.current_time_s,
            "hnh_count":           self._hnh_count,
            "cpu_util":            round(snap.cpu_util, 3),
            "gpu_util":            round(snap.gpu_util, 3),
            "pending_tasks":       snap.num_pending_tasks,
            "running_tasks":       snap.num_running_tasks,
            "global_pipeline_util": round(snap.global_pipeline_utility_g, 3),
            "upstream_delay_s":    (
                self.task_states[self._pending[1]].intrinsic_delay_s
                if self._pending and self._pending[1] in self.task_states
                else 0.0
            ),
        }
