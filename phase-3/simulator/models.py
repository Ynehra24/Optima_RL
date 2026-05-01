"""
models.py — Core dataclasses for the DAG HNH simulator.

Domain mapping (analogies to Malladi et al.):
  Task        ↔  Flight
  Job (DAG)   ↔  Tail plan (sequence of flights)
  Upstream    ↔  Incoming delayed flight
  Downstream  ↔  Departing flight (hold decision point)
  Hold        ↔  Hold the departing flight at the gate
  Preempt     ↔  Depart on time (miss the connection)
  DAG edge    ↔  Passenger itinerary (connecting dependency)
  SLO_deadline↔  OTP buffer (15-min buffer in aviation)
  Cluster     ↔  Airline network
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


# ===========================================================================
# Enums
# ===========================================================================

class TaskStatus(Enum):
    PENDING  = auto()   # waiting on upstream dependency
    READY    = auto()   # all parents done, resource not yet assigned
    RUNNING  = auto()   # executing on a machine
    DONE     = auto()   # completed successfully
    EVICTED  = auto()   # preempted — resources taken away
    FAILED   = auto()   # failed permanently

class SchedulingClass(Enum):
    BEST_EFFORT = 0   # T_sla = 300s
    BATCH       = 1   # T_sla = 120s
    MID_TIER    = 2   # T_sla = 30s
    PRODUCTION  = 3   # T_sla = 0s

class WorkloadType(Enum):
    TRAINING   = "training"
    INFERENCE  = "inference"
    ETL        = "etl"
    PIPELINE   = "pipeline"
    SERVING    = "serving"
    OTHER      = "other"

class GpuType(Enum):
    V100  = "V100"
    A100  = "A100"
    T4    = "T4"
    P100  = "P100"
    A10   = "A10"
    OTHER = "other"
    NONE  = "none"   # CPU-only task

class HNHDecision(Enum):
    HOLD   = "hold"
    NO_HOLD = "no_hold"


# ===========================================================================
# Machine — a single physical node in the cluster
# ===========================================================================

@dataclass
class Machine:
    machine_id: str
    cap_cpu: float          # cores
    cap_mem: float          # GB
    cap_gpu: float          # number of GPUs (0 for CPU-only)
    gpu_type: GpuType = GpuType.NONE

    # Dynamic utilisation (updated each tick)
    used_cpu: float = 0.0
    used_mem: float = 0.0
    used_gpu: float = 0.0
    machine_load_1: float = 0.0   # 1-min load average
    net_receive_util: float = 0.0  # fraction of link capacity

    @property
    def free_cpu(self) -> float:
        return max(0.0, self.cap_cpu - self.used_cpu)

    @property
    def free_mem(self) -> float:
        return max(0.0, self.cap_mem - self.used_mem)

    @property
    def free_gpu(self) -> float:
        return max(0.0, self.cap_gpu - self.used_gpu)

    @property
    def cpu_util(self) -> float:
        return self.used_cpu / self.cap_cpu if self.cap_cpu > 0 else 0.0

    @property
    def gpu_util(self) -> float:
        return self.used_gpu / self.cap_gpu if self.cap_gpu > 0 else 0.0

    @property
    def mem_util(self) -> float:
        return self.used_mem / self.cap_mem if self.cap_mem > 0 else 0.0

    @property
    def is_idle(self) -> bool:
        return self.used_cpu == 0.0 and self.used_gpu == 0.0


# ===========================================================================
# Task — one node in a DAG
# ===========================================================================

@dataclass
class Task:
    """Static description of a task. Immutable once generated."""

    task_id: str
    job_id: str
    task_index: int            # position within the job DAG (0-based)

    # Scheduling attributes
    scheduling_class: SchedulingClass
    workload_type: WorkloadType
    gpu_type: GpuType          # NONE for CPU-only
    priority: int              # raw [0-11] Borg priority

    # Resource demands (as fraction of one machine's capacity)
    plan_cpu: float            # fraction of machine_cpu_cores
    plan_mem: float            # fraction of machine_mem_gb
    plan_gpu: float            # fraction of one GPU (0 for CPU-only)

    # Actual utilisation (sampled at task creation; stable during run)
    cpu_usage: float           # fraction of plan_cpu
    gpu_wrk_util: float        # fraction of plan_gpu
    avg_mem_usage: float       # fraction of plan_mem
    max_mem_usage: float       # fraction of plan_mem (peak)

    # Duration model
    expected_duration_s: float
    inst_num: int = 1          # number of worker instances

    # SLO from scheduling class
    slo_deadline_s: float = 120.0

    # GPU type one-hot index (for state vector encoding)
    gpu_type_idx: int = 0

    # Workload type one-hot index
    workload_type_idx: int = 0

    @property
    def is_gpu_task(self) -> bool:
        return self.plan_gpu > 0.0

    @property
    def resource_cost_score(self) -> float:
        """Composite resource opportunity cost (λ weights from SimConfig)."""
        # λ_gpu=0.6, λ_cpu=0.25, λ_mem=0.15 — GPU-heavy ML cluster defaults
        return 0.6 * self.plan_gpu + 0.25 * self.plan_cpu + 0.15 * self.plan_mem


@dataclass
class TaskState:
    """Dynamic runtime state of a task. Mutated during the episode."""

    task: Task

    status: TaskStatus = TaskStatus.PENDING
    machine_id: Optional[str] = None

    # Timestamps (in simulated seconds from episode start)
    submit_time: float = 0.0
    scheduled_start_s: float = 0.0   # planned start (from DAG scheduling)
    actual_start_s: Optional[float] = None
    actual_end_s: Optional[float] = None

    # Delay tree variables (§6)
    departure_delay_s: float = 0.0   # D_k: actual_start - scheduled_start
    arrival_delay_s: float   = 0.0   # A_k: actual_end - expected_end
    hold_duration_s: float   = 0.0   # H_k: hold applied by agent
    ground_delay_s: float    = 0.0   # GD_k: queue wait excluding hold

    # Intrinsic slowdown (the event that triggers HNH)
    has_intrinsic_delay: bool = False
    intrinsic_delay_s: float  = 0.0

    # FIX 3: field that FIX 9 in simulator._spawn_job writes to.
    # Stores the maximum intrinsic delay of any parent task, propagated at
    # spawn time so the upstream_delay feature in the state vector can
    # reflect the cascade even before the parent has actually started.
    # state_builder._compute_upstream_delay uses this as a fallback when
    # no runtime timing information is yet available for the parent tasks.
    observed_upstream_delay_s: float = 0.0

    # HNH decision tracking
    hnh_decided: bool = False
    hnh_action_idx: int = 0          # index into hold_actions_s

    # Influence index (set post-episode by reward engine)
    rho_h_a: float = 0.0

    # Restart count (for reward penalty)
    restart_count: int = 0

    @property
    def expected_end_s(self) -> Optional[float]:
        if self.actual_start_s is None:
            return None
        return self.actual_start_s + self.task.expected_duration_s

    @property
    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.DONE, TaskStatus.FAILED)

    @property
    def completion_delay_s(self) -> float:
        """Total delay to final destination (analog of δ_i in the paper)."""
        if self.actual_end_s is None:
            return 0.0
        expected = self.scheduled_start_s + self.task.expected_duration_s
        return max(0.0, self.actual_end_s - expected)


# ===========================================================================
# Job — a DAG of tasks
# ===========================================================================

@dataclass
class Job:
    """Static structure of a job (DAG of tasks)."""

    job_id: str
    tasks: Dict[str, Task]              # task_id -> Task
    edges: List[Tuple[str, str]]        # (parent_task_id, child_task_id)

    # Scheduling attributes inherited from the job
    scheduling_class: SchedulingClass
    workload_type: WorkloadType
    arrival_time_s: float
    job_deadline_s: float               # hard SLO for the whole job

    # Cached graph properties (computed once after creation)
    _parents: Dict[str, Set[str]] = field(default_factory=dict)
    _children: Dict[str, Set[str]] = field(default_factory=dict)
    _depth: Dict[str, int] = field(default_factory=dict)
    _total_descendants: Dict[str, int] = field(default_factory=dict)
    _critical_path_len: Dict[str, float] = field(default_factory=dict)
    _slack_time: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self._build_graph_cache()

    @property
    def job_size(self) -> int:
        return len(self.tasks)

    def _build_graph_cache(self):
        """Pre-compute parents, children, depths, descendants, critical path."""
        task_ids = list(self.tasks.keys())
        self._parents = {tid: set() for tid in task_ids}
        self._children = {tid: set() for tid in task_ids}

        for parent_id, child_id in self.edges:
            if parent_id in self._children:
                self._children[parent_id].add(child_id)
            if child_id in self._parents:
                self._parents[child_id].add(parent_id)

        # Topological sort (Kahn's algorithm)
        in_degree = {tid: len(self._parents[tid]) for tid in task_ids}
        queue = [tid for tid in task_ids if in_degree[tid] == 0]
        topo_order = []
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            for child in self._children.get(node, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Depth (topological level from source nodes)
        depth = {tid: 0 for tid in task_ids}
        for tid in topo_order:
            for child in self._children.get(tid, set()):
                depth[child] = max(depth[child], depth[tid] + 1)
        self._depth = depth

        # Total descendants (count via reverse topological order)
        desc = {tid: 0 for tid in task_ids}
        for tid in reversed(topo_order):
            for child in self._children.get(tid, set()):
                desc[tid] += 1 + desc[child]
        self._total_descendants = desc

        # Critical path length (forward pass on expected durations)
        earliest_start = {tid: 0.0 for tid in task_ids}
        for tid in topo_order:
            dur = self.tasks[tid].expected_duration_s
            for child in self._children.get(tid, set()):
                earliest_start[child] = max(
                    earliest_start[child],
                    earliest_start[tid] + dur
                )
        crit = {}
        for tid in task_ids:
            crit[tid] = self._longest_path_from(tid)
        self._critical_path_len = crit

        # Slack time (latest_start - earliest_start)
        latest_start = {tid: self.job_deadline_s for tid in task_ids}
        for tid in reversed(topo_order):
            dur = self.tasks[tid].expected_duration_s
            for child in self._children.get(tid, set()):
                latest_start[tid] = min(
                    latest_start[tid],
                    latest_start[child] - dur
                )
        slack = {}
        for tid in task_ids:
            slack[tid] = max(0.0, latest_start[tid] - earliest_start[tid])
        self._slack_time = slack

    def _longest_path_from(self, start_id: str) -> float:
        """DFS to find the longest path (in seconds) from start_id to any sink."""
        visited: Dict[str, float] = {}

        def dfs(tid: str) -> float:
            if tid in visited:
                return visited[tid]
            dur = self.tasks[tid].expected_duration_s
            children = self._children.get(tid, set())
            if not children:
                visited[tid] = dur
                return dur
            result = dur + max(dfs(c) for c in children)
            visited[tid] = result
            return result

        return dfs(start_id)

    def get_parents(self, task_id: str) -> Set[str]:
        return self._parents.get(task_id, set())

    def get_children(self, task_id: str) -> Set[str]:
        return self._children.get(task_id, set())

    def get_depth(self, task_id: str) -> int:
        return self._depth.get(task_id, 0)

    def get_total_descendants(self, task_id: str) -> int:
        return self._total_descendants.get(task_id, 0)

    def get_critical_path_len(self, task_id: str) -> float:
        return self._critical_path_len.get(task_id, 0.0)

    def get_slack_time(self, task_id: str) -> float:
        return self._slack_time.get(task_id, 0.0)

    def is_on_critical_path(self, task_id: str) -> bool:
        return self.get_slack_time(task_id) == 0.0

    def max_depth(self) -> int:
        return max(self._depth.values()) if self._depth else 0

    def completed_tasks(self, task_states: Dict[str, TaskState]) -> int:
        return sum(
            1 for tid in self.tasks
            if task_states.get(tid) and task_states[tid].status == TaskStatus.DONE
        )


# ===========================================================================
# Cluster snapshot — global state at a given tick
# ===========================================================================

@dataclass
class ClusterSnapshot:
    """Global cluster metrics. Updated every tick via rolling window."""

    # Resource capacity totals
    total_cpu_capacity: float = 0.0
    total_gpu_capacity: float = 0.0
    total_mem_capacity: float = 0.0
    total_machines: int = 0

    # Current utilisation (rolling)
    cpu_util: float = 0.0
    gpu_util: float = 0.0
    machine_load_avg: float = 0.0
    network_receive_util: float = 0.0
    num_idle_machines: int = 0

    # Task queue state
    num_pending_tasks: int = 0
    num_running_tasks: int = 0

    # Rolling window metrics (24h)
    failed_task_rate_g: float = 0.0
    global_pipeline_utility_g: float = 1.0
    global_operator_utility_g: float = 1.0


# ===========================================================================
# Episode metrics tracker
# ===========================================================================

@dataclass
class MetricsTracker:
    """Accumulates episode-level statistics for evaluation."""

    total_jobs: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    evicted_tasks: int = 0
    started_tasks: int = 0

    # HNH decisions
    hold_decisions: int = 0
    no_hold_decisions: int = 0
    holds_that_saved_pipeline: int = 0   # hold → pipeline completed
    holds_that_wasted_resources: int = 0  # hold → pipeline still failed

    # Delay tracking
    total_departure_delay_s: float = 0.0
    total_arrival_delay_s: float   = 0.0
    total_pipeline_stalls: int = 0        # "missed connections"
    pipelines_saved_by_hold: int = 0

    # Reward tracking
    total_reward: float = 0.0
    episode_steps: int = 0

    def reset(self):
        for f in self.__dataclass_fields__:
            setattr(self, f, type(getattr(self, f))())

    def summary(self) -> dict:
        total_hnh = max(self.hold_decisions + self.no_hold_decisions, 1)
        total_t = max(self.total_tasks, 1)
        return {
            "total_jobs":          self.total_jobs,
            "total_tasks":         self.total_tasks,
            "completed_pct":       round(100 * self.completed_tasks / total_t, 2),
            "failed_pct":          round(100 * self.failed_tasks / total_t, 2),
            "evicted_pct":         round(100 * self.evicted_tasks / total_t, 2),
            "hold_rate_pct":       round(100 * self.hold_decisions / total_hnh, 2),
            "pipelines_saved":     self.pipelines_saved_by_hold,
            "pipeline_stalls":     self.total_pipeline_stalls,
            "avg_departure_delay_s": round(
                self.total_departure_delay_s / max(self.started_tasks, 1), 2),
            "avg_arrival_delay_s": round(
                self.total_arrival_delay_s / max(self.completed_tasks, 1), 2),
            "total_reward":        round(self.total_reward, 4),
            "episode_steps":       self.episode_steps,
        }
