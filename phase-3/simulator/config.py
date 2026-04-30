"""
config.py — SimConfig for the Phase 3 DAG Hold-or-Not-Hold simulator.

Grounded in Google Borg 2019 + Alibaba PAI 2020 cluster characteristics.
All numeric defaults are calibrated to realistic cluster behaviour
so the A2C agent faces a non-trivial learning problem.

Tunable knobs are grouped by concern so it is easy to sweep
the reward hyperparameters (α, β, λ, B_thresh) independently
of the topology and arrival-rate parameters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SimConfig:
    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    random_seed: int = 42

    # ------------------------------------------------------------------
    # Episode length
    # ------------------------------------------------------------------
    # One episode = one "week" of cluster time, matching the paper.
    # Simulated seconds. 7 days × 86400 s/day.
    episode_duration_s: float = 7 * 86_400.0

    # ------------------------------------------------------------------
    # Cluster topology
    # ------------------------------------------------------------------
    # Number of machines in the synthetic cluster.
    # Calibrated from Borg trace: ~12,500 machines in a cell.
    # We use a scaled-down version for simulator tractability.
    num_machines: int = 500

    # Per-machine resource capacity (matching Borg/Alibaba medians)
    machine_cpu_cores: float = 32.0        # cores per machine
    machine_mem_gb: float   = 128.0        # GB per machine
    machine_gpu_count: float = 4.0         # GPUs per machine (0 for CPU-only)

    # Fraction of machines that have GPUs (rest are CPU-only).
    # Alibaba GPU cluster: ~30% of machines have GPUs.
    gpu_machine_fraction: float = 0.30

    # ------------------------------------------------------------------
    # Job / DAG arrival process
    # ------------------------------------------------------------------
    # Mean inter-arrival time between jobs (seconds).
    # Borg: ~1 job per 2-3 seconds at peak; we use a moderate rate.
    job_arrival_rate_per_s: float = 0.5   # jobs/second

    # Job size distribution: (num_tasks, probability) pairs.
    # Calibrated from Alibaba pai_task_table: most jobs are small.
    job_size_distribution: List[Tuple[int, float]] = field(default_factory=lambda: [
        (1,  0.25),   # single-task jobs (no DAG)
        (2,  0.20),
        (3,  0.15),
        (5,  0.15),
        (8,  0.10),
        (12, 0.08),
        (20, 0.05),
        (50, 0.02),
    ])

    # DAG topology style: (style, probability).
    # "chain"   — linear A→B→C→…
    # "funnel"  — many tasks merge into one
    # "fan_out" — one task feeds many parallel tasks
    # "diamond" — merge then split then merge again
    # "random"  — Erdős–Rényi random DAG (realistic)
    dag_topology_distribution: List[Tuple[str, float]] = field(default_factory=lambda: [
        ("chain",    0.30),
        ("fan_out",  0.20),
        ("funnel",   0.15),
        ("diamond",  0.15),
        ("random",   0.20),
    ])

    # ------------------------------------------------------------------
    # Task duration and delay model
    # ------------------------------------------------------------------
    # Task duration in seconds: log-normal (shape, scale).
    # Alibaba: median ~300s, long tail to ~3600s.
    task_duration_lognormal_mu: float    = 5.7    # ln(300) ≈ 5.7
    task_duration_lognormal_sigma: float = 1.2

    # Intrinsic delay probability — fraction of tasks that experience
    # a hardware/software slowdown that triggers an HNH decision.
    # This is the core "event" that drives the problem.
    task_delay_probability: float = 0.12   # ~12% of tasks get delayed

    # Delay magnitude when it occurs: (mean_s, stddev_s).
    task_delay_mean_s:   float = 120.0    # 2 min average delay
    task_delay_stddev_s: float = 90.0     # high variance

    # ------------------------------------------------------------------
    # Scheduling classes (Borg 4-class model)
    # ------------------------------------------------------------------
    # (class_id, fraction_of_jobs, T_sla_s, eviction_priority_name)
    # class 3 = production (latency-sensitive, T_sla=0s)
    # class 2 = mid-tier (T_sla=30s)
    # class 1 = batch (T_sla=120s)
    # class 0 = best-effort (T_sla=300s)
    scheduling_class_distribution: List[Tuple[int, float, float, str]] = field(
        default_factory=lambda: [
            (3, 0.15, 0.0,   "production"),
            (2, 0.25, 30.0,  "mid_tier"),
            (1, 0.40, 120.0, "batch"),
            (0, 0.20, 300.0, "best_effort"),
        ]
    )

    # ------------------------------------------------------------------
    # Workload types (Alibaba 6-class model)
    # ------------------------------------------------------------------
    workload_type_distribution: List[Tuple[str, float]] = field(default_factory=lambda: [
        ("training",   0.35),
        ("inference",  0.20),
        ("etl",        0.15),
        ("pipeline",   0.12),
        ("serving",    0.10),
        ("other",      0.08),
    ])

    # ------------------------------------------------------------------
    # GPU type distribution (for gpu_type_spec one-hot[6])
    # ------------------------------------------------------------------
    gpu_type_distribution: List[Tuple[str, float]] = field(default_factory=lambda: [
        ("V100", 0.30),
        ("A100", 0.25),
        ("T4",   0.20),
        ("P100", 0.15),
        ("A10",  0.07),
        ("other", 0.03),
    ])

    # ------------------------------------------------------------------
    # Resource demand distributions (per task, as fraction of machine)
    # ------------------------------------------------------------------
    # CPU: fraction of one machine's cores requested.
    # Alibaba: median ~0.1 (small tasks), tail to 0.8 (large ML jobs).
    task_cpu_demand_mean: float   = 0.12
    task_cpu_demand_stddev: float = 0.15

    # Memory: fraction of one machine's memory.
    task_mem_demand_mean: float   = 0.10
    task_mem_demand_stddev: float = 0.12

    # GPU: fraction of one GPU requested (0 for non-GPU tasks).
    # 70% of tasks on the GPU cluster request at least some GPU.
    task_gpu_demand_mean: float   = 0.50
    task_gpu_demand_stddev: float = 0.30
    task_gpu_request_probability: float = 0.65

    # ------------------------------------------------------------------
    # Actual utilisation vs requested (the "plan vs actual" split)
    # ------------------------------------------------------------------
    # When a task is running, actual usage = plan × utilisation_factor.
    # Alibaba shows typical utilisation at 40-60% of requested.
    cpu_utilisation_factor_mean: float   = 0.50
    cpu_utilisation_factor_stddev: float = 0.20
    gpu_utilisation_factor_mean: float   = 0.55
    gpu_utilisation_factor_stddev: float = 0.25
    mem_utilisation_factor_mean: float   = 0.65  # memory is stickier
    mem_utilisation_factor_stddev: float = 0.15

    # ------------------------------------------------------------------
    # Global cluster metrics (rolling baseline)
    # ------------------------------------------------------------------
    # Rolling window for global PU / OPU / failed_task_rate_G.
    # Matches the paper: W = 24 hours.
    global_window_s: float = 24 * 3600.0

    # Target cluster utilisation at steady state (used for calibration).
    # Borg: ~60% average CPU, ~40% GPU.
    target_cpu_util: float = 0.60
    target_gpu_util: float = 0.40

    # Network receive utilisation (fraction of link capacity).
    # ML clusters: high network load during gradient sync.
    network_util_mean: float   = 0.45
    network_util_stddev: float = 0.15

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------
    # Discrete hold durations in seconds. Maps directly to A2C output.
    # {0, 15, 30, 60, 120} from the spec table.
    hold_actions_s: List[int] = field(default_factory=lambda: [0, 15, 30, 60, 120])

    # For DDPG (continuous): max hold = 120s.
    hold_max_s: float = 120.0

    # ------------------------------------------------------------------
    # Reward hyperparameters (tunable, match paper defaults)
    # ------------------------------------------------------------------
    # α: pipeline utility vs cluster efficiency weight.
    # α=1 → maximise downstream completion; α=0 → maximise cluster OTP.
    alpha: float = 0.75

    # β: local vs global reward weight.
    # β=1 → only care about this task; β=0 → only care about global.
    beta: float = 0.75

    # λ: GPU congestion sensitivity in OL(τ).
    lam: float = 0.30

    # B_thresh: GPU utilisation above which congestion penalty fires.
    b_thresh: float = 0.85

    # ∆C: normalising constant for task disutility (max tolerable delay, s).
    delta_c: float = 600.0

    # ∆F: normalising constant for operator utility (max flight delay).
    delta_f: float = 300.0

    # ------------------------------------------------------------------
    # State vector embedding sizes
    # ------------------------------------------------------------------
    # job_id embedding dimension. We fix at 8 dims for the simulator
    # (rather than the 8-16 range in the doc) to remove ambiguity.
    job_id_embedding_dim: int = 8

    # ------------------------------------------------------------------
    # Simulator clock
    # ------------------------------------------------------------------
    # Simulation tick resolution in seconds.
    # HNH decisions happen at integer seconds.
    tick_s: float = 1.0

    # Maximum number of HNH decisions per episode before truncation.
    # Prevents infinite loops in degenerate cases.
    max_hnh_decisions: int = 50_000

    # ------------------------------------------------------------------
    # Instance count (inst_num from Alibaba)
    # ------------------------------------------------------------------
    inst_num_distribution: List[Tuple[int, float]] = field(default_factory=lambda: [
        (1,  0.50),
        (2,  0.20),
        (4,  0.15),
        (8,  0.10),
        (16, 0.05),
    ])
