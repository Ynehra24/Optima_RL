# Optima_RL: Hold-or-Not-Hold RL

This project introduces reinforcement-learning simulators modeled around the **Hold-or-Not-Hold (HNH)** decision problem across three different operational domains. In complex interconnected networks—such as passenger aviation, multi-hub logistics, or large-scale cloud task dependency graphs—delays can propagate and cascade, leading to severe disruptions.

Our project explores three separate phases, progressively adapting the HNH problem to different contexts:

1. **Phase 1: Aviation.** Should an outbound connecting flight be held for delayed incoming transfer passengers, or should the aircraft leave on time to avoid downstream scheduling conflicts?
2. **Phase 2: Freight & Logistics.** Should a freight truck wait at a cross-docking hub for delayed cargo from an incoming truck, or depart to maintain strict delivery schedules?
3. **Phase 3: Cloud DAG Scheduling.** Should a complex DAG task execution be delayed for struggling prerequisites, or should the scheduler prioritize other sub-graphs to prevent total pipeline stall?

By formalizing this trade-off using a Reinforcement Learning architecture, we evaluate and implement various custom learning agents (A2C, DQN, AC, DDPG) that learn to actively minimize total delay propagation.

The project is inspired by the AAMAS 2021 paper *"To hold or not to hold? - Reducing Passenger Missed Connections in Airlines using Reinforcement Learning"* and extends the same delay-tree reasoning to logistics and cloud scheduling.

---

## Repository Layout

```text
.
├── phase-1/
│   ├── simulator/              # Airline network simulator and validation demo
│   ├── rewardEngineering/      # Airline delay-tree reward attribution
│   └── algoImplementation/     # A2C, DQN, AC, DDPG training code
├── phase-2/
│   ├── simulator/              # Cross-dock logistics environment
│   ├── rewardEngineering/      # Logistics delay-tree reward attribution
│   └── algoImplementation/     # A2C, DQN, AC training code
├── phase-3/
│   ├── simulator/              # DAG scheduling simulator
│   ├── rewardEngineering/      # DAG delay-tree attribution helpers
│   ├── preprocessing/          # Borg/Alibaba calibration scripts
│   └── algoImplementation/     # A2C, DQN, AC, DDPG training code
└── README.md
```

Each phase is self-contained and has its own `simulator`, `rewardEngineering`, and `algoImplementation` package layout. Because the phases reuse package names such as `simulator`, run phase-specific module commands from the phase directory when noted below.

---

## Setup

You can install all necessary dependencies using the provided `requirements.txt` file.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Core simulator/training code primarily uses custom numpy-based neural networks, `numpy`, and `matplotlib`. Phase 2 imports `gymnasium` with a fallback to `gym`. `pandas` is used by calibration and preprocessing scripts, and `scapy` is used by `phase-3/pcaphelper.py`.

On machines where Matplotlib cannot write to the default user cache, set a local cache directory before training:

```bash
mkdir -p .cache/matplotlib
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```

---

## Phase 1: Airline HNH

Phase 1 simulates an airline network with passenger connections. The simulator exposes a Gym-like `reset()` / `step()` interface and supports baseline policies such as no-hold and fixed-hold heuristics.

Key files:

- `phase-1/simulator/simulator.py` — airline network simulator
- `phase-1/simulator/run_demo.py` — simulator demo and validation run
- `phase-1/rewardEngineering/delay_tree.py` — delay-tree attribution
- `phase-1/algoImplementation/train.py` — RL training and evaluation

## State Space Mapping — Aviation Hold-or-Not-Hold (Baseline)

### State Definition
At each decision epoch `t`, the RL agent observes:

s_t = {\
    PL(τ),        # Local Passenger Utility vector\
    AL(τ),        # Local Airline Utility vector\
    PG,           # Global Passenger Utility (24h window)\
    AG,           # Global Airline Utility (24h window)\
    τ*,           # Locally optimal hold time\
}\

### Local State (Flight-Level)
- PL(τ): Expected passenger utility for each hold duration τ
- AL(τ): Expected airline utility (delay cost) for each τ
- τ*: argmax over α·PL(τ) + (1−α)·AL(τ)

### Global State (Network-Level)
- PG: Average passenger utility across network (past 24h)
- AG: Average airline utility across network (past 24h)

### Action
- τ ∈ {0, 5, 10, 15, 20, 25, 30} minutes (or continuous [0,30])

### Reward Structure
- R_T = β·R_L + (1−β)·R_G
- R_L = α·P_L + (1−α)·A_L
- R_G = α·P_G + (1−α)·A_G

### Delay Tree Variables (for attribution)
- D_i: Departure delay
- A_i: Arrival delay
- H_i: Hold duration
- GD_i: Ground delay
- GA_i: Arrival congestion delay
- T_i: Air-time delay

### Key Insight
State combines:
- Forward-looking local forecasts (PL, AL)
- Backward-looking global health (PG, AG)
- Precomputed optimal action prior (τ*)

Run the simulator demo:

```bash
cd phase-1
python3 -m simulator.run_demo
cd ..
```

Run the reward-engineering test:

```bash
cd phase-1
python3 -m rewardEngineering.test_tree
cd ..
```

Train and evaluate RL agents with the standard script defaults:

```bash
python3 phase-1/algoImplementation/train.py
```

Train one algorithm:

```bash
python3 phase-1/algoImplementation/train.py --algo a2c
```

Useful options:

```bash
python3 phase-1/algoImplementation/train.py --help
```

Results are written under `results/` relative to the directory where the command is run.

---

## Phase 2: Logistics Cross-Docking HNH

Phase 2 adapts HNH to truck departures, cargo transfers, bay utilization, and missed freight connections. The environment follows the Gymnasium-style API and includes both single-hub and multi-hub modes.

Key files:

- `phase-2/simulator/logistics_env.py` — single-hub logistics environment
- `phase-2/simulator/multi_hub_env.py` — multi-hub logistics environment
- `phase-2/simulator/validate_simulator.py` — simulator validation suite
- `phase-2/rewardEngineering/delay_tree.py` — logistics delay tree with bay-blockage attribution
- `phase-2/algoImplementation/train.py` — RL training and evaluation

## State Space Mapping — Logistics Cross-Docking (Phase 2)

### State Definition
At each decision epoch `t`, for outbound truck k:

s_t = {\
    CL(τ), OL(τ), τ*,        # Local forecast + optimal action\
    Vk, Qk, Xk, Ek,          # Cargo characteristics\
    Δin, Δslack,             # Timing dynamics\
    Lk, Fk, Nin,             # Operational constraints\
    CG, OG,                  # Global utility (24h)\
    BG, WG, YG, ZG,          # Hub congestion metrics\
    Dk, Ak,                  # Delay variables\
    Gk_bay, Gk_road          # Delay decomposition\
}\

---

### Local State (Truck-Level)
- CL(τ): Cargo utility (value-weighted SLA success)
- OL(τ): Operator utility (delay + congestion penalty)
- τ*: argmax over α·CL(τ) + (1−α)·OL(τ)

Cargo Features:
- Vk: Cargo value score
- Qk: Volume fraction
- Xk: SLA urgency level
- Ek: Perishability fraction

Timing Features:
- Δin: Inbound delay (ETA lag)
- Δslack: Transfer slack

Constraints:
- Lk: Driver hours remaining (hard cap)
- Fk: Downstream deadline pressure
- Nin: Number of inbound trucks

---

### Global State (Hub-Level)
- CG: Global cargo utility (24h)
- OG: Global operator utility (24h)
- BG: Bay utilisation (congestion)
- WG: Throughput rate
- YG: Failed transfer rate
- ZG: Inbound queue depth

---

### Delay Tree Variables
- Dk: Departure delay
- Ak: Arrival delay
- Gk_bay: Dock congestion delay
- Gk_road: Road delay
- Hk: Hold duration

---

### Action
- τ ∈ {0, 5, 10, ..., 30} or continuous

---

### Reward Structure
- R_T_k = β·R_L_k + (1−β)·R_G_k
- R_L_k = α·CL_k + (1−α)·OL_k
- R_G_k = α·CG_k + (1−α)·OG_k

---

### Key Insight
State expands baseline by adding:
- Cargo semantics (value, perishability, SLA)
- Physical constraints (driver hours, bays)
- System congestion signals (BG, ZG)

Run the simulator validation suite:

```bash
cd phase-2
python3 -m simulator.validate_simulator
cd ..
```

Run the reward-engineering tests:

```bash
cd phase-2
python3 -m rewardEngineering.test_tree
cd ..
```

Train and evaluate RL agents with the standard script defaults:

```bash
python3 phase-2/algoImplementation/train.py
```

Train one algorithm:

```bash
python3 phase-2/algoImplementation/train.py --algo a2c
```

Run the multi-hub training mode:

```bash
python3 phase-2/algoImplementation/train.py --multi-hub
```

Useful options:

```bash
python3 phase-2/algoImplementation/train.py --help
```

Results are written to `phase-2/algoImplementation/results/`.

---

## Phase 3: Cloud DAG Scheduling HNH

Phase 3 maps HNH to DAG scheduling, where a task can wait for upstream dependencies or start immediately and risk stalls, failed parents, or inefficient resource usage. The state vector has 88 dimensions and the action space is seven hold durations in seconds.

Key files:

- `phase-3/simulator/simulator.py` — DAG scheduling simulator
- `phase-3/simulator/run_demo.py` — baseline policy comparison
- `phase-3/simulator/reward_engine.py` — local/global reward computation
- `phase-3/rewardEngineering/delay_tree.py` — DAG delay tree
- `phase-3/algoImplementation/train.py` — RL training and evaluation
- `phase-3/algoImplementation/benchmark_check.py` — benchmark/result checks

## State Space Mapping — Cloud DAG Scheduling (Phase 3)

### State Definition
At each decision epoch `t`, for task k:

s_t = {
    # Core RL Meta\
    CL(τ), OL(τ), τ*, α,\
    # Task Identity & Priority\
    job_id, task_index, priority,\
    scheduling_class, workload_type,\
    gpu_type_spec, inst_num, task_status,\
    # DAG Structure\
    num_parents, num_children, total_descendants,\
    critical_path_len, slack_time, is_on_critical_path,\
    depth_in_dag, fan_out_ratio,\
    upstream_delay, job_size, dag_completion_fraction,\
    # Resource Demands\
    plan_cpu, plan_mem, plan_gpu,\
    cpu_usage, gpu_wrk_util,\
    avg_mem_usage, max_mem_usage,\
    resource_cost_score,\
    # Global Cluster State\
    total_cpu_capacity, total_gpu_capacity,\
    cpu_util, gpu_util,\
    num_pending_tasks, num_running_tasks,\
    num_idle_machines, machine_load_avg,\
    network_receive_util,\
    failed_task_rate_G,\
    global_pipeline_utility_G,\
    global_operator_utility_G,\
    # Delay Tree\
    D_k, A_k, H_k, GD_k,\
    rho_H_A, SLO_deadline_k\
}\

---

### Core Local State
- CL(τ): Pipeline utility (downstream completion success)
- OL(τ): Cluster efficiency cost
- τ*: argmax over α·CL(τ) + (1−α)·OL(τ)

---

### DAG Structure Features (Critical Addition)
- num_parents / children / descendants
- critical_path_len
- slack_time (key decision variable)
- is_on_critical_path
- upstream_delay

---

### Resource Features
- Requested: CPU, memory, GPU
- Actual: utilisation metrics
- resource_cost_score (composite opportunity cost)

---

### Global Cluster State
- Resource utilisation (cpu_util, gpu_util)
- Queue pressure (num_pending_tasks)
- Capacity signals (idle machines)
- Stability signals (failed_task_rate_G)
- Global performance (pipeline + operator utility)

---

### Delay Tree Variables
- D_k: Start delay
- A_k: Completion delay
- H_k: Hold duration
- GD_k: Queue delay
- ρ(H_k, A_k): causal attribution
- SLO_deadline_k: hard constraint

---

### Action
- τ ∈ {0, 15, 30, 60, 120} seconds or continuous

---

### Reward Structure
- R_T_k = β·R_L_k + (1−β)·R_G_k
- R_L_k = α·CL_k + (1−α)·OL_k
- R_G_k = α·CG_k + (1−α)·OG_k

---

### Key Insight
State generalizes previous domains by introducing:
- Graph structure (DAG topology + critical path)
- Fine-grained resource economics (GPU/CPU/memory)
- Cluster-level congestion + stability signals

Run the simulator demo:

```bash
python3 phase-3/simulator/run_demo.py
```

Run the delay-tree smoke tests:

```bash
cd phase-3/rewardEngineering
python3 test_tree.py
cd ../..
```

Train and evaluate RL agents with the standard preset:

```bash
python3 phase-3/algoImplementation/train.py --preset standard
```

Train one algorithm with the standard preset:

```bash
python3 phase-3/algoImplementation/train.py --preset standard --algo a2c
```

Other available presets are `smoke`, `long`, and `paper`, but `standard` is the normal training run used by default in the Phase 3 script.

Useful options:

```bash
python3 phase-3/algoImplementation/train.py --help
```

Results are written to `phase-3/algoImplementation/results/`.

---

## Training Defaults

Current defaults in the training scripts:

| Phase | Algorithms | Default train/test budget |
| --- | --- | --- |
| Phase 1 | A2C, DQN, AC, DDPG | 25 train episodes, 5 test episodes |
| Phase 2 | A2C, DQN, AC | 25 train episodes, 5 test episodes |
| Phase 3 | A2C, DQN, AC, DDPG | `standard` preset: 30 train episodes, 8 test episodes, 3,600s episode cap, 750 HNH decisions |

All phases evaluate baseline policies and trained RL policies, then save summaries and plots where supported.

---

## Notes and Known Caveats

- The repository currently does not include a pinned dependency file.
- Phase 1 demo commands should be run as modules from `phase-1` because `run_demo.py` uses package-relative imports.
- Phase 2 needs `gymnasium` or `gym` installed before validation or training.
- Phase 2 does not currently contain a `simulator/run_demo.py`; use `simulator.validate_simulator` for simulator checks.
- Phase 3 training defaults to the `standard` preset if no preset is supplied, but the README commands pass `--preset standard` explicitly for clarity.

---

## Reference

Malladi, T., Murugappan, K., Sudarsanam, D., Suriyanarayanan, R., & Vasan, A. (2021). *To hold or not to hold? - Reducing Passenger Missed Connections in Airlines using Reinforcement Learning.* AAMAS 2021, 862-870.
