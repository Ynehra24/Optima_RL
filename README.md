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

### State Space for Phase 1 — Aviation (Hold-or-Not-Hold RL)
Core idea: State = local + global utility forecasts + network context + derived helper

| Component                | Symbol(s) | Type            | Meaning                                          |
| ------------------------ | --------- | --------------- | ------------------------------------------------ |
| Full state               | `s_t`     | State           | Complete feature vector at decision time         |
| Local passenger utility  | `P_L(τ)`  | Forecast vector | Passenger benefit for each hold time             |
| Local airline utility    | `A_L(τ)`  | Forecast vector | Airline delay cost per hold                      |
| Global passenger utility | `P_G`     | Scalar          | Avg passenger utility over network (24h window)  |
| Global airline utility   | `A_G`     | Scalar          | Avg airline performance (OTP proxy)              |
| Locally optimal hold     | `τ*`      | Derived scalar  | Best τ from local objective (speeds convergence) |
| Reward weights           | `α, β`    | Scalars         | Trade-offs: PU vs AU, local vs global            |


### Sumulator
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

### State space for Phase 2 — Logistics Cross-Docking
Core idea: Extends aviation into rich operational state (truck + hub + constraints)

Local Truck Context (Decision-Critical)

| Component     | Symbol(s) | Type   | Meaning                        |
| ------------- | --------- | ------ | ------------------------------ |
| Full state    | `s_t`     | State  | Truck + hub + transfer context |
| Action        | `τ`       | Action | Hold duration (0–30 min)       |
| Realised hold | `H_k`     | Actual | Actual applied hold            |

Local Truck Context (Decision-Critical)

| Feature           | Symbol    | Meaning                            |
| ----------------- | --------- | ---------------------------------- |
| Cargo utility     | `C_L(τ)`  | Value of successful transfers      |
| Operator utility  | `O_L(τ)`  | Delay / cost to logistics operator |
| Optimal hold      | `τ*`      | Local best τ                       |
| Cargo value       | `V_k`     | Importance of goods                |
| Volume fraction   | `Q_k`     | % of truck affected                |
| SLA urgency       | `X_k`     | Delivery strictness                |
| Perishability     | `E_k`     | Time sensitivity                   |
| Inbound delay     | `Δ_in`    | ETA lag                            |
| Transfer slack    | `Δ_slack` | Buffer before departure            |
| Driver hours      | `L_k`     | Hard constraint                    |
| Deadline pressure | `F_k`     | Downstream urgency                 |
| # inbound trucks  | `N_in`    | Complexity of decision             |

Global Hub Context

| Feature                 | Symbol     | Meaning                  |
| ----------------------- | ---------- | ------------------------ |
| Global cargo utility    | `C_G`      | System-wide success rate |
| Global operator utility | `O_G`      | Network efficiency       |
| Bay utilisation         | `B_G`      | Congestion signal        |
| Throughput              | `W_G`      | Transfers per hour       |
| Failure rate            | `Y_G`      | Missed transfers         |
| Queue depth             | `Z_G`      | System delay             |
| Departure delay         | `D_k`      | Truck delay              |
| Arrival delay           | `A_k`      | End-to-end delay         |
| Bay delay               | `G_k^bay`  | Dock congestion          |
| Road delay              | `G_k^road` | Transit delay            |

Reward state variables

| Symbol  | Meaning       |
| ------- | ------------- |
| `R_T^k` | Total reward  |
| `R_L^k` | Local reward  |
| `R_G^k` | Global reward |
| `α, β`  | Trade-offs    |


### Simulator

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

### State space for Phase 3: Cloud DAG Scheduling HNH

Core idea: State = multi-layer graph + resource + cluster + delay attribution

RL Meta (Core Carryover)

| Symbol   | Meaning                      |
| -------- | ---------------------------- |
| `C_L(τ)` | Pipeline success probability |
| `O_L(τ)` | Cluster efficiency cost      |
| `τ*`     | Optimal hold                 |
| `α`      | Trade-off weight             |

Task-Level Identity & Priority

| Feature          | Symbol             |
| ---------------- | ------------------ |
| Job ID           | `job_id`           |
| Task index       | `task_index`       |
| Priority         | `priority`         |
| Scheduling class | `scheduling_class` |
| Workload type    | `workload_type`    |
| GPU type         | `gpu_type_spec`    |
| Instance count   | `inst_num`         |
| Task status      | `task_status`      |

DAG Structure

| Feature        | Symbol                    | Meaning                   |
| -------------- | ------------------------- | ------------------------- |
| Parents        | `num_parents`             | Blocking dependencies     |
| Children       | `num_children`            | Immediate impact          |
| Descendants    | `total_descendants`       | Long-term impact          |
| Critical path  | `critical_path_len`       | Completion bottleneck     |
| Slack          | `slack_time`              | Safe delay margin         |
| Critical flag  | `is_on_critical_path`     | Binary importance         |
| Depth          | `depth_in_dag`            | Stage of execution        |
| Fan-out        | `fan_out_ratio`           | Parallel unlock potential |
| Upstream delay | `Δ_in`                    | Parent delay              |
| Job size       | `job_size`                | DAG complexity            |
| Completion %   | `dag_completion_fraction` | Progress                  |

Resource Demand

| Feature         | Symbol                           |
| --------------- | -------------------------------- |
| CPU             | `plan_cpu`                       |
| Memory          | `plan_mem`                       |
| GPU             | `plan_gpu`                       |
| CPU usage       | `cpu_usage`                      |
| GPU utilisation | `gpu_wrk_util`                   |
| Memory usage    | `avg_mem_usage`, `max_mem_usage` |
| Resource cost   | `resource_cost_score`            |

Global Cluster State

| Feature                 | Symbol                 |
| ----------------------- | ---------------------- |
| CPU capacity            | `total_cpu_capacity`   |
| GPU capacity            | `total_gpu_capacity`   |
| CPU util                | `cpu_util`             |
| GPU util                | `gpu_util`             |
| Pending tasks           | `num_pending_tasks`    |
| Running tasks           | `num_running_tasks`    |
| Idle machines           | `num_idle_machines`    |
| Load avg                | `machine_load_avg`     |
| Network util            | `network_receive_util` |
| Failure rate            | `failed_task_rate_G`   |
| Global pipeline utility | `C_G`                  |
| Global operator utility | `O_G`                  |

Delay Tree (Causal Attribution)

| Symbol           | Meaning          |
| ---------------- | ---------------- |
| `D_k`            | Start delay      |
| `A_k`            | Completion delay |
| `H_k`            | Hold applied     |
| `GD_k`           | Queue delay      |
| `ρ(H_k, A_k)`    | Causal influence |
| `SLO_deadline_k` | Hard constraint  |

Reward Variables

| Symbol   | Meaning         |
| -------- | --------------- |
| `R_T_k`  | Total reward    |
| `R_L_k`  | Local reward    |
| `R_G_k`  | Global reward   |
| `β`      | Trade-off       |
| `σ_i(τ)` | Task disutility |


### Simulator
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
