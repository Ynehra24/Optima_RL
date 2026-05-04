# Optima RL — Hold-or-Not-Hold Reinforcement Learning

> *"To hold or not to hold?"* — Malladi et al., AAMAS 2021

A three-phase study that adapts the **Hold-or-Not-Hold (HNH)** decision problem across progressively complex operational domains, training custom RL agents (A2C, DQN, AC, DDPG) to minimise delay propagation and improve system-wide throughput.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Quick Start — One Command](#3-quick-start--one-command)
4. [Manual Setup](#4-manual-setup)
5. [Phase 1 — Airline Network](#5-phase-1--airline-network)
6. [Phase 2 — Logistics Cross-Docking](#6-phase-2--logistics-cross-docking)
7. [Phase 3 — Cloud DAG Scheduling](#7-phase-3--cloud-dag-scheduling)
8. [Algorithms &amp; Hyperparameters](#8-algorithms--hyperparameters)
9. [Outputs &amp; Artefacts](#9-outputs--artefacts)
10. [Reference](#10-reference)

---

## 1. Project Overview

Complex interconnected networks — passenger airlines, freight hubs, cloud clusters — share a common challenge: upstream delays propagate and cascade. At every decision point, an operator must choose between **holding** (waiting for late transfers at a cost to the current schedule) or **not holding** (departing on time and accepting missed connections).

This project formalises that trade-off as an RL problem across three domains:

| Phase             | Domain               | Core Decision                                           |
| ----------------- | -------------------- | ------------------------------------------------------- |
| **Phase 1** | Airline Network      | Hold connecting flight for delayed passengers?          |
| **Phase 2** | Freight & Logistics  | Hold cross-dock truck for delayed inbound cargo?        |
| **Phase 3** | Cloud DAG Scheduling | Delay downstream DAG task for struggling prerequisites? |

All three phases share the same agent zoo (A2C, DQN, AC, DDPG), the same reward structure (α/β-weighted local + global utility), and evaluate against three baseline policies: **No-Hold**, **Heuristic-15**, and **Heuristic-30** (or domain equivalents).

---

## 2. Repository Layout

```text
Optima_RL/
├── run.sh                          ← Full automated pipeline (start here)
├── requirements.txt                ← Python dependencies
├── .gitattributes                  ← Enforces LF line endings for shell scripts
│
├── phase-1/                        ← Airline Hold-or-Not-Hold
│   ├── simulator/                  # Airline network simulator
│   │   ├── simulator.py            # Core AirlineNetworkSimulator class
│   │   ├── config.py               # SimConfig (α, β, seeds, schedules)
│   │   ├── generators.py           # Flight/passenger generation
│   │   ├── context_engine.py       # State context builder
│   │   └── run_demo.py             # Quick demo / validation run
│   ├── rewardEngineering/
│   │   ├── delay_tree.py           # Delay-tree attribution
│   │   └── reward_calculator.py    # α/β reward computation
│   └── algoImplementation/
│       ├── train.py                # Training + evaluation entry point
│       ├── environment.py          # Gym-wrapper shim
│       ├── agents/                 # A2C, DQN, AC, DDPG implementations
│       └── results/                # Plots + summary JSON (generated)
│
├── phase-2/                        ← Logistics Cross-Docking HNH
│   ├── simulator/
│   │   ├── logistics_env.py        # Single-hub Gymnasium environment
│   │   ├── multi_hub_env.py        # 10-hub FAF5 cascading environment
│   │   ├── hub_chain.py            # Inter-hub dependency engine
│   │   ├── cargo_manager.py        # Cargo transfer logic
│   │   ├── bay_manager.py          # Dock bay utilisation
│   │   └── calibrated/             # Pre-calibrated delay / routing params
│   ├── rewardEngineering/
│   │   ├── delay_tree.py           # Logistics delay tree w/ bay attribution
│   │   └── reward_calculator.py
│   ├── data/                       # Raw FAF5 / CFS source datasets
│   └── algoImplementation/
│       ├── train.py                # Training + evaluation entry point
│       ├── agents/                 # A2C, DQN, AC, DDPG
│       └── results/                # Plots + summary JSON (generated)
│
└── phase-3/                        ← Cloud DAG Scheduling HNH
    ├── simulator/
    │   ├── simulator.py            # DAGSchedulingSimulator
    │   ├── config.py               # Episode duration, HNH budget, α/β
    │   ├── models.py               # Task / Job / DAG data models
    │   ├── generators.py           # Alibaba/Borg trace-calibrated workloads
    │   ├── state_builder.py        # 88-dim state vector construction
    │   ├── reward_engine.py        # DAG reward with SLO attribution
    │   └── calibrated/             # Alibaba + Borg calibration JSONs
    ├── rewardEngineering/
    │   └── delay_tree.py           # Causal delay attribution for DAGs
    ├── preprocessing/
    │   ├── alibaba_calibration.py  # Calibrate from Alibaba cluster traces
    │   └── borg_calibration.py     # Calibrate from Google Borg traces
    └── algoImplementation/
        ├── train.py                # Training + evaluation entry point
        ├── benchmark_check.py      # Pass/fail report vs baselines
        ├── agents/                 # A2C, DQN, AC, DDPG
        └── results/                # Plots + summary JSON (generated)
```

---

## 3. Quick Start — One Command

> **For evaluators and anyone on Ubuntu 22.04 (including Docker)**

### Option A — Docker (recommended for a clean, reproducible environment)

```bash
# 1. Pull the Ubuntu 22.04 image
docker pull ubuntu:22.04

# 2. Clone the repository (outside the container, or inside — both work)
git clone https://github.com/Ynehra24/Optima_RL.git

# 3. Run a container, mounting the repo and dropping into bash
docker run --rm -it \
    -v "$(pwd)/Optima_RL:/workspace" \
    -w /workspace \
    ubuntu:22.04 \
    bash run.sh
```

That single `bash run.sh` handles **everything**: system packages, virtual environment, dependencies, all three training phases, evaluation, benchmark check, and consolidated results.

---

### Option B — Native Ubuntu 22.04

```bash
# 1. Clone the repository
git clone https://github.com/Ynehra24/Optima_RL.git
cd Optima_RL

# 2. Run the full pipeline (requires sudo / root for apt-get)
bash run.sh
```

> **Note:** `run.sh` calls `apt-get` internally, so it must be run as **root** or with **sudo** in a Docker container (Docker containers run as root by default).

---

### What `run.sh` Does

| Step | Action                                                                                                    |
| ---- | --------------------------------------------------------------------------------------------------------- |
| 1    | `apt-get` installs `python3`, `python3-venv`, `python3-dev`, `build-essential`, `libpcap-dev` |
| 2    | Creates `./venv` and activates the virtual environment                                                  |
| 3    | `pip install` all Python dependencies                                                                   |
| 4    | Trains all agents on**Phase 1** (airline, 25 episodes each)                                         |
| 5    | Trains all agents on**Phase 2** (logistics multi-hub, 25 episodes)                                  |
| 6    | Trains all agents on**Phase 3** (DAG scheduling, `standard` preset, 30 episodes)                  |
| 7    | Runs `benchmark_check.py` to generate a pass/fail evaluation report                                     |
| 8    | Writes `results/run_summary.txt` with metric tables from all three phases                               |

All output lands in `./results/`:

```text
results/
├── logs/
│   ├── phase1_training.log
│   ├── phase2_training.log
│   ├── phase3_training.log
│   └── phase3_benchmark.log
├── phase1/            ← figures + summary.json
├── phase2/            ← figures + summary.json
├── phase3/            ← figures + summary.json + benchmark report
└── run_summary.txt    ← consolidated metric table
```

---

## 4. Manual Setup

If you prefer to run phases individually (e.g., on Windows, macOS, or an existing environment):

### Prerequisites

| Requirement | Version                              |
| ----------- | ------------------------------------ |
| Python      | ≥ 3.10                              |
| pip         | ≥ 23                                |
| libpcap     | (Linux/macOS — needed by `scapy`) |

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-dev build-essential libpcap-dev
```

On macOS (Homebrew):

```bash
brew install libpcap
```

### Create Virtual Environment

```bash
git clone https://github.com/Ynehra24/Optima_RL.git
cd Optima_RL

python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import numpy, matplotlib, gymnasium, pandas, scapy; print('All dependencies OK')"
```

---

## 5. Phase 1 — Airline Network

Simulates an airline network with passenger connections. The RL agent decides, at each flight's scheduled departure, whether to hold for delayed inbound transfer passengers.

### State Space (17-dim)

| Component                | Symbol      | Meaning                                   |
| ------------------------ | ----------- | ----------------------------------------- |
| Local passenger utility  | `P_L(τ)` | Passenger benefit per hold duration       |
| Local airline utility    | `A_L(τ)` | Airline delay cost per hold               |
| Global passenger utility | `P_G`     | Network-wide avg passenger utility (24 h) |
| Global airline utility   | `A_G`     | Network OTP proxy                         |
| Locally optimal hold     | `τ*`     | Best τ from local objective              |
| Reward weights           | `α, β`  | Passenger vs airline, local vs global     |

### Run

```bash
# Train all 4 agents with default settings (25 episodes)
python3 phase-1/algoImplementation/train.py

# Train a single agent
python3 phase-1/algoImplementation/train.py --algo a2c

# Quick smoke test (2 episodes)
python3 phase-1/algoImplementation/train.py --episodes 2 --no-sweep

# Run the simulator demo
cd phase-1 && python3 -m simulator.run_demo && cd ..

# Run reward-engineering tests
cd phase-1 && python3 -m rewardEngineering.test_tree && cd ..
```

### CLI Options

```
--algo     {all, a2c, dqn, ac, ddpg}   Agent(s) to train (default: all)
--episodes N                            Override train episode count
--no-plots                              Skip figure generation
--no-sweep                              Skip α/β tunability sweep (Figure 8)
```

### Outputs

```text
phase-1/algoImplementation/results/
├── figure6_missed_otp.png      ← Missed PAX + OTP bar chart
├── figure6c_delays.png         ← Arrival / departure delays
├── figure7_rl_metrics.png      ← Reward, value, loss training curves
├── figure8_tunability.png      ← α/β sweep (A2C)
├── summary.json                ← All metrics + delta vs baselines
└── *_agent.pkl                 ← Saved agent weights
```

---

## 6. Phase 2 — Logistics Cross-Docking

Adapts HNH to freight hubs. A truck at a cross-docking facility must decide whether to wait for late inbound cargo or depart on schedule. The full 10-hub cascading network (`--multi-hub`) is the recommended evaluation mode. **Miss Rate** (missed cargo transfers) is the primary KPI; **SLA%** (departures within 45 min of schedule) is the secondary trade-off metric.

### State Space (34-dim / 42-dim multi-hub)

| Group                         | Key Features                                                                                                                                                            |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Local truck context** | Cargo utility `C_L(τ)`, operator utility `O_L(τ)`, optimal hold `τ*`, cargo value `V_k`, volume fraction `Q_k`, SLA urgency `X_k`, perishability `E_k` |
| **Transfer context**    | Inbound delay `Δ_in`, transfer slack `Δ_slack`, driver hours `L_k`, deadline pressure `F_k`, number of inbound trucks `N_in`                                |
| **Global hub context**  | Bay utilisation `B_G`, throughput `W_G`, failure rate `Y_G`, queue depth `Z_G`, global cargo utility `C_G`, global operator utility `O_G`                   |
| **Network context**     | (multi-hub only, +8 dims) Downstream bay util, cascade risk, upstream inter-hub delay, hub centrality, trucks in transit                                        |
| **Delay attribution**   | Departure delay `D_k`, arrival delay `A_k`, bay delay `G_k^bay`, road delay `G_k^road`                                                                          |

### Run

```bash
# Train all agents — 10-hub cascading network (recommended)
python -X utf8 phase-2/algoImplementation/train.py --multi-hub

# Train all agents — single-hub mode
python -X utf8 phase-2/algoImplementation/train.py

# Train a single agent
python -X utf8 phase-2/algoImplementation/train.py --algo dqn --multi-hub

# Quick smoke test (2 episodes, no plots)
python -X utf8 phase-2/algoImplementation/train.py --episodes 2 --no-plots --no-sweep --multi-hub

# Run simulator validation suite
cd phase-2 && python3 -m simulator.validate_simulator && cd ..
```

> **Windows note:** Use `python -X utf8` to avoid UnicodeEncodeError on cp1252 terminals. The script auto-reconfigures stdout to UTF-8 as well.

### CLI Options

```
--algo       {all, a2c, dqn, ac, ddpg}   Agent(s) to train (default: all)
--episodes   N                            Override train episode count
--multi-hub                               Enable 10-hub cascading network (recommended)
--no-plots                                Skip figure generation
--no-sweep                                Skip alpha/beta tunability sweep
```

### Outputs

```text
phase-2/algoImplementation/results/
├── figure6_missed_transfers.png   <- Missed transfers + miss rate
├── figure6b_bay_delay.png         <- Bay utilisation + departure delay
├── figure7_rl_metrics.png         <- Training curves (reward, loss, value)
├── figure8_tunability.png         <- alpha/beta sweep (A2C)
├── summary.json                   <- All metrics (SLA%, throughput, miss rate, holds%)
└── *_agent.pkl                    <- Saved agent weights
```

---

## 7. Phase 3 — Cloud DAG Scheduling

Adapts HNH to cloud task scheduling. The agent decides whether to delay a downstream DAG task that is waiting on a struggling upstream dependency, or proceed and risk pipeline stalls and evictions. Calibrated against real Alibaba and Google Borg cluster traces.

### State Space (88-dim)

| Group                       | Key Features                                                                                                                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RL meta**           | Pipeline success probability `C_L(τ)`, cluster efficiency cost `O_L(τ)`, optimal hold `τ*`, trade-off weight `α`                                                                    |
| **Task identity**     | Job ID, task index, priority, scheduling class, workload type, GPU type, instance count, task status                                                                                            |
| **DAG structure**     | Parents, children, descendants, critical-path length, slack time, critical-path flag, depth, fan-out ratio, upstream delay `Δ_in`, job size, DAG completion fraction                         |
| **Resource demand**   | Planned CPU / memory / GPU, CPU usage, GPU utilisation, avg/max memory usage, resource cost score                                                                                               |
| **Global cluster**    | CPU/GPU capacity + utilisation, pending/running tasks, idle machines, machine load avg, network utilisation, failed task rate, global pipeline utility `C_G`, global operator utility `O_G` |
| **Delay attribution** | Start delay `D_k`, completion delay `A_k`, hold applied `H_k`, queue delay `GD_k`, causal influence `ρ(H_k, A_k)`, SLO deadline                                                      |

### Training Presets

| Preset         | Train eps | Test eps | Episode cap | HNH decisions |
| -------------- | --------- | -------- | ----------- | ------------- |
| `smoke`      | 2         | 1        | 300 s       | 30            |
| `standard`  | 30        | 8        | 3,600 s     | 750           |
| `long`       | 100       | 20       | 7,200 s     | 1,500         |
| `paper`      | 200       | 30       | 86,400 s    | 5,000         |

### Run

```bash
# Train all agents — standard preset (recommended)
python3 phase-3/algoImplementation/train.py --preset standard

# Quick smoke test (~1 min)
python3 phase-3/algoImplementation/train.py --preset smoke

# Train a single agent
python3 phase-3/algoImplementation/train.py --preset standard --algo a2c

# Run the simulator demo
python3 phase-3/simulator/run_demo.py

# Run delay-tree smoke tests
cd phase-3/rewardEngineering && python3 test_tree.py && cd ../..
```

### Benchmark Check

After training, verify whether RL agents beat the baselines:

```bash
# Check all agents against no_hold, heuristic, and gpu_guard
python3 phase-3/algoImplementation/benchmark_check.py \
    --summary phase-3/algoImplementation/results/summary.json \
    --all-agents \
    --output phase-3/algoImplementation/training_logs/benchmark.log
```

### CLI Options

```
--algo           {all, a2c, dqn, ac, ddpg}    Agent(s) to train (default: all)
--preset         {smoke, standard, long, paper} Training budget preset
--episodes       N                              Override train episode count
--test-episodes  N                              Override test episode count
--duration-s     SECONDS                        Override episode wall-clock cap
--max-decisions  N                              Override max HNH decisions/episode
--run-name       NAME                           Tag for output files
--no-plots                                      Skip figure generation
```

### Outputs

```text
phase-3/algoImplementation/results/
├── phase3_training_curves.png    ← Step reward, episode reward, loss
├── phase3_eval_bars.png          ← Pipeline stalls + eviction rate
├── summary.json                  ← All metrics (shared + run-specific)
├── <run-name>_summary.json       ← Run-specific JSON
└── *_agent.pkl                   ← Saved agent weights

phase-3/algoImplementation/training_logs/
└── <run-name>_benchmark.log      ← Pass/fail report vs baselines
```

---

## 8. Algorithms & Hyperparameters

All agents are implemented from scratch using pure NumPy — no deep-learning framework dependency.

| Agent          | Type                      | Key Architecture                                 |
| -------------- | ------------------------- | ------------------------------------------------ |
| **A2C**  | Actor-Critic (on-policy)  | Shared MLP backbone, policy + value heads, GAE   |
| **DQN**  | Value-based (off-policy)  | MLP Q-network, experience replay, ε-greedy      |
| **AC**   | Actor-Critic (on-policy)  | Separate actor/critic MLPs, REINFORCE baseline   |
| **DDPG** | Actor-Critic (off-policy) | Continuous actor → discretised, target networks |

### Shared Defaults

| Hyperparameter                      | Phase 1 | Phase 2 | Phase 3 |
| ----------------------------------- | ------- | ------- | ------- |
| Learning rate `lr`                  | 0.0001  | 0.0003  | 0.0003  |
| Discount `gamma`                    | 0.8     | 0.9     | 0.9     |
| Batch size                          | 32      | 32      | 32      |
| Hidden layers (MLP)                 | [64,64] | [128,128] | [128,128] |
| Cargo/passenger weight `alpha`      | 0.75    | 0.50    | 0.50    |
| Local/global weight `beta`          | 0.75    | 0.75    | 0.75    |
| DQN epsilon decay steps             | 5000    | 5000    | 4000    |
| DDPG OU noise sigma                 | 0.05    | 0.15    | —       |
| Random seed                         | 42      | 42      | 42      |

> **Phase 2** uses `lr=0.0003`, `gamma=0.9`, hidden layers `[128, 128]`, and dense action shaping in the reward function.

### Hold Actions (all phases)

```
Index:  0    1    2    3    4    5    6
Hold:   0   5   10   15   20   25   30  minutes
```

---

## 9. Outputs & Artefacts

When run via `run.sh`, all artefacts are mirrored into a top-level `results/` directory:

```text
results/
├── run_summary.txt              ← Human-readable table: all phases
├── logs/
│   ├── phase1_training.log      ← Full stdout from Phase 1 train.py
│   ├── phase2_training.log      ← Full stdout from Phase 2 train.py
│   ├── phase3_training.log      ← Full stdout from Phase 3 train.py
│   └── phase3_benchmark.log     ← benchmark_check.py output
├── phase1/
│   ├── summary.json             ← OTP, missed PAX, delays, holds %
│   ├── figure6_missed_otp.png
│   ├── figure6c_delays.png
│   ├── figure7_rl_metrics.png
│   └── figure8_tunability.png
├── phase2/
│   ├── summary.json             <- Miss rate, SLA%, throughput, holds%
│   ├── figure6_missed_transfers.png
│   ├── figure6b_bay_delay.png
│   ├── figure7_rl_metrics.png
│   └── figure8_tunability.png
└── phase3/
    ├── summary.json             ← Completed %, evictions, stalls, reward
    ├── eval_run_summary.json
    ├── phase3_training_curves.png
    ├── phase3_eval_bars.png
    └── eval_run_benchmark.log   ← Pass/fail vs baselines
```

---

## 10. Reference

Malladi, T., Murugappan, K., Sudarsanam, D., Suriyanarayanan, R., & Vasan, A. (2021).
*To hold or not to hold? — Reducing Passenger Missed Connections in Airlines using Reinforcement Learning.*
**AAMAS 2021**, 862–870.

---

<p align="center">
  <sub>Optima RL · Three-Phase Hold-or-Not-Hold Study · Built for AAMAS 2021 Extension</sub>
</p>
