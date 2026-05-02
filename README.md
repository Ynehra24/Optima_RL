# Hold-or-Not-Hold RL

Reinforcement-learning simulators for the **Hold-or-Not-Hold (HNH)** decision problem across three domains:

1. **Phase 1:** airline passenger connections
2. **Phase 2:** freight cross-docking and cargo transfers
3. **Phase 3:** cloud DAG task scheduling

The shared question is: when an outbound vehicle or task has delayed dependencies, should the system hold it briefly, or let it leave/start on time and accept missed connections, failed transfers, or pipeline stalls?

The project is inspired by the AAMAS 2021 paper *"To hold or not to hold? - Reducing Passenger Missed Connections in Airlines using Reinforcement Learning"* and extends the same idea to logistics and cloud scheduling.

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

There is no `requirements.txt` in the repository, so install the Python packages used by the current code directly.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy matplotlib gymnasium pandas scapy
```

Core simulator/training code uses `numpy` and `matplotlib`. Phase 2 imports `gymnasium` with a fallback to `gym`; installing `gymnasium` is the simplest path. `pandas` is used by calibration/preprocessing scripts, and `scapy` is used by `phase-3/pcaphelper.py`.

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
