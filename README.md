# Hold or Not to Hold — Multi-Domain RL Synchronisation

A reinforcement learning system for solving the **Hold-or-Not-Hold (HNH)** decision problem across aviation, freight logistics, and cloud DAG scheduling. The RL agent learns when to delay a departure/task to accommodate delayed dependencies, and when the cost of holding outweighs the benefit.

Based on the AAMAS 2021 paper *"To Hold or Not to Hold? Using AI to Improve the Efficiency of Airline Networks"*.

---

## Problem Statement

In both airline and freight logistics networks, operators face a recurring real-time decision: should a departing vehicle **wait** for delayed connecting cargo/passengers, or **leave on time** and force a missed connection?

- Holding too long → departure delays cascade through the network
- Leaving too early → missed connections, rebooking costs, SLA violations

This project frames the decision as a sequential RL problem. A local agent observes the state of a single vehicle/task and its dependencies, then outputs a discrete hold duration.

---

## Repository Structure

```
RL Proj/
├── phase-1/                        # Airline (Aviation) Domain
│   ├── simulator/                  # Discrete-event airline network simulator
│   │   ├── config.py               # SimConfig — hyperparams & airline knobs
│   │   ├── models.py               # Flight, Passenger, FlightState dataclasses
│   │   ├── context_engine.py       # State vector (PL, OL, PG, OG) construction
│   │   ├── event_engine.py         # Priority-queue discrete event engine
│   │   ├── generators.py           # Synthetic schedule & delay generators
│   │   ├── simulator.py            # AirlineNetworkSimulator (Gym-like API)
│   │   ├── validation.py           # Paper benchmark validation suite
│   │   └── run_demo.py             # Demo runner with baseline comparisons
│   └── rewardEngineering/          # Aviation reward engineering
│       ├── delay_tree.py           # Delay Tree — causal delay attribution
│       ├── reward_calculator.py    # RewardCalculator (R_L, R_G, R_T)
│       └── test_tree.py            # Unit tests for delay tree attribution
│
├── phase-2/                        # Logistics Cross-Docking Domain
│   ├── simulator/                  # Cross-dock microsimulator
│   │   ├── config.py               # SimConfig — logistics knobs (BG, SLA, decay)
│   │   ├── models.py               # Truck, CargoUnit, TruckState, Hub dataclasses
│   │   ├── context_engine.py       # 34-dim state vector with cargo enrichments
│   │   ├── event_engine.py         # Discrete event engine (DOCK, DEPARTURE, TRANSFER)
│   │   ├── generators.py           # Synthetic truck plan & cargo generators
│   │   ├── simulator.py            # CrossDockSimulator (Gym-like API)
│   │   ├── validation.py           # Logistics benchmark validation suite
│   │   └── run_demo.py             # Demo runner
│   ├── rewardEngineering/          # Logistics reward engineering
│   │   ├── delay_tree.py           # Extended Delay Tree with GB (bay-blockage) node
│   │   ├── reward_calculator.py    # LogisticsRewardCalculator (R_L, R_G + congestion)
│   │   └── test_tree.py            # 9 unit tests — all passing
│   └── algoImplementation/         # A2C/DQN/AC training for logistics
│
├── phase-3/                        # Cloud DAG Scheduling Domain
│   ├── simulator/                  # Calibrated DAG scheduling simulator
│   │   ├── config.py               # Borg/Alibaba calibrated SimConfig
│   │   ├── models.py               # Job, Task, Machine, TaskState dataclasses
│   │   ├── state_builder.py        # 88-dim DAG/resource/cluster state vector
│   │   ├── reward_engine.py        # Local/global reward with DT attribution
│   │   ├── simulator.py            # DAGSchedulingSimulator
│   │   └── run_demo.py             # Baseline policy comparison
│   ├── rewardEngineering/          # DAG reward engineering
│   │   ├── delay_tree.py           # DAG Delay Tree with U/RC/F nodes
│   │   ├── reward_calculator.py    # Convenience attribution helpers
│   │   └── test_tree.py            # Delay tree smoke tests
│   └── algoImplementation/         # A2C, DQN, AC for Phase 3
│
└── README.md
```

---

## Methodology

### Core Idea — Delay Tree Attribution

The key insight from the paper is **global credit assignment**: when a downstream truck arrives late, how much of that lateness was caused by *this* hold decision three hops ago?

A **Delay Tree** traces the causal chain:

```
Arrival / Completion Delay (A_k)
    └── Departure Delay (D_k)     [weight = D/A]
            ├── Hold Delay (H_k)             [weight = H/D]  ← RL action
            ├── Ground Delay (GD_k)          [weight = GD/D]
            └── Bay Blockage (GB_k) [P2 only] [weight = GB/D]
```

Influence weights are proportional — each node takes a fraction equal to its contribution to the total delay. The RL agent's hold decision is attributed exactly its causal share of the global outcome.

### Reward Structure

$$
R_T^k = \beta \cdot R_L^k + (1-\beta) \cdot R_G^k
$$

| Component               | Formula                                                     | Description                           |
| ----------------------- | ----------------------------------------------------------- | ------------------------------------- |
| **Local reward**  | $R_L = \alpha \cdot CL(\tau) + (1-\alpha) \cdot OL(\tau)$ | Immediate utility at decision time    |
| **Global reward** | $R_G = \alpha \cdot CG + (1-\alpha) \cdot OG$             | Network-wide impact attributed via DT |
| **α**            | 0.75                                                        | Cargo/passenger vs. schedule weight   |
| **β**            | 0.75                                                        | Local vs. global reward weight        |

**Phase 2 adds a bay-congestion penalty to $R_G$:**

$$
R_G^{P2} = \alpha \cdot CG + (1-\alpha) \cdot OG - \lambda_{bay} \cdot \text{BayCongestion}_k
$$

### State Vector

**Phase 1** — 17 dimensions: 7×PL(τ) + 7×AL(τ) + PG + AG + τ*.

**Phase 2** — 34 dimensions. Extends Phase 1 with:

- `V_k` — mean cargo value score
- `Q_k` — cargo volume fraction
- `X_k` — worst-case SLA urgency (0=standard, 1=priority, 2=express)
- `E_k` — perishable cargo fraction
- `L_k` — driver hours remaining
- `BG` — current bay utilisation
- `WG` — global cargo utility (rolling)
- `YG` — global transfer success rate
- `ZG` — delayed inbound queue depth

**Phase 3** — 88 dimensions. Extends the same HNH state idea to cloud DAG scheduling with task identity, scheduling class, workload/GPU type, DAG topology, critical-path features, resource demand, cluster utilisation, and delay-tree variables.

---

## Phase-by-Phase Progress

### ✅ Phase 1 — Airline (Aviation)

Full discrete-event microsimulator for an airline hub-and-spoke network.

- Generates synthetic flight schedules, passenger itineraries, and delay distributions matching paper benchmarks
- Implements the Delay Tree exactly as described in AAMAS §5.1
- `AirlineNetworkSimulator` exposes a standard Gym-like API (`reset()` / `step()`)
- Validates 4 core metrics: OTP, missed-connection rate, avg delay, avg hold
- Baseline policies: `no_hold`, `heuristic`, `random`

### ✅ Phase 2 — Logistics Cross-Docking (Simulator + Reward Engineering)

Adapts the aviation framework to a freight cross-docking hub network.

**Domain mapping:**

| Aviation          | Logistics                                                             |
| ----------------- | --------------------------------------------------------------------- |
| Passenger         | Cargo Unit (with `value_score`, `sla_urgency`, `is_perishable`) |
| Flight            | Truck (with `bay_dwell_delay`, `driver_hours_remaining`)          |
| Gate delay        | Bay-blockage delay (`GB_k`)                                         |
| Missed connection | Failed transfer → 24h next-cycle penalty                             |

**Key extensions over Phase 1:**

1. **GB (Bay-Blockage) node** — A new node type in the Delay Tree tracks time trucks wait for a free docking bay. This dilutes the hold's attributed blame proportionally. At GB=0, the tree reduces exactly to Phase 1.
2. **Bay-congestion cascading** — `register_bay_congestion()` attributes bay queue penalties back to the hold that caused the blockage, as an additional $R_G$ penalty term.
3. **Cargo disutility enrichments** — CL(τ) encodes value weighting ($V_k$), SLA urgency amplification $(1+X_k)$, and exponential perishable decay instead of the linear aviation model.

4 core metrics: schedule OTP, failed transfer rate, avg delivery delay, avg bay utilisation.

### ✅ Phase 3 — Cloud DAG Scheduling (Simulator + Delay Tree + RL)

Adapts HNH to Borg/Alibaba-style DAG scheduling.

**Domain mapping:**

| Aviation/Logistics | Phase 3 DAG Scheduling |
| ------------------ | ---------------------- |
| Flight / Truck     | Task                   |
| Tail / route       | Job DAG                |
| Incoming transfer  | Parent task dependency |
| Hold departure     | Delay task start       |
| Missed connection  | Eviction / pipeline stall |
| Gate/bay pressure  | CPU/GPU/memory resource congestion |

**Key extensions over Phase 2:**

1. **DAG Delay Tree** — Adds `U` (upstream parent), `RC` (resource congestion), and `F` (failed/evicted parent) nodes while preserving the core `A/D/H/GD/T` attribution rules.
2. **Reward integration** — `phase-3/simulator/reward_engine.py` now routes post-episode global attribution through the dedicated DAG Delay Tree when available.
3. **Algorithm implementation** — `phase-3/algoImplementation/` includes A2C, DQN, and online AC agents plus a training/evaluation driver for the 88-dim simulator state.

---

## Running the Code

### Prerequisites

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install numpy
```

### Phase 1 — Airline Demo

```bash
cd phase-1/simulator
python run_demo.py
```

### Phase 2 — Cross-Docking Demo

```bash
cd phase-2/simulator
python run_demo.py
```

### Phase 2 — Reward Engineering Tests

```bash
cd phase-2/rewardEngineering
python test_tree.py
```

Expected output:

```
============================================================
  LOGISTICS REWARD ENGINEERING — UNIT TESTS
============================================================
  ✓ PASSED (×9)
============================================================
  ALL 9 TESTS PASSED ✓
============================================================
```

### Phase 3 — DAG Simulator Demo

```bash
python3 phase-3/simulator/run_demo.py
```

### Phase 3 — Delay Tree Tests

```bash
python3 phase-3/rewardEngineering/test_tree.py
```

### Phase 3 — RL Training Smoke Run

```bash
python3 phase-3/algoImplementation/train.py --algo a2c --episodes 1 --no-plots --duration-s 300 --max-decisions 20
```

---

## Key Design Decisions

- **Agent API is domain-agnostic** — `LogisticsRewardCalculator` exposes the same interface as Phase 1's `RewardCalculator`. The A2C agent code is unchanged between phases.
- **Bay-blockage is separable** — Rather than folding bay wait time into `GD_k` (as aviation folds gate congestion), logistics treats `GB_k` as an independent Delay Tree node. This gives the agent a cleaner causal signal.
- **Perishability uses exponential decay** — Aviation uses a linear disutility ramp. Logistics uses exponential decay for perishable cargo, where delay cost accelerates non-linearly.
- **SLA urgency is amplified, not capped** — The `(1 + X_k)` multiplier in `CL(τ)` means express cargo (X_k=2) is penalised 3× harder than standard cargo. This is a tunable hyperparameter.

---

## References

> Peng, B., Xiong, H., & Zhang, J. (2021). *To Hold or Not to Hold? Using AI to Improve the Efficiency of Airline Operations Under Uncertainty.* AAMAS 2021, Paper #862.
