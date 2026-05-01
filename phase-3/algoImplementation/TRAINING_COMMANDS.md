# Phase 3 Training Commands

Use these from the repository root.

## Full Long Run: All Phase 1-Style RL Models

This trains A2C, DQN, AC, and DDPG, then evaluates them against `no_hold`,
`heuristic`, and `gpu_guard`.

```bash
python3 phase-3/algoImplementation/train.py \
  --algo all \
  --preset long \
  --run-name phase3_all_long_v1
```

Then validate every RL model present in the summary:

```bash
python3 phase-3/algoImplementation/benchmark_check.py \
  --summary phase-3/algoImplementation/results/phase3_all_long_v1_summary.json \
  --all-agents \
  --output phase-3/algoImplementation/training_logs/phase3_all_long_v1_benchmark.log
```

## Longer Paper-Style Run

This is much heavier. Run it when you can leave the machine alone.

```bash
python3 phase-3/algoImplementation/train.py \
  --algo all \
  --preset paper \
  --run-name phase3_all_paper_v1
```

```bash
python3 phase-3/algoImplementation/benchmark_check.py \
  --summary phase-3/algoImplementation/results/phase3_all_paper_v1_summary.json \
  --all-agents \
  --output phase-3/algoImplementation/training_logs/phase3_all_paper_v1_benchmark.log
```

## Per-Model Long Runs

These are useful if one model needs extra training or if you want separate logs.

```bash
python3 phase-3/algoImplementation/train.py \
  --algo a2c \
  --preset long \
  --run-name phase3_a2c_long_v1
```

```bash
python3 phase-3/algoImplementation/train.py \
  --algo dqn \
  --preset long \
  --run-name phase3_dqn_long_v1
```

```bash
python3 phase-3/algoImplementation/train.py \
  --algo ac \
  --preset long \
  --run-name phase3_ac_long_v1
```

```bash
python3 phase-3/algoImplementation/train.py \
  --algo ddpg \
  --preset long \
  --run-name phase3_ddpg_long_v1
```

## Custom Long Run

Use this if you want a stronger run than `long` without jumping to `paper`.

```bash
python3 phase-3/algoImplementation/train.py \
  --algo all \
  --episodes 150 \
  --test-episodes 25 \
  --duration-s 14400 \
  --max-decisions 2500 \
  --log-every 500 \
  --run-name phase3_all_custom150_v1
```

Validate:

```bash
python3 phase-3/algoImplementation/benchmark_check.py \
  --summary phase-3/algoImplementation/results/phase3_all_custom150_v1_summary.json \
  --all-agents \
  --output phase-3/algoImplementation/training_logs/phase3_all_custom150_v1_benchmark.log
```

## What To Send Back

Send:

- The final table printed by `train.py`.
- The output of `benchmark_check.py`.
- The file `phase-3/algoImplementation/results/<run-name>_summary.json`.

I can then verify whether the results are internally consistent and justify why
each model did or did not outperform the baselines.
