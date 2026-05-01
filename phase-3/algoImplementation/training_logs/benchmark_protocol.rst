Phase 3 A2C Benchmark Protocol
==============================

Purpose
-------

This file records the command sequence and acceptance criteria used for the
Phase 3 DAG scheduling training run.  It is intentionally stored beside the
logs so the benchmark can be reproduced without reverse-engineering terminal
history.

Training Command
----------------

::

    python3 phase-3/algoImplementation/train.py \
      --algo a2c \
      --episodes 8 \
      --no-plots \
      --duration-s 1200 \
      --max-decisions 160

Benchmark Command
-----------------

::

    python3 phase-3/algoImplementation/benchmark_check.py \
      --agent a2c \
      --output phase-3/algoImplementation/training_logs/a2c_final_8ep_benchmark_report.log

Baselines
---------

``no_hold``
    Always chooses action index 0.  This baseline has no hold cost but causes
    task evictions and pipeline stalls whenever the HNH event is real.

``heuristic``
    Holds for the simulator's fixed moderate hold action when upstream delay
    is visible and GPU utilisation is below the pressure threshold.

``gpu_guard``
    Holds more conservatively under high resource pressure and longer when the
    delay signal is large.

Acceptance Metrics
------------------

The benchmark checker requires A2C to be non-worse than each baseline on every
metric below:

``completed_pct``
    Higher is better.

``pipeline_stalls``
    Lower is better.

``evicted_pct``
    Lower is better.

``failed_pct``
    Lower is better.

``avg_reward``
    Higher is better.  This uses mean step reward rather than total episode
    reward so policies are not rewarded merely for creating more HNH events.

Final Passing Result
--------------------

The committed 8-episode run passed all 15 comparisons:

* A2C vs no_hold: 5 / 5
* A2C vs heuristic: 5 / 5
* A2C vs gpu_guard: 5 / 5

The most important behavioral result is that A2C kept pipeline stalls,
evictions, and failures at zero while improving completion percentage and
average reward over the heuristic baselines.
