"""Check Phase 3 training results against baseline policies.

The training script writes ``results/summary.json``.  This checker turns that
JSON into a deterministic pass/fail report so we can quickly tell whether an
RL policy is actually improving over the baselines instead of merely producing
nice-looking training curves.

Example:
    python3 phase-3/algoImplementation/benchmark_check.py --agent a2c
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, Tuple


_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUMMARY = os.path.join(_HERE, "results", "summary.json")


METRIC_DIRECTIONS = {
    "completed_pct": "higher",
    "pipeline_stalls": "lower",
    "evicted_pct": "lower",
    "failed_pct": "lower",
    "avg_reward": "higher",
}


def load_summary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "evaluation" not in payload:
        raise ValueError(f"{path} does not contain an evaluation block")
    return payload["evaluation"]


def compare_metric(agent_value: float, baseline_value: float, direction: str,
                   tolerance: float) -> Tuple[bool, float]:
    delta = agent_value - baseline_value
    if direction == "higher":
        return delta >= -tolerance, delta
    if direction == "lower":
        return delta <= tolerance, delta
    raise ValueError(f"Unknown direction: {direction}")


def score_against_baseline(agent: Dict, baseline: Dict,
                           tolerance: float) -> Tuple[int, int, Dict[str, Tuple[bool, float]]]:
    details = {}
    wins = 0
    total = 0
    for metric, direction in METRIC_DIRECTIONS.items():
        if metric not in agent or metric not in baseline:
            continue
        ok, delta = compare_metric(
            float(agent[metric]),
            float(baseline[metric]),
            direction,
            tolerance,
        )
        details[metric] = (ok, delta)
        wins += int(ok)
        total += 1
    return wins, total, details


def format_report(evaluation: Dict, agent_name: str,
                  baselines: Iterable[str], tolerance: float) -> str:
    if agent_name not in evaluation:
        raise ValueError(f"Agent {agent_name!r} not found in summary")

    agent = evaluation[agent_name]
    lines = []
    lines.append("=" * 78)
    lines.append(f"PHASE 3 BENCHMARK CHECK: {agent_name.upper()}")
    lines.append("=" * 78)
    lines.append(
        f"Agent metrics: completed={agent.get('completed_pct', 0):.2f}% | "
        f"stalls={agent.get('pipeline_stalls', 0):.2f} | "
        f"evicted={agent.get('evicted_pct', 0):.2f}% | "
        f"failed={agent.get('failed_pct', 0):.2f}% | "
        f"avg_reward={agent.get('avg_reward', 0):.4f}"
    )
    lines.append("-" * 78)

    all_passed = True
    for baseline_name in baselines:
        if baseline_name not in evaluation:
            lines.append(f"SKIP {baseline_name}: missing from summary")
            continue
        baseline = evaluation[baseline_name]
        wins, total, details = score_against_baseline(agent, baseline, tolerance)
        baseline_passed = wins == total and total > 0
        all_passed = all_passed and baseline_passed
        status = "PASS" if baseline_passed else "FAIL"
        lines.append(
            f"{status} vs {baseline_name}: {wins}/{total} metrics non-worse "
            f"(tolerance={tolerance})"
        )
        for metric, (ok, delta) in details.items():
            direction = METRIC_DIRECTIONS[metric]
            marker = "OK" if ok else "BAD"
            lines.append(
                f"  {marker:3s} {metric:<16s} direction={direction:<6s} "
                f"agent={float(agent[metric]):>9.4f} "
                f"baseline={float(baseline[metric]):>9.4f} "
                f"delta={delta:>9.4f}"
            )
        lines.append("-" * 78)

    lines.append("OVERALL: " + ("PASS" if all_passed else "FAIL"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Phase 3 RL summary against baselines")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--agent", default="a2c")
    parser.add_argument("--baseline", action="append",
                        default=["no_hold", "heuristic", "gpu_guard"])
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    evaluation = load_summary(args.summary)
    report = format_report(evaluation, args.agent, args.baseline, args.tolerance)
    print(report)
    if report.rstrip().endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
