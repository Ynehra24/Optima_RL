"""Phase 3 reward attribution helpers built on the DAG Delay Tree."""

from __future__ import annotations

from typing import Dict

from .delay_tree import DAGDelayTree


def attribute_job_delay_penalty(
    job,
    task_states: Dict[str, object],
    penalty_scale: float = 1.0,
) -> Dict[str, float]:
    """Return hold-task penalties attributed from delayed task completions."""
    tree = DAGDelayTree()
    return tree.attribute_job_outcomes(
        job=job,
        task_states=task_states,
        outcome_scale=penalty_scale,
    )


def build_delay_tree(job, task_states: Dict[str, object]) -> DAGDelayTree:
    """Construct and return a populated Phase 3 DAG Delay Tree."""
    tree = DAGDelayTree()
    tree.build_from_job_states(job, task_states)
    return tree
