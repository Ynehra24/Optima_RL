"""Smoke tests for the Phase 3 DAG Delay Tree.

Run from the repository root:
    python phase-3/rewardEngineering/test_tree.py
"""

from __future__ import annotations

from delay_tree import DAGDelayTree


def test_direct_hold_attribution() -> None:
    tree = DAGDelayTree()
    tree.build_hold_delay_tree("task_b", 30.0, [("task_a", 20.0)])
    root = tree.build_departure_delay_tree(
        task_id="task_b",
        departure_delay_s=45.0,
        parent_arrival_delays=[("task_a", 20.0)],
        hold_duration_s=30.0,
        queue_delay_s=0.0,
    )

    assert root is not None
    attr = tree.attribute_outcome(root, 1.0)
    assert "task_b" in attr
    assert 0.55 < attr["task_b"] < 0.65


def test_arrival_to_parent_hold_chain() -> None:
    tree = DAGDelayTree()
    tree.build_hold_delay_tree("task_b", 30.0, [("task_a", 20.0)])
    tree.build_departure_delay_tree(
        task_id="task_b",
        departure_delay_s=45.0,
        parent_arrival_delays=[("task_a", 20.0)],
        hold_duration_s=30.0,
        queue_delay_s=0.0,
    )
    root = tree.build_arrival_delay_tree(
        task_id="task_b",
        arrival_delay_s=60.0,
        departure_delay_s=45.0,
        runtime_slowdown_s=15.0,
    )

    assert root is not None
    attr = tree.attribute_outcome(root, 2.0)
    assert "task_b" in attr
    assert 0.80 < attr["task_b"] < 1.10


def test_resource_congestion_attribution() -> None:
    tree = DAGDelayTree()
    attr = tree.attribute_resource_congestion(
        "task_gpu_train",
        {"task_serving": 120.0, "task_etl": 60.0},
        normalizer_s=300.0,
    )
    assert attr == {"task_gpu_train": 0.3}


if __name__ == "__main__":
    test_direct_hold_attribution()
    test_arrival_to_parent_hold_chain()
    test_resource_congestion_attribution()
    print("Phase 3 Delay Tree smoke tests passed.")
