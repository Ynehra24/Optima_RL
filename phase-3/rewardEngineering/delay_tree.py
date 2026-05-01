"""
Delay Tree for the Phase 3 DAG scheduling domain.

Phase 3 maps the aviation/logistics hold-or-not-hold idea onto cloud DAG
scheduling:

  A  : arrival/completion delay of a task
  D  : departure/start delay of a task
  H  : hold applied by the RL agent
  GD : scheduler queue/ground delay excluding the hold
  T  : intrinsic runtime slowdown while the task executes
  U  : upstream parent delay propagated through a DAG edge
  RC : resource congestion delay caused by unavailable CPU/GPU/memory
  F  : failed or evicted parent dependency

The first five nodes mirror phases 1 and 2.  U, RC, and F are the cloud DAG
extensions.  They make attribution explicit when a task starts late because a
parent task, resource pressure, or a failed dependency blocked it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


EPSILON = 1e-9
ON_TIME_THRESHOLD_S = 15.0


@dataclass
class DelayNode:
    """One node in the Phase 3 Delay Tree."""

    type_name: str
    task_id: str
    value: float
    children: List["DelayNode"] = field(default_factory=list)
    influence_weights: List[float] = field(default_factory=list)

    def add_child(self, node: "DelayNode", weight: float) -> None:
        """Attach a child once, accumulating weight if the edge already exists."""
        if weight <= 0:
            return
        for i, child in enumerate(self.children):
            if child is node:
                self.influence_weights[i] += weight
                return
        self.children.append(node)
        self.influence_weights.append(weight)

    @property
    def label(self) -> str:
        return f"{self.type_name}:{self.task_id}"


class DAGDelayTree:
    """Delay Tree for cloud DAG hold decisions.

    The builder methods follow the same three rules as phases 1 and 2:

      Rule 1: completion delay A_k is caused by start delay D_k, runtime
              slowdown T_k, and residual scheduler/resource effects.
      Rule 2: start delay D_k is caused by upstream parent delays U_i,
              the hold H_k, queue delay GD_k, and resource congestion RC_k.
      Rule 3: hold H_k is caused by delayed parent tasks.

    The tree can then attribute a local or global outcome back to hold nodes.
    """

    def __init__(self) -> None:
        self.nodes_by_event: Dict[Tuple[str, str], DelayNode] = {}

    def get_or_create(self, type_name: str, task_id: str, value: float) -> DelayNode:
        key = (type_name, task_id)
        if key not in self.nodes_by_event:
            self.nodes_by_event[key] = DelayNode(
                type_name=type_name,
                task_id=task_id,
                value=float(value),
            )
        else:
            self.nodes_by_event[key].value = float(value)
        return self.nodes_by_event[key]

    def build_arrival_delay_tree(
        self,
        task_id: str,
        arrival_delay_s: float,
        departure_delay_s: float,
        runtime_slowdown_s: float,
        resource_congestion_s: float = 0.0,
        queue_delay_s: float = 0.0,
    ) -> Optional[DelayNode]:
        """Rule 1: A_k depends on D_k, T_k, RC_k, and GD_k.

        Delay propagation is arrested for small completion delays, matching the
        phase 1/2 fifteen-minute aviation buffer but expressed in simulator
        seconds.  Phase 3 uses seconds throughout, so callers can tune the
        threshold by filtering before calling this method if needed.
        """
        if arrival_delay_s <= 0:
            return None

        arrival = self.get_or_create("A", task_id, arrival_delay_s)
        if arrival_delay_s <= ON_TIME_THRESHOLD_S:
            return arrival

        components = [
            ("D", task_id, departure_delay_s),
            ("T", task_id, runtime_slowdown_s),
            ("RC", task_id, resource_congestion_s),
            ("GD", task_id, queue_delay_s),
        ]
        self._add_ratio_weighted_children(arrival, components)
        return arrival

    def build_departure_delay_tree(
        self,
        task_id: str,
        departure_delay_s: float,
        parent_arrival_delays: Iterable[Tuple[str, float]],
        hold_duration_s: float,
        queue_delay_s: float,
        resource_congestion_s: float = 0.0,
        failed_parent_delays: Iterable[Tuple[str, float]] = (),
    ) -> Optional[DelayNode]:
        """Rule 2: D_k depends on DAG parents, H_k, GD_k, RC_k, and F_i."""
        if departure_delay_s <= 0:
            return None

        departure = self.get_or_create("D", task_id, departure_delay_s)
        components: List[Tuple[str, str, float]] = []

        for parent_id, delay in parent_arrival_delays:
            if delay > 0:
                components.append(("U", parent_id, delay))
        if hold_duration_s > 0:
            components.append(("H", task_id, hold_duration_s))
        if queue_delay_s > 0:
            components.append(("GD", task_id, queue_delay_s))
        if resource_congestion_s > 0:
            components.append(("RC", task_id, resource_congestion_s))
        for parent_id, delay in failed_parent_delays:
            if delay > 0:
                components.append(("F", parent_id, delay))

        self._add_ratio_weighted_children(departure, components)
        for parent_id, delay in parent_arrival_delays:
            if delay > 0:
                upstream = self.get_or_create("U", parent_id, delay)
                parent_arrival = self.get_or_create("A", parent_id, delay)
                upstream.add_child(parent_arrival, 1.0)
        return departure

    def build_hold_delay_tree(
        self,
        task_id: str,
        hold_duration_s: float,
        delayed_parents: Iterable[Tuple[str, float]],
    ) -> Optional[DelayNode]:
        """Rule 3: H_k depends uniformly on delayed parent task arrivals."""
        if hold_duration_s <= 0:
            return None

        hold = self.get_or_create("H", task_id, hold_duration_s)
        candidates = [
            (parent_id, delay)
            for parent_id, delay in delayed_parents
            if 0 < delay <= hold_duration_s
        ]
        if not candidates:
            return hold

        weight = 1.0 / len(candidates)
        for parent_id, delay in candidates:
            parent_arrival = self.get_or_create("A", parent_id, delay)
            hold.add_child(parent_arrival, weight)
        return hold

    def build_from_job_states(self, job, task_states: Dict[str, object]) -> None:
        """Populate the tree from a simulator Job and its TaskState map.

        This method intentionally accepts duck-typed simulator objects so the
        reward engineering package can be imported without hard-linking to the
        simulator package path.  It reads the TaskState attributes defined in
        phase-3/simulator/models.py.
        """
        for task_id in job.tasks:
            state = task_states.get(task_id)
            if state is None:
                continue

            parents = list(job.get_parents(task_id))
            parent_delays = []
            failed_parents = []
            for parent_id in parents:
                parent_state = task_states.get(parent_id)
                if parent_state is None:
                    continue
                arrival_delay = max(0.0, getattr(parent_state, "arrival_delay_s", 0.0))
                completion_delay = max(0.0, getattr(parent_state, "completion_delay_s", 0.0))
                if arrival_delay > 0 or completion_delay > 0:
                    parent_delays.append((parent_id, max(arrival_delay, completion_delay)))
                if getattr(parent_state, "status", None) is not None:
                    status_name = getattr(parent_state.status, "name", "")
                    if status_name in {"FAILED", "EVICTED"}:
                        failed_parents.append((parent_id, max(completion_delay, arrival_delay, 1.0)))

            hold_s = max(0.0, getattr(state, "hold_duration_s", 0.0))
            departure_s = max(0.0, getattr(state, "departure_delay_s", 0.0))
            arrival_s = max(0.0, getattr(state, "arrival_delay_s", 0.0))
            queue_s = max(0.0, getattr(state, "ground_delay_s", 0.0))
            intrinsic_s = max(0.0, getattr(state, "intrinsic_delay_s", 0.0))
            resource_s = self.estimate_resource_congestion_delay(state)

            self.build_hold_delay_tree(task_id, hold_s, parent_delays)
            self.build_departure_delay_tree(
                task_id=task_id,
                departure_delay_s=departure_s,
                parent_arrival_delays=parent_delays,
                hold_duration_s=hold_s,
                queue_delay_s=queue_s,
                resource_congestion_s=resource_s,
                failed_parent_delays=failed_parents,
            )
            self.build_arrival_delay_tree(
                task_id=task_id,
                arrival_delay_s=arrival_s,
                departure_delay_s=departure_s,
                runtime_slowdown_s=intrinsic_s,
                resource_congestion_s=resource_s,
                queue_delay_s=queue_s,
            )

    def attribute_outcome(
        self,
        source_node: DelayNode,
        outcome_value: float,
        min_weight: float = 1e-6,
    ) -> Dict[str, float]:
        """Attribute an outcome to hold tasks by traversing child links."""
        attribution: Dict[str, float] = {}
        visiting = set()

        def dfs(node: DelayNode, current_weight: float) -> None:
            node_id = id(node)
            if node_id in visiting or current_weight < min_weight:
                return
            visiting.add(node_id)

            if node.type_name == "H":
                attribution[node.task_id] = (
                    attribution.get(node.task_id, 0.0)
                    + current_weight * outcome_value
                )

            for child, weight in zip(node.children, node.influence_weights):
                dfs(child, current_weight * weight)

            visiting.remove(node_id)

        dfs(source_node, 1.0)
        return attribution

    def attribute_job_outcomes(
        self,
        job,
        task_states: Dict[str, object],
        outcome_scale: float = 1.0,
    ) -> Dict[str, float]:
        """Attribute all delayed task completions in a job to hold decisions."""
        self.build_from_job_states(job, task_states)
        totals: Dict[str, float] = {}

        for task_id in job.tasks:
            state = task_states.get(task_id)
            if state is None:
                continue
            arrival_delay = max(0.0, getattr(state, "arrival_delay_s", 0.0))
            if arrival_delay <= 0:
                continue
            source = self.nodes_by_event.get(("A", task_id))
            if source is None:
                continue
            for hold_id, value in self.attribute_outcome(
                source,
                arrival_delay * outcome_scale,
            ).items():
                totals[hold_id] = totals.get(hold_id, 0.0) + value

        return totals

    def attribute_resource_congestion(
        self,
        held_task_id: str,
        blocked_task_delays: Dict[str, float],
        normalizer_s: float = 300.0,
    ) -> Dict[str, float]:
        """Attribute resource congestion caused by a held task to that hold."""
        positive = [max(0.0, delay) for delay in blocked_task_delays.values()]
        total_blockage = sum(positive)
        if total_blockage <= 0:
            return {}
        penalty = min(1.0, total_blockage / max(normalizer_s * len(positive), 1.0))
        return {held_task_id: penalty}

    @staticmethod
    def estimate_resource_congestion_delay(task_state: object) -> float:
        """Estimate RC_k from task state when the simulator lacks explicit RC."""
        departure = max(0.0, getattr(task_state, "departure_delay_s", 0.0))
        hold = max(0.0, getattr(task_state, "hold_duration_s", 0.0))
        queue = max(0.0, getattr(task_state, "ground_delay_s", 0.0))
        upstream = max(0.0, getattr(task_state, "observed_upstream_delay_s", 0.0))
        return max(0.0, departure - hold - queue - upstream)

    def _add_ratio_weighted_children(
        self,
        parent: DelayNode,
        components: Iterable[Tuple[str, str, float]],
    ) -> None:
        positive = [
            (type_name, entity_id, float(value))
            for type_name, entity_id, value in components
            if value > 0
        ]
        total = sum(value for _, _, value in positive)
        if total <= EPSILON:
            return

        for type_name, entity_id, value in positive:
            child = self.get_or_create(type_name, entity_id, value)
            parent.add_child(child, value / total)

    def to_adjacency(self) -> Dict[str, List[Tuple[str, float]]]:
        """Return a serialisable adjacency view useful for tests/debugging."""
        out: Dict[str, List[Tuple[str, float]]] = {}
        for node in self.nodes_by_event.values():
            out[node.label] = [
                (child.label, float(weight))
                for child, weight in zip(node.children, node.influence_weights)
            ]
        return out
