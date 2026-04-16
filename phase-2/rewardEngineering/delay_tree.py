"""
Delay Tree for the Logistics Cross-Docking domain (Phase 2).

Extends the Phase 1 aviation Delay Tree (Section 5.1 of the paper) with:
  1. A new "GB" (bay-blockage) node type — represents bay_dwell_delay
     which has no aviation analog.  GB is a child of departure delay D_k
     alongside H_k (hold) and GD_k (intrinsic ground delay).
  2. Bay-congestion attribution — when a hold forces a truck to occupy a bay
     longer, downstream trucks queuing for that bay inherit part of the penalty.

Node type reference (logistics ← aviation):
  "A"  : arrival delay              ← A_i
  "D"  : departure delay            ← D_i
  "H"  : hold delay (RL action)     ← H_i
  "GD" : intrinsic departure delay  ← G^D_i (ground-time departure)
  "GA" : arrival bay dwell delay    ← G^A_i (ground-time arrival)
  "T"  : road-time delay            ← T_i   (air-time delay)
  "GB" : bay-blockage delay [NEW]   — time spent waiting for a free docking bay

Influence index ρ(X, Y) is identical to paper Section 5.1:
  - For arrival/departure: ratio of positive component to sum of positives
  - For hold delays: uniform 1/|S_k| across feeder trucks with delay < hold
  - For bay-blockage: treated identically to GD (ratio-based)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DelayNode:
    """A node in the logistics Delay Tree.

    Identical structure to Phase 1's DelayNode with the same fields.
    The only change is that type_name can now also be "GB" (bay-blockage).
    """
    type_name: str     # "A", "D", "H", "GD", "GA", "T", "GB"
    truck_id: str      # identifier of the truck this node belongs to
    value: float       # delay magnitude in minutes
    children: List['DelayNode'] = field(default_factory=list)
    influence_weights: List[float] = field(default_factory=list)

    def add_child(self, node: 'DelayNode', weight: float):
        """Add a child node with its influence weight.

        Prevents duplicate children (same object identity).
        """
        if node not in self.children:
            self.children.append(node)
            self.influence_weights.append(weight)


class LogisticsDelayTree:
    """Delay Tree for the logistics cross-docking domain.

    Extends Phase 1's DelayTree with:
      - Rule 1 (arrival): A_k depends on D_k, T_k (road delay), GA_k
        Identical to aviation.
      - Rule 2 (departure): D_k depends on A_prev (prev route leg),
        H_k (hold), GD_k (intrinsic ground), AND GB_k (bay-blockage) [NEW]
      - Rule 3 (hold): H_k depends on incoming feeder truck delays
        Identical to aviation.

    The GB_k node captures the additional departure delay caused by
    waiting for a free docking bay.  In the aviation domain, gate
    congestion is folded into GD; in logistics, bay blockage is a
    separable and significant delay source (especially at high BG).
    """

    def __init__(self):
        # Maps (type_name, truck_id) → DelayNode
        self.nodes_by_event: Dict[tuple, DelayNode] = {}

    def get_or_create(self, type_name: str, truck_id: str, value: float) -> DelayNode:
        """Retrieve existing node or create a new one.

        If the node already exists, its value is updated to reflect
        the latest observed delay.
        """
        key = (type_name, truck_id)
        if key not in self.nodes_by_event:
            node = DelayNode(type_name=type_name, truck_id=truck_id, value=value)
            self.nodes_by_event[key] = node
        else:
            self.nodes_by_event[key].value = value
        return self.nodes_by_event[key]

    # ── Rule 1: Arrival delay tree ────────────────────────────────────

    def build_arrival_delay_tree(
        self,
        truck_id: str,
        arrival_delay: float,
        departure_delay: float,
        road_delay: float,
        arrival_bay_delay: float,
    ) -> Optional[DelayNode]:
        """Rule 1: A_k depends on D_k, T_k (road), GA_k (arrival bay dwell).

        Identical to aviation Rule 1.
        If A_k <= 15 minutes (on-time), delay propagation is arrested.
        """
        if arrival_delay <= 0:
            return None

        A_node = self.get_or_create("A", truck_id, arrival_delay)

        # On-time threshold: if A_k <= 15 min, no further propagation
        if arrival_delay <= 15.0:
            return A_node

        components = [
            ("D", departure_delay),
            ("T", road_delay),
            ("GA", arrival_bay_delay),
        ]

        sum_positive = sum(max(val, 0) for _, val in components)
        if sum_positive > 0:
            for type_name, value in components:
                if value > 0:
                    child_node = self.get_or_create(type_name, truck_id, value)
                    weight = max(value, 0) / sum_positive
                    A_node.add_child(child_node, weight)

        return A_node

    # ── Rule 2: Departure delay tree (EXTENDED with GB) ───────────────

    def build_departure_delay_tree(
        self,
        truck_id: str,
        departure_delay: float,
        prev_truck_id: Optional[str],
        prev_arrival_delay: float,
        hold_duration: float,
        departure_ground_delay: float,
        bay_blockage_delay: float = 0.0,
    ) -> Optional[DelayNode]:
        """Rule 2: D_k depends on A_prev (prev route leg), H_k, GD_k, GB_k.

        EXTENSION vs Phase 1:
          Phase 1: D_k ← {A_prev, H_k, GD_k}
          Phase 2: D_k ← {A_prev, H_k, GD_k, GB_k}

        GB_k (bay-blockage delay) captures time the truck waited for a
        free docking bay before it could begin loading.  This is a
        logistics-specific delay source with no aviation analog.

        Influence is split proportionally among all positive components,
        identical to the paper's formula for Rule 1.
        """
        if departure_delay <= 0:
            return None

        D_node = self.get_or_create("D", truck_id, departure_delay)

        # Build list of delay components: (type, id, value)
        components: List[Tuple[str, str, float]] = []

        if prev_truck_id and prev_arrival_delay > 0:
            components.append(("A_prev", prev_truck_id, prev_arrival_delay))
        if hold_duration > 0:
            components.append(("H", truck_id, hold_duration))
        if departure_ground_delay > 0:
            components.append(("GD", truck_id, departure_ground_delay))
        if bay_blockage_delay > 0:
            components.append(("GB", truck_id, bay_blockage_delay))

        sum_positive = sum(max(val, 0) for _, _, val in components)
        if sum_positive > 0:
            for type_name, t_id, value in components:
                if value > 0:
                    # "A_prev" → stored as "A" node (arrival of previous truck)
                    node_type = "A" if type_name == "A_prev" else type_name
                    child_node = self.get_or_create(node_type, t_id, value)
                    weight = max(value, 0) / sum_positive
                    D_node.add_child(child_node, weight)

        return D_node

    # ── Rule 3: Hold delay tree ───────────────────────────────────────

    def build_hold_delay_tree(
        self,
        truck_id: str,
        hold_duration: float,
        incoming_trucks: List[Tuple[str, float]],
    ) -> Optional[DelayNode]:
        """Rule 3: H_k depends on incoming feeder truck delays A_ki.

        Identical to aviation Rule 3.
        S_k = {A_ki | A_ki < H_k} (feeder delays less than the hold).
        Influence is uniform: 1/|S_k| for each feeder in S_k.
        """
        if hold_duration <= 0:
            return None

        H_node = self.get_or_create("H", truck_id, hold_duration)

        # S_k: feeder trucks whose delay is strictly less than hold duration
        s_k = [
            (t_id, delay)
            for t_id, delay in incoming_trucks
            if delay < hold_duration and delay > 0
        ]

        if len(s_k) > 0:
            weight = 1.0 / len(s_k)
            for t_id, delay in s_k:
                child_node = self.get_or_create("A", t_id, delay)
                H_node.add_child(child_node, weight)

        return H_node

    # ── Reward attribution via DFS ────────────────────────────────────

    def attribute_outcome(
        self,
        source_node: DelayNode,
        outcome_value: float,
    ) -> Dict[str, float]:
        """Attribute an outcome value to past hold decisions via the DT.

        Walks the tree from source_node (e.g., an arrival delay) down
        to hold nodes ("H"), multiplying influence weights along the
        path.  Returns {truck_id: attributed_value} for each hold found.

        This is identical to Phase 1's attribute_outcome — the DFS logic
        is domain-agnostic; only the tree structure differs.
        """
        attribution: Dict[str, float] = {}
        visited = set()

        def dfs(node: DelayNode, current_weight: float):
            node_id = id(node)
            if node_id in visited or current_weight < 1e-6:
                return

            visited.add(node_id)

            # Terminal condition: found a hold decision node
            if node.type_name == "H":
                if node.truck_id not in attribution:
                    attribution[node.truck_id] = 0.0
                attribution[node.truck_id] += current_weight * outcome_value

            # Recurse into children
            for child, weight in zip(node.children, node.influence_weights):
                dfs(child, current_weight * weight)

            visited.remove(node_id)

        dfs(source_node, 1.0)
        return attribution

    # ── Bay-congestion cascading attribution [NEW] ────────────────────

    def attribute_bay_congestion(
        self,
        held_truck_id: str,
        bay_blockage_delays: Dict[str, float],
    ) -> Dict[str, float]:
        """Attribute bay-congestion penalties back to the hold that caused them.

        When truck T_k is held for H_k minutes, it occupies a docking bay
        for H_k extra minutes.  Other trucks (T_j) that queue for that bay
        accumulate bay_dwell_delay (GB_j).  We attribute these GB_j penalties
        back to the hold H_k that caused the bay blockage.

        Args:
            held_truck_id: The truck whose hold caused bay congestion.
            bay_blockage_delays: {truck_id: GB_j minutes} for trucks that
                were blocked due to this hold.

        Returns:
            {held_truck_id: total attributed bay-congestion penalty}
            The penalty is a normalised sum of blockage delays.
        """
        if not bay_blockage_delays:
            return {}

        # Sum all bay-blockage delays caused by this hold
        total_blockage = sum(bay_blockage_delays.values())

        # Normalise: each minute of bay blockage contributes 1/Δ_F penalty
        # (using Δ_F = 60 min as the operator delay normaliser)
        DELTA_F = 60.0
        penalty = min(1.0, total_blockage / (DELTA_F * max(1, len(bay_blockage_delays))))

        return {held_truck_id: penalty}
