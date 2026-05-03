"""
delay_tree.py — FIXED VERSION
==============================
Implements delay attribution from Section 5.1 of the paper.

Key fix: Properly trace causality from arrival delays → departure delays → holds.

Debug output shows:
  - Which flight nodes are created and why
  - How attribution weights are computed
  - Which holds are responsible for which outcomes
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class DelayNode:
    """Represents a node in the Delay Tree (DT) as defined in Section 5.1."""
    type_name: str     # "A", "D", "H", "GD", "GA", "T"
    flight_id: str
    value: float       # delay in minutes
    children: List['DelayNode'] = field(default_factory=list)
    influence_weights: List[float] = field(default_factory=list)

    def add_child(self, node: 'DelayNode', weight: float):
        if node not in self.children:
            self.children.append(node)
            self.influence_weights.append(weight)

    def __repr__(self):
        return f"DelayNode({self.type_name}_{self.flight_id}, val={self.value:.1f})"


class DelayTree:
    """
    Implements the Delay Tree (DT) logic from Section 5.1.
    Used to attribute global delays (AU) and PAX misses (PU) across past hold decisions.
    
    Key logic:
      Rule 1: A_i (arrival delay) depends on D_i (departure), T_i (air time), GA_i (ground)
      Rule 2: D_i (departure delay) depends on A_j (prev tail arrival), H_i (hold), GD_i (ground)
      Rule 3: H_i (hold) depends on incoming flight arrivals that triggered the hold
    """
    
    def __init__(self, debug: bool = True):
        self.nodes_by_event: Dict[tuple, DelayNode] = {}
        self.debug = debug
        self.log_lines = []
    
    def _log(self, msg: str):
        """Internal logging."""
        if self.debug:
            self.log_lines.append(msg)
            # print(f"[DelayTree] {msg}")
    
    def get_or_create(self, type_name: str, flight_id: str, value: float) -> DelayNode:
        key = (type_name, flight_id)
        if key not in self.nodes_by_event:
            node = DelayNode(type_name=type_name, flight_id=flight_id, value=value)
            self.nodes_by_event[key] = node
            self._log(f"Created node: {type_name}({flight_id}) = {value:.1f} min")
        else:
            old_val = self.nodes_by_event[key].value
            self.nodes_by_event[key].value = value
            if old_val != value:
               self._log(f"Updated node: {type_name}({flight_id}) = {old_val:.1f} → {value:.1f}")
        return self.nodes_by_event[key]

    def build_arrival_delay_tree(
        self, 
        flight_id: str, 
        arrival_delay: float, 
        departure_delay: float, 
        air_time_delay: float, 
        arrival_ground_delay: float
    ) -> Optional[DelayNode]:
        """
        Rule 1: A_i depends on D_i, T_i, GA_i
        
        If A_i <= 15 minutes (OTP threshold), the delay is "arrested" and
        doesn't propagate further.
        """
        if arrival_delay <= 0:
            return None
        
       # self._log(f"\n=== ARRIVAL {flight_id} ===")
       # self._log(f"A_i = {arrival_delay:.1f} min (composed of D={departure_delay:.1f}, T={air_time_delay:.1f}, GA={arrival_ground_delay:.1f})")
            
        A_node = self.get_or_create("A", flight_id, arrival_delay)
        
        # If on-time, don't build dependency tree
        if arrival_delay <= 15.0:
            self._log(f"  → ON-TIME (≤ 15 min), no propagation")
            return A_node
        
        self._log(f"  → DELAYED (> 15 min), building component tree")
        
        # Components that contributed to arrival delay
        components = [
            ("D", departure_delay),
            ("T", air_time_delay),
            ("GA", arrival_ground_delay)
        ]
        
        # Sum of positive components
        sum_positive = sum(max(val, 0) for _, val in components)
        
        if sum_positive > 0:
            for type_name, value in components:
                if value > 0:
                    child_node = self.get_or_create(type_name, flight_id, value)
                    weight = max(value, 0) / sum_positive
                    A_node.add_child(child_node, weight)
                    self._log(f"  Added dependency: {type_name}({flight_id}) → A({flight_id}) [weight={weight:.3f}]")
        
        return A_node

    def build_departure_delay_tree(
        self,
        flight_id: str,
        departure_delay: float,
        prev_flight_id: Optional[str],
        prev_arrival_delay: float,
        hold_duration: float,
        departure_ground_delay: float
    ) -> Optional[DelayNode]:
        """
        Rule 2: D_i depends on A_j (prev tail flight), H_i (hold), GD_i (ground delay)
        
        This is critical: departure delay is attributed to:
          1. Previous tail flight's arrival delay (if any)
          2. The hold decision we made (H_i)
          3. Ground delay variability
        """
        if departure_delay <= 0:
            return None
        
        self._log(f"\n=== DEPARTURE {flight_id} ===")
        self._log(f"D_i = {departure_delay:.1f} min")
        if prev_flight_id:
            self._log(f"  Prev tail: {prev_flight_id} (arrival_delay={prev_arrival_delay:.1f})")
        self._log(f"  Hold: {hold_duration:.1f} min")
        self._log(f"  Ground delay: {departure_ground_delay:.1f} min")
        
        D_node = self.get_or_create("D", flight_id, departure_delay)
        
        # Components contributing to departure delay
        components: List[Tuple[str, str, float]] = []
        
        if prev_flight_id and prev_arrival_delay > 0:
            components.append(("A_prev", prev_flight_id, prev_arrival_delay))
        if hold_duration > 0:
            components.append(("H", flight_id, hold_duration))
        if departure_ground_delay > 0:
            components.append(("GD", flight_id, departure_ground_delay))
        
        sum_positive = sum(max(val, 0) for _, _, val in components)
        
        if sum_positive > 0:
            for type_name, f_id, value in components:
                if value > 0:
                    # For prev arrival, reference the existing A node
                    if type_name == "A_prev":
                        child_node = self.get_or_create("A", f_id, value)
                    else:
                        child_node = self.get_or_create(type_name, f_id, value)
                    
                    weight = max(value, 0) / sum_positive
                    D_node.add_child(child_node, weight)
                    self._log(f"  Added dependency: {type_name}({f_id}) → D({flight_id}) [weight={weight:.3f}]")
        
        return D_node

    def build_hold_delay_tree(
        self,
        flight_id: str,
        hold_duration: float,
        incoming_flights: List[Tuple[str, float]]  # list of (incoming_flight_id, arrival_delay)
    ) -> Optional[DelayNode]:
        """
        Rule 3: H_i (hold) depends on incoming flight arrivals that triggered it
        
        The hold is justified if there are connecting PAX from late arrivals.
        This node tracks which arrival delays caused this hold.
        """
        if hold_duration <= 0:
            return None
        
        self._log(f"\n=== HOLD {flight_id} ===")
        self._log(f"H_i = {hold_duration:.1f} min")
        
        H_node = self.get_or_create("H", flight_id, hold_duration)
        
        # Incoming flights that are late (A_ik < H_i per paper)
        # These are the flights we're waiting for
        s_i = [(f_id, delay) for f_id, delay in incoming_flights 
               if 0 < delay < hold_duration]
        
        if len(s_i) > 0:
            self._log(f"  Incoming flights triggering hold (delay < {hold_duration:.1f}): {len(s_i)}")
            weight = 1.0 / len(s_i)
            for f_id, delay in s_i:
                child_node = self.get_or_create("A", f_id, delay)
                H_node.add_child(child_node, weight)
                self._log(f"  Added dependency: A({f_id}, delay={delay:.1f}) → H({flight_id}) [weight={weight:.3f}]")
        else:
            self._log(f"  No incoming flights with delay < {hold_duration:.1f}, hold not fully justified")
        
        return H_node

    def attribute_outcome(self, source_node: DelayNode, outcome_value: float) -> Dict[str, float]:
        """
        Trace from an outcome (arrival delay, missed connection) back through the 
        delay tree to find which HOLDS are responsible.
        
        Returns: {flight_id: attributed_reward} where flight_id is the flight that was held.
        
        Example:
          - Flight arrives 25 minutes late (A node)
          - A depends on D (departure delay 20 min)
          - D depends on H (hold 15 min) and prev_arrival (5 min)
          - H depends on incoming A_k (12 min)
          - Then: output = {flight_id: 0.25 * outcome_value} (portion of delay attributed to this hold)
        """
        attribution: Dict[str, float] = {}
        visited_ids = set()
        
        def dfs(node: DelayNode, current_weight: float, depth: int = 0):
            """Recursive DFS to trace blame back through delay tree."""
            indent = "  " * depth
            node_id = id(node)
            
            # Cycle prevention
            if node_id in visited_ids or current_weight < 1e-6:
                return
            
            visited_ids.add(node_id)
            
            # Found a hold! Attribute this portion of outcome to it
            if node.type_name == "H":
                if node.flight_id not in attribution:
                    attribution[node.flight_id] = 0.0
                attr_val = current_weight * outcome_value
                attribution[node.flight_id] += attr_val
                self._log(f"{indent}→ Hold {node.flight_id}: +{attr_val:.6f} (weight={current_weight:.3f})")
            
            # Traverse to children (go deeper in causality)
            for child, weight in zip(node.children, node.influence_weights):
                child_weight = current_weight * weight
                dfs(child, child_weight, depth + 1)
            
            visited_ids.remove(node_id)
        
        self._log(f"\n=== ATTRIBUTING {source_node} with outcome={outcome_value:.6f} ===")
        dfs(source_node, 1.0)
        
        if not attribution:
            self._log(f"  → No holds found in causality tree (or all zero-weighted)")
        else:
            self._log(f"  → Total attributed to holds: {sum(attribution.values()):.6f}")
        
        return attribution