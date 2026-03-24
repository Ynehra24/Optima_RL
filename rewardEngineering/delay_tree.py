from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

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

class DelayTree:
    """
    Implements the Delay Tree (DT) logic from Section 5.1 of "To hold or not to hold?".
    Used to attribute global delays (AU) and PAX misses (PU) across past hold decisions.
    """
    def __init__(self):
        # Maps (type_name, flight_id) to DelayNode
        self.nodes_by_event: Dict[tuple, DelayNode] = {}
        
    def get_or_create(self, type_name: str, flight_id: str, value: float) -> DelayNode:
        key = (type_name, flight_id)
        if key not in self.nodes_by_event:
            node = DelayNode(type_name=type_name, flight_id=flight_id, value=value)
            self.nodes_by_event[key] = node
        else:
            self.nodes_by_event[key].value = value
        return self.nodes_by_event[key]

    def build_arrival_delay_tree(
        self, 
        flight_id: str, 
        arrival_delay: float, 
        departure_delay: float, 
        air_time_delay: float, 
        arrival_ground_delay: float
    ) -> Optional[DelayNode]:
        """Rule 1: A_i depends on D_i, T_i, GA_i."""
        if arrival_delay <= 0:
            return None
            
        A_node = self.get_or_create("A", flight_id, arrival_delay)
        
        # Calculate influence index for Rule 1: rho(X_i, A_i)
        # If A_i <= 15 minutes, rho = 0, propagated delay is arrested.
        if arrival_delay <= 15.0:
            return A_node
            
        components = [
            ("D", departure_delay),
            ("T", air_time_delay),
            ("GA", arrival_ground_delay)
        ]
        
        sum_positive = sum(max(val, 0) for _, val in components)
        if sum_positive > 0:
            for type_name, value in components:
                if value > 0:
                    child_node = self.get_or_create(type_name, flight_id, value)
                    weight = max(value, 0) / sum_positive
                    A_node.add_child(child_node, weight)
                    
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
        """Rule 2: D_i depends on A_j (prev tail), H_i, GD_i."""
        if departure_delay <= 0:
            return None
            
        D_node = self.get_or_create("D", flight_id, departure_delay)
        
        # Calculate influence index for Rule 2: rho(X_i, D_i) uses same logic as Rule 1
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
                    node_type = "A" if type_name == "A_prev" else type_name
                    child_node = self.get_or_create(node_type, f_id, value)
                    weight = max(value, 0) / sum_positive
                    D_node.add_child(child_node, weight)
                    
        return D_node

    def build_hold_delay_tree(
        self,
        flight_id: str,
        hold_duration: float,
        incoming_flights: List[Tuple[str, float]] # list of (incoming_flight_id, arrival_delay)
    ) -> Optional[DelayNode]:
        """Rule 3: H_i depends on incoming flight delays A_ik."""
        if hold_duration <= 0:
            return None
            
        H_node = self.get_or_create("H", flight_id, hold_duration)
        
        # S_i = {A_ik | A_ik < H_i}
        # In practice, comparing strictly '<' might miss flights we hold exactly for, 
        # but adhering strictly to the paper text "A_ik < H_i".
        # Let's consider delays that influenced the hold.
        s_i = [(f_id, delay) for f_id, delay in incoming_flights if delay < hold_duration and delay > 0]
        
        if len(s_i) > 0:
            weight = 1.0 / len(s_i)
            for f_id, delay in s_i:
                child_node = self.get_or_create("A", f_id, delay)
                H_node.add_child(child_node, weight)
                
        return H_node

    def attribute_outcome(self, source_node: DelayNode, outcome_value: float) -> Dict[str, float]:
        """
        Calculates how much of outcome_value (e.g. PU, AU lost due to A_i) 
        is attributed to past holds. Returns a dictionary of {flight_id: attributed_value}.
        Where flight_id is the flight that was held (H).
        """
        attribution: Dict[str, float] = {}
        visited = set()
        
        def dfs(node: DelayNode, current_weight: float):
            # cycle prevention and stop early if weight is negligible
            node_id = id(node)
            if node_id in visited or current_weight < 1e-6:
                return
                
            visited.add(node_id)
            
            if node.type_name == "H":
                if node.flight_id not in attribution:
                    attribution[node.flight_id] = 0.0
                attribution[node.flight_id] += current_weight * outcome_value
            
            # DFS traversal down the delay tree
            for child, weight in zip(node.children, node.influence_weights):
                dfs(child, current_weight * weight)
                
            visited.remove(node_id)
                
        dfs(source_node, 1.0)
        return attribution
