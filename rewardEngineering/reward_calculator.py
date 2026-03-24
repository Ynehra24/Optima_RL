from typing import Dict, List
import numpy as np

from simulator.config import SimConfig
from simulator.context_engine import FlightContext
from rewardEngineering.delay_tree import DelayTree, DelayNode

class RewardCalculator:
    """
    Implements the Reward Engineering logic from Section 5 of "To hold or not to hold?".
    Calculates Local (RL) and Global (RG) rewards, and combines them into Total Reward (RT).
    """
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.alpha = cfg.alpha  # Knob for PU-AU trade-off
        self.beta = getattr(cfg, 'beta', 0.5)  # Knob for local-global trade-off (assuming it might be in config, otherwise default 0.5)
        self.delay_tree = DelayTree()

        # Track global utilities per flight
        self.global_pu_attributed: Dict[str, float] = {}
        self.global_au_attributed: Dict[str, float] = {}

    def compute_local_reward(self, ctx: FlightContext, hold_minutes: float) -> float:
        """
        R_L = alpha * P_L + (1 - alpha) * A_L
        """
        action_idx = self.cfg.hold_actions.index(int(hold_minutes))
        
        # Protect against out-of-bounds if config mismatches
        local_pu = ctx.PL[action_idx] if action_idx < len(ctx.PL) else 0.5
        local_au = ctx.AL[action_idx] if action_idx < len(ctx.AL) else 0.5
        
        return self.alpha * local_pu + (1 - self.alpha) * local_au

    def register_flight_arrival(
        self,
        flight_id: str,
        arrival_delay: float,
        departure_delay: float,
        air_time_delay: float,
        arrival_ground_delay: float,
        arrival_pu: float,
        arrival_au: float
    ):
        """
        Called when a flight actually arrives. Updates the delay tree and attributes 
        the global PU and AU to past hold actions.
        """
        # Rule 1: Build arrival delay tree node
        A_node = self.delay_tree.build_arrival_delay_tree(
            flight_id=flight_id,
            arrival_delay=arrival_delay,
            departure_delay=departure_delay,
            air_time_delay=air_time_delay,
            arrival_ground_delay=arrival_ground_delay
        )
        
        if A_node:
            # Attribute the (negative) consequence of this delay to past holds
            # The utilities are usually measured such that higher is better. 
            # If the flight is delayed, the arriving PU and AU will reflect losses.
            
            attributed_pu = self.delay_tree.attribute_outcome(A_node, arrival_pu)
            attributed_au = self.delay_tree.attribute_outcome(A_node, arrival_au)
            
            for f_id, pu_val in attributed_pu.items():
                self.global_pu_attributed[f_id] = self.global_pu_attributed.get(f_id, 0.0) + pu_val
                
            for f_id, au_val in attributed_au.items():
                self.global_au_attributed[f_id] = self.global_au_attributed.get(f_id, 0.0) + au_val

    def register_flight_departure(
        self,
        flight_id: str,
        departure_delay: float,
        prev_flight_id: str,
        prev_arrival_delay: float,
        hold_duration: float,
        departure_ground_delay: float,
        incoming_flights: List[tuple]
    ):
        """
        Called when a flight departs. Constructs the departure and hold tree nodes.
        incoming_flights should strictly represent delays of feeding flights: list of (f_id, delay)
        """
        # Rule 3: Hold depends on incoming passenger flights
        if hold_duration > 0:
            self.delay_tree.build_hold_delay_tree(
                flight_id=flight_id,
                hold_duration=hold_duration,
                incoming_flights=incoming_flights
            )
            
        # Rule 2: Departure depends on past tail arrival, hold and ground delay
        self.delay_tree.build_departure_delay_tree(
            flight_id=flight_id,
            departure_delay=departure_delay,
            prev_flight_id=prev_flight_id,
            prev_arrival_delay=prev_arrival_delay,
            hold_duration=hold_duration,
            departure_ground_delay=departure_ground_delay
        )

    def get_global_reward(self, flight_id: str) -> float:
        """
        R_G = alpha * P_G + (1 - alpha) * A_G
        Estimates the global penalty attributed to the hold decision of this flight.
        Ideally called at the end of the simulation or after a window W.
        """
        pg = self.global_pu_attributed.get(flight_id, 0.0)
        ag = self.global_au_attributed.get(flight_id, 0.0)
        
        rg = self.alpha * pg + (1 - self.alpha) * ag
        return rg

    def get_total_reward(self, ctx: FlightContext, hold_minutes: float, flight_id: str) -> float:
        """
        R_T = beta * R_L + (1 - beta) * R_G
        Note: The true R_G might only be known in the future.
        """
        r_l = self.compute_local_reward(ctx, hold_minutes)
        r_g = self.get_global_reward(flight_id)
        
        return self.beta * r_l + (1 - self.beta) * r_g
