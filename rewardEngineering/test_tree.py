from typing import Dict
from rewardEngineering.delay_tree import DelayTree

def test_delay_tree():
    dt = DelayTree()
    
    print("--- Testing Delay Tree Logic ---")
    
    # 1. Incoming flight F_in arrives delayed
    # Normally this would be built when F_in arrives, but we'll mock it.
    f_in_arrival = dt.get_or_create("A", "F_in", 20.0)
    
    # 2. Flight F_out holds for F_in
    # hold duration = 15.0. Incoming flights = [("F_in", 20.0)]
    # Rule 3 says S_i = {A_ik | A_ik < H_i}. Here 20.0 < 15.0 is False, wait. 
    # The paper rule says A_ik < H_i which actually means delays less than hold!
    # Let's verify our code logic:
    # "s_i = [(f_id, delay) for f_id, delay in incoming_flights if delay < hold_duration ...]"
    # So if delay = 10, hold = 15, S_i is valid.
    H_node = dt.build_hold_delay_tree(
        flight_id="F_out",
        hold_duration=15.0,
        incoming_flights=[("F_in", 10.0)]
    )
    
    # 3. Flight F_out departs
    # departure_delay = 20.0 (Hold=15, Ground=5)
    D_node = dt.build_departure_delay_tree(
        flight_id="F_out",
        departure_delay=20.0,
        prev_flight_id="F_prev",
        prev_arrival_delay=0.0,
        hold_duration=15.0,
        departure_ground_delay=5.0
    )
    
    # 4. Flight F_out arrives
    # arrival_delay = 30.0 (Departure=20, AirTime=10, GroundArr=0)
    A_node = dt.build_arrival_delay_tree(
        flight_id="F_out",
        arrival_delay=30.0,
        departure_delay=20.0,
        air_time_delay=10.0,
        arrival_ground_delay=0.0
    )
    
    print("Nodes built successfully.")
    
    # Now, try to attribute the outcome of A_node
    # Let's say Arrival lost -100 AU due to delay.
    outcomes = dt.attribute_outcome(A_node, -100.0)
    print(f"Outcome attribution for F_out Arrival (-100 AU): {outcomes}")
    
    # Calculation check:
    # A_i = 30. Depends on D_i=20, T=10, GA=0. Sum=30. 
    # D_i weight = 20/30 = 2/3.
    # D_i = 20. Depends on H=15, GD=5. Sum=20.
    # H weight = 15/20 = 3/4.
    # Path A -> D -> H  = (2/3) * (3/4) = 6/12 = 0.5.
    # So F_out's hold should take 0.5 * (-100) = -50.0.
    
    assert "F_out" in outcomes, "F_out hold should be in outcomes"
    assert abs(outcomes["F_out"] - (-50.0)) < 1e-5, f"Expected -50, got {outcomes['F_out']}"
    print("SUCCESS: Delay Tree properly attributed exact causal fraction to the hold!")

if __name__ == '__main__':
    test_delay_tree()
