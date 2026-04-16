"""
Unit tests for the Logistics Delay Tree and Reward Calculator (Phase 2).

Tests mirror Phase 1's test_tree.py structure, extended with:
  1. Bay-blockage (GB) node attribution
  2. Bay-congestion cascading penalty
  3. Value-weighted local reward via TruckContext
"""

import sys
import os

# Add the phase-2 directory to path so imports resolve correctly
# (simulator/ and rewardEngineering/ live inside phase-2/)
_phase2_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _phase2_dir not in sys.path:
    sys.path.insert(0, _phase2_dir)

from rewardEngineering.delay_tree import LogisticsDelayTree
from rewardEngineering.reward_calculator import LogisticsRewardCalculator


def test_basic_delay_tree():
    """Test 1: Basic DT attribution — identical scenario to Phase 1.

    Scenario:
      - Inbound truck T_in arrives 10 min late
      - Outbound truck T_out holds 15 min for T_in
      - T_out departs with delay 20 min (Hold=15, Ground=5)
      - T_out arrives with delay 30 min (Departure=20, Road=10, BayArr=0)

    Expected attribution path:
      A (30) → D (20) → H (15)
      A weight on D = 20/30 = 2/3
      D weight on H = 15/20 = 3/4
      Path product = 2/3 * 3/4 = 0.5
      Attribution = 0.5 * (-100) = -50.0
    """
    dt = LogisticsDelayTree()

    print("=== Test 1: Basic DT (same as Phase 1) ===")

    # Incoming truck arrives delayed
    dt.get_or_create("A", "T_in", 20.0)

    # T_out holds for T_in
    dt.build_hold_delay_tree(
        truck_id="T_out",
        hold_duration=15.0,
        incoming_trucks=[("T_in", 10.0)]
    )

    # T_out departs (no bay blockage in this test)
    dt.build_departure_delay_tree(
        truck_id="T_out",
        departure_delay=20.0,
        prev_truck_id="T_prev",
        prev_arrival_delay=0.0,
        hold_duration=15.0,
        departure_ground_delay=5.0,
        bay_blockage_delay=0.0,   # no bay blockage
    )

    # T_out arrives
    A_node = dt.build_arrival_delay_tree(
        truck_id="T_out",
        arrival_delay=30.0,
        departure_delay=20.0,
        road_delay=10.0,
        arrival_bay_delay=0.0,
    )

    outcomes = dt.attribute_outcome(A_node, -100.0)
    print(f"  Attribution: {outcomes}")

    assert "T_out" in outcomes, "T_out hold should be in outcomes"
    assert abs(outcomes["T_out"] - (-50.0)) < 1e-5, \
        f"Expected -50.0, got {outcomes['T_out']}"
    print("  ✓ PASSED: Basic DT attribution matches Phase 1\n")


def test_bay_blockage_in_dt():
    """Test 2: Bay-blockage (GB) node splits attribution away from hold.

    Scenario:
      - T_out holds 15 min, but also waits 5 min for a bay (GB=5)
      - Departure delay = 25 min (Hold=15, GD=5, GB=5)
      - Arrival delay = 35 min (Departure=25, Road=10)

    Expected departure attribution:
      D (25) → H (15), GD (5), GB (5)
      H weight = 15/25 = 0.6
      GD weight = 5/25 = 0.2
      GB weight = 5/25 = 0.2

    Full path A → D → H:
      A weight on D = 25/35 = 5/7
      D weight on H = 15/25 = 3/5
      Path product = 5/7 * 3/5 = 3/7 ≈ 0.4286
      Attribution = 0.4286 * (-100) ≈ -42.86
    """
    dt = LogisticsDelayTree()

    print("=== Test 2: Bay-blockage (GB) splits attribution ===")

    # T_feeder arrives late
    dt.get_or_create("A", "T_feeder", 12.0)

    # T_out holds
    dt.build_hold_delay_tree(
        truck_id="T_out",
        hold_duration=15.0,
        incoming_trucks=[("T_feeder", 12.0)]
    )

    # T_out departs WITH bay blockage
    dt.build_departure_delay_tree(
        truck_id="T_out",
        departure_delay=25.0,
        prev_truck_id=None,
        prev_arrival_delay=0.0,
        hold_duration=15.0,
        departure_ground_delay=5.0,
        bay_blockage_delay=5.0,   # NEW: bay blockage
    )

    # T_out arrives
    A_node = dt.build_arrival_delay_tree(
        truck_id="T_out",
        arrival_delay=35.0,
        departure_delay=25.0,
        road_delay=10.0,
        arrival_bay_delay=0.0,
    )

    outcomes = dt.attribute_outcome(A_node, -100.0)
    print(f"  Attribution: {outcomes}")

    expected = (5.0/7.0) * (15.0/25.0) * (-100.0)   # ≈ -42.857
    assert "T_out" in outcomes, "T_out hold should be in outcomes"
    assert abs(outcomes["T_out"] - expected) < 0.01, \
        f"Expected {expected:.4f}, got {outcomes['T_out']:.4f}"
    print(f"  ✓ PASSED: GB node correctly diluted hold attribution from -50 to {expected:.2f}\n")


def test_bay_congestion_attribution():
    """Test 3: Bay-congestion cascading penalty.

    When T_out's hold blocks bays, downstream trucks T_x and T_y
    accumulate bay-blockage delays.  These are attributed back to T_out.
    """
    dt = LogisticsDelayTree()

    print("=== Test 3: Bay-congestion cascading ===")

    blocked = {"T_x": 8.0, "T_y": 4.0}   # 12 min total blockage
    result = dt.attribute_bay_congestion(
        held_truck_id="T_out",
        bay_blockage_delays=blocked,
    )

    print(f"  Bay congestion attribution: {result}")
    assert "T_out" in result, "T_out should receive bay congestion penalty"
    assert result["T_out"] > 0, "Penalty should be positive"
    # penalty = min(1.0, 12 / (60 * 2)) = min(1.0, 0.1) = 0.1
    expected_penalty = min(1.0, 12.0 / (60.0 * 2))
    assert abs(result["T_out"] - expected_penalty) < 1e-5, \
        f"Expected {expected_penalty}, got {result['T_out']}"
    print(f"  ✓ PASSED: Bay congestion penalty = {result['T_out']:.4f}\n")


def test_no_bay_blockage_matches_phase1():
    """Test 4: When GB=0, behaviour is identical to Phase 1.

    Verifies backward compatibility — with no bay-blockage, the
    logistics DT produces the same attribution as Phase 1's DT.
    """
    dt = LogisticsDelayTree()

    print("=== Test 4: No GB → identical to Phase 1 ===")

    dt.get_or_create("A", "T_in", 20.0)
    dt.build_hold_delay_tree(
        truck_id="T_out", hold_duration=15.0,
        incoming_trucks=[("T_in", 10.0)]
    )
    dt.build_departure_delay_tree(
        truck_id="T_out", departure_delay=20.0,
        prev_truck_id=None, prev_arrival_delay=0.0,
        hold_duration=15.0, departure_ground_delay=5.0,
        bay_blockage_delay=0.0,
    )
    A_node = dt.build_arrival_delay_tree(
        truck_id="T_out", arrival_delay=30.0,
        departure_delay=20.0, road_delay=10.0,
        arrival_bay_delay=0.0,
    )

    outcomes = dt.attribute_outcome(A_node, -100.0)
    assert abs(outcomes.get("T_out", 0) - (-50.0)) < 1e-5
    print("  ✓ PASSED: Zero GB gives Phase 1 result\n")


def test_on_time_arrests_propagation():
    """Test 5: On-time arrival (≤15 min) arrests further delay propagation."""
    dt = LogisticsDelayTree()

    print("=== Test 5: On-time arrests propagation ===")

    dt.build_hold_delay_tree(
        truck_id="T_out", hold_duration=10.0,
        incoming_trucks=[("T_in", 5.0)]
    )
    dt.build_departure_delay_tree(
        truck_id="T_out", departure_delay=12.0,
        prev_truck_id=None, prev_arrival_delay=0.0,
        hold_duration=10.0, departure_ground_delay=2.0,
        bay_blockage_delay=0.0,
    )
    # Arrival delay = 14 min → on-time (≤15), no children built
    A_node = dt.build_arrival_delay_tree(
        truck_id="T_out", arrival_delay=14.0,
        departure_delay=12.0, road_delay=2.0,
        arrival_bay_delay=0.0,
    )

    outcomes = dt.attribute_outcome(A_node, -100.0)
    print(f"  Attribution for on-time arrival: {outcomes}")
    assert len(outcomes) == 0, "On-time arrival should have no attribution"
    print("  ✓ PASSED: On-time blocks propagation\n")


def test_multi_feeder_hold():
    """Test 6: Hold caused by multiple feeder trucks — uniform split."""
    dt = LogisticsDelayTree()

    print("=== Test 6: Multiple feeders ===")

    # Three feeder trucks with delays < hold duration
    feeders = [("T_f1", 5.0), ("T_f2", 8.0), ("T_f3", 12.0)]
    dt.build_hold_delay_tree(
        truck_id="T_out", hold_duration=15.0,
        incoming_trucks=feeders,
    )
    # All three feeders have delay < 15 → S_k has 3 members
    # Each gets 1/3 influence

    H_node = dt.get_or_create("H", "T_out", 15.0)
    print(f"  H_node children: {len(H_node.children)}")
    print(f"  H_node weights: {H_node.influence_weights}")

    assert len(H_node.children) == 3, "Should have 3 feeder children"
    for w in H_node.influence_weights:
        assert abs(w - 1.0/3.0) < 1e-5, f"Each weight should be 1/3, got {w}"
    print("  ✓ PASSED: Uniform 1/3 split across 3 feeders\n")


def test_reward_calculator_local():
    """Test 7: Local reward computation via LogisticsRewardCalculator."""
    from types import SimpleNamespace

    print("=== Test 7: Local reward computation ===")

    # Mock SimConfig with the fields the reward calculator reads
    cfg = SimpleNamespace(
        alpha=0.75,
        beta=0.75,
        lambda_congestion=0.30,
        hold_actions=[0, 5, 10, 15, 20, 25, 30],
    )
    calc = LogisticsRewardCalculator(cfg)

    # Mock TruckContext with known CL and OL
    ctx = SimpleNamespace(
        truck_id="T_test",
        CL=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],   # 7 actions
        OL=[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],  # 7 actions
    )

    # Hold = 0 min → action index 0
    r0 = calc.compute_local_reward(ctx, 0.0)
    expected_0 = 0.75 * 1.0 + 0.25 * 1.0   # = 1.0
    assert abs(r0 - expected_0) < 1e-5, f"Expected {expected_0}, got {r0}"

    # Hold = 15 min → action index 3
    r3 = calc.compute_local_reward(ctx, 15.0)
    expected_3 = 0.75 * 0.7 + 0.25 * 0.85   # = 0.525 + 0.2125 = 0.7375
    assert abs(r3 - expected_3) < 1e-5, f"Expected {expected_3}, got {r3}"

    # Hold = 30 min → action index 6
    r6 = calc.compute_local_reward(ctx, 30.0)
    expected_6 = 0.75 * 0.4 + 0.25 * 0.7   # = 0.3 + 0.175 = 0.475
    assert abs(r6 - expected_6) < 1e-5, f"Expected {expected_6}, got {r6}"

    print(f"  R_L(τ=0)  = {r0:.4f} (expected {expected_0})")
    print(f"  R_L(τ=15) = {r3:.4f} (expected {expected_3})")
    print(f"  R_L(τ=30) = {r6:.4f} (expected {expected_6})")
    print("  ✓ PASSED: Local reward matches α-weighted CL/OL\n")


def test_total_reward_with_global():
    """Test 8: Total reward R_T = β·R_L + (1-β)·R_G with bay congestion."""
    from types import SimpleNamespace

    print("=== Test 8: Total reward with bay congestion ===")

    cfg = SimpleNamespace(
        alpha=0.75,
        beta=0.75,
        lambda_congestion=0.30,
        hold_actions=[0, 5, 10, 15, 20, 25, 30],
    )
    calc = LogisticsRewardCalculator(cfg)

    # Register a truck departure + arrival to create global attribution
    calc.register_truck_departure(
        truck_id="T_out",
        departure_delay=20.0,
        prev_truck_id=None,
        prev_arrival_delay=0.0,
        hold_duration=15.0,
        departure_ground_delay=5.0,
        bay_blockage_delay=0.0,
        incoming_trucks=[("T_in", 10.0)],
    )

    calc.register_truck_arrival(
        truck_id="T_out",
        arrival_delay=30.0,
        departure_delay=20.0,
        road_delay=10.0,
        arrival_bay_delay=0.0,
        arrival_cu=-0.8,    # negative = cargo suffered
        arrival_ou=-0.5,    # negative = schedule suffered
    )

    # Register bay congestion
    calc.register_bay_congestion(
        held_truck_id="T_out",
        blocked_trucks={"T_x": 10.0},
    )

    # Check global reward
    r_g = calc.get_global_reward("T_out")
    print(f"  R_G = {r_g:.4f}")

    # Check total reward using a mock context
    ctx = SimpleNamespace(
        truck_id="T_out",
        CL=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        OL=[1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7],
    )
    r_t = calc.get_total_reward(ctx, 15.0, "T_out")
    r_l = calc.compute_local_reward(ctx, 15.0)

    print(f"  R_L = {r_l:.4f}")
    print(f"  R_G = {r_g:.4f}")
    print(f"  R_T = β·R_L + (1-β)·R_G = {0.75}·{r_l:.4f} + {0.25}·{r_g:.4f} = {r_t:.4f}")

    expected_rt = 0.75 * r_l + 0.25 * r_g
    assert abs(r_t - expected_rt) < 1e-5, f"Expected {expected_rt}, got {r_t}"
    print("  ✓ PASSED: Total reward combines local + global correctly\n")


def test_reset():
    """Test 9: Reset clears all state."""
    from types import SimpleNamespace

    print("=== Test 9: Reset ===")

    cfg = SimpleNamespace(
        alpha=0.75,
        beta=0.75,
        lambda_congestion=0.30,
        hold_actions=[0, 5, 10, 15, 20, 25, 30],
    )
    calc = LogisticsRewardCalculator(cfg)
    calc.global_cu_attributed["T_out"] = 1.0
    calc.global_ou_attributed["T_out"] = 1.0
    calc.bay_congestion_attributed["T_out"] = 0.5

    calc.reset()

    assert len(calc.global_cu_attributed) == 0
    assert len(calc.global_ou_attributed) == 0
    assert len(calc.bay_congestion_attributed) == 0
    assert len(calc.delay_tree.nodes_by_event) == 0
    print("  ✓ PASSED: Reset clears all state\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  LOGISTICS REWARD ENGINEERING — UNIT TESTS")
    print("=" * 60 + "\n")

    test_basic_delay_tree()
    test_bay_blockage_in_dt()
    test_bay_congestion_attribution()
    test_no_bay_blockage_matches_phase1()
    test_on_time_arrests_propagation()
    test_multi_feeder_hold()
    test_reward_calculator_local()
    test_total_reward_with_global()
    test_reset()

    print("=" * 60)
    print("  ALL 9 TESTS PASSED ✓")
    print("=" * 60)

