"""
Validation utilities for the logistics cross-docking simulator.

Mirrors simulator/validation.py — same three axes:
  1. Cargo connectivity matrix (replaces PAX connectivity)
  2. Failed transfer rate validation (replaces missed-connection check)
  3. Network delay histograms + OTP by route type (same structure)

New: validate_four_metrics() — reports all 4 logistics metrics prominently.
New: validate_against_faf5() — compares simulator outputs to FAF5 ground truth.
"""

from __future__ import annotations
import sys, os
# Make `phase2_simulator` importable when this file is run directly from any directory:
#   python /path/to/phase2_simulator/validation.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from phase2_simulator.config import SimConfig
from phase2_simulator.simulator import CrossDockSimulator


def validate_cargo_connectivity(sim: CrossDockSimulator) -> Dict[str, Any]:
    """Compute cargo connectivity matrix across the hub network.
    Mirrors validate_pax_connectivity() from simulator/validation.py.
    """
    connectivity: Dict[tuple, int] = defaultdict(int)
    for cu in sim.cargo.values():
        if len(cu.legs) >= 2:
            first_ts = sim.trucks.get(cu.legs[0])
            last_ts  = sim.trucks.get(cu.legs[-1])
            if first_ts and last_ts:
                origin = first_ts.truck.origin_hub
                dest   = last_ts.truck.dest_hub
                connectivity[(origin, dest)] += cu.unit_count

    top_pairs = sorted(connectivity.items(), key=lambda x: -x[1])
    return {
        "matrix": dict(connectivity),
        "top_pairs": top_pairs[:20],
        "total_connecting_cargo_units": sum(connectivity.values()),
    }


def validate_failed_transfers(sim: CrossDockSimulator) -> Dict[str, Any]:
    """Validate failed-transfer rates against expected baseline.
    Mirrors validate_missed_connections() from simulator/validation.py.
    """
    m = sim.metrics
    expected = sim.cfg.hub.baseline_failed_transfer_rate
    rel_error = (abs(m.failed_transfer_rate - expected) / expected
                 if expected > 0 else 0.0)
    return {
        "total_connecting_cargo": m.connecting_cargo_units,
        "failed_transfers": m.failed_transfers,
        "successful_transfers": m.successful_transfers,
        "simulated_failed_rate": round(m.failed_transfer_rate, 4),
        "expected_failed_rate": expected,
        "relative_error": round(rel_error, 4),
        "priority_failed (X_k=2)": m.priority_failed,
        "daily_failed": dict(m.daily_failed),
    }


def validate_network_delays(sim: CrossDockSimulator) -> Dict[str, Any]:
    """Validate delay distributions and OTP by route type.
    Mirrors validate_network_delays() from simulator/validation.py.
    OTP by route type now correctly includes DELIVERED trucks (bug fixed).
    """
    m = sim.metrics
    bins   = [15, 30, 45, 60, float("inf")]
    labels = ["<=15", "16-30", "31-45", "46-60", ">60"]

    def histogram(delays: List[float]) -> Dict[str, float]:
        if not delays:
            return {lbl: 0.0 for lbl in labels}
        counts = {lbl: 0 for lbl in labels}
        for d in delays:
            for i, upper in enumerate(bins):
                if d <= upper:
                    counts[labels[i]] += 1
                    break
        total = len(delays)
        return {lbl: round(c / total * 100, 1) for lbl, c in counts.items()}

    # OTP by route type — DEPARTED and DELIVERED (bug-fixed)
    otp_by_route: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"ontime": 0, "total": 0})
    for ts in sim.trucks.values():
        if ts.status.name in ("DEPARTED", "DELIVERED"):
            rt = ts.truck.route_type
            otp_by_route[rt]["total"] += 1
            if ts.departure_delay_D <= sim.cfg.ontime_threshold:
                otp_by_route[rt]["ontime"] += 1

    otp_pct = {rt: round(v["ontime"] / max(1, v["total"]) * 100, 2)
               for rt, v in otp_by_route.items()}

    return {
        "OTP_overall_%":           round(m.schedule_otp * 100, 2),
        "OTP_by_route_type":       otp_pct,
        "expected_OTP_%":          sim.cfg.hub.baseline_schedule_otp * 100,
        "avg_departure_delay_min": round(m.avg_departure_delay, 2),
        "avg_delivery_delay_min":  round(m.avg_delivery_delay, 2),
        "avg_bay_utilisation_%":   round(m.avg_bay_utilisation * 100, 2),
        "departure_delay_hist":    histogram(m.departure_delays),
        "arrival_delay_hist":      histogram(m.arrival_delays),
    }


def validate_four_metrics(sim: CrossDockSimulator) -> Dict[str, Any]:
    """Report all four logistics metrics prominently.

    The aviation simulator had 2 metrics (OTP + missed connections).
    The logistics simulator has 4:
      1. Schedule OTP (same concept)
      2. Failed transfer % (replaces missed connections %)
      3. Avg delivery delay (A_k mean — replaces avg arrival delay)
      4. Bay congestion % (BG mean — NEW, no aviation analog)
    """
    m = sim.metrics
    return {
        "METRIC_1_schedule_OTP_%":        round(m.schedule_otp * 100, 2),
        "METRIC_2_failed_transfer_%":     round(m.failed_transfer_rate * 100, 2),
        "METRIC_3_avg_delivery_delay_min": round(m.avg_delivery_delay, 2),
        "METRIC_4_bay_congestion_%":       round(m.avg_bay_utilisation * 100, 2),
    }


def validate_against_faf5(sim: CrossDockSimulator) -> Dict[str, Any]:
    """Compare simulator route distributions to FAF5 ground truth.

    FAF5 NTAD (487,394 US road links) provides:
      Route duration distribution: 55% short, 35% medium, 10% long (for cross-docking)
      Speed mean = 43.2 mph, speed_std = 6.7 mph
      Road delay range: 0-90 min (from speed variance over typical route lengths)

    Each check returns True (PASS) or False (FAIL).
    """
    # Count route types in the simulation
    route_counts = defaultdict(int)
    for ts in sim.trucks.values():
        route_counts[ts.truck.route_type] += 1
    total = max(1, sum(route_counts.values()))

    short_pct  = route_counts.get("short",  0) / total * 100
    medium_pct = route_counts.get("medium", 0) / total * 100
    long_pct   = route_counts.get("long",   0) / total * 100

    # FAF5 targets
    faf5_short  = 55.0
    faf5_medium = 35.0
    faf5_long   = 10.0
    tolerance   = 10.0   # ±10 percentage points acceptable

    short_ok  = abs(short_pct  - faf5_short)  <= tolerance
    medium_ok = abs(medium_pct - faf5_medium) <= tolerance
    long_ok   = abs(long_pct   - faf5_long)   <= tolerance

    # Road delay range check (should be within FAF5-calibrated ±90 min)
    road_delays = [ts.road_delay for ts in sim.trucks.values()]
    mean_road  = float(np.mean(road_delays)) if road_delays else 0.0
    std_road   = float(np.std(road_delays)) if road_delays else 0.0
    # FAF5 implies road_delay_sigma ≈ 10 min for medium routes; check within factor 2
    road_std_ok = 5.0 <= std_road <= 20.0

    all_pass = short_ok and medium_ok and long_ok and road_std_ok
    return {
        "overall_pass": all_pass,
        "route_distribution": {
            "short_%":    round(short_pct,  1),
            "medium_%":   round(medium_pct, 1),
            "long_%":     round(long_pct,   1),
            "faf5_target_short_%":  faf5_short,
            "faf5_target_medium_%": faf5_medium,
            "faf5_target_long_%":   faf5_long,
            "short_pass":  short_ok,
            "medium_pass": medium_ok,
            "long_pass":   long_ok,
        },
        "road_delay_calibration": {
            "mean_road_delay_min": round(mean_road, 2),
            "std_road_delay_min":  round(std_road, 2),
            "faf5_implied_std_min": 10.0,
            "road_std_pass": road_std_ok,
        },
    }


def run_full_validation(cfg: SimConfig | None = None, policy: str = "no_hold",
                        seed: int = 42, verbose: bool = True) -> Dict[str, Any]:
    """Run full validation. Mirrors run_full_validation() from simulator/validation.py."""
    if cfg is None:
        cfg = SimConfig(random_seed=seed)

    sim = CrossDockSimulator(cfg)
    sim.run_episode(policy=policy, seed=seed)

    four    = validate_four_metrics(sim)
    conn    = validate_cargo_connectivity(sim)
    failed  = validate_failed_transfers(sim)
    delays  = validate_network_delays(sim)
    faf5    = validate_against_faf5(sim)

    results = {
        "hub": cfg.hub.name,
        "policy": policy,
        "num_days": cfg.num_days,
        "four_metrics": four,
        "cargo_connectivity": conn,
        "failed_transfers": failed,
        "network_delays": delays,
        "faf5_validation": faf5,
        "full_summary": sim.metrics.summary(),
    }

    if verbose:
        _print_validation(results)

    return results


def _print_validation(results: Dict[str, Any]) -> None:
    print("=" * 70)
    print(f"  SIMULATION VALIDATION — {results['hub']}")
    print(f"  Policy: {results['policy']}  |  Days: {results['num_days']}")
    print("=" * 70)

    four = results["four_metrics"]
    print(f"\n{'─'*70}")
    print("  THE FOUR LOGISTICS METRICS")
    print(f"{'─'*70}")
    print(f"  Metric 1 — Schedule OTP:         {four['METRIC_1_schedule_OTP_%']}%")
    print(f"  Metric 2 — Failed Transfer Rate: {four['METRIC_2_failed_transfer_%']}%")
    print(f"  Metric 3 — Avg Delivery Delay:   {four['METRIC_3_avg_delivery_delay_min']} min")
    print(f"  Metric 4 — Bay Congestion:       {four['METRIC_4_bay_congestion_%']}%")

    s = results["full_summary"]
    print(f"\n--- Truck Operations ---")
    print(f"  Total trucks:       {s['total_trucks']}")
    print(f"  Docked:             {s['docked_trucks']}")
    print(f"  Departed:           {s['departed_trucks']}")
    print(f"  Avg hold:           {s['avg_hold_min']} min")

    ft = results["failed_transfers"]
    print(f"\n--- Transfer Analysis ---")
    print(f"  Connecting cargo:   {ft['total_connecting_cargo']}")
    print(f"  Successful:         {ft['successful_transfers']}")
    print(f"  Failed:             {ft['failed_transfers']}")
    print(f"  Simulated rate:     {ft['simulated_failed_rate']*100:.2f}%  "
          f"(expected {ft['expected_failed_rate']*100:.1f}%)")
    print(f"  Priority failed:    {ft['priority_failed (X_k=2)']}")

    dd = results["network_delays"]
    print(f"\n--- Network Delays ---")
    print(f"  OTP:                {dd['OTP_overall_%']}%  (expected {dd['expected_OTP_%']}%)")
    print(f"  OTP by route type:  {dd['OTP_by_route_type']}")
    print(f"  Dep delay hist:     {dd['departure_delay_hist']}")

    faf5 = results["faf5_validation"]
    rd   = faf5["route_distribution"]
    rl   = faf5["road_delay_calibration"]
    print(f"\n--- FAF5 Calibration Validation ---")
    print(f"  Route dist:  short={rd['short_%']}% {'✓' if rd['short_pass'] else '✗'} "
          f" medium={rd['medium_%']}% {'✓' if rd['medium_pass'] else '✗'} "
          f" long={rd['long_%']}% {'✓' if rd['long_pass'] else '✗'}")
    print(f"  Road delay:  mean={rl['mean_road_delay_min']}min  "
          f"std={rl['std_road_delay_min']}min  "
          f"{'✓' if rl['road_std_pass'] else '✗'} (target std ~10 min)")
    print(f"  OVERALL FAF5 CHECK: {'✓ PASS' if faf5['overall_pass'] else '✗ FAIL'}")

    cc = results["cargo_connectivity"]
    print(f"\n--- Cargo Connectivity ---")
    print(f"  Total connecting units: {cc['total_connecting_cargo_units']}")
    for (orig, dest), count in cc["top_pairs"][:8]:
        print(f"    {orig} → {dest}: {count}")
    print("=" * 70)


if __name__ == "__main__":
    # Allow running as a standalone script from ANY directory:
    #   python /path/to/validation.py
    # Works because we add the parent of phase2_simulator/ to sys.path.
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Re-import with path now fixed
    from phase2_simulator.config import SimConfig, HUB_SMALL, HUB_LARGE  # noqa: F811

    print("\nRunning Hub-Small validation (7 days, no_hold)...")
    run_full_validation(cfg=SimConfig(hub=HUB_SMALL, num_days=7, random_seed=42),
                        policy="no_hold", seed=42)

    print("\nRunning Hub-Large validation (3 days, no_hold)...")
    run_full_validation(cfg=SimConfig(hub=HUB_LARGE, num_days=3, random_seed=42),
                        policy="no_hold", seed=42)
