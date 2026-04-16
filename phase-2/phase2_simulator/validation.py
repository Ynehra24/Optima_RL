"""
Validation utilities for the logistics cross-docking simulator.

Direct counterpart of simulator/validation.py.
Validates the simulator along three axes matching the original:
  1. Cargo transfer connectivity — transfer matrix across hubs
  2. Failed transfer rates — vs baseline expected rate
  3. Network delays — departure delay histogram, OTP schedule

References:
  Mirrors simulator/validation.py (Section 7.1 of Malladi et al. paper)
  adapted for the logistics domain.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from phase2_simulator.config import SimConfig
from phase2_simulator.simulator import CrossDockSimulator, MetricsTracker


def validate_cargo_connectivity(sim: CrossDockSimulator) -> Dict[str, Any]:
    """Compute the cargo transfer connectivity matrix across hubs.

    Mirrors validate_pax_connectivity() from simulator/validation.py.

    Returns a dict with:
      - matrix    : Dict[(origin, dest), count] of connecting cargo units
      - top_pairs : sorted top pairs by cargo volume
    """
    connectivity: Dict[tuple, int] = defaultdict(int)

    for cu in sim.cargo.values():
        if len(cu.legs) >= 2:
            first_tid = cu.legs[0]
            last_tid = cu.legs[-1]
            first_ts = sim.trucks.get(first_tid)
            last_ts = sim.trucks.get(last_tid)
            if first_ts and last_ts:
                origin = first_ts.truck.origin_hub
                dest = last_ts.truck.dest_hub
                connectivity[(origin, dest)] += cu.unit_count

    top_pairs = sorted(connectivity.items(), key=lambda x: -x[1])

    return {
        "matrix": dict(connectivity),
        "top_pairs": top_pairs[:20],
        "total_connecting_cargo_units": sum(connectivity.values()),
    }


def validate_failed_transfers(
    sim: CrossDockSimulator,
) -> Dict[str, Any]:
    """Validate failed-transfer rates against expected baseline.

    Mirrors validate_missed_connections() from simulator/validation.py.

    Expected baseline from HubProfile.baseline_failed_transfer_rate.
    """
    metrics = sim.metrics
    expected_rate = sim.cfg.hub.baseline_failed_transfer_rate

    return {
        "total_connecting_cargo": metrics.connecting_cargo,
        "failed_transfers": metrics.failed_transfers,
        "successful_transfers": metrics.successful_transfers,
        "simulated_failed_rate": round(metrics.failed_transfer_rate, 4),
        "expected_failed_rate": expected_rate,
        "relative_error": (
            abs(metrics.failed_transfer_rate - expected_rate) / expected_rate
            if expected_rate > 0
            else 0.0
        ),
        "daily_failed": dict(metrics.daily_failed),
        "daily_transfers": dict(metrics.daily_transfers),
    }


def validate_network_delays(
    sim: CrossDockSimulator,
) -> Dict[str, Any]:
    """Validate delay distributions and schedule OTP.

    Mirrors validate_network_delays() from simulator/validation.py.
    Replaces haul-type OTP breakdown with route-type OTP breakdown.
    """
    metrics = sim.metrics
    delay_bins = [15, 30, 45, 60, float("inf")]
    bin_labels = ["<=15", "16-30", "31-45", "46-60", ">60"]

    def histogram(delays: List[float]) -> Dict[str, float]:
        if not delays:
            return {label: 0.0 for label in bin_labels}
        counts = {label: 0 for label in bin_labels}
        for d in delays:
            for i, upper in enumerate(delay_bins):
                if d <= upper:
                    counts[bin_labels[i]] += 1
                    break
        total = len(delays)
        return {label: round(c / total * 100, 1) for label, c in counts.items()}

    # OTP by route type
    otp_by_route: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ontime": 0, "total": 0})
    for tid, ts in sim.trucks.items():
        if ts.status.name == "DEPARTED":
            rtype = ts.truck.route_type
            otp_by_route[rtype]["total"] += 1
            if ts.departure_delay_D <= sim.cfg.ontime_threshold:
                otp_by_route[rtype]["ontime"] += 1

    otp_route_pct = {}
    for rtype, counts in otp_by_route.items():
        if counts["total"] > 0:
            otp_route_pct[rtype] = round(counts["ontime"] / counts["total"] * 100, 2)

    return {
        "OTP_overall": round(metrics.schedule_otp * 100, 2),
        "OTP_by_route_type": otp_route_pct,
        "expected_OTP": sim.cfg.hub.baseline_schedule_otp * 100,
        "avg_bay_utilisation": round(metrics.avg_bay_utilisation, 3),
        "departure_delay_histogram": histogram(metrics.departure_delays),
        "arrival_delay_histogram": histogram(metrics.arrival_delays),
        "avg_departure_delay": round(metrics.avg_departure_delay, 2),
        "avg_arrival_delay": round(metrics.avg_arrival_delay, 2),
    }


def run_full_validation(
    cfg: SimConfig | None = None,
    policy: str = "no_hold",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a full validation of the logistics simulator.

    Mirrors run_full_validation() from simulator/validation.py.

    Steps:
      1. Build and run the sim with a baseline policy
      2. Validate cargo connectivity
      3. Validate failed transfers
      4. Validate network delays
      5. Report
    """
    from phase2_simulator.config import SimConfig as _SC

    if cfg is None:
        cfg = _SC(random_seed=seed)

    sim = CrossDockSimulator(cfg)
    sim.run_episode(policy=policy, seed=seed)

    connectivity_val = validate_cargo_connectivity(sim)
    failed_val = validate_failed_transfers(sim)
    delay_val = validate_network_delays(sim)

    results = {
        "hub": cfg.hub.name,
        "policy": policy,
        "num_days": cfg.num_days,
        "cargo_connectivity": connectivity_val,
        "failed_transfers": failed_val,
        "network_delays": delay_val,
        "full_summary": sim.metrics.summary(),
    }

    if verbose:
        _print_validation(results)

    return results


def _print_validation(results: Dict[str, Any]) -> None:
    """Pretty-print validation results.

    Mirrors _print_validation() from simulator/validation.py.
    """
    print("=" * 70)
    print(f"  SIMULATOR VALIDATION — {results['hub']}")
    print(f"  Policy: {results['policy']}  |  Days simulated: {results['num_days']}")
    print("=" * 70)

    summary = results["full_summary"]
    print(f"\n--- Business Metrics ---")
    print(f"  Total trucks:           {summary['total_trucks']}")
    print(f"  Docked trucks:          {summary['docked']}")
    print(f"  Departed trucks:        {summary['departed']}")
    print(f"  Schedule OTP:           {summary['schedule_OTP']}%")
    print(f"  Avg departure delay:    {summary['avg_departure_delay_min']} min")
    print(f"  Avg arrival delay:      {summary['avg_arrival_delay_min']} min")
    print(f"  Avg bay utilisation:    {summary['avg_bay_utilisation']:.1%}")

    ft = results["failed_transfers"]
    print(f"\n--- Failed Transfers ---")
    print(f"  Total connecting cargo: {ft['total_connecting_cargo']}")
    print(f"  Failed transfers:       {ft['failed_transfers']}")
    print(f"  Successful transfers:   {ft['successful_transfers']}")
    print(f"  Simulated rate:         {ft['simulated_failed_rate'] * 100:.2f}%")
    print(f"  Expected rate:          {ft['expected_failed_rate'] * 100:.2f}%")
    print(f"  Relative error:         {ft['relative_error'] * 100:.1f}%")

    dd = results["network_delays"]
    print(f"\n--- Network Delays ---")
    print(f"  OTP overall:            {dd['OTP_overall']}% (expected: {dd['expected_OTP']}%)")
    print(f"  OTP by route type:      {dd['OTP_by_route_type']}")
    print(f"  Departure delay hist:   {dd['departure_delay_histogram']}")
    print(f"  Avg departure delay:    {dd['avg_departure_delay']} min")

    cc = results["cargo_connectivity"]
    print(f"\n--- Cargo Connectivity ---")
    print(f"  Total connecting units: {cc['total_connecting_cargo_units']}")
    print(f"  Top 10 hub-pairs:")
    for (orig, dest), count in cc["top_pairs"][:10]:
        print(f"    {orig} -> {dest}: {count} units")

    print("=" * 70)
