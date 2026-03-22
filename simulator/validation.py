"""
Validation utilities — Section 7.1 of the paper.

The paper validates the simulator along three axes:
  1. PAX profiles — connectivity matrix across cities matches input
  2. Missed connections — baseline misconnect rate matches real data
  3. Network delays — histogram of arrival/departure delays matches real data

This module provides functions to compute those metrics and print /
plot validation reports.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from simulator.config import SimConfig
from simulator.simulator import AirlineNetworkSimulator, MetricsTracker


def validate_pax_connectivity(sim: AirlineNetworkSimulator) -> Dict[str, Any]:
    """Compute the connectivity matrix across airports.

    Returns a dict with:
      - matrix: Dict[(origin, dest), count] of connecting PAX
      - top_pairs: sorted list of top pairs by count
    """
    connectivity: Dict[tuple, int] = defaultdict(int)

    for pax in sim.pax.values():
        if len(pax.legs) >= 2:
            first_leg_fid = pax.legs[0]
            last_leg_fid = pax.legs[-1]
            first_fs = sim.flights.get(first_leg_fid)
            last_fs = sim.flights.get(last_leg_fid)
            if first_fs and last_fs:
                origin = first_fs.flight.origin
                dest = last_fs.flight.destination
                connectivity[(origin, dest)] += pax.group_size

    # Sort by count
    top_pairs = sorted(connectivity.items(), key=lambda x: -x[1])

    return {
        "matrix": dict(connectivity),
        "top_pairs": top_pairs[:20],
        "total_connecting_pax": sum(connectivity.values()),
    }


def validate_missed_connections(
    sim: AirlineNetworkSimulator,
) -> Dict[str, Any]:
    """Validate missed-connection rates against expected baseline.

    From the paper:
      Air-East: ~3% of connecting PAX miss connections
      Air-West: ~5.5% of connecting PAX miss connections
    """
    metrics = sim.metrics
    expected_rate = sim.cfg.airline.baseline_misconnect_rate

    return {
        "total_connecting_pax": metrics.connecting_pax,
        "missed_connections": metrics.missed_connections,
        "successful_connections": metrics.successful_connections,
        "simulated_misconnect_rate": round(metrics.misconnect_rate, 4),
        "expected_misconnect_rate": expected_rate,
        "relative_error": (
            abs(metrics.misconnect_rate - expected_rate) / expected_rate
            if expected_rate > 0
            else 0.0
        ),
        "daily_missed": dict(metrics.daily_missed),
        "daily_connections": dict(metrics.daily_connections),
    }


def validate_network_delays(
    sim: AirlineNetworkSimulator,
) -> Dict[str, Any]:
    """Validate delay distributions and OTP.

    From the paper (Air-East):
      OTP aggregate ~88.5%, short ~87.68%, long ~81.9%
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

    # OTP by haul type
    otp_by_haul: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ontime": 0, "total": 0})
    for fid, fs in sim.flights.items():
        if fs.status.name == "ARRIVED":
            haul = fs.flight.haul_type
            otp_by_haul[haul]["total"] += 1
            if fs.arrival_delay_A <= sim.cfg.ontime_threshold:
                otp_by_haul[haul]["ontime"] += 1

    otp_haul_pct = {}
    for haul, counts in otp_by_haul.items():
        if counts["total"] > 0:
            otp_haul_pct[haul] = round(counts["ontime"] / counts["total"] * 100, 2)

    return {
        "OTP_overall": round(metrics.otp * 100, 2),
        "OTP_by_haul": otp_haul_pct,
        "expected_OTP": sim.cfg.airline.baseline_otp * 100,
        "arrival_delay_histogram": histogram(metrics.arrival_delays),
        "departure_delay_histogram": histogram(metrics.departure_delays),
        "avg_arrival_delay": round(metrics.avg_arrival_delay, 2),
        "avg_departure_delay": round(metrics.avg_departure_delay, 2),
    }


def run_full_validation(
    cfg: SimConfig | None = None,
    policy: str = "no_hold",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a full validation of the simulator.

    Steps:
      1. Build and run the sim with a baseline policy (e.g. no_hold)
      2. Validate PAX connectivity
      3. Validate missed connections
      4. Validate network delays
      5. Report

    Returns combined validation results.
    """
    from simulator.config import SimConfig as _SC

    if cfg is None:
        cfg = _SC(random_seed=seed)

    sim = AirlineNetworkSimulator(cfg)
    sim.run_episode(policy=policy, seed=seed)

    pax_val = validate_pax_connectivity(sim)
    misconn_val = validate_missed_connections(sim)
    delay_val = validate_network_delays(sim)

    results = {
        "airline": cfg.airline.name,
        "policy": policy,
        "num_days": cfg.num_days,
        "pax_connectivity": pax_val,
        "missed_connections": misconn_val,
        "network_delays": delay_val,
        "full_summary": sim.metrics.summary(),
    }

    if verbose:
        _print_validation(results)

    return results


def _print_validation(results: Dict[str, Any]) -> None:
    """Pretty-print validation results."""
    print("=" * 70)
    print(f"  SIMULATOR VALIDATION — {results['airline']}")
    print(f"  Policy: {results['policy']}  |  Days simulated: {results['num_days']}")
    print("=" * 70)

    summary = results["full_summary"]
    print(f"\n--- Business Metrics ---")
    print(f"  Total flights:          {summary['total_flights']}")
    print(f"  Departed flights:       {summary['departed']}")
    print(f"  Arrived flights:        {summary['arrived']}")
    print(f"  OTP:                    {summary['OTP']}%")
    print(f"  Avg arrival delay:      {summary['avg_arrival_delay_min']} min")
    print(f"  Avg departure delay:    {summary['avg_departure_delay_min']} min")

    mc = results["missed_connections"]
    print(f"\n--- Missed Connections ---")
    print(f"  Total connecting PAX:   {mc['total_connecting_pax']}")
    print(f"  Missed connections:     {mc['missed_connections']}")
    print(f"  Successful connections: {mc['successful_connections']}")
    print(f"  Simulated rate:         {mc['simulated_misconnect_rate'] * 100:.2f}%")
    print(f"  Expected rate:          {mc['expected_misconnect_rate'] * 100:.2f}%")
    print(f"  Relative error:         {mc['relative_error'] * 100:.1f}%")

    dd = results["network_delays"]
    print(f"\n--- Network Delays ---")
    print(f"  OTP overall:            {dd['OTP_overall']}% (expected: {dd['expected_OTP']}%)")
    print(f"  OTP by haul:            {dd['OTP_by_haul']}")
    print(f"  Arrival delay hist:     {dd['arrival_delay_histogram']}")
    print(f"  Departure delay hist:   {dd['departure_delay_histogram']}")

    pax_c = results["pax_connectivity"]
    print(f"\n--- PAX Connectivity ---")
    print(f"  Total connecting PAX:   {pax_c['total_connecting_pax']}")
    print(f"  Top 10 city-pairs:")
    for (orig, dest), count in pax_c["top_pairs"][:10]:
        print(f"    {orig} -> {dest}: {count} PAX")

    print("=" * 70)
