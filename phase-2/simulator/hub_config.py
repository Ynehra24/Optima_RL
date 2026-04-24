"""
hub_config.py — Static hub and zone definitions for the Phase 2 simulator.

Loads the routing matrix from calibrated data and defines the hub network.
The simulator uses a single primary hub (analogous to Air-East in the paper)
with 10 surrounding zones that generate inbound/outbound truck flows.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Human-readable names for top FAF5 zones (approximate)
FAF_ZONE_NAMES = {
    11: "Washington DC",  12: "Delaware",        19: "New York",
    20: "New Jersey",     21: "Philadelphia",     22: "Pittsburgh",
    29: "Pennsylvania",   30: "Ohio",             31: "Indiana",
    32: "Illinois",       33: "Chicago",          34: "Michigan",
    40: "Iowa",           41: "Missouri",         42: "Minnesota",
    49: "Wisconsin",      50: "Kansas",           51: "Nebraska",
    52: "South Dakota",   53: "North Dakota",     61: "Virginia",
    62: "W Virginia",     63: "North Carolina",   64: "South Carolina",
    65: "Georgia",        66: "Florida",          67: "Alabama",
    68: "Mississippi",    69: "Tennessee",        71: "Arkansas",
    72: "Louisiana",      73: "Oklahoma",         74: "Texas",
    75: "Colorado",       79: "Mountain States",  81: "California",
    82: "Oregon",         83: "Washington",       89: "Pacific NW",
}


@dataclass
class Zone:
    """A freight zone in the simulated network."""
    zone_id: int
    name: str
    is_hub: bool = False  # True for the primary hub


@dataclass
class HubNetwork:
    """The simulated hub and its surrounding zones.

    The primary hub is the cross-docking facility. All trucks either:
    - Arrive AT the hub (inbound from surrounding zones)
    - Depart FROM the hub (outbound to surrounding zones)
    """
    hub_zone: Zone
    zones: List[Zone]
    routing_matrix: Dict[str, Dict]  # {origin_zone_id: {dest_zone_id: prob}}

    def get_dest_zones(self, origin_zone_id: int) -> List[int]:
        """Get possible destination zones from an origin, sorted by probability."""
        probs = self.routing_matrix.get(str(origin_zone_id), {}).get("probabilities", {})
        return [int(k) for k, v in sorted(probs.items(), key=lambda x: -x[1])]

    def sample_destination(self, origin_zone_id: int, rng) -> Optional[int]:
        """Sample a destination zone given origin, using routing matrix probabilities."""
        entry = self.routing_matrix.get(str(origin_zone_id), {})
        probs_dict = entry.get("probabilities", {})
        if not probs_dict:
            return None
        dest_ids = [int(k) for k in probs_dict.keys()]
        dest_probs = list(probs_dict.values())
        # Normalize
        total = sum(dest_probs)
        dest_probs = [p / total for p in dest_probs]
        return int(rng.choice(dest_ids, p=dest_probs))

    def get_route_type(self, origin_zone_id: int, dest_zone_id: int) -> str:
        """Return 'short', 'medium', or 'long' for an O-D pair."""
        entry = self.routing_matrix.get(str(origin_zone_id), {})
        route_types = entry.get("route_types", {})
        return route_types.get(str(dest_zone_id), "medium")

    @property
    def zone_ids(self) -> List[int]:
        return [z.zone_id for z in self.zones]


def load_hub_network(calibrated_dir: str) -> HubNetwork:
    """Load routing matrix and build HubNetwork.

    If calibrated data is not available, falls back to a default
    synthetic network for development/testing.

    Args:
        calibrated_dir: Path to directory containing routing_matrix.json

    Returns:
        HubNetwork ready for use by ScheduleGenerator
    """
    routing_matrix_path = os.path.join(calibrated_dir, "routing_matrix.json")

    if os.path.exists(routing_matrix_path):
        with open(routing_matrix_path) as f:
            routing_matrix = json.load(f)
        print(f"[hub_config] Loaded routing matrix from {routing_matrix_path}")
    else:
        print("[hub_config] WARNING: routing_matrix.json not found. Using default network.")
        routing_matrix = _default_routing_matrix()

    zone_ids = [int(k) for k in routing_matrix.keys()]

    # Hub zone = highest volume zone
    hub_zone_id = zone_ids[0]

    zones = []
    for zid in zone_ids:
        name = FAF_ZONE_NAMES.get(zid, f"Zone_{zid}")
        zones.append(Zone(zone_id=zid, name=name, is_hub=(zid == hub_zone_id)))

    hub_zone = Zone(zone_id=hub_zone_id,
                    name=FAF_ZONE_NAMES.get(hub_zone_id, f"Hub_{hub_zone_id}"),
                    is_hub=True)

    print(f"[hub_config] Primary hub: {hub_zone.name} (zone {hub_zone.zone_id})")
    print(f"[hub_config] Network: {len(zones)} zones")

    return HubNetwork(hub_zone=hub_zone, zones=zones, routing_matrix=routing_matrix)


def _default_routing_matrix() -> dict:
    """Fallback synthetic routing matrix for testing without calibrated data."""
    zones = [33, 74, 65, 81, 30, 41, 34, 62, 19, 22]  # Chicago hub + 9 zones
    matrix = {}
    for i, orig in enumerate(zones):
        dests = [z for z in zones if z != orig]
        # Simple distance-based probs: nearer zones get more traffic
        probs = {}
        route_types = {}
        for j, dest in enumerate(dests):
            weight = 1.0 / (abs(i - zones.index(dest)) + 1)
            probs[str(dest)] = weight
            idx_diff = abs(i - zones.index(dest))
            route_types[str(dest)] = "short" if idx_diff <= 2 else ("medium" if idx_diff <= 5 else "long")

        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        matrix[str(orig)] = {
            "probabilities": probs,
            "route_types": route_types,
            "total_tons_2022": 1000.0,
        }
    return matrix
