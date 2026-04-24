"""
hub_chain.py — Full 10-zone FAF5 mesh network for Phase 2 logistics.

This module provides a TRUE extension of Phase 1's Air-East airline network:

  Phase 1 (Air-East):  Full airline network, many airports, hub-and-spoke
  Phase 2 (FAF5 mesh): Full 10-zone FAF5 freight network, all zones as hubs

Network topology:
  - 10 hubs (all FAF5 zones from routing_matrix.json)
  - Mesh routing: trucks go to whichever hub the routing matrix assigns them
  - Transit times based on route type: short=60min, medium=120min, long=240min
  - ~30% of outbound trucks travel inter-hub (become feeders at destination hub)

Cascade mechanism (identical physics to Phase 1):
  Hub X holds truck T by 15 min
  → T arrives at Hub Y 15 min late
  → Hub Y's agent sees upstream_inter_hub_delay > 0
  → Hub Y may hold its outbound truck → cascade propagates through mesh
  → Network-wide OTP can IMPROVE with strategic holds (exactly like Phase 1!)

Scale comparison with Phase 1:
  Phase 1 Air-East:   ~460 flights/day across full airline network
  Phase 2 FAF5 mesh:  ~800 trucks/day across 10 zones (80/hub × 10)
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from simulator.config import SimConfig
from simulator.delay_sampler import DelaySampler
from simulator.bay_manager import BayManager
from simulator.cargo_manager import CargoManager
from simulator.context_engine import ContextEngine
from simulator.event_queue import EventQueue, EpisodeStats
from simulator.schedule_generator import (
    ScheduleGenerator, DailySchedule, load_cargo_profiles
)
from simulator.hub_config import load_hub_network, HubNetwork
from rewardEngineering.reward_calculator import LogisticsRewardCalculator


# Transit time by route type — calibrated to US highway freight
TRANSIT_BY_ROUTE = {
    "short":  60,    # ~1 hr  (intra-region, e.g. Ohio → Indiana)
    "medium": 120,   # ~2 hrs (cross-region, e.g. Ohio → Illinois)
    "long":   240,   # ~4 hrs (cross-country, e.g. Ohio → California)
}


@dataclass
class InterHubTruck:
    """A truck in transit between two hubs in the freight mesh."""
    truck_id: str
    origin_hub: str
    dest_hub: str
    actual_departure: float
    scheduled_departure: float
    eta_dest: float
    hold_applied: float = 0.0


@dataclass
class HubNetworkState:
    """8-dim network context appended to the 34-dim local state (total 42-dim)."""
    hub_id: str
    hub_centrality: float          # Out-degree / (n_hubs-1) — replaces hub_position
    downstream_bay_util: float     # Weighted avg bay util across downstream hubs
    downstream_queue_len: float    # Trucks in transit to downstream hubs / n_bays
    downstream_avg_delay: float    # Weighted avg departure delay at downstream hubs
    upstream_inter_hub_delay: float  # Mean lateness of inbound inter-hub trucks
    n_transit_trucks: float        # Total trucks in transit into this hub
    downstream_miss_rate: float    # Weighted avg miss rate at downstream hubs
    cascade_risk: float            # Risk that our hold cascades into congested hub

    def to_vector(self) -> np.ndarray:
        """Return 8-dim network state vector, all values clipped to [0,1]."""
        return np.clip(np.array([
            self.downstream_bay_util,
            self.downstream_queue_len,
            self.downstream_avg_delay,
            self.upstream_inter_hub_delay,
            self.hub_centrality,
            self.n_transit_trucks,
            self.downstream_miss_rate,
            self.cascade_risk,
        ], dtype=np.float32), 0.0, 1.0)


class HubChain:
    """Full 10-zone FAF5 freight mesh network.

    Replaces the 3-hub linear chain with the full 10-zone network loaded
    from routing_matrix.json. This matches Phase 1's use of the full
    Air-East airline network rather than a simplified 3-airport chain.

    Key properties matching Phase 1:
      - All zones participate as independent decision-making hubs
      - Inter-hub routing uses actual FAF5 routing matrix probabilities
      - Transit times calibrated to US highway freight by route type
      - Cascade propagation across the full mesh (not just linear chain)
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._master_rng = np.random.default_rng(cfg.seed)

        # ── Load full FAF5 network ──────────────────────────────────────
        self._hub_network: HubNetwork = load_hub_network(cfg.calibrated_dir)
        all_zones = self._hub_network.zones

        # Hub ID = "Hub_{zone_id}" for all 10 zones
        self.hub_ids: List[str] = [f"Hub_{z.zone_id}" for z in all_zones]
        self._zone_of: Dict[str, int] = {
            f"Hub_{z.zone_id}": z.zone_id for z in all_zones
        }

        n_hubs = len(self.hub_ids)
        print(f"[HubChain] Full FAF5 mesh: {n_hubs} hubs, "
              f"routing via routing_matrix.json")

        # ── Build full transit time matrix ──────────────────────────────
        # Uses actual route types from routing matrix (short/medium/long)
        self._transit_matrix: Dict[Tuple[str, str], float] = {}
        for h1 in self.hub_ids:
            for h2 in self.hub_ids:
                if h1 == h2:
                    continue
                z1 = self._zone_of[h1]
                z2 = self._zone_of[h2]
                route_type = self._hub_network.get_route_type(z1, z2)
                self._transit_matrix[(h1, h2)] = float(
                    TRANSIT_BY_ROUTE.get(route_type, 120)
                )

        # ── Compute hub centrality (out-degree) ────────────────────────
        # Number of distinct zones this hub sends trucks to (from routing matrix)
        self._hub_centrality: Dict[str, float] = {}
        for hub_id in self.hub_ids:
            zone_id = self._zone_of[hub_id]
            destinations = self._hub_network.get_dest_zones(zone_id)
            # Only count destinations that are also hub nodes
            hub_zone_ids = set(self._zone_of.values())
            n_connected = sum(1 for d in destinations if d in hub_zone_ids)
            self._hub_centrality[hub_id] = n_connected / max(n_hubs - 1, 1)

        # ── Build per-hub components ────────────────────────────────────
        delay_sampler = DelaySampler(cfg.calibrated_dir,
                                      rng=np.random.default_rng(cfg.seed))
        cargo_profiles = load_cargo_profiles(cfg.calibrated_dir)

        self.event_queues: Dict[str, EventQueue] = {}
        self.bay_managers: Dict[str, BayManager] = {}
        self.schedule_generators: Dict[str, ScheduleGenerator] = {}

        for i, hub_id in enumerate(self.hub_ids):
            hub_seed = cfg.seed + i
            bay_mgr = BayManager(n_bays=cfg.n_bays,
                                  operating_start=float(cfg.operating_start))
            cargo_mgr = CargoManager(cfg)
            ctx_engine = ContextEngine(cfg, bay_mgr, cargo_mgr)
            reward_calc = LogisticsRewardCalculator(cfg)

            self.event_queues[hub_id] = EventQueue(
                cfg=cfg,
                delay_sampler=DelaySampler(cfg.calibrated_dir,
                                            rng=np.random.default_rng(hub_seed)),
                bay_manager=bay_mgr,
                cargo_manager=cargo_mgr,
                context_engine=ctx_engine,
                reward_calculator=reward_calc,
                rng=np.random.default_rng(hub_seed + 100),
            )
            self.bay_managers[hub_id] = bay_mgr
            self.schedule_generators[hub_id] = ScheduleGenerator(
                cfg=cfg,
                hub_network=self._hub_network,
                delay_sampler=DelaySampler(cfg.calibrated_dir,
                                            rng=np.random.default_rng(hub_seed + 200)),
                cargo_profiles=cargo_profiles,
                rng=np.random.default_rng(hub_seed + 300),
            )

        # Trucks in transit between hubs
        self._in_transit: List[InterHubTruck] = []

        # Per-hub RNGs for inter-hub decisions
        self._hub_rngs: Dict[str, np.random.Generator] = {
            hub_id: np.random.default_rng(cfg.seed + i + 500)
            for i, hub_id in enumerate(self.hub_ids)
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self):
        """Full episode reset — clears all hubs and in-transit queue."""
        for hub_id, eq in self.event_queues.items():
            eq.reset()
            self.bay_managers[hub_id].reset()
        self._in_transit.clear()

    def reset_day(self):
        """Day-level reset — preserves episode stats, clears per-day queues."""
        for eq in self.event_queues.values():
            eq.reset_day()
        self._in_transit.clear()

    def load_day(self, day_index: int):
        """Generate and load today's schedule for all 10 hubs."""
        for hub_id in self.hub_ids:
            schedule = self.schedule_generators[hub_id].generate_day(day_index)
            self.event_queues[hub_id].load_day(schedule)

    def get_next_decision(self) -> Optional[Tuple[str, object, str]]:
        """Return the earliest pending HOLD_DECISION across all 10 hubs.

        Returns:
            (hub_id, TruckContext, truck_id) or None if all hubs are done.
        """
        best_time = float("inf")
        best_hub = None

        for hub_id, eq in self.event_queues.items():
            evt = self._peek_next_decision(eq)
            if evt is not None and evt.time < best_time:
                best_time = evt.time
                best_hub = hub_id

        if best_hub is None:
            return None

        result = self.event_queues[best_hub].advance_to_next_decision()
        if result is None:
            return None

        ctx, truck_id = result
        return best_hub, ctx, truck_id

    def apply_hold(self, hub_id: str, truck_id: str, hold_minutes: float) -> float:
        """Apply hold at hub_id. Propagate cascade to destination hub if inter-hub.

        Returns:
            Local reward from the hub's EventQueue.
        """
        eq = self.event_queues[hub_id]
        reward = eq.apply_hold(truck_id, hold_minutes)

        # Cascade: propagate to destination hub using routing matrix probabilities
        self._maybe_propagate(hub_id, truck_id,
                               actual_dep=eq.current_time + hold_minutes,
                               gd_delay=eq._departure_delays.get(truck_id, 0.0))
        return reward

    def get_network_state(self, hub_id: str) -> HubNetworkState:
        """Build the 8-dim network context for the given hub.

        For mesh topology, aggregates stats across ALL directly connected
        downstream hubs (weighted by routing matrix probability).
        """
        zone_id = self._zone_of[hub_id]

        # Get directly reachable downstream hubs (from routing matrix)
        dest_zones = self._hub_network.get_dest_zones(zone_id)
        downstream_hubs = [
            f"Hub_{z}" for z in dest_zones
            if f"Hub_{z}" in self.event_queues and f"Hub_{z}" != hub_id
        ]

        # Routing weights for downstream hubs
        weights = self._get_routing_weights(hub_id, downstream_hubs)

        # Aggregate downstream stats
        if downstream_hubs and weights:
            ds_bay_util = sum(
                w * self.bay_managers[h].get_bay_utilization(
                    self.event_queues[h].current_time)
                for h, w in zip(downstream_hubs, weights)
            )
            ds_miss_rate = sum(
                w * self.event_queues[h].stats.missed_transfer_rate
                for h, w in zip(downstream_hubs, weights)
            )
            ds_avg_delay = sum(
                w * min(
                    self.event_queues[h].stats.total_departure_delay /
                    max(self.event_queues[h].stats.n_total_departures, 1) / 60.0,
                    1.0
                )
                for h, w in zip(downstream_hubs, weights)
            )
            # Trucks in transit to all downstream hubs
            n_transit = len([t for t in self._in_transit
                             if t.dest_hub in downstream_hubs])
            ds_queue = min(n_transit / max(self.cfg.n_bays, 1), 1.0)
            cascade_risk = float(ds_bay_util > 0.6 and ds_queue > 0.2)
        else:
            ds_bay_util = ds_miss_rate = ds_avg_delay = ds_queue = 0.0
            cascade_risk = 0.0

        # Upstream: how late are trucks arriving at THIS hub from other hubs?
        inbound_late = [
            t for t in self._in_transit if t.dest_hub == hub_id
        ]
        if inbound_late:
            mean_late = np.mean([
                max(0.0, t.actual_departure - t.scheduled_departure)
                for t in inbound_late
            ])
            upstream_delay = min(mean_late / 60.0, 1.0)
        else:
            upstream_delay = 0.0

        # Total trucks currently in transit INTO this hub
        n_inbound_transit = len(inbound_late)
        n_transit_norm = min(n_inbound_transit / max(self.cfg.n_bays, 1), 1.0)

        return HubNetworkState(
            hub_id=hub_id,
            hub_centrality=self._hub_centrality.get(hub_id, 0.5),
            downstream_bay_util=ds_bay_util,
            downstream_queue_len=ds_queue,
            downstream_avg_delay=ds_avg_delay,
            upstream_inter_hub_delay=upstream_delay,
            n_transit_trucks=n_transit_norm,
            downstream_miss_rate=ds_miss_rate,
            cascade_risk=cascade_risk,
        )

    def get_combined_stats(self) -> EpisodeStats:
        """Aggregate EpisodeStats across all 10 hubs."""
        combined = EpisodeStats()
        for eq in self.event_queues.values():
            s = eq.stats
            combined.n_transfers_success   += s.n_transfers_success
            combined.n_transfers_missed    += s.n_transfers_missed
            combined.n_rebooked            += s.n_rebooked
            combined.total_departure_delay += s.total_departure_delay
            combined.total_rebook_delay    += s.total_rebook_delay
            combined.total_reward          += s.total_reward
            combined.n_rewards             += s.n_rewards
            combined.n_on_time_departures  += s.n_on_time_departures
            combined.n_total_departures    += s.n_total_departures
            combined.bay_utilization_samples.extend(s.bay_utilization_samples)
        return combined

    def per_hub_stats(self) -> Dict[str, dict]:
        """Return per-hub miss rate and OTP for results breakdown."""
        return {
            hub_id: {
                "missed_rate": eq.stats.missed_transfer_rate,
                "OTP":         eq.stats.OTP,
                "missed":      eq.stats.n_transfers_missed,
            }
            for hub_id, eq in self.event_queues.items()
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _peek_next_decision(self, eq: EventQueue):
        """Peek at earliest HOLD_DECISION in an EventQueue without consuming."""
        for event in eq._heap:
            if (event.event_type == "HOLD_DECISION" and
                    event.truck_id not in eq._decided_trucks):
                return event
        return None

    def _maybe_propagate(self, hub_id: str, truck_id: str,
                          actual_dep: float, gd_delay: float):
        """Propagate late departure to the destination hub (mesh routing).

        Uses the FAF5 routing matrix probabilities to select the destination
        hub — exactly matching Phase 1's O-D-based flight routing.
        """
        rng = self._hub_rngs[hub_id]

        # Only inter_hub_fraction of outbound trucks travel inter-hub
        if rng.random() > self.cfg.inter_hub_fraction:
            return

        # Select destination hub using routing matrix probabilities
        origin_zone = self._zone_of[hub_id]
        dest_zone = self._hub_network.sample_destination(origin_zone, rng)
        if dest_zone is None:
            return

        dest_hub = f"Hub_{dest_zone}"
        if dest_hub not in self.event_queues or dest_hub == hub_id:
            return

        # Transit time based on route type
        transit_base = self._transit_matrix.get(
            (hub_id, dest_hub), self.cfg.transit_time_mean
        )
        transit_noise = float(self._master_rng.normal(0, self.cfg.transit_time_std))
        actual_arrival = actual_dep + gd_delay + transit_base + transit_noise

        # The ON-TIME arrival = what the destination hub planned for
        # (no hold delay, no noise — just scheduled_dep + transit)
        scheduled_arrival = (actual_dep - gd_delay) + transit_base  # no hold, no noise

        inter_truck = InterHubTruck(
            truck_id=f"{truck_id}@{hub_id}→{dest_hub}",
            origin_hub=hub_id,
            dest_hub=dest_hub,
            actual_departure=actual_dep,
            scheduled_departure=actual_dep - gd_delay,
            eta_dest=actual_arrival,
            hold_applied=0.0,
        )
        self._in_transit.append(inter_truck)

        # ← THE CASCADE: inject into destination hub with BOTH times
        self.event_queues[dest_hub].inject_inter_hub_arrival(
            truck_id=inter_truck.truck_id,
            arrival_time=actual_arrival,        # when it ACTUALLY arrives (late)
            scheduled_arrival=scheduled_arrival, # when it SHOULD have arrived
            origin_hub=hub_id,
        )

    def _get_routing_weights(self, hub_id: str,
                              downstream_hubs: List[str]) -> List[float]:
        """Get routing matrix probability weights for downstream hubs."""
        zone_id = self._zone_of[hub_id]
        entry = self._hub_network.routing_matrix.get(str(zone_id), {})
        probs_dict = entry.get("probabilities", {})

        weights = []
        for dh in downstream_hubs:
            dz = self._zone_of.get(dh, -1)
            weights.append(probs_dict.get(str(dz), 0.1))

        total = sum(weights)
        return [w / total for w in weights] if total > 0 else \
               [1.0 / len(downstream_hubs)] * len(downstream_hubs)
