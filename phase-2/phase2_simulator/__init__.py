"""
Logistics Cross-Docking Microsimulator for the Hold-or-Not-Hold (HOH) RL Problem.

Adaptation of the airline microsimulator described in:
  Malladi et al., "To Hold or Not to Hold? - Reducing Passenger Missed
  Connections in Airlines using Reinforcement Learning," AAMAS 2021.

Applied to: Supply-chain cross-docking logistics hub.
State space: logistics_state_space_reference.pdf (Phase 2)
Dataset: dynamic_supply_chain_logistics_dataset.csv

Sections mirrored: 6.1 (Simulator) and 7.1 (Validation)
"""

from phase2_simulator.config import SimConfig, HubProfile, HUB_SMALL, HUB_LARGE
from phase2_simulator.simulator import CrossDockSimulator

__all__ = [
    "SimConfig",
    "HubProfile",
    "HUB_SMALL",
    "HUB_LARGE",
    "CrossDockSimulator",
]
