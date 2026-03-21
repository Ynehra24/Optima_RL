"""
Airline Network Microsimulator for Hold-or-Not-Hold (HNH) RL Problem.

Replication of the simulator described in:
  Malladi et al., "To Hold or Not to Hold? - Reducing Passenger Missed
  Connections in Airlines using Reinforcement Learning," AAMAS 2021.

Sections: 6.1 (Simulator) and 7.1 (Validation)
"""

from simulator.config import SimConfig, AirlineProfile
from simulator.simulator import AirlineNetworkSimulator

__all__ = ["SimConfig", "AirlineProfile", "AirlineNetworkSimulator"]
