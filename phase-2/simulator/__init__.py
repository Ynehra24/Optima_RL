"""
Phase 2 Logistics Cross-Docking Simulator.

Discrete-event microsimulator for the Hold-or-Not-Hold (HNH) problem
adapted to the trucking logistics domain.

Structurally isomorphic to the Phase 1 airline simulator (paper Section 6.1)
with the following key extensions:
  - Bay-blockage (GB) delay node in the Delay Tree
  - Heterogeneous cargo (value, SLA urgency, perishability)
  - Bay utilization dynamics and congestion cascading
  - 34-dimensional state vector (vs 17 in Phase 1)

Public API:
    from simulator.logistics_env import LogisticsEnv
    from simulator.config import SimConfig

    cfg = SimConfig()
    env = LogisticsEnv(cfg)
    obs = env.reset()
    obs, reward, done, info = env.step(action)
"""

from simulator.config import SimConfig
from simulator.logistics_env import LogisticsEnv

__all__ = ["SimConfig", "LogisticsEnv"]
