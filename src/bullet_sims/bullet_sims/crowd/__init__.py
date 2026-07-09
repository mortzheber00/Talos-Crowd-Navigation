"""crowd
Crowd-navigation module: a self-contained 2D ORCA solver and a MuJoCo
pedestrian simulator that drives kinematic disk agents with it.

Public API:
    from bullet_sims.crowd import orca_new_velocity, ORCAParams
    from bullet_sims.crowd import PedestrianSimulator, Pedestrian
    from bullet_sims.crowd import scenarios
"""

from bullet_sims.crowd.orca import orca_new_velocity, ORCAParams
from bullet_sims.crowd.pedestrian import PedestrianSimulator, Pedestrian

__all__ = [
    "orca_new_velocity",
    "ORCAParams",
    "PedestrianSimulator",
    "Pedestrian",
]
