"""crowd_demo.py
Standalone demo of the ORCA pedestrian crowd in the MuJoCo simulator — no robot.

Run (from the workspace root, after building + sourcing):

    ros2 run bullet_sims crowd_demo                # default scenario: crossing
    ros2 run bullet_sims crowd_demo circle
    ros2 run bullet_sims crowd_demo random
    ros2 run bullet_sims crowd_demo corridor 60    # scenario + max seconds

Scenarios: head_on, crossing, circle, corridor, random (see crowd/scenarios.py).
Each pedestrian is a coloured disk (cylinder). A small matching marker on the
floor shows its goal. Pedestrians steer with ORCA and reverse start<->goal on
arrival, so the crowd keeps moving.
"""

import sys
import time

import mujoco
import numpy as np

from simulator.mujoco_wrapper import MujocoWrapper
from bullet_sims.crowd.pedestrian import PedestrianSimulator
from bullet_sims.crowd import scenarios


def _add_goal_markers(sim, crowd):
    """Add a small, flat, visual-only disk at each pedestrian goal (same colour)."""
    for ped in crowd.peds:
        color = list(ped.color[:3]) + [0.4]
        g = sim._spec.worldbody.add_geom(
            name=f"{ped.name}_goal",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[ped.radius, 0.01, 0.0],
            pos=[ped.goal[0], ped.goal[1], 0.01],
            rgba=color,
            contype=0, conaffinity=0,
        )
    sim._model = None  # include the markers in the compile


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "crossing"
    max_seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0

    # 300 Hz matches the walking controller; render the passive viewer
    sim = MujocoWrapper(sim_rate=300, num_sub_steps=4, render=True)

    crowd = PedestrianSimulator(sim, radius=0.3, height=1.7, max_speed=1.3,
                                pref_perturb=0.05, goal_tol=0.3, personal_space=0.3)
    scenarios.build(crowd, scenario)
    _add_goal_markers(sim, crowd)

    print(f"[crowd_demo] scenario='{scenario}'  pedestrians={len(crowd.peds)}  "
          f"max_seconds={max_seconds}")

    dt = sim.stepTime()
    worst = np.inf
    next_report = 1.0
    wall0 = time.time()

    while sim.isRunning() and sim.simTime() < max_seconds:
        crowd.update(dt)
        states=crowd.states()
        print(f"states: {states}")
        sim.step()

        worst = min(worst, crowd.min_separation())
        if sim.simTime() >= next_report:
            next_report += 1.0
            print(f"  t={sim.simTime():5.1f}s  active={crowd.num_active():2d}  "
                  f"min_gap_so_far={worst:+.3f} m")

        # pace to real time so the crowd is watchable
        target = wall0 + sim.simTime()
        lag = target - time.time()
        if lag > 0:
            time.sleep(lag)

        if crowd.all_arrived():
            print("[crowd_demo] all pedestrians reached their goals")
            break

    print(f"[crowd_demo] done. worst surface-to-surface gap over run: {worst:+.3f} m "
          f"({'NO overlaps' if worst >= 0 else 'OVERLAP detected'})")


if __name__ == "__main__":
    main()
