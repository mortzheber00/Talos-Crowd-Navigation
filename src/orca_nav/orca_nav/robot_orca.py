"""robot_orca.py
Computes the ORCA velocity the robot should follow to reach a goal while avoiding
the pedestrians.

This is the robot side of the crowd-navigation loop. Each tick you give it the
robot's current ground-plane state and the pedestrians' ground-truth states
(exactly what ``PedestrianSimulator.states()`` returns), and it returns a single
2D velocity command: the collision-free velocity closest to "head toward the
goal". That command is what the reactive footstep layer will turn into footsteps.

It reuses the same ORCA solver as the pedestrians (``crowd/orca.py``), so the
avoidance is consistent on both sides.

Everything is 2D, in the MuJoCo world ground plane (x, y), SI units.
"""

import numpy as np

from orca_nav.orca import orca_new_velocity, ORCAParams


class RobotORCA:
    """Goal-seeking ORCA velocity generator for the robot."""

    def __init__(self,
                 goal,
                 radius=0.35,
                 max_speed=0.4,
                 desired_speed=0.25,
                 safety_margin=0.2,
                 goal_tol=0.3,
                 brake_radius=0.6,
                 responsibility=0.2,
                 params=None):
        """
        Args:
            goal (array-like, 2): target position in the world plane [m].
            radius (float): the robot's footprint radius [m].
            max_speed (float): upper bound on the commanded speed [m/s].
            desired_speed (float): cruising speed toward the goal [m/s]. Keep this
                at/below what the walker can actually achieve.
            safety_margin (float): extra radius [m] ORCA plans with, to account for
                the robot being committed ~1 step ahead (it cannot change velocity
                instantly). A good value is about max_speed * step_duration.
            goal_tol (float): distance [m] at which the goal counts as reached.
            brake_radius (float): distance [m] within which the preferred speed is
                ramped down linearly so the robot settles on the goal.
            responsibility (float): ORCA reciprocity share. 0.5 = mutual, which
                matches the pedestrians avoiding the robot reciprocally. Raise
                toward 1.0 to make the robot take more of the avoidance itself.
            params (ORCAParams): ORCA parameters (sensible defaults if None).
        """
        self.goal = np.asarray(goal, dtype=float)[:2].copy()
        self.radius = float(radius)
        self.max_speed = float(max_speed)
        self.desired_speed = float(desired_speed)
        self.safety_margin = float(safety_margin)
        self.goal_tol = float(goal_tol)
        self.brake_radius = float(brake_radius)
        self.responsibility = float(responsibility)
        self.params = params or ORCAParams()

    # ------------------------------------------------------------------ #
    # configuration
    # ------------------------------------------------------------------ #

    def set_goal(self, goal):
        """Update the navigation goal (world x, y)."""
        self.goal = np.asarray(goal, dtype=float)[:2].copy()

    def at_goal(self, position):
        """True once the robot is within goal_tol of the goal."""
        position = np.asarray(position, dtype=float)[:2]
        return np.hypot(*(self.goal - position)) <= self.goal_tol

    # ------------------------------------------------------------------ #
    # velocity computation
    # ------------------------------------------------------------------ #

    def preferred_velocity(self, position):
        """Velocity that heads straight to the goal, ramped down near it."""
        position = np.asarray(position, dtype=float)[:2]
        to_goal = self.goal - position
        dist = np.hypot(*to_goal)
        if dist < 1e-6:
            return np.zeros(2)
        speed = self.desired_speed * min(1.0, dist / max(self.brake_radius, 1e-6))
        return to_goal / dist * speed

    def compute_velocity(self, position, velocity, ped_states):
        """Compute the robot's collision-free ORCA velocity for this tick.

        Args:
            position (array-like, 2): robot ground-plane position [m] (e.g. the
                CoM projection: ``simulator.computeComPosition(body_id)[:2]``).
            velocity (array-like, 2): robot current ground-plane velocity [m/s]
                (e.g. ``simulator.computeComVelocity(body_id)[:2]``).
            ped_states (iterable): pedestrian states as (pos(2), vel(2), radius),
                exactly what ``PedestrianSimulator.states()`` returns.

        Returns:
            np.ndarray shape (2,): the velocity command to follow, |v| <= max_speed.
        """
        position = np.asarray(position, dtype=float)[:2]
        velocity = np.asarray(velocity, dtype=float)[:2]

        # stop cleanly once the goal is reached
        if self.at_goal(position):
            return np.zeros(2)

        pref = self.preferred_velocity(position)

        # pedestrians are the neighbours; pass their true radii unchanged. The
        # robot carries the whole extra clearance via its own inflated radius.
        neighbors = [(np.asarray(p, float)[:2], np.asarray(v, float)[:2], float(r))
                     for (p, v, r) in ped_states]

        eff_radius = self.radius + self.safety_margin
        v_cmd = orca_new_velocity(
            position, velocity, eff_radius, self.max_speed,
            pref, neighbors, self.params, self.responsibility)
        return v_cmd


# ---------------------------------------------------------------------------
# functional convenience wrapper (stateless, if you prefer not to keep an object)
# ---------------------------------------------------------------------------

def robot_orca_velocity(position, velocity, goal, ped_states,
                        radius=0.35, max_speed=0.4, desired_speed=0.25,
                        safety_margin=0.2, goal_tol=0.3, brake_radius=0.6,
                        responsibility=0.2, params=None):
    """One-shot version of ``RobotORCA.compute_velocity``. Builds a RobotORCA with
    the given settings and returns the velocity command. Handy for a quick call
    inside the walking loop without keeping state around."""
    ctrl = RobotORCA(goal, radius=radius, max_speed=max_speed,
                     desired_speed=desired_speed, safety_margin=safety_margin,
                     goal_tol=goal_tol, brake_radius=brake_radius,
                     responsibility=responsibility, params=params)
    return ctrl.compute_velocity(position, velocity, ped_states)
