"""reactive_footstep_planner.py
Online, one-step-at-a-time footstep planning that follows a commanded velocity.

Instead of a fixed footstep plan (planLine / planCircle), this generates the next
footstep every step in the direction of the current velocity command ``v_cmd``
coming from the robot's ORCA layer. Each step:

  * if the robot is at the goal, it steps in place (halts);
  * otherwise it places the next (swing) foot one step along ``v_cmd``, offset to
    its side so the feet keep straddling the direction of travel.

Placement is **omnidirectional / fixed facing**: the foot orientation stays at the
initial heading (no body rotation), which reuses the robust straight-line gait —
the robot side-steps to follow a curved path rather than turning. This is the
stable mode for bounded avoidance manoeuvres.

The planner keeps a short lookahead buffer (a few steps ahead, all extrapolated
from the current ``v_cmd``) so the LIP-MPC can see the ~2 steps of ZMP reference it
needs. The buffer is regenerated from the current support foot each step, so the
committed next step always reflects the latest ``v_cmd`` (≈ one-step reactivity).

Everything is 2D in the world ground plane (x, y), SI units.
"""

import numpy as np
import pinocchio as pin

from bullet_sims.footstep_planner import Side, FootStep, other_foot_id


def _footprint(conf, side):
    """Foot-vertex matrix for a side (same convention as FootStepPlanner)."""
    if side == Side.LEFT:
        return np.array([[conf.lfxp, conf.lfxn], [conf.lfyp, conf.lfyn], [0, 0]])
    return np.array([[conf.lfxn, conf.lfxp], [conf.lfyn, conf.lfyp], [0, 0]])


class ReactiveFootStepPlanner:
    """Generates footsteps that follow a velocity command toward a goal."""

    def __init__(self, conf, goal, support_pose,
                 samples_per_step, samples_per_horizon,
                 max_step_len=None, goal_tol=0.3, n_lookahead=None):
        """
        Args:
            conf: talos_conf module (uses step_size_x/y, step_dur, foot half-sizes).
            goal (array-like, 2): navigation goal in the world plane [m].
            support_pose (pin.SE3): the current support foot pose; fixes the facing
                heading and the foot ground height for all generated steps.
            samples_per_step (int): MPC samples per step (conf.no_mpc_samples_per_step).
            samples_per_horizon (int): MPC samples over the horizon
                (conf.no_mpc_samples_per_horizon).
            max_step_len (float): clamp on the per-step forward length [m].
            goal_tol (float): distance [m] at which the goal counts as reached.
            n_lookahead (int): how many future steps to keep buffered (auto if None).
        """
        self.conf = conf
        self.goal = np.asarray(goal, dtype=float)[:2].copy()
        self.sy = conf.step_size_y
        self.step_dur = conf.step_dur
        self.max_step_len = (1.4 * conf.step_size_x) if max_step_len is None else max_step_len
        self.goal_tol = goal_tol
        self.samples_per_step = int(samples_per_step)
        self.samples_per_horizon = int(samples_per_horizon)

        need = int(np.ceil(self.samples_per_horizon / self.samples_per_step)) + 2
        self.n_lookahead = need if n_lookahead is None else int(n_lookahead)

        # fixed facing taken from the initial support foot (omnidirectional walk)
        R = np.asarray(support_pose.rotation)
        self.yaw0 = float(np.arctan2(R[1, 0], R[0, 0]))
        c, s = np.cos(self.yaw0), np.sin(self.yaw0)
        self.R_fixed = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.facing = np.array([c, s])
        self.z = float(support_pose.translation[2])

        self.buffer = []   # [support, next_swing, future_1, ...]

    # ------------------------------------------------------------------ #
    # goal
    # ------------------------------------------------------------------ #

    def at_goal(self, position):
        position = np.asarray(position, dtype=float)[:2]
        return np.hypot(*(self.goal - position)) <= self.goal_tol

    # ------------------------------------------------------------------ #
    # step placement
    # ------------------------------------------------------------------ #

    def _place_next(self, from_fs, v_cmd):
        """Place the swing foot (opposite of from_fs.side) one step along v_cmd."""
        swing_side = other_foot_id(from_fs.side)
        v = np.asarray(v_cmd, dtype=float)[:2]
        speed = np.hypot(*v)
        if speed < 1e-4:
            # no command -> step in place (feet come together at current spot)
            d = self.facing
            length = 0.0
        else:
            d = v / speed
            length = float(np.clip(speed * self.step_dur, 0.0, self.max_step_len))
        perp = np.array([-d[1], d[0]])                 # 90 deg left of travel
        sign = 1.0 if swing_side == Side.LEFT else -1.0
        base = np.asarray(from_fs.pose.translation)[:2]
        pos = base + d * length + perp * (2.0 * self.sy) * sign
        pose = pin.SE3(self.R_fixed, np.array([pos[0], pos[1], self.z]))
        return FootStep(pose, _footprint(self.conf, swing_side), swing_side)

    def regenerate(self, support_fs, v_cmd, at_goal=False):
        """Rebuild the lookahead buffer from the current support foot and v_cmd.
        Call once per step transition. Returns the buffer."""
        vc = np.zeros(2) if at_goal else np.asarray(v_cmd, dtype=float)[:2]
        self.buffer = [support_fs]
        foot = support_fs
        for _ in range(self.n_lookahead):
            foot = self._place_next(foot, vc)
            self.buffer.append(foot)
        return self.buffer

    def make_footstep(self, pose, side):
        """Wrap a pose + side into a FootStep with the correct footprint."""
        return FootStep(pose, _footprint(self.conf, side), side)

    def swing_target(self):
        """The next footstep to step to (buffer[1]); valid after regenerate()."""
        return self.buffer[1]

    # ------------------------------------------------------------------ #
    # MPC reference
    # ------------------------------------------------------------------ #

    def mpc_reference(self, mpc_phase):
        """ZMP + yaw reference over the MPC horizon, phase-aligned within the step.

        Args:
            mpc_phase (int): MPC-sample index within the current step (0 at the
                start of a step, increments each MPC update).

        Returns:
            (zmp_ref (N,2), yaw_ref (N,)) where N = samples_per_horizon. The ZMP
            reference holds the current support-foot centre for the remaining
            samples of this step, then each subsequent buffered footstep centre.
        """
        remaining = self.samples_per_step - int(mpc_phase)
        remaining = int(np.clip(remaining, 1, self.samples_per_step))

        zmp = []
        c0 = np.asarray(self.buffer[0].pose.translation)[:2]
        for _ in range(remaining):
            zmp.append(c0)

        bi = 1
        while len(zmp) < self.samples_per_horizon and bi < len(self.buffer):
            c = np.asarray(self.buffer[bi].pose.translation)[:2]
            for _ in range(self.samples_per_step):
                if len(zmp) < self.samples_per_horizon:
                    zmp.append(c)
            bi += 1
        while len(zmp) < self.samples_per_horizon:      # safety pad
            zmp.append(zmp[-1])

        zmp = np.asarray(zmp[:self.samples_per_horizon], dtype=float)
        yaw = np.full(self.samples_per_horizon, self.yaw0, dtype=float)
        return zmp, yaw
