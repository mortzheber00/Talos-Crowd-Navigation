"""reactive_footstep_planner.py
Online, one-step-at-a-time footstep planning that STEERS toward a commanded velocity.

Instead of a fixed footstep plan (planLine / planCircle), this generates the next
footstep every step so the robot TURNS to face the current velocity command
``v_cmd`` (coming from the robot's ORCA layer) and then walks forward along it — the
way a person turns to face where they are going. Each step:

  * rotate the body heading ``psi`` a bounded amount toward the direction of
    ``v_cmd`` (rate-limited, with a deadband so noise / a zero command don't spin
    the robot);
  * step the swing foot forward along the new heading, offset to its own side by a
    fixed step width. The forward length shrinks with heading misalignment, so a
    sharply off-axis command turns (nearly) in place until the body faces it.

This is a **non-holonomic / unicycle** gait: the swing foot is always placed
``step_size_y`` to its own side of the *current facing* (never the travel
direction), so the swing foot can never land across the stance foot — leg-crossing
is eliminated by construction, and the robot follows a smooth arc instead of
side-stepping. The geometry mirrors ``FootStepPlanner.planCircle(face_path=True)``,
generalised so the turn rate comes from ``v_cmd`` rather than a fixed radius.

The planner keeps a short lookahead buffer (a few steps ahead, all extrapolated
from the current ``v_cmd``) so the LIP-MPC can see the ~2 steps of ZMP/yaw reference
it needs. The buffer is regenerated from the current support foot each step, so the
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


def _wrap(angle):
    """Wrap an angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class ReactiveFootStepPlanner:
    """Generates footsteps that steer the body toward a velocity command."""

    def __init__(self, conf, goal, support_pose,
                 samples_per_step, samples_per_horizon,
                 max_step_len=None, goal_tol=0.3, n_lookahead=None,
                 max_dyaw=None, turn_deadband=0.05):
        """
        Args:
            conf: talos_conf module (uses step_size_x/y, step_dur, foot half-sizes).
            goal (array-like, 2): navigation goal in the world plane [m].
            support_pose (pin.SE3): the current support foot pose; sets the initial
                body heading and the foot ground height.
            samples_per_step (int): MPC samples per step (conf.no_mpc_samples_per_step).
            samples_per_horizon (int): MPC samples over the horizon
                (conf.no_mpc_samples_per_horizon).
            max_step_len (float): clamp on the per-step forward length [m].
            goal_tol (float): distance [m] at which the goal counts as reached.
            n_lookahead (int): how many future steps to keep buffered (auto if None).
            max_dyaw (float): max heading change per step [rad] (rate limit on the
                turn). Small -> gentle arcs, which keeps the LIP laterally stable.
                Defaults to 10 deg/step.
            turn_deadband (float): command speed [m/s] below which the heading is
                held (no turning) and the robot steps in place.
        """
        self.conf = conf
        self.goal = np.asarray(goal, dtype=float)[:2].copy()
        self.sy = conf.step_size_y
        self.step_dur = conf.step_dur
        self.max_step_len = (1.4 * conf.step_size_x) if max_step_len is None else max_step_len
        self.goal_tol = goal_tol
        self.samples_per_step = int(samples_per_step)
        self.samples_per_horizon = int(samples_per_horizon)
        self.max_dyaw = np.deg2rad(10.0) if max_dyaw is None else float(max_dyaw)
        self.turn_deadband = float(turn_deadband)

        need = int(np.ceil(self.samples_per_horizon / self.samples_per_step)) + 2
        self.n_lookahead = need if n_lookahead is None else int(n_lookahead)

        # persistent body heading (world yaw), advanced one step per regenerate().
        # seeded from the initial support foot so straight walking starts unchanged.
        R = np.asarray(support_pose.rotation)
        self.psi = float(np.arctan2(R[1, 0], R[0, 0]))
        self.z = float(support_pose.translation[2])

        self.buffer = []   # [support, next_swing, future_1, ...]

    # ------------------------------------------------------------------ #
    # goal
    # ------------------------------------------------------------------ #

    def at_goal(self, position):
        position = np.asarray(position, dtype=float)[:2]
        return np.hypot(*(self.goal - position)) <= self.goal_tol

    # ------------------------------------------------------------------ #
    # steering + step placement
    # ------------------------------------------------------------------ #

    @staticmethod
    def _left_dir(psi):
        """Unit vector 90 deg to the left of heading psi."""
        return np.array([-np.sin(psi), np.cos(psi)])

    @staticmethod
    def _rot_z(psi):
        c, s = np.cos(psi), np.sin(psi)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _advance_heading(self, psi, v_cmd):
        """One steering update: rotate ``psi`` toward the v_cmd direction (rate-
        limited, deadbanded) and return the new heading plus the forward step
        length along it.

        The forward length uses the command speed projected onto the NEW heading,
        so a command pointing far off the current facing yields ~zero length (the
        robot turns in place) and only walks forward once it is roughly aligned.
        """
        v = np.asarray(v_cmd, dtype=float)[:2]
        speed = np.hypot(*v)
        if speed < self.turn_deadband:
            # no meaningful command -> hold heading, step in place
            return psi, 0.0
        psi_des = np.arctan2(v[1], v[0])
        dpsi = np.clip(_wrap(psi_des - psi), -self.max_dyaw, self.max_dyaw)
        psi_new = psi + dpsi
        # forward speed = command projected onto the new facing (>= 0)
        fwd = float(v @ np.array([np.cos(psi_new), np.sin(psi_new)]))
        length = float(np.clip(max(0.0, fwd) * self.step_dur, 0.0, self.max_step_len))
        return psi_new, length

    def _place_foot(self, centre_xy, psi, swing_side):
        """Place the swing foot ``step_size_y`` to its own side of the pelvis path
        point ``centre_xy``, facing heading ``psi``."""
        lateral = self.sy if swing_side == Side.LEFT else -self.sy
        pos = np.asarray(centre_xy, float)[:2] + lateral * self._left_dir(psi)
        pose = pin.SE3(self._rot_z(psi), np.array([pos[0], pos[1], self.z]))
        return FootStep(pose, _footprint(self.conf, swing_side), swing_side)

    def regenerate(self, support_fs, v_cmd, at_goal=False):
        """Rebuild the lookahead buffer from the current support foot and v_cmd,
        advancing the persistent body heading by one (rate-limited) step toward
        the command. Call once per step transition. Returns the buffer.
        """
        vc = np.zeros(2) if at_goal else np.asarray(v_cmd, dtype=float)[:2]

        # pelvis ground point implied by the support foot at the current heading
        support_xy = np.asarray(support_fs.pose.translation)[:2]
        lat_support = self.sy if support_fs.side == Side.LEFT else -self.sy
        centre = support_xy - lat_support * self._left_dir(self.psi)

        self.buffer = [support_fs]
        psi = self.psi
        swing_side = other_foot_id(support_fs.side)
        first_psi = None
        for _ in range(self.n_lookahead):
            psi, length = self._advance_heading(psi, vc)
            centre = centre + length * np.array([np.cos(psi), np.sin(psi)])
            self.buffer.append(self._place_foot(centre, psi, swing_side))
            if first_psi is None:
                first_psi = psi
            swing_side = other_foot_id(swing_side)

        # commit the heading of the next (executed) step so the next regenerate
        # continues smoothly from it
        if first_psi is not None:
            self.psi = first_psi
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
            samples of this step, then each subsequent buffered footstep centre;
            the yaw reference carries each footstep's heading so the MPC's ZMP
            feasibility box tracks the turning foot.
        """
        remaining = self.samples_per_step - int(mpc_phase)
        remaining = int(np.clip(remaining, 1, self.samples_per_step))

        def _foot_yaw(fs):
            R = np.asarray(fs.pose.rotation)
            return float(np.arctan2(R[1, 0], R[0, 0]))

        zmp, yaw = [], []
        c0 = np.asarray(self.buffer[0].pose.translation)[:2]
        y0 = _foot_yaw(self.buffer[0])
        for _ in range(remaining):
            zmp.append(c0)
            yaw.append(y0)

        bi = 1
        while len(zmp) < self.samples_per_horizon and bi < len(self.buffer):
            c = np.asarray(self.buffer[bi].pose.translation)[:2]
            yb = _foot_yaw(self.buffer[bi])
            for _ in range(self.samples_per_step):
                if len(zmp) < self.samples_per_horizon:
                    zmp.append(c)
                    yaw.append(yb)
            bi += 1
        while len(zmp) < self.samples_per_horizon:      # safety pad
            zmp.append(zmp[-1])
            yaw.append(yaw[-1])

        zmp = np.asarray(zmp[:self.samples_per_horizon], dtype=float)
        yaw = np.asarray(yaw[:self.samples_per_horizon], dtype=float)
        return zmp, yaw
