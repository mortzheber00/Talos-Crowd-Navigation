"""reactive_footstep_planner.py
Online, one-step-at-a-time footstep planning that follows a commanded twist.

Instead of a fixed footstep plan (planLine / planCircle), this generates the next
footstep every step from the current planar velocity command ``(v_x, v_y)`` plus
yaw rate ``omega_z`` coming from the robot's ORCA layer / heading controller.
Each step:

  * if the robot is at the goal, it steps in place (halts);
  * otherwise the planner rolls a virtual base pose (x, y, psi) forward along
    the circular arc defined by the commanded twist (SE(2) exponential map,
    radius R = |v| / omega, straight-line limit for small omega) and places the
    swing foot at its nominal lateral offset ``+-sy`` in the *heading* frame,
    with the foot yaw tangent to the arc.

Stability safeguards:

  * **No-crossing constraint**: in the heading frame the left foot must stay at
    least ``min_foot_sep`` left of the right foot (and vice versa). Commands
    that would cross the feet ("scissor steps") are clamped outward into
    sidesteps; lateral overstretch is clamped by ``max_foot_sep``.
  * **Per-step limits**: yaw delta per step <= ``max_dyaw_per_step``, step
    length <= ``max_step_len``.
  * **Dynamic step duration**: tight turns (high |omega|) and fast walking get
    shorter/faster steps. Durations are quantised to multiples of ``dt_mpc`` so
    step transitions stay aligned with the MPC updates; every buffered step
    carries its own duration (``buffer_samples``).

The planner keeps a short lookahead buffer (a few steps ahead, all extrapolated
from the current command) so the LIP-MPC can see the ZMP/yaw reference it needs.
The buffer is regenerated from the current support foot each step, so the
committed next step always reflects the latest command (~ one-step reactivity).

Everything is 2D in the world ground plane (x, y, yaw), SI units.
"""

import numpy as np
import pinocchio as pin

from bullet_sims.footstep_planner import Side, FootStep, other_foot_id


def _footprint(conf, side):
    """Foot-vertex matrix for a side (same convention as FootStepPlanner)."""
    if side == Side.LEFT:
        return np.array([[conf.lfxp, conf.lfxn], [conf.lfyp, conf.lfyn], [0, 0]])
    return np.array([[conf.lfxn, conf.lfxp], [conf.lfyn, conf.lfyp], [0, 0]])


def wrap_angle(a):
    """Wrap an angle to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_of(pose):
    """Yaw angle of a pin.SE3 (rotation about world z)."""
    R = np.asarray(pose.rotation)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rot_z(yaw):
    """3x3 rotation about z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _e_left(psi):
    """Unit vector pointing left of heading psi (heading-frame +y axis)."""
    return np.array([-np.sin(psi), np.cos(psi)])


class ReactiveFootStepPlanner:
    """Generates footsteps that follow a (v_x, v_y, omega_z) command toward a goal."""

    def __init__(self, conf, goal, support_pose,
                 samples_per_step, samples_per_horizon,
                 max_step_len=None, goal_tol=0.3, n_lookahead=None,
                 min_sep=None, max_sep=None, omega_max=None,
                 max_dyaw_step=None, step_dur_bounds=None):
        """
        Args:
            conf: talos_conf module (uses step_size_x/y, dt_mpc, foot half-sizes,
                and the reactive-gait limits unless overridden below).
            goal (array-like, 2): navigation goal in the world plane [m].
            support_pose (pin.SE3): the current support foot pose; fixes the
                initial heading and the foot ground height for generated steps.
            samples_per_step (int): nominal MPC samples per step
                (conf.no_mpc_samples_per_step).
            samples_per_horizon (int): MPC samples over the horizon
                (conf.no_mpc_samples_per_horizon).
            max_step_len (float): clamp on the per-step travel distance [m].
            goal_tol (float): distance [m] at which the goal counts as reached.
            n_lookahead (int): how many future steps to keep buffered (auto if None).
            min_sep (float): no-crossing lateral clearance between feet [m].
            max_sep (float): max lateral spread between feet [m].
            omega_max (float): clamp on the commanded yaw rate [rad/s].
            max_dyaw_step (float): max yaw delta per step [rad].
            step_dur_bounds ((float, float)): (min, max) step duration [s], each
                a multiple of conf.dt_mpc.
        """
        self.conf = conf
        self.goal = np.asarray(goal, dtype=float)[:2].copy()
        self.sy = conf.step_size_y
        self.step_dur = conf.step_dur
        self.max_step_len = (1.4 * conf.step_size_x) if max_step_len is None else max_step_len
        self.goal_tol = goal_tol
        self.samples_per_step = int(samples_per_step)
        self.samples_per_horizon = int(samples_per_horizon)

        self.min_sep = conf.min_foot_sep if min_sep is None else float(min_sep)
        self.max_sep = conf.max_foot_sep if max_sep is None else float(max_sep)
        self.omega_max = conf.omega_max if omega_max is None else float(omega_max)
        self.max_dyaw_step = (conf.max_dyaw_per_step if max_dyaw_step is None
                              else float(max_dyaw_step))
        self.v_back_max = getattr(conf, 'v_back_max', 0.15)
        self.v_lat_max = getattr(conf, 'v_lat_max', 0.15)
        self.max_reach = getattr(conf, 'max_foot_reach', 0.40)
        # CoM feedback into the base seed (see regenerate): gain on the measured
        # CoM offset from the foot-anchored base, and a clamp on the shift
        self.base_fb_gain = getattr(conf, 'base_fb_gain', 0.7)
        self.base_fb_max = getattr(conf, 'base_fb_max', 0.10)

        self.dt_mpc = conf.dt_mpc
        dur_min, dur_max = ((conf.step_dur_min, conf.step_dur_max)
                            if step_dur_bounds is None else step_dur_bounds)
        self.n_min = max(1, int(round(dur_min / self.dt_mpc)))
        self.n_max = max(self.n_min, int(round(dur_max / self.dt_mpc)))

        # the buffer must cover the horizon even at the fastest step cadence
        need = int(np.ceil(self.samples_per_horizon / self.n_min)) + 2
        self.n_lookahead = need if n_lookahead is None else int(n_lookahead)

        # heading state; yaw0 kept as the initial value (spawn heading)
        self.yaw0 = yaw_of(support_pose)
        self.heading = self.yaw0
        self.z = float(support_pose.translation[2])

        self.buffer = []          # [support, next_swing, future_1, ...]
        self.buffer_samples = []  # buffer_samples[k] = MPC samples buffer[k] is support

    # ------------------------------------------------------------------ #
    # goal
    # ------------------------------------------------------------------ #

    def at_goal(self, position):
        position = np.asarray(position, dtype=float)[:2]
        return np.hypot(*(self.goal - position)) <= self.goal_tol

    # ------------------------------------------------------------------ #
    # command shaping
    # ------------------------------------------------------------------ #

    def _clamp_command(self, v_world, omega):
        """Clamp the twist to per-step limits and pick the step duration.

        Tight turns and fast walking shorten the step (down to n_min MPC
        samples) so the gait can catch the centrifugal drift; the yaw delta and
        travel distance per step stay bounded regardless of the raw command.

        Returns (v (2,), omega, n_samples).
        """
        v = np.asarray(v_world, dtype=float)[:2].copy()
        omega = float(np.clip(omega, -self.omega_max, self.omega_max))
        speed = float(np.hypot(*v))

        # dynamic duration: nominal, reduced by turn rate and speed
        v_ref = self.max_step_len / (self.n_max * self.dt_mpc)
        turn_frac = abs(omega) / self.omega_max if self.omega_max > 0.0 else 0.0
        speed_frac = min(speed / v_ref, 1.0) if v_ref > 0.0 else 0.0
        T_raw = (self.n_max * self.dt_mpc) * (1.0 - 0.5 * turn_frac - 0.25 * speed_frac)
        n = int(np.clip(int(round(T_raw / self.dt_mpc)), self.n_min, self.n_max))
        T = n * self.dt_mpc

        # per-step yaw delta clamp
        if abs(omega) * T > self.max_dyaw_step:
            omega = np.sign(omega) * self.max_dyaw_step / T
        # per-step travel clamp
        if speed * T > self.max_step_len:
            v *= self.max_step_len / (speed * T)
        return v, omega, n

    # ------------------------------------------------------------------ #
    # step placement
    # ------------------------------------------------------------------ #

    @staticmethod
    def _integrate_arc(xy, psi, v_world, omega, T):
        """Roll the base pose along the circular arc of the twist for time T.

        Exact SE(2) exponential: the base moves on an arc of radius |v|/omega
        (handles lateral velocity components too); straight-line limit for
        small omega. Returns (xy_new, psi_new).
        """
        c, s = np.cos(psi), np.sin(psi)
        R = np.array([[c, -s], [s, c]])
        vb = R.T @ np.asarray(v_world, dtype=float)[:2]   # body-frame velocity
        dpsi = omega * T
        u = vb * T
        if abs(dpsi) < 1e-6:
            dp_body = u
        else:
            sd, cd = np.sin(dpsi), np.cos(dpsi)
            V = (1.0 / dpsi) * np.array([[sd, -(1.0 - cd)], [1.0 - cd, sd]])
            dp_body = V @ u
        return np.asarray(xy, dtype=float) + R @ dp_body, wrap_angle(psi + dpsi)

    def _place_next(self, base_xy, psi, support_fs, v_cmd, omega,
                    pos_offset=None):
        """Advance the base one step along the commanded arc and place the
        swing foot (opposite of support_fs.side), enforcing the no-crossing
        constraint against the support foot.

        pos_offset (2,) shifts the nominal placement (capture-step balance
        correction) BEFORE the no-crossing clamp, so the correction can never
        cross the feet.

        Returns (FootStep, base_xy_new, psi_new, n_samples).
        """
        swing_side = other_foot_id(support_fs.side)
        # bipedal gait limits in the BODY frame: backward and lateral walking
        # have far less stability margin (asymmetric foot, and the no-crossing
        # clamp halves the effective lateral step rate), so cap those
        # components regardless of what the avoidance layer asks for
        c, s = np.cos(psi), np.sin(psi)
        Rp = np.array([[c, -s], [s, c]])
        vb = Rp.T @ np.asarray(v_cmd, dtype=float)[:2]
        vb[0] = np.clip(vb[0], -self.v_back_max, None)
        vb[1] = np.clip(vb[1], -self.v_lat_max, self.v_lat_max)
        v, om, n = self._clamp_command(Rp @ vb, omega)
        if np.hypot(*v) < 1e-4 and abs(om) < 1e-4:
            # no command -> step in place at nominal cadence
            base_new, psi_new, n = base_xy, psi, self.n_max
        else:
            base_new, psi_new = self._integrate_arc(base_xy, psi, v, om, n * self.dt_mpc)

        # nominal placement: lateral offset in the HEADING frame (feet straddle
        # the base, tangent to the arc) - not perpendicular to travel
        e_left = _e_left(psi_new)
        sign = 1.0 if swing_side == Side.LEFT else -1.0
        pos = base_new + sign * self.sy * e_left
        if pos_offset is not None:
            pos = pos + np.asarray(pos_offset, dtype=float)[:2]

        # kinematic reach clamp: a target beyond leg reach never lands and
        # collapses the stance contact - cap the distance to the support foot
        p_support = np.asarray(support_fs.pose.translation)[:2]
        d = pos - p_support
        dn = float(np.hypot(*d))
        if dn > self.max_reach:
            pos = p_support + d * (self.max_reach / dn)

        # no-crossing / spread clamp relative to the actual support foot.
        # While turning, the new heading and the support-foot yaw differ by up
        # to max_dyaw_step; a placement that clears min_sep in the new-heading
        # frame can still cross the legs in the support-foot frame (which is
        # where the hips actually are). Enforce the clamp in BOTH frames, the
        # support frame last (crossing is deadlier than lateral overspread).
        yaw_sup = yaw_of(support_fs.pose)
        for e_lat in (e_left, _e_left(yaw_sup)):
            lat = float(e_lat @ (pos - p_support))
            if sign * lat < self.min_sep:
                pos += (self.min_sep - sign * lat) * sign * e_lat
            elif sign * lat > self.max_sep:
                pos -= (sign * lat - self.max_sep) * sign * e_lat

        # foot yaw tangent to the arc, but bounded w.r.t. the support foot so a
        # noisy measured support yaw cannot demand a huge leg twist
        foot_yaw = wrap_angle(
            yaw_sup + np.clip(wrap_angle(psi_new - yaw_sup),
                              -self.max_dyaw_step, self.max_dyaw_step))

        pose = pin.SE3(rot_z(foot_yaw), np.array([pos[0], pos[1], self.z]))
        fs = FootStep(pose, _footprint(self.conf, swing_side), swing_side)
        return fs, base_new, psi_new, n

    def regenerate(self, support_fs, v_cmd, at_goal=False, omega_cmd=0.0,
                   first_step_offset=None, com_xy=None):
        """Rebuild the lookahead buffer from the current support foot and the
        commanded twist (v_cmd, omega_cmd). Call once per step transition.

        first_step_offset (2,): capture-step balance correction applied to the
        committed next step only (e.g. the measured-vs-planned DCM error), so
        a disturbed robot steps toward where it is falling. The no-crossing
        clamp still bounds the corrected placement.

        com_xy (2,): measured CoM ground position at the step transition.
        The virtual base is otherwise anchored ONLY to the support-foot chain,
        which is open loop w.r.t. the body: if the CoM drifts, the footsteps
        follow the feet and the MPC's limited ZMP box is left to fight the
        offset alone. At a transition the CoM crosses the midline between the
        feet - exactly the base line - so (com - base) is a sway-free drift
        measurement; a fraction of it is fed into the base seed so the steps
        track the body, not just the feet.

        Returns the buffer."""
        if at_goal:
            vc, om = np.zeros(2), 0.0
        else:
            vc, om = np.asarray(v_cmd, dtype=float)[:2], float(omega_cmd)

        # heading seeded from the measured support-foot yaw, but bounded
        # against the previous heading: a bad yaw measurement (foot still
        # settling after an early touchdown, sole tilted mid-disturbance) must
        # not swing the whole plan around in one step
        psi_meas = yaw_of(support_fs.pose)
        dpsi = np.clip(wrap_angle(psi_meas - self.heading),
                       -2.0 * self.max_dyaw_step, 2.0 * self.max_dyaw_step)
        psi = wrap_angle(self.heading + dpsi)

        sign_sup = 1.0 if support_fs.side == Side.LEFT else -1.0
        base_xy = (np.asarray(support_fs.pose.translation)[:2]
                   - sign_sup * self.sy * _e_left(psi))
        if com_xy is not None:
            err = np.asarray(com_xy, dtype=float)[:2] - base_xy
            shift = self.base_fb_gain * err
            sn = float(np.hypot(*shift))
            if sn > self.base_fb_max:
                shift *= self.base_fb_max / sn
            base_xy = base_xy + shift
        self.heading = psi

        self.buffer = [support_fs]
        self.buffer_samples = []
        foot = support_fs
        for k in range(self.n_lookahead):
            foot, base_xy, psi, n = self._place_next(
                base_xy, psi, foot, vc, om,
                pos_offset=first_step_offset if k == 0 else None)
            self.buffer.append(foot)
            # n is the swing duration toward `foot`, i.e. the support duration
            # of the previous buffer entry
            self.buffer_samples.append(n)
        self.buffer_samples.append(self.buffer_samples[-1])   # pad last entry
        return self.buffer

    def make_footstep(self, pose, side):
        """Wrap a pose + side into a FootStep with the correct footprint."""
        return FootStep(pose, _footprint(self.conf, side), side)

    def swing_target(self):
        """The next footstep to step to (buffer[1]); valid after regenerate()."""
        return self.buffer[1]

    def step_samples(self):
        """MPC samples of the committed step; valid after regenerate()."""
        return self.buffer_samples[0]

    def step_duration(self):
        """Duration [s] of the committed step; valid after regenerate()."""
        return self.buffer_samples[0] * self.dt_mpc

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
            samples of this step, then each subsequent buffered footstep centre
            for its own duration; the yaw reference carries the corresponding
            support-foot yaw (feeds the yaw-rotated ZMP box in the MPC).
        """
        n0 = self.buffer_samples[0]
        remaining = int(np.clip(n0 - int(mpc_phase), 1, n0))

        zmp, yaw = [], []

        def emit(fs, count):
            c = np.asarray(fs.pose.translation)[:2]
            y = yaw_of(fs.pose)
            for _ in range(count):
                if len(zmp) < self.samples_per_horizon:
                    zmp.append(c)
                    yaw.append(y)

        emit(self.buffer[0], remaining)
        bi = 1
        while len(zmp) < self.samples_per_horizon and bi < len(self.buffer):
            emit(self.buffer[bi], self.buffer_samples[bi])
            bi += 1
        while len(zmp) < self.samples_per_horizon:      # safety pad
            zmp.append(zmp[-1])
            yaw.append(yaw[-1])

        return (np.asarray(zmp, dtype=float),
                np.asarray(yaw, dtype=float))
