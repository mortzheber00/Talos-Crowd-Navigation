"""
talos walking simulation
"""

import os
from types import SimpleNamespace

import numpy as np
import pinocchio as pin

import rclpy
from rclpy.node import Node

# simulator
from simulator.mujoco_wrapper import MujocoWrapper as PybulletWrapper
from simulator.robot import Robot

# robot configs
import bullet_sims.talos_conf as conf
from bullet_sims.tsid_wrapper import TSIDWrapper

# modules
from bullet_sims.footstep_planner import Side, FootStepPlanner, FootStep, other_foot_id
from bullet_sims.foot_trajectory import SwingFootTrajectory
from bullet_sims.talos import Talos
from bullet_sims.lip_mpc import LIPMPC, LIPInterpolator, generate_zmp_reference, generate_yaw_reference
from bullet_sims.reactive_footstep_planner import (ReactiveFootStepPlanner,
                                                   wrap_angle, yaw_of, rot_z)

# crowd
from orca_nav.robot_orca import RobotORCA
from orca_nav import debug_viz

# crowd
from orca_nav import scenarios
from orca_nav.pedestrian import PedestrianSimulator




# footstep path for the walking demo: "line" (straight) or "circle" (curved)
WALK_MODE = "circle"
CIRCLE_RADIUS = 1.5     # [m] radius of the centre path when WALK_MODE == "circle"
CIRCLE_CLOCKWISE = False   # False = curve left/CCW, True = curve right/CW
# False = omnidirectional (fixed facing, side-steps along the curve) -> robust: reuses
#         the straight-line gait, no body rotation, best for bounded curves (avoidance).
# True  = turn the whole body to face the path -> more natural, harder to stabilise.
CIRCLE_FACE_PATH = False

# robot navigation: goal it heads toward, and the arrow display scale.
# v_cmd (from ORCA) now DRIVES the reactive footstep planner; green arrow = v_cmd.
ROBOT_GOAL = (4.0, 0.0)    # world (x, y) navigation goal
ARROW_SCALE = 3.0          # visual stretch so the slow v_cmd arrow is visible
WALK_DURATION = 45.0       # [s] max walk time (reactive walking has no fixed step count)
REACTIVE = True            # True: footsteps follow v_cmd; False: fixed WALK_MODE plan

# ---- runtime parameters (defaults; override at launch via ROS 2 params, e.g.
#   ros2 run bullet_sims walking --ros-args -p scenario:=crossing -p pedestrian_sensor_data:=true )
SCENARIO = "random"              # crowd scenario: head_on | crossing | circle | corridor | random
PEDESTRIAN_SENSOR_DATA = True   # True: depth-camera perception; False: ground-truth states
# WALK selects what drives the reactive gait:
#   "goal"   - ORCA velocity toward ROBOT_GOAL, avoiding pedestrians (default)
#   "circle" - constant twist (CIRCLE_SPEED forward + omega = speed/radius):
#              the robot walks a circular arc with the feet tangent to it,
#              ignoring ORCA. E.g.:
#   ros2 run bullet_sims walking --ros-args -p walk:=circle -p circle_radius:=1.5
WALK = "goal"
CIRCLE_SPEED = 0.3               # [m/s] forward speed on the circle
# USE_HEADING toggles the whole heading integration (runtime: -p use_heading:=false):
#   True  - the robot turns into its direction of travel: HeadingController
#           yields omega_z, footsteps/torso yaw with the path, the MPC's ZMP
#           box rotates with the support foot.
#   False - omnidirectional gait: omega_z = 0 everywhere, the feet keep their
#           spawn yaw and the robot side-steps along curved paths (the old,
#           pre-heading behavior; also the fallback if turning misbehaves).
USE_HEADING = False

# reactive walking: per-MPC-tick gain pulling the open-loop LIP interpolator
# state toward the measured CoM. Must exceed 1 - exp(-omega*dt_mpc) ~ 0.29
# (the LIP divergence rate per MPC interval), otherwise plan and robot still
# drift apart exponentially - just more slowly.
MPC_STATE_FB = 0.5
MPC_STATE_RESET_DIST = 0.25   # [m] plan/robot CoM gap that forces a hard reset
# capture stepping: the DCM orbits the support foot by up to ~this radius in a
# nominal gait; only the excess is treated as a disturbance and stepped toward
DCM_NOMINAL_RADIUS = 0.12     # [m]
CAPTURE_STEP_MAX = 0.12       # [m] clamp on the correction at step transitions
# at a step COMMIT the DCM sits close to the support->next segment in a
# nominal gait (the big excursions happen mid-step), so the commit-time
# capture correction uses this tighter deadband
DCM_COMMIT_RADIUS = 0.08      # [m]
CAPTURE_RETARGET_MAX = 0.15   # [m] clamp on mid-step swing retargeting
CAPTURE_RETARGET_CUTOFF = 0.7  # retarget only in the first fraction of the step
EARLY_CONTACT_FZ = 100.0      # [N] swing-foot load that ends the step early
# retargeting + early-touchdown are protections for DYNAMIC walking; at low
# speed (braking, standing at the goal) their signals are ill-conditioned
# (tiny reference segments, trippling contacts) and plain MPC/ZMP standing -
# which was always rock solid - takes over. A TURNING gait is always dynamic,
# regardless of speed: the yaw-rate-proportional disturbances (touchdown yaw
# error, stance-foot yaw torsion) grow the step-to-step sway mode, and the
# DCM retargeting is the layer that damps it - so it must stay armed while
# omega is commanded (a 0.1 m/s circle walk otherwise falls in ~10 steps).
DYNAMIC_GAIT_SPEED = 0.12     # [m/s] |v_filt| above which they are active
DYNAMIC_GAIT_OMEGA = 0.01     # [rad/s] |omega_cmd| above which they are active


################################################################################
# setup helpers
################################################################################

def _setup_simulator():
    """Init ROS, set the sim-rate-dependent timing on ``conf``, and build the
    MuJoCo simulator."""
    if not rclpy.ok():
        rclpy.init()

    frequency = 300
    conf.dt = 1.0 / frequency
    conf.no_sim_per_mpc = int(round(conf.dt_mpc / conf.dt))
    conf.no_sim_per_step = int(round(conf.step_dur / conf.dt))
    # step transitions are scheduled in multiples of no_sim_per_mpc ticks, which
    # only lines up with the MPC updates if dt_mpc is an exact multiple of dt
    assert abs(conf.no_sim_per_mpc * conf.dt - conf.dt_mpc) < 1e-9, \
        "dt_mpc must be an integer multiple of the sim dt"

    # WALKING_RENDER=0 runs headless (no viewer window, no real-time pacing) -
    # useful for batch tests/CI where the run should go as fast as possible
    render = os.environ.get('WALKING_RENDER', '1') not in ('0', 'false', 'False')
    return PybulletWrapper(sim_rate=frequency, num_solver_iterations=50,
                           num_sub_steps=8, render=render)


def _setup_crowd(simulator):
    """Spawn the pedestrian crowd.

    IMPORTANT: this must run *before* the robot is created. Adding pedestrian
    bodies changes the MuJoCo model, which forces a recompile that resets
    simulation state; doing it after the robot would wipe the robot's pose.
    """
    crowd = PedestrianSimulator(simulator, radius=0.3, max_speed=1.3,
                                pref_perturb=0.05, personal_space=0.3)
    scenarios.build(crowd, SCENARIO)
    return crowd


def _create_robot(simulator):
    """Create the TSID wrapper and the MuJoCo Talos robot.
    Returns (tw, robot), or None if the TSID wrapper fails to build."""
    try:
        tw = TSIDWrapper(conf)
    except Exception as e:
        print('Failed to create TSIDWrapper:', e)
        return None
    robot = Talos(simulator)
    return tw, robot


def _build_plan(robot):
    """Build the straight-line footstep plan. The final step is duplicated twice
    so the MPC's receding horizon never looks past the end of the plan."""
    no_steps = 20
    planner = FootStepPlanner(conf)
    # plan[0] must be the current SUPPORT side/pose (stepping starts at plan_idx=1),
    # else the first swing lifts the loaded foot.
    T_support_w = robot.supportFootPose()
    if WALK_MODE == "circle":
        plan = planner.planCircle(T_support_w, robot.support_foot, no_steps=no_steps,
                                  radius=CIRCLE_RADIUS, clockwise=CIRCLE_CLOCKWISE,
                                  face_path=CIRCLE_FACE_PATH)
    else:
        plan = planner.planLine(T_support_w, robot.support_foot, no_steps=no_steps)
    plan.append(plan[-2])
    plan.append(plan[-1])
    return planner, plan, no_steps


def _setup_lip_mpc(robot):
    """Set up the LIP MPC + interpolator, and the CoM endpoints for the pre-walk
    lateral sway (from the spawn CoM to over the first support foot)."""
    mpc = LIPMPC(conf)

    T_support_w = robot.supportFootPose()
    x0 = np.array([T_support_w.translation[0], 0.0, T_support_w.translation[1], 0.0])
    interpolator = LIPInterpolator(x0, conf)

    com_start = robot.stack.comState().value().copy()
    c, _, _ = interpolator.comState()
    com_target = c.copy()
    return mpc, interpolator, com_start, com_target


################################################################################
# logging
################################################################################

def _allocate_logs(N_sim, n_peds=0):
    """Allocate all per-iteration log buffers as NaN-filled arrays."""
    def nan(*shape):
        return np.nan * np.empty(shape if len(shape) > 1 else shape[0])

    return SimpleNamespace(
        # --- crowd navigation ---
        ROBOT_XY=nan(N_sim, 2),                 # robot CoM ground path
        PED_XY=nan(N_sim, max(n_peds, 1), 2),   # pedestrian ground paths
        MIN_SEP=nan(N_sim),                     # robot<->nearest-ped surface gap [m]
        DIST_GOAL=nan(N_sim),                   # robot distance to goal [m]
        VCMD=nan(N_sim, 2),                     # ORCA velocity command
        VPREF=nan(N_sim, 2),                    # goal-direction preferred velocity
        OMEGA_CMD=nan(N_sim),                   # heading-controller yaw-rate command
        STEP_DUR=nan(N_sim),                    # committed step duration [s]
        HEADING=nan(N_sim),                     # planner heading (support-foot yaw)
        GOAL=np.asarray((0.0, 0.0)),            # filled in _run_loop
        PED_RADII=np.zeros(max(n_peds, 1)),     # filled in _run_loop
        # CoM position/velocity/acceleration: reference (LIP), pinocchio, pybullet
        COM_POS_REF=nan(N_sim, 3), COM_VEL_REF=nan(N_sim, 3), COM_ACC_REF=nan(N_sim, 3),
        COM_POS_PIN=nan(N_sim, 3), COM_VEL_PIN=nan(N_sim, 3), COM_ACC_PIN=nan(N_sim, 3),
        COM_POS_PB=nan(N_sim, 3), COM_VEL_PB=nan(N_sim, 3), COM_ACC_PB=nan(N_sim, 3),
        # centroidal angular momentum (pinocchio)
        ANG_MOM_PIN=nan(N_sim, 3),
        # left/right foot pos/vel/acc: reference and pinocchio
        LF_POS_REF=nan(N_sim, 3), LF_VEL_REF=nan(N_sim, 3), LF_ACC_REF=nan(N_sim, 3),
        RF_POS_REF=nan(N_sim, 3), RF_VEL_REF=nan(N_sim, 3), RF_ACC_REF=nan(N_sim, 3),
        LF_POS_PIN=nan(N_sim, 3), LF_VEL_PIN=nan(N_sim, 3), LF_ACC_PIN=nan(N_sim, 3),
        RF_POS_PIN=nan(N_sim, 3), RF_VEL_PIN=nan(N_sim, 3), RF_ACC_PIN=nan(N_sim, 3),
        # ZMP (reference / estimated) and DCM (estimated)
        ZMP_REF_LOG=nan(N_sim, 2), ZMP_EST=nan(N_sim, 2), DCM_EST=nan(N_sim, 2),
        # normal foot forces: pybullet ft sensor vs pinocchio qp
        FZ_LEFT_PB=nan(N_sim), FZ_RIGHT_PB=nan(N_sim),
        FZ_LEFT_PIN=nan(N_sim), FZ_RIGHT_PIN=nan(N_sim),
        # yaw chain: pelvis yaw, hip-yaw joints (left q[7], right q[13]),
        # torso_1 joint - hip yaw limits are the hard constraint in turning
        # (leg_right_1: [-1.571, +0.349], leg_left_1: [-0.349, +1.571])
        YAW_CHAIN=nan(N_sim, 4),
        TIME=nan(N_sim),
    )


def _record_step(L, i, t, c, c_dot, c_ddot, t_step_elapsed, swing_trajectory,
                 robot, simulator, interpolator, pin_data):
    """Store one iteration of state for the post-run plots."""
    L.TIME[i] = t

    # reference (planned) com from the lip interpolator
    L.COM_POS_REF[i] = c
    L.COM_VEL_REF[i] = c_dot
    L.COM_ACC_REF[i] = c_ddot

    # com from the pinocchio / tsid model
    com_state = robot.stack.comState()
    L.COM_POS_PIN[i] = com_state.value()
    L.COM_VEL_PIN[i] = com_state.derivative()
    L.COM_ACC_PIN[i] = com_state.second_derivative()

    # com from pybullet (no direct com acceleration -> left as nan)
    body_id = robot.robot.id()
    L.COM_POS_PB[i] = simulator.computeComPosition(body_id)
    L.COM_VEL_PB[i] = simulator.computeComVelocity(body_id)

    # centroidal angular momentum from pinocchio
    q, v = robot.robot.q(), robot.robot.v()
    hg = pin.computeCentroidalMomentum(robot.stack.model, pin_data, q, v)
    L.ANG_MOM_PIN[i] = hg.angular

    # yaw chain: base yaw from the free-flyer quaternion + hip/torso yaw joints
    qw, qx, qy, qz = q[6], q[3], q[4], q[5]
    base_yaw = np.arctan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz))
    L.YAW_CHAIN[i] = (base_yaw, q[7], q[13], q[19])

    # foot poses / velocities from pinocchio (acc needs the qp dv -> nan)
    H_lf, v_lf = robot.stack.get_pose_vel_acc_LF()
    H_rf, v_rf = robot.stack.get_pose_vel_acc_RF()
    L.LF_POS_PIN[i] = H_lf.translation
    L.RF_POS_PIN[i] = H_rf.translation
    L.LF_VEL_PIN[i] = v_lf.linear
    L.RF_VEL_PIN[i] = v_rf.linear

    # planned foot reference: the swinging foot follows the spline, the support
    # foot is held fixed at its current placement
    if swing_trajectory is not None:
        T_sw, V_sw, A_sw = swing_trajectory.evaluate(t_step_elapsed)
        if robot.swing_foot == Side.LEFT:
            L.LF_POS_REF[i], L.LF_VEL_REF[i], L.LF_ACC_REF[i] = T_sw.translation, V_sw.linear, A_sw.linear
            L.RF_POS_REF[i], L.RF_VEL_REF[i], L.RF_ACC_REF[i] = H_rf.translation, 0.0, 0.0
        else:
            L.RF_POS_REF[i], L.RF_VEL_REF[i], L.RF_ACC_REF[i] = T_sw.translation, V_sw.linear, A_sw.linear
            L.LF_POS_REF[i], L.LF_VEL_REF[i], L.LF_ACC_REF[i] = H_lf.translation, 0.0, 0.0

    # zmp: planned (interpolator command) vs estimated
    L.ZMP_REF_LOG[i] = interpolator.zmp()[:2]
    if robot.zmp is not None:
        L.ZMP_EST[i] = robot.zmp[:2]
    # dcm: estimated
    if robot.dcm is not None:
        L.DCM_EST[i] = robot.dcm[:2]

    # normal forces under each foot: pybullet ft sensor vs pinocchio qp
    f_l, _ = robot._foot_wrench_world(conf.lf_frame_name)
    f_r, _ = robot._foot_wrench_world(conf.rf_frame_name)
    L.FZ_LEFT_PB[i] = f_l[2]
    L.FZ_RIGHT_PB[i] = f_r[2]
    wr_lf = robot.stack.get_wrench_LF(robot.stack.sol)
    wr_rf = robot.stack.get_wrench_RF(robot.stack.sol)
    L.FZ_LEFT_PIN[i] = wr_lf[2]
    L.FZ_RIGHT_PIN[i] = wr_rf[2]


def _plot_results(L):
    """Plot CoM position/velocity/acceleration and ZMP, then save and show."""
    import os
    import matplotlib.pyplot as plt
    try:
        plt.style.use('seaborn-dark')
    except OSError:
        pass  # style was renamed/removed in newer matplotlib

    fig, axs = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    axis_name = ['x', 'y']

    for col in range(2):
        # CoM position
        axs[0, col].plot(L.TIME, L.COM_POS_REF[:, col], label='reference (LIP)')
        axs[0, col].plot(L.TIME, L.COM_POS_PB[:, col], label='pybullet (truth)')
        axs[0, col].plot(L.TIME, L.COM_POS_PIN[:, col], '--', label='pinocchio')
        axs[0, col].set_title(f'CoM position  {axis_name[col]}')
        axs[0, col].set_ylabel('p [m]')

        # CoM velocity
        axs[1, col].plot(L.TIME, L.COM_VEL_REF[:, col], label='reference (LIP)')
        axs[1, col].plot(L.TIME, L.COM_VEL_PB[:, col], label='pybullet (truth)')
        axs[1, col].plot(L.TIME, L.COM_VEL_PIN[:, col], '--', label='pinocchio')
        axs[1, col].set_title(f'CoM velocity  {axis_name[col]}')
        axs[1, col].set_ylabel('v [m/s]')

        # CoM acceleration (no direct pybullet acceleration -> reference vs pinocchio)
        axs[2, col].plot(L.TIME, L.COM_ACC_REF[:, col], label='reference (LIP)')
        axs[2, col].plot(L.TIME, L.COM_ACC_PIN[:, col], '--', label='pinocchio')
        axs[2, col].set_title(f'CoM acceleration  {axis_name[col]}')
        axs[2, col].set_ylabel('a [m/s^2]')

        # ZMP: planned reference vs estimated from the foot wrenches
        axs[3, col].plot(L.TIME, L.ZMP_REF_LOG[:, col], label='reference')
        axs[3, col].plot(L.TIME, L.ZMP_EST[:, col], label='estimated')
        axs[3, col].set_title(f'ZMP  {axis_name[col]}')
        axs[3, col].set_ylabel('p_zmp [m]')
        axs[3, col].set_xlabel('t [s]')

    for ax in axs.ravel():
        ax.legend(fontsize=7)

    fig.tight_layout()
    # save next to this package (repo-relative) instead of a hardcoded absolute
    # path, and never let a failed save/show abort the run
    plots_dir = os.environ.get(
        'WALKING_PLOTS_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'T7'))
    try:
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, 'walking_results_random.png')
        fig.savefig(out_path, dpi=150)
        print(f"saved plots to {out_path}")
    except OSError as e:
        print(f"could not save plots to {plots_dir}: {e}")
    # Only pop up a (blocking) window if explicitly requested; otherwise just save.
    # Interactive windows can be un-closeable over VNC, so default to save-only.
    if os.environ.get('WALKING_SHOW_PLOTS'):
        try:
            plt.show()
        except Exception as e:
            print(f"could not show plots: {e}")
    else:
        plt.close(fig)


def _plot_navigation(L):
    """Crowd-navigation plots: top-down trajectory and the robot<->nearest-ped gap
    (plus goal progress and command speed)."""
    import os
    import matplotlib.pyplot as plt

    fig, axd = plt.subplot_mosaic([['traj', 'sep'],
                                   ['traj', 'turn'],
                                   ['traj', 'prog']], figsize=(14, 9))

    # --- top-down trajectory ---
    ax = axd['traj']
    n_peds = L.PED_XY.shape[1]
    for j in range(n_peds):
        ax.plot(L.PED_XY[:, j, 0], L.PED_XY[:, j, 1], '-', lw=1.0, alpha=0.5,
                color='tab:gray', label='pedestrians' if j == 0 else None)
    ax.plot(L.ROBOT_XY[:, 0], L.ROBOT_XY[:, 1], '-', lw=2.5, color='tab:green',
            label='robot')
    # start (first non-nan robot sample) and goal
    valid = np.where(~np.isnan(L.ROBOT_XY[:, 0]))[0]
    if len(valid):
        s = L.ROBOT_XY[valid[0]]
        ax.plot(s[0], s[1], 'o', color='tab:green', ms=9, label='start')
    ax.plot(L.GOAL[0], L.GOAL[1], '*', color='tab:red', ms=16, label='goal')
    ax.set_title('Top-down trajectory')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # --- robot <-> nearest pedestrian surface gap (safety) ---
    ax = axd['sep']
    ax.plot(L.TIME, L.MIN_SEP, color='tab:blue')
    ax.axhline(0.0, color='tab:red', ls='--', lw=1, label='contact (gap=0)')
    ax.set_title('Distance to nearest pedestrian (surface gap)')
    ax.set_ylabel('gap [m]'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # --- turn command + step cadence ---
    ax = axd['turn']
    ax.plot(L.TIME, L.OMEGA_CMD, color='tab:blue', lw=1, label='omega_cmd')
    ax.set_ylabel('omega [rad/s]', color='tab:blue')
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.step(L.TIME, L.STEP_DUR, color='tab:red', lw=1, where='post',
             label='step duration')
    ax2.set_ylabel('step dur [s]', color='tab:red')
    ax.set_title('Turn command and step cadence')

    # --- progress to goal + command speed ---
    ax = axd['prog']
    ax.plot(L.TIME, L.DIST_GOAL, color='tab:purple', label='distance to goal')
    ax.set_ylabel('dist to goal [m]', color='tab:purple')
    ax.set_xlabel('t [s]'); ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    speed = np.hypot(L.VCMD[:, 0], L.VCMD[:, 1])
    ax2.plot(L.TIME, speed, color='tab:orange', lw=1, label='|v_cmd|')
    ax2.set_ylabel('|v_cmd| [m/s]', color='tab:orange')
    ax.set_title('Goal progress and command speed')

    fig.tight_layout()
    plots_dir = os.environ.get(
        'WALKING_PLOTS_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'T7'))
    try:
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, 'navigation_results_random.png')
        fig.savefig(out_path, dpi=150)
        print(f"saved navigation plots to {out_path}")
    except OSError as e:
        print(f"could not save navigation plots to {plots_dir}: {e}")
    if os.environ.get('WALKING_SHOW_PLOTS'):
        try:
            plt.show()
        except Exception as e:
            print(f"could not show navigation plots: {e}")
    else:
        plt.close(fig)


################################################################################
# main control loop
################################################################################

class HeadingController:
    """P controller that turns the robot into its direction of travel.

    ORCA outputs only a planar velocity; this derives the yaw-rate command
    omega_z = kp * wrap(atan2(v_y, v_x) - heading) for the footstep planner.
    Below the speed deadband no turning is commanded (the robot never
    spins in place from noise while stopping/at the goal).

    The velocity is low-pass filtered first: the raw ORCA command jitters
    tick-to-tick (the walking CoM sway feeds back into the avoidance solver),
    and unfiltered that noise saturates omega with alternating sign - the
    per-step sampled command then yaws the feet in a destabilising zigzag.
    """

    def __init__(self, dt, kp=None, omega_max=None, v_deadband=None, tau=0.4,
                 a_max=0.4, domega_max=0.5, enabled=True):
        self.kp = conf.heading_kp if kp is None else kp
        self.omega_max = conf.omega_max if omega_max is None else omega_max
        self.v_deadband = conf.heading_v_deadband if v_deadband is None else v_deadband
        self.alpha = float(np.clip(dt / tau, 0.0, 1.0))   # EMA gain per tick
        self.dv_max = a_max * dt                          # slew limit per tick
        self.domega_max = domega_max * dt                 # omega slew per tick
        self.enabled = enabled                            # False -> never turn
        self.v_filt = np.zeros(2)
        self.heading_err = 0.0
        self.omega_out = 0.0

    def compute(self, v_cmd, heading):
        # EMA against tick noise, then a hard slew limit: a biped cannot jump
        # its velocity, and ORCA emits step changes (0.25 -> 0.4 m/s) exactly
        # in the tightest moments
        dv = self.alpha * (np.asarray(v_cmd, dtype=float)[:2] - self.v_filt)
        dv_norm = float(np.hypot(*dv))
        if dv_norm > self.dv_max:
            dv *= self.dv_max / dv_norm
        self.v_filt += dv
        speed = float(np.hypot(*self.v_filt))
        self.heading_err = 0.0
        if not self.enabled or speed < self.v_deadband:
            omega_raw = 0.0
        else:
            err = wrap_angle(np.arctan2(self.v_filt[1], self.v_filt[0]) - heading)
            self.heading_err = float(err)
            omega = np.clip(self.kp * err, -self.omega_max, self.omega_max)
            # the direction of a slow command is ill-conditioned (ORCA dithers in
            # front of crowds and atan2 of a small vector swings wildly), so fade
            # the turning authority in with speed: full turning only when actually
            # walking somewhere (~3x the deadband)
            scale = np.clip((speed - self.v_deadband) / (3.0 * self.v_deadband),
                            0.0, 1.0)
            omega_raw = float(omega * scale)
        # slew-limit omega: dodging around a pedestrian, the heading error can
        # flip sign between consecutive steps, and a per-step-sampled omega
        # then yaws the feet in an alternating zigzag - the least stable thing
        # a turning biped can do. The slew turns that into a smooth swing.
        self.omega_out += np.clip(omega_raw - self.omega_out,
                                  -self.domega_max, self.domega_max)
        return self.omega_out

    def v_planner(self):
        """Velocity the FOOTSTEP planner should follow: stop-and-turn gating.

        Walking fast across the legs while turning (large angle between the
        commanded direction and the body heading) is the least stable thing a
        biped can do - so fade the translation out and let the robot turn in
        place first, then walk. Full speed below 0.5 rad error, zero above 1.0.
        """
        gate = float(np.clip(1.0 - (abs(self.heading_err) - 0.5) / 0.5, 0.0, 1.0))
        return self.v_filt * gate


def _capture_excess(com, vcom, sup_xy, next_xy, omega_lip, clamp,
                    radius=None):
    """Disturbance part of the measured DCM.

    In a nominal gait the DCM travels from near the support foot toward the
    next footstep (and sits BETWEEN the feet when stepping in place), i.e. it
    stays close to the segment support -> next touchdown. Only the excess
    beyond the nominal orbit radius around that segment is returned (clamped)
    - where the robot is actually falling toward and the feet should go.
    """
    dcm = np.asarray(com)[:2] + np.asarray(vcom)[:2] / omega_lip
    a = np.asarray(sup_xy, dtype=float)[:2]
    b = np.asarray(next_xy, dtype=float)[:2]
    ab = b - a
    denom = float(ab @ ab)
    s = float(np.clip((dcm - a) @ ab / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
    err = dcm - (a + s * ab)          # DCM offset from the nominal segment
    err_n = float(np.hypot(*err))
    excess = err_n - (DCM_NOMINAL_RADIUS if radius is None else radius)
    if excess <= 0.0 or err_n < 1e-9:
        return np.zeros(2)
    return err / err_n * min(excess, clamp)


def _run_loop(simulator, crowd, robot, mpc, interpolator, plan, ZMP_ref, YAW_ref,
              com_start, com_target, N_pre, N_sim):
    """Run the walking simulation loop and return the filled log buffers."""
    ROBOT_RADIUS_PHYS = 0.3                     # physical robot radius for the safety plot
    n_peds = len(crowd.peds)
    L = _allocate_logs(N_sim, n_peds)
    L.GOAL = np.asarray(ROBOT_GOAL, dtype=float)
    L.PED_RADII = np.array([r for (_, _, r) in crowd.states()])

    k = 0                                       # MPC index (fixed-plan mode)
    plan_idx = 1                                # step index in the plan (fixed-plan mode)
    mpc_phase = 0                               # MPC-sample index within the current step
    t_step_elapsed = 0.0                        # elapsed time within current step
    t_publish = 0.0                             # last ros publish time
    u_k = np.zeros(2)                           # held mpc control (commanded zmp)
    swing_trajectory = None                     # current swing-foot spline
    next_step = None                            # current swing-target footstep (for viz)
    pin_data = robot.stack.model.createData()   # scratch for centroidal queries
    c = c_dot = c_ddot = None                   # latest com reference

    # torso yaw reference = low-passed circular MEAN of the measured foot yaws.
    # The hip yaw joints are tight and asymmetric (leg_right_1 caps at +0.349)
    # and TSID runs without joint position bounds while the simulator enforces
    # them - so the trunk must track the feet through turns, otherwise the
    # accumulated pelvis lag (the soft posture task alone leaves ~0.2 rad)
    # slams a hip yaw joint into its limit mid-step and the robot falls.
    # Following the MEASURED feet (not the command) means the reference never
    # leads the gait and moves continuously (the swing foot yaws in flight),
    # so no per-step reference jumps excite the sway mode. Straight walking:
    # foot yaws are constant -> constant reference, unchanged behaviour.
    T_torso0, _ = robot.stack.torsoState()
    p_torso0 = T_torso0.translation.copy()
    yaw_torso0 = np.arctan2(T_torso0.rotation[1, 0], T_torso0.rotation[0, 0])
    # spawn roll/pitch with the spawn yaw factored out
    R_torso0_level = rot_z(-yaw_torso0) @ T_torso0.rotation.copy()
    yaw_feet_filt = yaw_torso0        # filtered mean foot yaw
    # slow enough that a touchdown-impact yaw glitch in a measured foot pose
    # cannot jerk the pelvis, fast enough that the tracking lag omega*tau
    # stays small against the +-0.35 rad hip-yaw budget
    TORSO_YAW_TAU = 0.8               # [s] low-pass on the mean foot yaw

    # robot ORCA velocity generator: v_cmd toward the goal, avoiding pedestrians.
    # max_speed stays close to desired_speed: a biped gains little from faster
    # evasion bursts but loses balance commanding them mid-turn.
    robot_ctrl = RobotORCA(ROBOT_GOAL, radius=0.35, max_speed=0.3,
                           desired_speed=0.25, safety_margin=0.2)
    # heading controller: turns the body into the direction of travel
    # (disabled -> omega stays 0 and the gait is purely omnidirectional)
    heading_ctrl = HeadingController(conf.dt, enabled=USE_HEADING)
    omega_cmd = 0.0
    circle_psi = yaw_of(robot.supportFootPose())  # path tangent, heading-off circle

    # reactive footstep planner: each step, rolls the base along the commanded
    # (v_cmd, omega_cmd) arc, places the next footstep and feeds the MPC.
    reactive_planner = ReactiveFootStepPlanner(
        conf, ROBOT_GOAL, robot.supportFootPose(),
        conf.no_mpc_samples_per_step, conf.no_mpc_samples_per_horizon)
    current_swing_side = robot.swing_foot        # which foot swings next
    reactive_planner.regenerate(
        reactive_planner.make_footstep(robot.supportFootPose(), robot.support_foot),
        np.zeros(2))                             # seed the buffer until the first step

    # per-step timing: steps have variable duration (multiples of dt_mpc), so
    # transitions run off a tick countdown instead of a fixed modulo
    step_ticks_left = 0                          # 0 -> transition on this tick
    n_step_samples = conf.no_mpc_samples_per_step
    step_dur_cur = conf.step_dur                 # committed step duration [s]

    # mid-step swing retargeting: a committed step is otherwise frozen for its
    # whole duration, so a disturbance can only be answered a full step later -
    # too slow to catch a fall. The touchdown target is shifted toward the
    # measured DCM excess while the foot is in flight (blended in smoothly).
    support_fs = None                            # current support (REACTIVE)
    swing_delta = np.zeros(2)                    # total touchdown shift [m]
    retarget_tau0 = None                         # step phase when shifting began
    early_contact = False                        # swing foot loaded early (latch)

    for i in range(-N_pre, N_sim):
        t = i * conf.dt
        dt = conf.dt

        # --- robot state + crowd + ORCA velocity command (first, so the reactive
        #     planner can use the latest v_cmd) ---
        body_id = robot.robot.id()
        com = simulator.computeComPosition(body_id)
        vcom = simulator.computeComVelocity(body_id)
        # move the crowd (pedestrians still avoid the robot via ground-truth here)
        crowd.update(conf.dt, external_agents=[(com[:2], vcom[:2], 0.35)])
        # pedestrian states for the robot come from the ONBOARD DEPTH CAMERA
        # (only what the robot actually sees), not ground truth. update_obstacles()
        # refreshes these at the perception rate; obstacles2d() returns the latest
        # world-frame estimates {id, x, y, r, vx, vy, stale}.
        if PEDESTRIAN_SENSOR_DATA:
            ped_states = [(np.array([o['x'], o['y']]), np.array([o['vx'], o['vy']]), o['r'])
                          for o in robot.obstacles2d()]
        else:
            ped_states = crowd.states()   # list of (pos(2), vel(2), radius)
        if WALK == "circle":
            # constant-twist circle demo: forward along the path tangent plus a
            # fixed yaw rate -> an arc of radius CIRCLE_SPEED/omega (no ORCA).
            # With heading ON the feet turn tangent to the arc (planner heading
            # is the tangent); with heading OFF the feet keep their spawn yaw
            # and the robot side-steps the arc, so the tangent is integrated
            # here instead of read back from the footsteps.
            omega_path = ((-1.0 if CIRCLE_CLOCKWISE else 1.0)
                          * CIRCLE_SPEED / CIRCLE_RADIUS)
            if USE_HEADING:
                psi_path = reactive_planner.heading
                omega_cmd = omega_path
            else:
                psi_path = circle_psi
                if i >= 0:
                    circle_psi += omega_path * conf.dt
                omega_cmd = 0.0
            v_cmd = CIRCLE_SPEED * np.array([np.cos(psi_path), np.sin(psi_path)])
            v_pref = v_cmd
            heading_ctrl.v_filt = v_cmd.copy()   # feeds v_planner()
            heading_ctrl.heading_err = 0.0
        else:
            v_cmd = robot_ctrl.compute_velocity(com[:2], vcom[:2], ped_states)
            v_pref = robot_ctrl.preferred_velocity(com[:2])
            # yaw-rate command: turn into the direction of travel; feedback is
            # the planner heading (support-foot yaw) - what the feet committed
            # to. Returns 0 (no turning) when USE_HEADING is off.
            omega_cmd = heading_ctrl.compute(v_cmd, reactive_planner.heading)
        robot.record_com(com)                   # CoM trail for the viewer overlay

        # --- fall detection: CoM below any plausible walking height ---
        if i >= 0 and com[2] < 0.60:
            print(f"[nav] ROBOT FELL at t={t:.2f}s (com_z={com[2]:.2f}); aborting walk")
            break

        # --- navigation logging ---
        if i >= 0:
            L.ROBOT_XY[i] = com[:2]
            L.OMEGA_CMD[i] = omega_cmd
            L.STEP_DUR[i] = step_dur_cur
            L.HEADING[i] = reactive_planner.heading
            for j, (p, _, r) in enumerate(ped_states):
                L.PED_XY[i, j] = np.asarray(p, dtype=float)[:2]
            gaps = [np.hypot(*(com[:2] - np.asarray(p, float)[:2])) - ROBOT_RADIUS_PHYS - r
                    for (p, _, r) in ped_states]
            L.MIN_SEP[i] = min(gaps) if gaps else np.nan
            L.DIST_GOAL[i] = float(np.hypot(*(L.GOAL - com[:2])))
            L.VCMD[i] = v_cmd
            L.VPREF[i] = v_pref

        # debug overlay: green = v_cmd (drives the footsteps), blue = goal direction,
        # red sphere = goal
        arrows = [a for a in (
            debug_viz.velocity_arrow(com[:2], v_pref, z=float(com[2]),
                                     scale=ARROW_SCALE, rgba=(0.3, 0.5, 1.0, 0.6)),
            debug_viz.velocity_arrow(com[:2], v_cmd, z=float(com[2]),
                                     scale=ARROW_SCALE, rgba=(0.1, 0.9, 0.1, 1.0)),
        ) if a is not None]
        goal_marker = debug_viz.point_marker(ROBOT_GOAL, z=0.15, radius=0.12,
                                             rgba=(1.0, 0.2, 0.2, 1.0))
        # yellow foot-shaped box at the next planned footstep, fading blue/red
        # boxes at the executed footstep history, white spheres for the CoM trail
        boxes = [debug_viz.footstep_box(next_step)] if next_step is not None else []
        hist = robot.footstep_history
        for age, fs in enumerate(reversed(hist)):
            alpha = 0.5 * (1.0 - age / max(len(hist), 1))
            color = ((0.2, 0.4, 1.0, alpha) if fs.side == Side.LEFT
                     else (1.0, 0.3, 0.2, alpha))
            boxes.append(debug_viz.footstep_box(fs, rgba=color))
        points = [goal_marker]
        points += [debug_viz.point_marker(p[:2], z=0.02, radius=0.02,
                                          rgba=(1.0, 1.0, 1.0, 0.5))
                   for p in robot.com_trail]
        debug_viz.draw_scene(simulator, arrows=arrows, points=points, boxes=boxes)

        # --- early touchdown: if the swing foot hits the ground in the late
        #     swing phase, end the step NOW instead of commanding the foot
        #     through the floor (that contact fight shoves the robot over).
        #     Force spikes are latched every tick; the actual switch happens on
        #     the next MPC tick so the transition stays MPC-aligned.
        dynamic_gait = (np.hypot(*heading_ctrl.v_filt) > DYNAMIC_GAIT_SPEED
                        or abs(omega_cmd) > DYNAMIC_GAIT_OMEGA)
        if (REACTIVE and i >= 0 and swing_trajectory is not None
                and step_ticks_left > 0
                and t_step_elapsed > 0.6 * step_dur_cur
                and dynamic_gait):
            fname = (conf.lf_frame_name if robot.swing_foot == Side.LEFT
                     else conf.rf_frame_name)
            f_sw, _ = robot._foot_wrench_world(fname)
            if f_sw[2] > EARLY_CONTACT_FZ:
                early_contact = True
            if early_contact and i % conf.no_sim_per_mpc == 0:
                step_ticks_left = 0

        # --- step transition: pick/plan the next footstep, build swing spline ---
        if i >= 0 and step_ticks_left == 0:
            if REACTIVE:
                # plan the next step along the (v_cmd, omega_cmd) arc (steps in
                # place at goal); the planner also picks the step duration
                swing_side = current_swing_side
                support_side = other_foot_id(swing_side)
                robot.setSwingFoot(swing_side)
                robot.setSupportFoot(support_side)
                support_fs = reactive_planner.make_footstep(
                    robot.supportFootPose(), support_side)
                at_goal = (False if WALK == "circle"
                           else reactive_planner.at_goal(com[:2]))
                # plan from the low-pass-filtered velocity: the raw ORCA
                # command jumps tick-to-tick and a step samples it only once,
                # which otherwise makes stride length/direction erratic.
                # com_xy closes the placement loop around the body (steps
                # follow the CoM drift instead of only the support-foot chain);
                # balance corrections happen mid-step (retargeting below).
                reactive_planner.regenerate(support_fs, heading_ctrl.v_planner(),
                                            at_goal, omega_cmd=omega_cmd,
                                            com_xy=com[:2])
                # capture stepping at the COMMIT: any DCM excess present right
                # now would otherwise be ignored for a whole step (it grows by
                # e^(omega*T) ~ 11x per step) before the mid-step retargeting
                # deadband sees it - step toward it immediately instead
                if dynamic_gait:
                    nom_xy = np.asarray(
                        reactive_planner.swing_target().pose.translation)[:2]
                    sup_xy = np.asarray(support_fs.pose.translation)[:2]
                    cap = _capture_excess(com, vcom, sup_xy, nom_xy,
                                          interpolator.omega, CAPTURE_STEP_MAX,
                                          radius=DCM_COMMIT_RADIUS)
                    if np.hypot(*cap) > 1e-9:
                        reactive_planner.regenerate(
                            support_fs, heading_ctrl.v_planner(), at_goal,
                            omega_cmd=omega_cmd, com_xy=com[:2],
                            first_step_offset=cap)
                next_step = reactive_planner.swing_target()
                n_step_samples = reactive_planner.step_samples()
                current_swing_side = other_foot_id(swing_side)
                mpc_phase = 0
            else:
                next_step = plan[plan_idx]
                robot.setSwingFoot(next_step.side)
                robot.setSupportFoot(other_foot_id(next_step.side))
                plan_idx += 1
                n_step_samples = conf.no_mpc_samples_per_step
            step_dur_cur = n_step_samples * conf.dt_mpc
            # countdown in whole MPC intervals so every transition tick is also
            # an MPC tick (mpc_phase = 0 is consumed on this same iteration)
            step_ticks_left = n_step_samples * conf.no_sim_per_mpc
            robot.record_footstep(next_step)
            T0_swing = robot.swingFootPose()
            T1_swing = next_step.poseInWorld()
            # raise the swing apex when the foot yaws a lot, so the toes cannot
            # scuff while the foot rotates
            dyaw = abs(wrap_angle(yaw_of(T1_swing) - yaw_of(T0_swing)))
            apex = float(np.clip(
                conf.step_height + 0.05 * dyaw / conf.max_dyaw_per_step,
                conf.step_height, conf.step_height + 0.05))
            swing_trajectory = SwingFootTrajectory(T0_swing, T1_swing,
                                                   step_dur_cur, height=apex)
            t_step_elapsed = 0.0
            swing_delta = np.zeros(2)
            retarget_tau0 = None
            early_contact = False
        if i >= 0:
            step_ticks_left -= 1

        # --- update the mpc every no_sim_per_mpc steps ---
        if i >= 0 and i % conf.no_sim_per_mpc == 0:
            if REACTIVE:
                # close the loop: pull the open-loop LIP state toward the
                # measured CoM so plan and robot cannot drift apart (the
                # footstep references are anchored to the measured feet)
                x_meas = np.array([com[0], vcom[0], com[1], vcom[1]])
                gap = np.hypot(x_meas[0] - interpolator.x[0],
                               x_meas[2] - interpolator.x[2])
                interpolator.reanchor(
                    x_meas, 1.0 if gap > MPC_STATE_RESET_DIST else MPC_STATE_FB)
            x_k = interpolator.x
            if REACTIVE:
                # ZMP/yaw reference over the horizon from the reactive buffer,
                # phase-aligned within the current step
                ZMP_ref_k, YAW_ref_k = reactive_planner.mpc_reference(mpc_phase)
                terminal_idx = -1
                mpc_phase += 1
            else:
                ZMP_ref_k = ZMP_ref[k:k + conf.no_mpc_samples_per_horizon]
                YAW_ref_k = YAW_ref[k:k + conf.no_mpc_samples_per_horizon]
                remaining = len(ZMP_ref) - k
                terminal_idx = remaining - 1 if remaining < conf.no_mpc_samples_per_horizon else -1
                k += 1
            try:
                u_k = mpc.buildSolveOCP(x_k, ZMP_ref_k, terminal_idx,
                                        yaw_ref_k=YAW_ref_k)
            except RuntimeError as e:
                # keep the partial logs/plots for diagnosis instead of crashing
                print(f"[nav] MPC failed at t={t:.2f}s ({e}); aborting walk")
                break

        # --- mid-step swing retargeting (every MPC tick, early in the step):
        #     shift the committed touchdown toward the measured DCM excess so
        #     a disturbed robot steps under itself NOW, not one step later ---
        if (REACTIVE and i >= 0 and i % conf.no_sim_per_mpc == 0
                and support_fs is not None and swing_trajectory is not None
                and 0.0 < t_step_elapsed < CAPTURE_RETARGET_CUTOFF * step_dur_cur
                and dynamic_gait):
            committed_xy = np.asarray(next_step.pose.translation)[:2]
            # LIVE support pose, not the transition-time snapshot: after an
            # early touchdown the support foot can still settle/slide for a few
            # cm, and clamping against its stale pose lets the swing target
            # cross the legs
            T_sup_live = robot.supportFootPose()
            sup_c = np.asarray(T_sup_live.translation)[:2]
            want = _capture_excess(com, vcom, sup_c, committed_xy,
                                   interpolator.omega, CAPTURE_RETARGET_MAX)
            if np.hypot(*(want - swing_delta)) > 0.01:
                new_xy = committed_xy + want
                # keep the shifted touchdown kinematically reachable: a target
                # beyond leg reach never lands, the stance leg overextends and
                # contact collapses (worse than an imperfect capture point)
                d = new_xy - sup_c
                dn = float(np.hypot(*d))
                if dn > conf.max_foot_reach:
                    new_xy = sup_c + d * (conf.max_foot_reach / dn)
                # and bound it by the no-crossing constraint (applied last)
                yaw_sup = yaw_of(T_sup_live)
                e_l = np.array([-np.sin(yaw_sup), np.cos(yaw_sup)])
                sgn = 1.0 if next_step.side == Side.LEFT else -1.0
                lat = float(e_l @ (new_xy - sup_c))
                if sgn * lat < conf.min_foot_sep:
                    new_xy += (conf.min_foot_sep - sgn * lat) * sgn * e_l
                swing_delta = new_xy - committed_xy
                if retarget_tau0 is None:
                    retarget_tau0 = t_step_elapsed / step_dur_cur

        # --- references, every iteration ---
        if i < 0:
            # pre-walk: cosine smoothstep sway of the CoM from spawn to the first
            # support foot (zero velocity at both ends so the wbc settles)
            s = (i + N_pre) / N_pre
            s = 0.5 - 0.5 * np.cos(np.pi * s)
            robot.stack.setComRefState((1.0 - s) * com_start + s * com_target)

        if i >= 0:
            t_step_elapsed += dt
            # advance the lip at sim rate with the held mpc control, feed the
            # interpolated com into the wbc com task
            interpolator.integrate(u_k)
            c, c_dot, c_ddot = interpolator.comState()
            robot.stack.setComRefState(c, c_dot, c_ddot)
            # evaluate the swing-foot spline and update its motion reference;
            # blend the mid-step retarget shift in smoothly (zero at the phase
            # where shifting began, full at touchdown - no reference jumps)
            if swing_trajectory is not None:
                T_swing, V_swing, A_swing = swing_trajectory.evaluate(t_step_elapsed)
                if retarget_tau0 is not None:
                    tau = min(t_step_elapsed / step_dur_cur, 1.0)
                    u = np.clip((tau - retarget_tau0) / max(1.0 - retarget_tau0, 1e-6),
                                0.0, 1.0)
                    s = 3.0 * u**2 - 2.0 * u**3
                    s_dot = ((6.0 * u - 6.0 * u**2)
                             / (max(1.0 - retarget_tau0, 1e-6) * step_dur_cur))
                    V_vec = V_swing.vector.copy()
                    A_vec = A_swing.vector.copy()
                    T_swing = pin.SE3(T_swing.rotation, T_swing.translation.copy())
                    T_swing.translation[:2] += s * swing_delta
                    V_vec[:2] += s_dot * swing_delta
                    robot.updateSwingFootRef(T_swing, V_vec, A_vec)
                else:
                    robot.updateSwingFootRef(T_swing, V_swing.vector, A_swing.vector)

            # trunk level (roll/pitch) + yaw tracking the measured mean foot
            # yaw, so the hip yaw joints stay centred between their limits
            H_lf, _ = robot.stack.get_pose_vel_acc_LF()
            H_rf, _ = robot.stack.get_pose_vel_acc_RF()
            yaw_lf, yaw_rf = yaw_of(H_lf), yaw_of(H_rf)
            yaw_mean = np.arctan2(np.sin(yaw_lf) + np.sin(yaw_rf),
                                  np.cos(yaw_lf) + np.cos(yaw_rf))
            yaw_feet_filt = wrap_angle(
                yaw_feet_filt + (dt / TORSO_YAW_TAU)
                * wrap_angle(yaw_mean - yaw_feet_filt))
            robot.stack.setTorsoRefState(
                pin.SE3(rot_z(yaw_feet_filt) @ R_torso0_level, p_torso0))

        # --- advance the simulation + whole-body control ---
        simulator.step()
        robot.update()

        # --- refresh onboard-camera obstacle estimates at ~30 Hz (after the step,
        #     so the depth/segmentation render sees the updated scene) ---
        if PEDESTRIAN_SENSOR_DATA and i % max(1, int(round((1.0 / 30.0) / conf.dt))) == 0:
            robot.update_obstacles()

        # --- navigation diagnostic (once per second while walking) ---
        if i >= 0 and i % int(round(1.0 / conf.dt)) == 0:
            dist_goal = float(np.hypot(*(np.asarray(ROBOT_GOAL, float) - com[:2])))
            print(f"[nav] t={t:5.1f}s  com=({com[0]:5.2f},{com[1]:5.2f})  "
                  f"dist_goal={dist_goal:4.2f}  |v_cmd|={np.hypot(*v_cmd):4.2f}  "
                  f"v_cmd=({v_cmd[0]:+.2f},{v_cmd[1]:+.2f})")

        # publish to ros at ~30 Hz
        if t - t_publish > 1. / 30.:
            t_publish = t
            robot.publish()

        # store for the post-run plots
        if i >= 0:
            _record_step(L, i, t, c, c_dot, c_ddot, t_step_elapsed,
                         swing_trajectory, robot, simulator, interpolator, pin_data)

    return L


################################################################################
# main
################################################################################

def _read_runtime_params():
    """Read runtime ROS 2 parameters and apply them to the module globals.
    Override at launch, e.g.:
        ros2 run bullet_sims walking --ros-args \
            -p scenario:=crossing -p pedestrian_sensor_data:=true
    """
    global SCENARIO, PEDESTRIAN_SENSOR_DATA, WALK, WALK_DURATION
    global CIRCLE_RADIUS, CIRCLE_SPEED, CIRCLE_CLOCKWISE, USE_HEADING
    node = rclpy.create_node("walking_params")
    node.declare_parameter("scenario", SCENARIO)
    node.declare_parameter("pedestrian_sensor_data", PEDESTRIAN_SENSOR_DATA)
    node.declare_parameter("walk", WALK)
    node.declare_parameter("circle_radius", CIRCLE_RADIUS)
    node.declare_parameter("circle_speed", CIRCLE_SPEED)
    node.declare_parameter("circle_clockwise", CIRCLE_CLOCKWISE)
    node.declare_parameter("use_heading", USE_HEADING)
    node.declare_parameter("duration", 0.0)    # 0 -> automatic
    SCENARIO = node.get_parameter("scenario").get_parameter_value().string_value
    PEDESTRIAN_SENSOR_DATA = node.get_parameter(
        "pedestrian_sensor_data").get_parameter_value().bool_value
    WALK = node.get_parameter("walk").get_parameter_value().string_value
    CIRCLE_RADIUS = node.get_parameter(
        "circle_radius").get_parameter_value().double_value
    CIRCLE_SPEED = node.get_parameter(
        "circle_speed").get_parameter_value().double_value
    CIRCLE_CLOCKWISE = node.get_parameter(
        "circle_clockwise").get_parameter_value().bool_value
    USE_HEADING = node.get_parameter(
        "use_heading").get_parameter_value().bool_value
    duration = node.get_parameter("duration").get_parameter_value().double_value
    if duration > 0.0:
        WALK_DURATION = duration
    elif WALK == "circle":
        # default: one full lap plus some margin
        WALK_DURATION = 2.0 * np.pi * CIRCLE_RADIUS / max(CIRCLE_SPEED, 1e-3) + 5.0
    node.destroy_node()
    print(f"[walking] scenario='{SCENARIO}'  walk='{WALK}'  "
          f"duration={WALK_DURATION:.0f}s  use_heading={USE_HEADING}  "
          f"pedestrian_sensor_data={PEDESTRIAN_SENSOR_DATA}")


def main():
    # NOTE: ros2 run skips the __main__ block, so all setup happens here.
    simulator = _setup_simulator()

    # runtime parameters (scenario, sensor vs ground truth) -> module globals
    _read_runtime_params()

    # crowd must be created before the robot (see _setup_crowd)
    crowd = _setup_crowd(simulator)

    created = _create_robot(simulator)
    if created is None:
        return
    tw, robot = created

    # register the pedestrian geoms so the depth-camera perception tracks them
    # (each pedestrian mocap body carries a geom named "<ped.name>_geom")
    if PEDESTRIAN_SENSOR_DATA:
        robot.set_obstacle_geoms([f"{p.name}_geom" for p in crowd.peds])

    # fixed footstep plan (only used when REACTIVE is False; the reactive planner
    # is created inside the loop and generates its footsteps online from v_cmd)
    planner, plan, no_steps = _build_plan(robot)
    ZMP_ref = generate_zmp_reference(plan, conf.no_mpc_samples_per_step)
    YAW_ref = generate_yaw_reference(plan, conf.no_mpc_samples_per_step)
    if not REACTIVE:
        planner.plot(simulator)

    # lip mpc + interpolator + pre-walk sway endpoints
    mpc, interpolator, com_start, com_target = _setup_lip_mpc(robot)
    robot.stack.setComRefState(com_start)

    # iteration counts
    pre_dur = 3.0   # time to wait before walking starts
    N_pre = int(round(pre_dur / conf.dt))
    if REACTIVE:
        N_sim = int(round(WALK_DURATION / conf.dt))
    else:
        N_sim = int(round(no_steps * conf.step_dur / conf.dt))

    # run the simulation, then plot
    L = _run_loop(simulator, crowd, robot, mpc, interpolator, plan, ZMP_ref, YAW_ref,
                  com_start, com_target, N_pre, N_sim)
    # raw log dump for offline analysis (same dir as the plots)
    import os
    plots_dir = os.environ.get(
        'WALKING_PLOTS_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'T7'))
    try:
        os.makedirs(plots_dir, exist_ok=True)
        np.savez_compressed(os.path.join(plots_dir, 'walk_log.npz'), **vars(L))
        print(f"saved raw logs to {os.path.join(plots_dir, 'walk_log.npz')}")
    except OSError as e:
        print(f"could not save raw logs: {e}")
    _plot_navigation(L)

    _plot_results(L)

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()