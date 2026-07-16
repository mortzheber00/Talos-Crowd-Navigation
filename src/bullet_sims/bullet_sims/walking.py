"""
talos walking simulation
"""

import signal
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
from bullet_sims.reactive_footstep_planner import ReactiveFootStepPlanner
from bullet_sims.floor_ambience import setup_floor_ambience

# crowd
from orca_nav.robot_orca import RobotORCA
from orca_nav import debug_viz

# crowd
from orca_nav import scenarios
from orca_nav.pedestrian import PedestrianSimulator




# footstep path for the walking demo: "line" (straight) or "circle" (curved)
WALK_MODE = "line"
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
WALK_DURATION = 30.0       # [s] max walk time (reactive walking has no fixed step count)
REACTIVE = True            # True: footsteps follow v_cmd; False: fixed WALK_MODE plan

# blended gait knobs (A/B): TURN_GAIN=0 -> pure fixed-facing side-step (the old
# stable mode); 1 -> turn at the full MAX_DYAW_DEG rate limit every step.
TURN_GAIN = 0.5            # fraction of the heading error turned per step
MAX_DYAW_DEG = 12.0         # [deg] max facing change per step
# dcm_walker's anti-crossing rule: only the INSIDE foot of a turn advances the
# heading, so yaw steps once per foot PAIR and the feet are never more than one
# dpsi apart in yaw. Halves the effective turn rate for a given MAX_DYAW_DEG.
ALTERNATE_YAW = True

# angular-momentum yaw feedforward: the AM task's reference is ZERO by default,
# which is a standing "do not rotate" order that OPPOSES every turn. Feeding
# L_z = I_zz * yaw_rate instead lets the task PERMIT the turn's spin while still
# regulating x/y AM to zero. AM_YAW_FF = 0 -> zero reference (legacy).
# W_AM is written into conf.w_am BEFORE the TSIDWrapper is built, so it must be
# set in main(), not here.
AM_YAW_FF = 8.1            # gain on the L_z = I_zz * yaw_rate feedforward
W_AM = 2e-6               # AM task weight (1e-4 = decorative; 1e-2 = it bites)
AM_YAW_RATE_TAU = 0.0     # [s] low-pass on the yaw rate; 0 = RAW (unfiltered)

# What the torso/pelvis yaw is steered to:
#   "support_meas"  - the MEASURED yaw of the foot that is actually planted.
#                     Keeps hip yaw near zero, so a turn never spends the left
#                     hip's 20 deg of inboard range on a pelvis lead.
#   "swing_target"  - the yaw of the NEXT footstep (legacy). Puts the pelvis one
#                     step AHEAD of the planted feet; measured 2026-07-16 to wind
#                     leg_left_1 to its -20 deg stop at ~19 deg facing, after
#                     which the loaded inside foot spins (1188 deg/s) and falls.
# Straight walking is unaffected either way (both yaws are 0).
TORSO_YAW_SOURCE = "support_meas"

# MPC state feedback: fraction of the MEASURED CoM state fed into the MPC (and
# back into the LIP interpolator) at every MPC tick.
#   0 = the LEGACY OPEN LOOP. x_k was `interpolator.x`, so the MPC planned from
#       its own integration and the robot was never in the loop at all:
#           MPC --u--> interpolator --c_ref--> TSID --> robot
#            ^                     |
#            +-------- x_k --------+          (robot not in this loop)
#       LIP-vs-robot model error then had nothing to correct it and integrated
#       without bound. Measured 2026-07-16: |CoM_ref - CoM_real| 5 -> 105 mm in
#       6 s of turning, with NO QP failure, the swing foot tracking to 5 mm and
#       hip yaw nowhere near its stops. Walking STRAIGHT it stayed 3-7 mm flat,
#       because the 2D LIP is an accurate model of straight symmetric walking --
#       a turn adds centripetal/angular-momentum dynamics the LIP has no state
#       for, so the mismatch is larger and integrates faster. This is why the
#       failure looked turn-specific, and why no gain touched it: raising kp_com
#       only makes the robot track a DRIFTING reference more faithfully.
#   1 = full state feedback (textbook MPC = receding horizon + state feedback).
# Lower it toward ~0.1 if feeding raw measurement makes the QP jittery.
MPC_STATE_FEEDBACK = 1.0

# touchdown detection from the swing foot's FT sensor: once its measured normal
# force exceeds this in the descent half of the step, the swing reference is
# FROZEN at its current pose -- otherwise an early contact makes the motion task
# keep driving the foot along the spline into the ground, loading a sole edge
# and tilting the foot. Must stay above the sensed swing-leg foot weight
# (~25-30 N). Set to 0 to disable.
TOUCHDOWN_FZ = 475.0        # [N]

# ---- runtime parameters (defaults; override at launch via ROS 2 params, e.g.
#   ros2 run bullet_sims walking --ros-args -p scenario:=crossing -p pedestrian_sensor_data:=true )
SCENARIO = "random"              # crowd scenario: head_on | crossing | circle | corridor | random
PEDESTRIAN_SENSOR_DATA = False    # True: depth-camera perception; False: ground-truth states


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

    return PybulletWrapper(sim_rate=frequency, num_solver_iterations=50,
                           num_sub_steps=8, render=True,
                           collision_boxes=conf.sole_collision_boxes)


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


def _plan_from(planner, T_support_w, support_side, no_steps):
    """Build the footstep plan ANCHORED at the given support foot, plus the
    ZMP/yaw references that match it.

    plan[0] is the support foot itself (stepping starts at plan[1]), else the
    first swing would lift the loaded foot. The last step is duplicated twice so
    the MPC's receding horizon never looks past the end of the plan.

    Called once at startup AND again at every step transition with the MEASURED
    support foot -- see the re-anchor comment in _run_loop.
    """
    if WALK_MODE == "circle":
        plan = planner.planCircle(T_support_w, support_side, no_steps=no_steps,
                                  radius=CIRCLE_RADIUS, clockwise=CIRCLE_CLOCKWISE,
                                  face_path=CIRCLE_FACE_PATH)
    else:
        plan = planner.planLine(T_support_w, support_side, no_steps=no_steps)
    plan = list(plan)
    plan.append(plan[-2])
    plan.append(plan[-1])
    ZMP_ref = generate_zmp_reference(plan, conf.no_mpc_samples_per_step)
    YAW_ref = generate_yaw_reference(plan, conf.no_mpc_samples_per_step)
    return plan, ZMP_ref, YAW_ref


def _build_plan(robot):
    """Initial footstep plan, anchored at the robot's current support foot."""
    no_steps = 20
    planner = FootStepPlanner(conf)
    plan, ZMP_ref, YAW_ref = _plan_from(planner, robot.supportFootPose(),
                                        robot.support_foot, no_steps)
    return planner, plan, no_steps, ZMP_ref, YAW_ref


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

_ABORT = False          # set by Ctrl-C; polled by _run_loop's sim loop


def _install_sigint_handler():
    """Make Ctrl-C stop the sim loop *cleanly* instead of killing the process, so
    an aborted run still reaches the plots/reports (they live after _run_loop
    returns, and the log buffers are NaN-filled -- an early exit just leaves a NaN
    tail that matplotlib and the isfinite-gated reports both skip).

    A second Ctrl-C restores the default handler and aborts hard, so a wedged run
    is still killable.
    """
    global _ABORT
    _ABORT = False

    def _on_sigint(signum, frame):
        global _ABORT
        _ABORT = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("\ninterrupted -- stopping the sim and plotting what was logged "
              "(Ctrl-C again to abort hard)", flush=True)

    signal.signal(signal.SIGINT, _on_sigint)


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
        # --- heading / swing-drag diagnostics ---
        # Is the torso-yaw slew after a heading_des jump dragging the swing leg
        # inward? These are the signals that answer it.
        TORSO_YAW=nan(N_sim),        # commanded torso yaw (delta from spawn) [rad]
        HEADING_DES=nan(N_sim),      # step's target heading (delta from spawn) [rad]
        TORSO_YAW_RATE=nan(N_sim),   # d(torso_yaw)/dt [rad/s]
        SWING_LAT_DEV=nan(N_sim),    # swing foot measured-minus-spline, along the
                                     # facing's LEFT axis [m] (+ = drifted left)
        SWING_SIDE=nan(N_sim),       # +1 = LEFT swinging, -1 = RIGHT
        TURN_SIGN=nan(N_sim),        # +1 = turning left, -1 = right, 0 = straight
        # measured foot YAW in the world. A foot that is in contact must not
        # rotate; if it does, it is spinning against friction.
        LF_YAW=nan(N_sim), RF_YAW=nan(N_sim),
        # hip-yaw joints. ASYMMETRIC URDF limits: leg_left_1 [-20, +90] deg,
        # leg_right_1 [-90, +20] deg. Saturation here is the standing suspect
        # for the turn wall.
        HIP_L=nan(N_sim), HIP_R=nan(N_sim),
        # Torso yaw, MEASURED, and its two (and only two) sources: the frame
        # Jacobian's yaw row is nonzero on exactly root_joint (pelvis) and
        # torso_1_joint (waist), both with coefficient 1.0. So
        #     torso_yaw_meas = pelvis_yaw_meas + q_torso1
        # Splitting it says WHICH route is blocked: the pelvis (hip-limited) or
        # the waist (pinned by kp_posture[12] = 5000).
        TORSO_YAW_MEAS=nan(N_sim),   # measured yaw of torso_2_link [rad]
        PELVIS_YAW_MEAS=nan(N_sim),  # measured yaw of base_link [rad]
        Q_TORSO1=nan(N_sim),         # waist yaw joint angle [rad]
        TIME=nan(N_sim),
    )


def _record_step(L, i, t, c, c_dot, c_ddot, t_step_elapsed, swing_trajectory,
                 robot, simulator, interpolator, pin_data,
                 torso_yaw=np.nan, heading_des=np.nan, yaw_sup0=0.0,
                 torso_yaw_rate=np.nan, T_cmd=None):
    """Store one iteration of state for the post-run plots."""
    L.TIME[i] = t

    # measured foot yaw in the world (for the planted-foot spin check)
    def _yaw(H):
        return float(np.arctan2(H.rotation[1, 0], H.rotation[0, 0]))
    L.LF_YAW[i] = _yaw(robot.stack.get_placement_LF())
    L.RF_YAW[i] = _yaw(robot.stack.get_placement_RF())
    # hip yaws (q[0:7] is the free-flyer, so q[7] = leg_left_1, q[13] = leg_right_1)
    q_now = robot.robot.q()
    L.HIP_L[i] = np.degrees(q_now[7])
    L.HIP_R[i] = np.degrees(q_now[13])
    # measured torso yaw and its decomposition (q[19] = torso_1_joint: the
    # free-flyer takes q[0:7], and torso_1 is the 13th actuated joint)
    T_torso_m, _ = robot.stack.torsoState()
    T_base_m, _ = robot.stack.baseState()
    L.TORSO_YAW_MEAS[i] = _yaw(T_torso_m)
    L.PELVIS_YAW_MEAS[i] = _yaw(T_base_m)
    L.Q_TORSO1[i] = q_now[19]

    # --- heading / swing-drag diagnostics ---
    L.TORSO_YAW[i] = torso_yaw
    L.TORSO_YAW_RATE[i] = torso_yaw_rate
    psi_des_rel = (heading_des - yaw_sup0 + np.pi) % (2 * np.pi) - np.pi
    L.HEADING_DES[i] = psi_des_rel
    L.TURN_SIGN[i] = np.sign(psi_des_rel)
    if swing_trajectory is not None:
        L.SWING_SIDE[i] = 1.0 if robot.swing_foot == Side.LEFT else -1.0
        # measured swing foot vs the pose ACTUALLY COMMANDED this tick (T_cmd),
        # projected onto the current facing's left axis. It must be the commanded
        # pose, not swing_trajectory.evaluate(): once TOUCHDOWN_FZ freezes the
        # reference the spline keeps running, and scoring against the spline
        # would book the intentional freeze as tracking error.
        T_ref = T_cmd if T_cmd is not None else swing_trajectory.evaluate(
            t_step_elapsed)[0]
        H_sw = (robot.stack.get_placement_LF() if robot.swing_foot == Side.LEFT
                else robot.stack.get_placement_RF())
        psi = yaw_sup0 + (torso_yaw if np.isfinite(torso_yaw) else 0.0)
        left = np.array([-np.sin(psi), np.cos(psi)])
        L.SWING_LAT_DEV[i] = float(
            left @ (H_sw.translation[:2] - T_ref.translation[:2]))

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
                                   ['traj', 'prog']], figsize=(14, 7))

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


def _plot_heading(L):
    """Heading/swing-drag diagnostic: does the torso-yaw slew after a heading_des
    jump drag the swing leg toward the inside of the turn?

    Read it as: HEADING_DES is a stair (it only moves at a step transition, and
    with ALTERNATE_YAW only on the inside foot's step). TORSO_YAW slews to each
    riser over ~0.2 s. If SWING_LAT_DEV goes inward during exactly those slews,
    and only for one swing side, the slew is dragging the swing leg.
    """
    import os
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    axs[0].plot(L.TIME, np.degrees(L.HEADING_DES), label='heading_des (stair)',
                drawstyle='steps-post', color='tab:red')
    axs[0].plot(L.TIME, np.degrees(L.TORSO_YAW), label='torso_yaw (slew)',
                color='tab:blue')
    axs[0].set_ylabel('yaw [deg]')
    axs[0].set_title('Commanded heading vs the torso reference chasing it')

    axs[1].plot(L.TIME, np.degrees(L.TORSO_YAW_RATE), color='tab:purple', lw=0.8)
    axs[1].axhline(0, color='k', lw=0.5)
    axs[1].set_ylabel('d(torso_yaw)/dt [deg/s]')
    axs[1].set_title('Torso yaw slew rate (spikes = a heading_des jump)')

    # swing deviation, split by which foot is swinging
    lat_mm = 1e3 * L.SWING_LAT_DEV
    is_L = L.SWING_SIDE > 0
    is_R = L.SWING_SIDE < 0
    axs[2].plot(np.where(is_L, L.TIME, np.nan), np.where(is_L, lat_mm, np.nan),
                '.', ms=2, color='tab:green', label='LEFT foot swinging')
    axs[2].plot(np.where(is_R, L.TIME, np.nan), np.where(is_R, lat_mm, np.nan),
                '.', ms=2, color='tab:orange', label='RIGHT foot swinging')
    axs[2].axhline(0, color='k', lw=0.5)
    axs[2].set_ylabel('swing lateral dev [mm]\n(+ = drifted LEFT of spline)')
    axs[2].set_xlabel('t [s]')
    axs[2].set_title('Swing foot: measured minus spline, along the facing left axis')

    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    plots_dir = os.environ.get(
        'WALKING_PLOTS_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'T7'))
    try:
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, 'heading_swing_drag.png')
        fig.savefig(out_path, dpi=150)
        print(f"saved heading diagnostic to {out_path}")
    except OSError as e:
        print(f"could not save heading diagnostic: {e}")
    # same opt-in display as _plot_results/_plot_navigation: interactive windows
    # can be un-closeable over VNC, so default to save-only.
    if os.environ.get('WALKING_SHOW_PLOTS'):
        try:
            plt.show()
        except Exception as e:
            print(f"could not show heading diagnostic: {e}")
    else:
        plt.close(fig)

    _report_heading(L)


def _report_heading(L):
    """Numeric verdict: correlate the torso-yaw slew with the swing drift, per
    swing side. This is what actually answers 'is the slew dragging the leg'."""
    ok = np.isfinite(L.SWING_LAT_DEV) & np.isfinite(L.TORSO_YAW_RATE)
    # EXCLUDE the fall: once the robot is going down, the swing foot deviates by
    # tens of cm and the hips blow past their limits, which would dominate every
    # mean here and say nothing about the gait. Nominal CoM height is ~0.87 m.
    healthy = L.COM_POS_PB[:, 2] > 0.75
    n_dropped = int((ok & ~healthy).sum())
    ok = ok & healthy
    if not ok.any():
        print("[heading] no healthy swing samples logged")
        return
    if n_dropped:
        t_bad = L.TIME[~healthy & np.isfinite(L.TIME)]
        print(f"[heading] dropped {n_dropped} samples with CoM < 0.75 m "
              f"(falling, from t={t_bad.min():.2f}s)")
    # "inward" = toward the turn centre = the sign of the turn.
    turn = np.sign(np.nanmedian(L.TURN_SIGN[ok]))
    print(f"\n[heading] turn direction: {'LEFT' if turn > 0 else 'RIGHT'}")
    print(f"[heading] {'swing side':>10} | {'n':>6} | {'mean lat dev':>12} | "
          f"{'inward dev':>11} | {'corr(dev, yaw_rate)':>19}")
    for name, sel in (('LEFT', L.SWING_SIDE > 0), ('RIGHT', L.SWING_SIDE < 0)):
        m = ok & sel
        if m.sum() < 10:
            continue
        dev = L.SWING_LAT_DEV[m]
        rate = L.TORSO_YAW_RATE[m]
        inward = turn * dev            # >0 means drifted toward the turn centre
        corr = (np.corrcoef(dev, rate)[0, 1]
                if np.std(rate) > 1e-12 and np.std(dev) > 1e-12 else np.nan)
        print(f"[heading] {name:>10} | {m.sum():6d} | {1e3*dev.mean():9.2f} mm | "
              f"{1e3*inward.mean():8.2f} mm | {corr:19.3f}")
    print("[heading] 'inward dev' > 0 on ONE side only = the slew is dragging "
          "that leg toward the turn.\n")
    _report_foot_spin(L)
    _report_drift(L)


def _report_foot_spin(L):
    """Does a foot in contact SPIN about the vertical as its load drops?

    A foot resists yaw only through distributed lateral friction over its
    contact patch, and that resistance scales with the normal force. MuJoCo's
    condim defaults to 3, so the torsional-friction coefficient never applies.
    Prediction if that is the mechanism: |d(yaw)/dt| is small while the foot is
    LOADED and grows sharply while it is LIGHTLY loaded but still down.
    """
    healthy = L.COM_POS_PB[:, 2] > 0.75
    dt = np.nanmedian(np.diff(L.TIME))
    if not np.isfinite(dt) or dt <= 0:
        return
    turn = np.sign(np.nanmedian(L.TURN_SIGN[np.isfinite(L.TURN_SIGN)]))
    print(f"[spin] foot yaw rate vs normal force "
          f"(condim=3 -> yaw held only by lateral friction, which scales with Fz)")
    print(f"[spin] {'foot':>5} | {'load band':>18} | {'n':>6} | "
          f"{'mean |dyaw/dt|':>14} | {'mean dyaw/dt (signed)':>21}")
    for name, yaw_arr, fz_arr in (('LEFT', L.LF_YAW, L.FZ_LEFT_PB),
                                  ('RIGHT', L.RF_YAW, L.FZ_RIGHT_PB)):
        dyaw = np.full_like(yaw_arr, np.nan)
        unwrapped = np.unwrap(np.where(np.isfinite(yaw_arr), yaw_arr, 0.0))
        dyaw[1:] = np.degrees(np.diff(unwrapped)) / dt
        base = healthy & np.isfinite(dyaw) & np.isfinite(fz_arr)
        for label, sel in (
                ('LOADED  Fz>300N', base & (fz_arr > 300)),
                ('UNLOADING 30-150N', base & (fz_arr > 30) & (fz_arr < 150))):
            if sel.sum() < 10:
                continue
            d = dyaw[sel]
            # signed toward the INSIDE of the turn
            print(f"[spin] {name:>5} | {label:>18} | {sel.sum():6d} | "
                  f"{np.abs(d).mean():11.2f}/s | {turn * d.mean():18.2f}/s")
    print("[spin] If UNLOADING >> LOADED, the foot is spinning as its load "
          "drops (friction cannot hold the leg's stored yaw twist).\n")
    _report_spin_timeline(L)


def _report_drift(L):
    """WHERE is the reference diverging from the robot?

    Three candidates, all already logged:
      * CoM   : the LIP interpolator is integrated open-loop from x0 and never
                re-anchored, so its CoM can walk away from the real one.
      * ZMP   : commanded vs what the feet actually produce.
      * feet  : the planted foot vs where the plan wanted it.
    Whichever grows monotonically is the drift that matters; the others are
    consequences.
    """
    healthy = L.COM_POS_PB[:, 2] > 0.75
    t = L.TIME
    t_end = np.nanmax(t[np.isfinite(t)])
    print("[drift] reference-vs-robot divergence per second "
          "(all in mm; growing monotonically = an uncorrected integrator)")
    print(f"[drift] {'t [s]':>6} | {'|CoM_ref - CoM_real|':>20} | "
          f"{'|ZMP_ref - ZMP_est|':>19} | {'CoM z':>6}")
    for t0 in np.arange(0, t_end, 1.0):
        w = np.isfinite(t) & (t >= t0) & (t < t0 + 1.0) & healthy
        if w.sum() < 5:
            continue
        d_com = 1e3 * np.nanmean(np.linalg.norm(
            L.COM_POS_REF[w, :2] - L.COM_POS_PB[w, :2], axis=1))
        d_zmp = 1e3 * np.nanmean(np.linalg.norm(
            L.ZMP_REF_LOG[w] - L.ZMP_EST[w], axis=1))
        print(f"[drift] {t0:6.1f} | {d_com:17.1f} mm | {d_zmp:16.1f} mm | "
              f"{np.nanmean(L.COM_POS_PB[w, 2]):6.3f}")
    print()


def _report_spin_timeline(L):
    """When does the loaded-foot spin START, relative to the facing angle?

    This is the ordering question: if the spin only appears once facing is
    already large / the robot is already going, it is a SYMPTOM. If it shows up
    while facing is still small and the gait still looks healthy, it is a
    CANDIDATE CAUSE.
    """
    dt = np.nanmedian(np.diff(L.TIME))
    if not np.isfinite(dt) or dt <= 0:
        return

    def rate(yaw_arr):
        d = np.full_like(yaw_arr, np.nan)
        good = np.isfinite(yaw_arr)
        u = np.unwrap(np.where(good, yaw_arr, 0.0))
        d[1:] = np.degrees(np.diff(u)) / dt
        d[~good] = np.nan
        return d

    dL, dR = rate(L.LF_YAW), rate(L.RF_YAW)
    print("[timeline] per-second: facing, loaded-foot spin, swing deviation, CoM h")
    print(f"[timeline] {'t [s]':>6} | {'facing':>7} | {'spin L':>8} | "
          f"{'CoM z':>6} | {'hipL [-20,+90]':>16} | {'LFyaw':>7} | "
          f"{'torso cmd':>9} | {'torso MEAS':>10} | {'= pelvis':>9} | "
          f"{'+ waist':>8}")
    t = L.TIME
    t_end = np.nanmax(t[np.isfinite(t)])
    for t0 in np.arange(0, t_end, 1.0):
        w = np.isfinite(t) & (t >= t0) & (t < t0 + 1.0)
        if w.sum() < 5:
            continue
        # facing = commanded heading (delta from spawn)
        face = np.degrees(np.nanmax(np.abs(L.HEADING_DES[w]))) if np.isfinite(
            L.HEADING_DES[w]).any() else np.nan
        lo_L = w & (L.FZ_LEFT_PB > 300) & np.isfinite(dL)
        lo_R = w & (L.FZ_RIGHT_PB > 300) & np.isfinite(dR)
        sL = np.abs(dL[lo_L]).mean() if lo_L.sum() > 5 else np.nan
        sR = np.abs(dR[lo_R]).mean() if lo_R.sum() > 5 else np.nan
        dev = 1e3 * np.nanmean(np.abs(L.SWING_LAT_DEV[w]))
        comz = np.nanmean(L.COM_POS_PB[w, 2])
        hl = L.HIP_L[w]
        hl_s = f"{np.nanmin(hl):+6.1f}..{np.nanmax(hl):+6.1f}"
        # measured foot yaws, as a delta from spawn -> directly comparable to
        # 'facing' (the commanded heading). If facing climbs and these do not,
        # the feet are simply not achieving the yaw they are told to.
        def _delta(arr):
            """mean over this window, as a delta from the run's first samples"""
            fin = arr[np.isfinite(arr)]
            if fin.size < 50:
                return np.nan
            return np.degrees(np.nanmean(arr[w]) - np.nanmean(fin[:50]))

        lfy = _delta(L.LF_YAW)
        t_cmd = np.degrees(np.nanmean(L.TORSO_YAW[w]))   # already a delta
        t_meas = _delta(L.TORSO_YAW_MEAS)
        p_meas = _delta(L.PELVIS_YAW_MEAS)
        waist = _delta(L.Q_TORSO1)
        print(f"[timeline] {t0:6.1f} | {face:6.1f}d | {sL:6.1f}/s | "
              f"{comz:6.3f} | {hl_s:>16} | {lfy:+6.1f}d | "
              f"{t_cmd:+8.1f}d | {t_meas:+9.1f}d | {p_meas:+8.1f}d | "
              f"{waist:+7.1f}d")
    print()


################################################################################
# main control loop
################################################################################

def _run_loop(simulator, crowd, robot, mpc, interpolator, plan, ZMP_ref, YAW_ref,
              com_start, com_target, N_pre, N_sim, planner=None, no_steps=0):
    """Run the walking simulation loop and return the filled log buffers."""
    ROBOT_RADIUS_PHYS = 0.3                     # physical robot radius for the safety plot
    n_peds = len(crowd.peds)
    L = _allocate_logs(N_sim, n_peds)
    L.GOAL = np.asarray(ROBOT_GOAL, dtype=float)
    L.PED_RADII = np.array([r for (_, _, r) in crowd.states()])

    k = 0                                       # MPC index (fixed-plan mode); reset
                                                # to 0 at each step transition, since
                                                # the plan is re-anchored there
    mpc_phase = 0                               # MPC-sample index within the current step
    t_step_elapsed = 0.0                        # elapsed time within current step
    t_publish = 0.0                             # last ros publish time
    u_k = np.zeros(2)                           # held mpc control (commanded zmp)
    swing_trajectory = None                     # current swing-foot spline
    next_step = None                            # current swing-target footstep (for viz)
    pin_data = robot.stack.model.createData()   # scratch for centroidal queries
    c = c_dot = c_ddot = None                   # latest com reference

    # torso-yaw tracking: the torso task pins the body's heading, so as the feet
    # yaw around a turn the legs twist and the robot topples. Command the torso
    # orientation reference to follow the support-foot heading (low-pass filtered
    # so the reference is smooth). yaw 0 -> unchanged from the spawn orientation,
    # so straight walking is unaffected.
    T_torso0, _ = robot.stack.torsoState()
    R_torso0 = T_torso0.rotation.copy()
    p_torso0 = T_torso0.translation.copy()
    T_base0, _ = robot.stack.baseState()
    R_base0 = T_base0.rotation.copy()
    p_base0 = T_base0.translation.copy()
    T_sup0 = robot.supportFootPose()
    yaw_sup0 = np.arctan2(T_sup0.rotation[1, 0], T_sup0.rotation[0, 0])
    torso_yaw = 0.0                             # filtered heading (delta from start)
    heading_des = yaw_sup0                      # desired heading = current step target yaw
    TORSO_YAW_LP = 0.01                         # low-pass gain per sim step

    # AM yaw feedforward: I_zz about the CoM from the centroidal composite
    # inertia at the current (home) posture. Held constant -- it varies with the
    # posture, but the legs stay near the same configuration through the gait.
    # (ccrba returns the 6xnv momentum matrix; the composite inertia lands in
    # pin_data.Ig as a side effect -- that is what carries I_zz.)
    pin.ccrba(robot.stack.model, pin_data, robot.robot.q(), robot.robot.v())
    I_zz = float(pin_data.Ig.inertia[2, 2])
    yaw_rate = 0.0                              # d(torso_yaw)/dt, optionally LP'd
    torso_yaw_prev = 0.0
    torso_yaw_rate = 0.0                        # raw d(torso_yaw)/dt (diagnostic)

    # robot ORCA velocity generator: v_cmd toward the goal, avoiding pedestrians
    robot_ctrl = RobotORCA(ROBOT_GOAL, radius=0.35, max_speed=0.4,
                           desired_speed=0.25, safety_margin=0.2)

    # reactive footstep planner: each step, places the next footstep in the
    # direction of v_cmd (omnidirectional / fixed facing) and feeds the MPC.
    reactive_planner = ReactiveFootStepPlanner(
        conf, ROBOT_GOAL, robot.supportFootPose(),
        conf.no_mpc_samples_per_step, conf.no_mpc_samples_per_horizon,
        turn_gain=TURN_GAIN, max_dyaw=np.deg2rad(MAX_DYAW_DEG),
        alternate_yaw=ALTERNATE_YAW)
    current_swing_side = robot.swing_foot        # which foot swings next
    reactive_planner.regenerate(
        reactive_planner.make_footstep(robot.supportFootPose(), robot.support_foot),
        np.zeros(2))                             # seed the buffer until the first step
    v_cmd_acc = np.zeros(2)                      # v_cmd sum since the last step
    v_cmd_n = 0                                  # ... and its sample count
    swing_frozen = False                         # touchdown detected this step
    T_frozen = None                              # swing pose held after touchdown
    T_cmd = None                                 # swing pose commanded this tick

    _install_sigint_handler()

    for i in range(-N_pre, N_sim):
        if _ABORT:
            print(f"aborted at t={i * conf.dt:.2f} s "
                  f"({i + N_pre}/{N_pre + N_sim} samples logged)", flush=True)
            break
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
        v_cmd = robot_ctrl.compute_velocity(com[:2], vcom[:2], ped_states)
        v_pref = robot_ctrl.preferred_velocity(com[:2])
        # accumulate for the per-step average: ORCA sees the CoM's gait-frequency
        # lateral sway (+/-0.3 m/s), so the instantaneous v_cmd oscillates at the
        # step frequency. Sampling it once per step at the transition ALIASES
        # that oscillation (lateral progress freezes when the sample lands near
        # zero), so the planner gets the average over the elapsed step instead.
        v_cmd_acc += v_cmd
        v_cmd_n += 1

        # --- navigation logging ---
        if i >= 0:
            L.ROBOT_XY[i] = com[:2]
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
        # yellow foot-shaped box at the next planned footstep
        boxes = [debug_viz.footstep_box(next_step)] if next_step is not None else []
        debug_viz.draw_scene(simulator, arrows=arrows, points=[goal_marker], boxes=boxes)

        # --- step transition: pick/plan the next footstep, build swing spline ---
        if i >= 0 and i % conf.no_sim_per_step == 0:
            if REACTIVE:
                # plan the next step in the direction of v_cmd (steps in place at goal)
                swing_side = current_swing_side
                support_side = other_foot_id(swing_side)
                robot.setSwingFoot(swing_side)
                robot.setSupportFoot(support_side)
                support_fs = reactive_planner.make_footstep(
                    robot.supportFootPose(), support_side)
                at_goal = reactive_planner.at_goal(com[:2])
                # de-aliased command: average of v_cmd over the elapsed step
                v_cmd_step = v_cmd_acc / max(v_cmd_n, 1)
                v_cmd_acc[:] = 0.0
                v_cmd_n = 0
                reactive_planner.regenerate(support_fs, v_cmd_step, at_goal)
                next_step = reactive_planner.swing_target()
                current_swing_side = other_foot_id(swing_side)
                mpc_phase = 0
            else:
                # RE-ANCHOR the fixed plan to the MEASURED support foot, every
                # step. This used to index a plan computed ONCE at t=0
                # (plan[plan_idx]) with nothing ever comparing it to the robot:
                # per-step placement error accumulated unbounded, the MPC kept
                # commanding a CoP inside the PLANNED foot while the real foot
                # drifted away, and once the gap exceeded the sole (~0.2x0.13 m)
                # the commanded CoP fell OUTSIDE the real support polygon and
                # became unrealizable -> the "fine for 8 s, then sudden" fall.
                # Smaller steps only slowed the accumulation, which is why the
                # fall moved later but never went away.
                # Rebuilding from the measured foot keeps the curvature
                # (dpsi = step_x/radius) but re-zeroes the error each step --
                # the same thing the reactive planner's regenerate() does.
                swing_side = current_swing_side
                support_side = other_foot_id(swing_side)
                robot.setSwingFoot(swing_side)
                robot.setSupportFoot(support_side)
                plan, ZMP_ref, YAW_ref = _plan_from(
                    planner, robot.supportFootPose(), support_side, no_steps)
                next_step = plan[1]
                current_swing_side = other_foot_id(swing_side)
                # plan[0] is now the CURRENT support foot, and the reference
                # holds it for the first no_mpc_samples_per_step samples, so the
                # horizon index restarts at 0 to stay phase-aligned.
                k = 0
            T0_swing = robot.swingFootPose()
            T1_swing = next_step.poseInWorld()
            swing_trajectory = SwingFootTrajectory(T0_swing, T1_swing, conf.step_dur)
            t_step_elapsed = 0.0
            swing_frozen = False
            T_frozen = None
            # Heading the torso/pelvis is steered to. The pelvis yaw costs hip
            # yaw: hip = foot_yaw - pelvis_yaw, and leg_left_1 has only 20 deg
            # of NEGATIVE range ([-20,+90]) vs leg_right_1's 90 ([-90,+20]).
            # Tracking next_step (the SWING TARGET) puts the pelvis a step AHEAD
            # of the planted feet, which spends that 20 deg budget on a lead the
            # gait does not need -- the left hip then winds to its -20 stop in a
            # left turn and the loaded foot spins instead of yielding.
            if TORSO_YAW_SOURCE == "support_meas":
                T_sup_now = robot.supportFootPose()      # MEASURED, just planted
                heading_des = np.arctan2(T_sup_now.rotation[1, 0],
                                         T_sup_now.rotation[0, 0])
            else:                                        # "swing_target" (legacy)
                heading_des = np.arctan2(next_step.pose.rotation[1, 0],
                                         next_step.pose.rotation[0, 0])

        # --- update the mpc every no_sim_per_mpc steps ---
        if i >= 0 and i % conf.no_sim_per_mpc == 0:
            # Close the loop: plan from the MEASURED CoM, not from the LIP's own
            # integration (see MPC_STATE_FEEDBACK). The interpolator is re-seeded
            # to the same state, otherwise c_ref keeps free-running away from the
            # robot even though the MPC now plans from the truth -- the feedback
            # has to reach the reference TSID actually tracks, not just the QP.
            if MPC_STATE_FEEDBACK > 0.0:
                com_m = robot.stack.comState()
                p_m, v_m = com_m.value(), com_m.derivative()
                x_meas = np.array([p_m[0], v_m[0], p_m[1], v_m[1]])
                a = float(np.clip(MPC_STATE_FEEDBACK, 0.0, 1.0))
                x_k = (1.0 - a) * interpolator.x + a * x_meas
                interpolator.set_state(x_k)
            else:
                x_k = interpolator.x            # legacy open loop
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
            u_k = mpc.buildSolveOCP(x_k, ZMP_ref_k, terminal_idx, yaw_ref_k=YAW_ref_k)

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
            # evaluate the swing-foot spline and update its motion reference.
            # Touchdown detection: in the descent half of the step, once the
            # swing foot's FT sensor reports ground contact, FREEZE the
            # reference at the current pose -- continuing the spline would
            # drive the foot into the ground and tilt it about a sole edge.
            if swing_trajectory is not None:
                if (TOUCHDOWN_FZ > 0 and not swing_frozen
                        and t_step_elapsed > 0.5 * conf.step_dur):
                    swing_frame = (conf.lf_frame_name if robot.swing_foot == Side.LEFT
                                   else conf.rf_frame_name)
                    f_swing, _ = robot._foot_wrench_world(swing_frame)
                    if f_swing[2] > TOUCHDOWN_FZ:
                        swing_frozen = True
                        T_frozen, _, _ = swing_trajectory.evaluate(t_step_elapsed)
                        print(f"[touchdown] early contact: t_step="
                              f"{t_step_elapsed:.2f}s Fz={f_swing[2]:.0f}N "
                              "-> swing ref frozen")
                if swing_frozen:
                    robot.updateSwingFootRef(T_frozen, np.zeros(6), np.zeros(6))
                    T_cmd = T_frozen
                else:
                    T_swing, V_swing, A_swing = swing_trajectory.evaluate(t_step_elapsed)
                    robot.updateSwingFootRef(T_swing, V_swing.vector, A_swing.vector)
                    T_cmd = T_swing

            # steer the torso toward the desired heading (fixed in omnidirectional
            # walking; only turns if the footsteps carry yaw)
            psi_des = (heading_des - yaw_sup0 + np.pi) % (2 * np.pi) - np.pi
            torso_yaw += TORSO_YAW_LP * (psi_des - torso_yaw)          # low-pass
            # raw slew rate of the commanded torso yaw. Computed every tick (not
            # just when the AM feedforward is on) because it is the signal that
            # says whether a heading_des jump is dragging the swing leg.
            torso_yaw_rate = (torso_yaw - torso_yaw_prev) / dt
            torso_yaw_prev = torso_yaw
            cz, sz = np.cos(torso_yaw), np.sin(torso_yaw)
            Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
            robot.stack.setTorsoRefState(pin.SE3(Rz @ R_torso0, p_torso0))
            # AM reference: permit the turn's yaw spin instead of fighting it.
            # yaw_rate is the RAW torso_yaw derivative unless AM_YAW_RATE_TAU>0
            # (the increment is bursty at step transitions, so the filter is
            # there to A/B against).
            if AM_YAW_FF != 0.0:
                if AM_YAW_RATE_TAU > 0.0:
                    a = dt / (AM_YAW_RATE_TAU + dt)
                    yaw_rate += a * (torso_yaw_rate - yaw_rate)
                else:
                    yaw_rate = torso_yaw_rate
                robot.stack.setAmRefState(
                    np.array([0.0, 0.0, AM_YAW_FF * I_zz * yaw_rate]))
            # steer the base (pelvis) yaw with the same heading, so the waist
            # does not absorb the whole turn and the hip-yaw joints don't take
            # the full foot-vs-body twist (yaw-only task, position row masked)
            robot.stack.setBaseOriRefState(pin.SE3(Rz @ R_base0, p_base0))

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
            # depth image + CameraInfo (front RGB-D): depth_image_proc's
            # point_cloud_xyz turns these into /camera/depth/points for RViz
            # (ros2 launch bullet_sims depth_image_proc.launch.py)
            robot.publish_depth_image()
            if PEDESTRIAN_SENSOR_DATA:
                # obstacle-estimate overlay (cyan) + ground-truth ghosts (grey)
                robot.publish_obstacle_markers(
                    {p.name: dict(x=p.position[0], y=p.position[1], r=p.radius)
                     for p in crowd.peds})

        # store for the post-run plots
        if i >= 0:
            _record_step(L, i, t, c, c_dot, c_ddot, t_step_elapsed,
                         swing_trajectory, robot, simulator, interpolator, pin_data,
                         torso_yaw=torso_yaw, heading_des=heading_des,
                         yaw_sup0=yaw_sup0, torso_yaw_rate=torso_yaw_rate,
                         T_cmd=T_cmd)

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
    global SCENARIO, PEDESTRIAN_SENSOR_DATA
    node = rclpy.create_node("walking_params")
    node.declare_parameter("scenario", SCENARIO)
    node.declare_parameter("pedestrian_sensor_data", PEDESTRIAN_SENSOR_DATA)
    SCENARIO = node.get_parameter("scenario").get_parameter_value().string_value
    PEDESTRIAN_SENSOR_DATA = node.get_parameter(
        "pedestrian_sensor_data").get_parameter_value().bool_value
    node.destroy_node()
    print(f"[walking] scenario='{SCENARIO}'  "
          f"pedestrian_sensor_data={PEDESTRIAN_SENSOR_DATA}")


def main():
    # NOTE: ros2 run skips the __main__ block, so all setup happens here.
    simulator = _setup_simulator()

    # apply beautiful tiled floor and lighting
    setup_floor_ambience(simulator._spec)
    simulator._dirty = True
    simulator._ensure_compiled()

    # runtime parameters (scenario, sensor vs ground truth) -> module globals
    _read_runtime_params()

    # AM task weight must land on conf BEFORE _create_robot builds the TSIDWrapper
    # (the task is constructed with its weight and never re-weighted afterwards).
    conf.w_am = W_AM

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
    planner, plan, no_steps, ZMP_ref, YAW_ref = _build_plan(robot)
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
                  com_start, com_target, N_pre, N_sim,
                  planner=planner, no_steps=no_steps)
    _plot_navigation(L)

    _plot_heading(L)

    _plot_results(L)

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()