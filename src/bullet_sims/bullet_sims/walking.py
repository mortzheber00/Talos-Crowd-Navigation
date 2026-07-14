"""
talos walking simulation
"""

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
WALK_DURATION = 30.0       # [s] max walk time (reactive walking has no fixed step count)
REACTIVE = True            # True: footsteps follow v_cmd; False: fixed WALK_MODE plan

# ---- runtime parameters (defaults; override at launch via ROS 2 params, e.g.
#   ros2 run bullet_sims walking --ros-args -p scenario:=crossing -p pedestrian_sensor_data:=true )
SCENARIO = "random"              # crowd scenario: head_on | crossing | circle | corridor | random
PEDESTRIAN_SENSOR_DATA = False   # True: depth-camera perception; False: ground-truth states


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
                           num_sub_steps=8, render=True)


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
        GOAL=np.asarray((0.0, 0.0)),            # filled in _run_loop
        PED_RADII=np.zeros(max(n_peds, 1)),     # filled in _run_loop
        ROBOT_RADIUS=0.3,                       # physical robot radius [m], filled in _run_loop
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


def _animate_navigation(L, fps=20, target_frames=200):
    """Animated top-down view of the run: robot and pedestrians as moving bodies
    (drawn at their physical radii) with fading trails, saved as a GIF.

    Set WALKING_NO_GIF=1 to skip (the animation is the slow part of plotting)."""
    import os
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.animation import FuncAnimation, PillowWriter

    if os.environ.get('WALKING_NO_GIF'):
        return

    # samples that actually got recorded (buffers are NaN-padded up front)
    valid = np.where(~np.isnan(L.TIME) & ~np.isnan(L.ROBOT_XY[:, 0]))[0]
    if len(valid) < 2:
        print("not enough navigation samples to animate")
        return
    # downsample so the gif stays small regardless of control-loop rate
    stride = max(1, len(valid) // target_frames)
    frames = valid[::stride]
    if frames[-1] != valid[-1]:
        frames = np.append(frames, valid[-1])

    n_peds = L.PED_XY.shape[1]
    robot_r = float(getattr(L, 'ROBOT_RADIUS', 0.3))
    ped_r = np.asarray(L.PED_RADII, dtype=float)

    # fixed axis limits covering the whole run (robot + peds + goal), with margin
    xs = [L.ROBOT_XY[valid, 0], np.atleast_1d(L.GOAL[0])]
    ys = [L.ROBOT_XY[valid, 1], np.atleast_1d(L.GOAL[1])]
    for j in range(n_peds):
        xs.append(L.PED_XY[valid, j, 0]); ys.append(L.PED_XY[valid, j, 1])
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    xs = xs[~np.isnan(xs)]; ys = ys[~np.isnan(ys)]
    margin = max(1.0, robot_r + (ped_r.max() if ped_r.size else 0.3))
    xlim = (xs.min() - margin, xs.max() + margin)
    ylim = (ys.min() - margin, ys.max() + margin)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.grid(True, alpha=0.3)
    ax.plot(L.GOAL[0], L.GOAL[1], '*', color='tab:red', ms=18, zorder=5, label='goal')

    # trails (drawn up to the current frame)
    robot_trail, = ax.plot([], [], '-', color='tab:green', lw=2.0, alpha=0.7, label='robot')
    ped_trails = [ax.plot([], [], '-', color='tab:gray', lw=1.0, alpha=0.4,
                          label='pedestrians' if j == 0 else None)[0]
                  for j in range(n_peds)]
    # bodies drawn as circles at physical radii
    robot_body = Circle((0.0, 0.0), robot_r, color='tab:green', alpha=0.9, zorder=6)
    ax.add_patch(robot_body)
    ped_bodies = []
    for j in range(n_peds):
        rj = float(ped_r[j]) if j < ped_r.size else 0.3
        c = Circle((0.0, 0.0), rj, color='tab:blue', alpha=0.6, zorder=4)
        ax.add_patch(c); ped_bodies.append(c)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, va='top', fontsize=11,
                        bbox=dict(boxstyle='round', fc='white', alpha=0.7))
    ax.set_title('Top-down view (robot & crowd)')
    ax.legend(loc='upper right', fontsize=9)

    def update(i):
        upto = valid[valid <= i]
        robot_trail.set_data(L.ROBOT_XY[upto, 0], L.ROBOT_XY[upto, 1])
        robot_body.center = (L.ROBOT_XY[i, 0], L.ROBOT_XY[i, 1])
        for j in range(n_peds):
            px, py = L.PED_XY[upto, j, 0], L.PED_XY[upto, j, 1]
            ped_trails[j].set_data(px, py)
            pj = L.PED_XY[i, j]
            if not np.isnan(pj[0]):
                ped_bodies[j].center = (pj[0], pj[1])
                ped_bodies[j].set_visible(True)
            else:
                ped_bodies[j].set_visible(False)
        time_text.set_text(f"t = {L.TIME[i]:5.2f} s")
        return [robot_trail, robot_body, time_text, *ped_trails, *ped_bodies]

    anim = FuncAnimation(fig, update, frames=frames, blit=False,
                         interval=1000.0 / max(fps, 1))

    plots_dir = os.environ.get(
        'WALKING_PLOTS_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'T7'))
    try:
        os.makedirs(plots_dir, exist_ok=True)
        out_path = os.path.join(plots_dir, 'navigation_top_down.gif')
        anim.save(out_path, writer=PillowWriter(fps=fps))
        print(f"saved top-down animation to {out_path}")
    except Exception as e:
        print(f"could not save top-down animation: {e}")
    finally:
        plt.close(fig)


################################################################################
# main control loop
################################################################################

def _run_loop(simulator, crowd, robot, mpc, interpolator, plan, ZMP_ref, YAW_ref,
              com_start, com_target, N_pre, N_sim):
    """Run the walking simulation loop and return the filled log buffers."""
    ROBOT_RADIUS_PHYS = 0.3                     # physical robot radius for the safety plot
    n_peds = len(crowd.peds)
    L = _allocate_logs(N_sim, n_peds)
    L.GOAL = np.asarray(ROBOT_GOAL, dtype=float)
    L.PED_RADII = np.array([r for (_, _, r) in crowd.states()])
    L.ROBOT_RADIUS = ROBOT_RADIUS_PHYS

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

    # torso-yaw tracking: the torso task pins the body's heading, so as the feet
    # yaw around a turn the legs twist and the robot topples. Command the torso
    # orientation reference to follow the support-foot heading (low-pass filtered
    # so the reference is smooth). yaw 0 -> unchanged from the spawn orientation,
    # so straight walking is unaffected.
    T_torso0, _ = robot.stack.torsoState()
    R_torso0 = T_torso0.rotation.copy()
    p_torso0 = T_torso0.translation.copy()
    T_sup0 = robot.supportFootPose()
    yaw_sup0 = np.arctan2(T_sup0.rotation[1, 0], T_sup0.rotation[0, 0])
    torso_yaw = 0.0                             # filtered heading (delta from start)
    heading_des = yaw_sup0                      # desired heading = current step target yaw
    TORSO_YAW_LP = 0.05                         # low-pass gain per sim step

    # robot ORCA velocity generator: v_cmd toward the goal, avoiding pedestrians
    robot_ctrl = RobotORCA(ROBOT_GOAL, radius=0.35, max_speed=0.4,
                           desired_speed=0.25, safety_margin=0.2)

    # reactive footstep planner: each step, places the next footstep in the
    # direction of v_cmd (omnidirectional / fixed facing) and feeds the MPC.
    reactive_planner = ReactiveFootStepPlanner(
        conf, ROBOT_GOAL, robot.supportFootPose(),
        conf.no_mpc_samples_per_step, conf.no_mpc_samples_per_horizon)
    current_swing_side = robot.swing_foot        # which foot swings next
    reactive_planner.regenerate(
        reactive_planner.make_footstep(robot.supportFootPose(), robot.support_foot),
        np.zeros(2))                             # seed the buffer until the first step

    try:
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
            v_cmd = robot_ctrl.compute_velocity(com[:2], vcom[:2], ped_states)
            v_pref = robot_ctrl.preferred_velocity(com[:2])

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
                    reactive_planner.regenerate(support_fs, v_cmd, at_goal)
                    next_step = reactive_planner.swing_target()
                    current_swing_side = other_foot_id(swing_side)
                    mpc_phase = 0
                else:
                    next_step = plan[plan_idx]
                    robot.setSwingFoot(next_step.side)
                    robot.setSupportFoot(other_foot_id(next_step.side))
                    plan_idx += 1
                T0_swing = robot.swingFootPose()
                T1_swing = next_step.poseInWorld()
                swing_trajectory = SwingFootTrajectory(T0_swing, T1_swing, conf.step_dur)
                t_step_elapsed = 0.0
                heading_des = np.arctan2(next_step.pose.rotation[1, 0],
                                         next_step.pose.rotation[0, 0])

            # --- update the mpc every no_sim_per_mpc steps ---
            if i >= 0 and i % conf.no_sim_per_mpc == 0:
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
                # evaluate the swing-foot spline and update its motion reference
                if swing_trajectory is not None:
                    T_swing, V_swing, A_swing = swing_trajectory.evaluate(t_step_elapsed)
                    robot.updateSwingFootRef(T_swing, V_swing.vector, A_swing.vector)

                # steer the torso toward the desired heading (fixed in omnidirectional
                # walking; only turns if the footsteps carry yaw)
                psi_des = (heading_des - yaw_sup0 + np.pi) % (2 * np.pi) - np.pi
                torso_yaw += TORSO_YAW_LP * (psi_des - torso_yaw)          # low-pass
                cz, sz = np.cos(torso_yaw), np.sin(torso_yaw)
                Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
                robot.stack.setTorsoRefState(pin.SE3(Rz @ R_torso0, p_torso0))

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
    except KeyboardInterrupt:
        # allow an early Ctrl-C: return whatever was logged so far so the
        # post-run plots and top-down GIF are still produced.
        print("\ninterrupted (Ctrl-C): plotting partial run...")

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
    _plot_navigation(L)
    _animate_navigation(L)

    _plot_results(L)

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()