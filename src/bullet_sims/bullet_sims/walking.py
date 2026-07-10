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

# crowd
from bullet_sims.crowd.pedestrian import PedestrianSimulator
from bullet_sims.crowd import scenarios
from bullet_sims.robot_orca import RobotORCA
from bullet_sims import debug_viz


# footstep path for the walking demo: "line" (straight) or "circle" (curved)
WALK_MODE = "circle"
CIRCLE_RADIUS = 1.5     # [m] radius of the centre path when WALK_MODE == "circle"
CIRCLE_CLOCKWISE = False   # False = curve left/CCW, True = curve right/CW
# False = omnidirectional (fixed facing, side-steps along the curve) -> robust: reuses
#         the straight-line gait, no body rotation, best for bounded curves (avoidance).
# True  = turn the whole body to face the path -> more natural, harder to stabilise.
CIRCLE_FACE_PATH = False

# robot ORCA velocity (debug): the goal it heads toward, and the arrow display scale.
# The v_cmd is only visualized here (green arrow); it does not yet drive the footsteps.
ROBOT_GOAL = (4.0, 0.0)    # world (x, y) navigation goal
ARROW_SCALE = 3.0          # visual stretch so the slow v_cmd arrow is visible


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
    scenarios.build(crowd, "crossing")
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

def _allocate_logs(N_sim):
    """Allocate all per-iteration log buffers as NaN-filled arrays."""
    def nan(*shape):
        return np.nan * np.empty(shape if len(shape) > 1 else shape[0])

    return SimpleNamespace(
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
        out_path = os.path.join(plots_dir, 'walking_results.png')
        fig.savefig(out_path, dpi=150)
        print(f"saved plots to {out_path}")
    except OSError as e:
        print(f"could not save plots to {plots_dir}: {e}")
    try:
        plt.show()
    except Exception as e:
        print(f"could not show plots: {e}")


################################################################################
# main control loop
################################################################################

def _run_loop(simulator, crowd, robot, mpc, interpolator, plan, ZMP_ref, YAW_ref,
              com_start, com_target, N_pre, N_sim):
    """Run the walking simulation loop and return the filled log buffers."""
    L = _allocate_logs(N_sim)

    k = 0                                       # current MPC index
    plan_idx = 1                                # current step index in the plan
    t_step_elapsed = 0.0                        # elapsed time within current step
    t_publish = 0.0                             # last ros publish time
    u_k = np.zeros(2)                           # held mpc control (commanded zmp)
    swing_trajectory = None                     # current swing-foot spline
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

    # robot ORCA velocity generator (debug visualization only for now)
    robot_ctrl = RobotORCA(ROBOT_GOAL, radius=0.35, max_speed=0.4,
                           desired_speed=0.25, safety_margin=0.2)

    for i in range(-N_pre, N_sim):
        t = i * conf.dt
        dt = conf.dt

        # --- update the mpc every no_sim_per_mpc steps ---
        if i >= 0 and i % conf.no_sim_per_mpc == 0:
            x_k = interpolator.x
            ZMP_ref_k = ZMP_ref[k:k + conf.no_mpc_samples_per_horizon]
            YAW_ref_k = YAW_ref[k:k + conf.no_mpc_samples_per_horizon]
            # apply the terminal constraint once the remaining reference is
            # shorter than a full horizon (the duplicated footsteps keep this
            # slice in bounds)
            remaining = len(ZMP_ref) - k
            terminal_idx = remaining - 1 if remaining < conf.no_mpc_samples_per_horizon else -1
            u_k = mpc.buildSolveOCP(x_k, ZMP_ref_k, terminal_idx, yaw_ref_k=YAW_ref_k)
            k += 1

        # --- step transition: build the next swing-foot spline ---
        if i >= 0 and i % conf.no_sim_per_step == 0:
            next_step = plan[plan_idx]
            robot.setSwingFoot(next_step.side)
            robot.setSupportFoot(other_foot_id(next_step.side))
            T0_swing = robot.swingFootPose()
            T1_swing = next_step.poseInWorld()
            swing_trajectory = SwingFootTrajectory(T0_swing, T1_swing, conf.step_dur)
            t_step_elapsed = 0.0
            plan_idx += 1
            # desired body heading = the yaw of the step we are stepping to; this
            # LEADS the turn instead of lagging behind the planted foot
            heading_des = np.arctan2(next_step.pose.rotation[1, 0],
                                     next_step.pose.rotation[0, 0])

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

            # steer the torso toward the desired (leading) path heading, so the
            # trunk actively pulls the body around the turn instead of lagging
            psi_des = (heading_des - yaw_sup0 + np.pi) % (2 * np.pi) - np.pi
            torso_yaw += TORSO_YAW_LP * (psi_des - torso_yaw)          # low-pass
            cz, sz = np.cos(torso_yaw), np.sin(torso_yaw)
            Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
            robot.stack.setTorsoRefState(pin.SE3(Rz @ R_torso0, p_torso0))

        # --- crowd: pedestrians avoid the robot; expose their states ---
        body_id = robot.robot.id()
        com = simulator.computeComPosition(body_id)
        vcom = simulator.computeComVelocity(body_id)
        # robot_radius inflated as a safety margin
        crowd.update(conf.dt, external_agents=[(com[:2], vcom[:2], 0.35)])
        ped_states = crowd.states()   # list of (pos(2), vel(2), radius)

        # robot ORCA velocity command. Only visualized for now (green = collision-free
        # v_cmd, blue = preferred/goal direction); it does not yet drive the footsteps.
        v_cmd = robot_ctrl.compute_velocity(com[:2], vcom[:2], ped_states)
        v_pref = robot_ctrl.preferred_velocity(com[:2])
        arrows = [a for a in (
            debug_viz.velocity_arrow(com[:2], v_pref, z=float(com[2]),
                                     scale=ARROW_SCALE, rgba=(0.3, 0.5, 1.0, 0.6)),
            debug_viz.velocity_arrow(com[:2], v_cmd, z=float(com[2]),
                                     scale=ARROW_SCALE, rgba=(0.1, 0.9, 0.1, 1.0)),
        ) if a is not None]
        # red sphere at the navigation goal
        goal_marker = debug_viz.point_marker(ROBOT_GOAL, z=0.15, radius=0.12,
                                             rgba=(1.0, 0.2, 0.2, 1.0))
        debug_viz.draw_scene(simulator, arrows=arrows, points=[goal_marker])

        # --- advance the simulation + whole-body control ---
        simulator.step()
        robot.update()

        # --- turning diagnostic (once per second while walking) ---
        if i >= 0 and i % int(round(1.0 / conf.dt)) == 0:
            T_sup_d = robot.supportFootPose()
            sup_yaw = np.degrees(np.arctan2(T_sup_d.rotation[1, 0], T_sup_d.rotation[0, 0]))
            T_base_d, _ = robot.stack.baseState()
            base_yaw = np.degrees(np.arctan2(T_base_d.rotation[1, 0], T_base_d.rotation[0, 0]))
            tgt_yaw = np.degrees(np.arctan2(plan[plan_idx - 1].pose.rotation[1, 0],
                                            plan[plan_idx - 1].pose.rotation[0, 0]))
            print(f"[turn] t={t:5.1f}s  target_step_yaw={tgt_yaw:7.1f}  "
                  f"support_yaw={sup_yaw:7.1f}  base_yaw={base_yaw:7.1f}  "
                  f"torso_ref_yaw={np.degrees(yaw_sup0 + torso_yaw):7.1f}  [deg]")

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

def main():
    # NOTE: ros2 run skips the __main__ block, so all setup happens here.
    simulator = _setup_simulator()

    # crowd must be created before the robot (see _setup_crowd)
    crowd = _setup_crowd(simulator)

    created = _create_robot(simulator)
    if created is None:
        return
    tw, robot = created

    # footstep plan + zmp reference
    planner, plan, no_steps = _build_plan(robot)
    ZMP_ref = generate_zmp_reference(plan, conf.no_mpc_samples_per_step)
    YAW_ref = generate_yaw_reference(plan, conf.no_mpc_samples_per_step)
    print("Plotting the footstep plan and ZMP reference\n")
    planner.plot(simulator)

    # lip mpc + interpolator + pre-walk sway endpoints
    mpc, interpolator, com_start, com_target = _setup_lip_mpc(robot)
    robot.stack.setComRefState(com_start)

    # iteration counts
    pre_dur = 3.0   # time to wait before walking starts
    N_pre = int(round(pre_dur / conf.dt))
    N_sim = int(round(no_steps * conf.step_dur / conf.dt))

    # run the simulation, then plot
    L = _run_loop(simulator, crowd, robot, mpc, interpolator, plan, ZMP_ref, YAW_ref,
                  com_start, com_target, N_pre, N_sim)
    _plot_results(L)

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
