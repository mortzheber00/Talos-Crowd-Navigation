"""talos walking + depth perception

Combines the two sibling demos:

* the full walking pipeline from ``walking.py`` (LIP-MPC, footstep planner,
  swing-foot trajectories, contact state machine, plots), and
* the crowd + onboard depth perception from ``stand_and_perceive.py``
  (``CrowdEnv`` cylinder agents in front of the head camera, depth image +
  per-cylinder obstacle estimates streamed to ROS 2).

The robot walks its footstep plan while the head depth camera perceives a
scripted crowd of cylinder agents; depth images, obstacle estimates and RViz
markers are published alongside the usual joint states / TF.
"""

import rclpy
import pinocchio as pin

# robot configs
import bullet_sims.talos_conf as conf

# crowd + perception (from stand_and_perceive)
from bullet_sims.crowd_env import CrowdEnv

# walking pipeline: reuse walking.py's setup/plan/mpc/logging/plotting helpers
from bullet_sims.footstep_planner import Side, other_foot_id
from bullet_sims.foot_trajectory import SwingFootTrajectory
from bullet_sims.lip_mpc import generate_zmp_reference, generate_yaw_reference
from bullet_sims.walking import (
    _setup_simulator,
    _create_robot,
    _build_plan,
    _setup_lip_mpc,
    _allocate_logs,
    _record_step,
    _plot_results,
)

import numpy as np


################################################################################
# main control loop
################################################################################

def _run_loop(simulator, crowd, robot, mpc, interpolator, plan, ZMP_ref, YAW_ref,
              com_start, com_target, N_pre, N_sim):
    """Run the walking loop while streaming depth perception of the crowd.

    Mirrors ``walking._run_loop`` but swaps the ORCA crowd for the ``CrowdEnv``
    cylinder agents and adds the depth-image / obstacle publishing from
    ``stand_and_perceive``.
    """
    L = _allocate_logs(N_sim)

    k = 0                                       # current MPC index
    plan_idx = 1                                # current step index in the plan
    t_step_elapsed = 0.0                        # elapsed time within current step
    t_publish = 0.0                             # last ros publish time
    u_k = np.zeros(2)                           # held mpc control (commanded zmp)
    swing_trajectory = None                     # current swing-foot spline
    pin_data = robot.stack.model.createData()   # scratch for centroidal queries
    c = c_dot = c_ddot = None                   # latest com reference

    # torso-yaw tracking (see walking._run_loop for the rationale)
    T_torso0, _ = robot.stack.torsoState()
    R_torso0 = T_torso0.rotation.copy()
    p_torso0 = T_torso0.translation.copy()
    T_sup0 = robot.supportFootPose()
    yaw_sup0 = np.arctan2(T_sup0.rotation[1, 0], T_sup0.rotation[0, 0])
    torso_yaw = 0.0                             # filtered heading (delta from start)
    heading_des = yaw_sup0                      # desired heading = current step target yaw
    TORSO_YAW_LP = 0.05                         # low-pass gain per sim step

    for i in range(-N_pre, N_sim):
        t = i * conf.dt
        dt = conf.dt

        # --- update the mpc every no_sim_per_mpc steps ---
        if i >= 0 and i % conf.no_sim_per_mpc == 0:
            x_k = interpolator.x
            ZMP_ref_k = ZMP_ref[k:k + conf.no_mpc_samples_per_horizon]
            YAW_ref_k = YAW_ref[k:k + conf.no_mpc_samples_per_horizon]
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
            heading_des = np.arctan2(next_step.pose.rotation[1, 0],
                                     next_step.pose.rotation[0, 0])

        # --- references, every iteration ---
        if i < 0:
            # pre-walk: cosine smoothstep sway of the CoM to the first support foot
            s = (i + N_pre) / N_pre
            s = 0.5 - 0.5 * np.cos(np.pi * s)
            robot.stack.setComRefState((1.0 - s) * com_start + s * com_target)

        if i >= 0:
            t_step_elapsed += dt
            interpolator.integrate(u_k)
            c, c_dot, c_ddot = interpolator.comState()
            robot.stack.setComRefState(c, c_dot, c_ddot)
            if swing_trajectory is not None:
                T_swing, V_swing, A_swing = swing_trajectory.evaluate(t_step_elapsed)
                robot.updateSwingFootRef(T_swing, V_swing.vector, A_swing.vector)

            # steer the torso toward the desired (leading) path heading
            psi_des = (heading_des - yaw_sup0 + np.pi) % (2 * np.pi) - np.pi
            torso_yaw += TORSO_YAW_LP * (psi_des - torso_yaw)
            cz, sz = np.cos(torso_yaw), np.sin(torso_yaw)
            Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
            robot.stack.setTorsoRefState(pin.SE3(Rz @ R_torso0, p_torso0))

        # --- crowd: scripted cylinder agents moving in front of the camera ---
        crowd.update(simulator.simTime())

        # --- advance the simulation + whole-body control ---
        simulator.step()
        robot.update()

        # publish to ros at ~30 Hz: robot state + depth perception of the crowd
        if t - t_publish > 1. / 30.:
            t_publish = t
            robot.publish()                 # joint states + TF (RViz)
            robot.publish_depth_image()      # depth image (32FC1)
            robot.update_obstacles()         # (x, y, r, vx, vy) per cylinder
            robot.publish_obstacle_markers(crowd.ground_truth())
            rclpy.spin_once(robot._node, timeout_sec=0.0)

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

    # Crowd MUST be created before the robot (spawning cylinder bodies forces a
    # MuJoCo recompile that would otherwise wipe the robot's pose).
    crowd = CrowdEnv(simulator)

    created = _create_robot(simulator)
    if created is None:
        return
    tw, robot = created

    # tell the perception pipeline which geoms are the crowd cylinders
    robot.set_obstacle_geoms(crowd.geom_names)

    # footstep plan + zmp / yaw reference
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

    print("TALOS läuft und streamt Tiefenbilder auf /camera/depth/image_rect ...")

    # run the simulation, then plot
    L = _run_loop(simulator, crowd, robot, mpc, interpolator, plan, ZMP_ref, YAW_ref,
                  com_start, com_target, N_pre, N_sim)
    _plot_results(L)

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
