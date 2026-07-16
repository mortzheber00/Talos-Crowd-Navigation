"""
talos standing + depth perception

A stripped-down sibling of walking.py: it re-uses the exact same simulator init
and robot loading, but drops the whole walking pipeline (LIP-MPC, foot-step
planner, swing-foot trajectories, contact state machine, matplotlib plots).

The whole-body controller simply holds the double-support home posture, so the
robot drives into its stance and just stands there.  A crowd of scripted
cylinder agents (CrowdEnv) moves in front of the head camera; the onboard depth
image is streamed to ROS 2 on /camera/depth/image_rect, alongside the usual
joint states / TF so the robot stays visible in RViz, and per-cylinder 2D
obstacle estimates (x, y, r, vx, vy) are extracted from aligned depth +
segmentation renders for ORCA.
"""

import rclpy

# simulator
from simulator.mujoco_wrapper import MujocoWrapper as PybulletWrapper

# robot configs
import bullet_sims.talos_conf as conf

from bullet_sims.talos import Talos
from bullet_sims.crowd_env import CrowdEnv


def _setup_simulator():
    """Init ROS, set the sim-rate-dependent timing on ``conf``, and build the
    MuJoCo simulator (identical to walking.py's setup so the WBC stays tuned)."""
    if not rclpy.ok():
        rclpy.init()

    frequency = 300
    conf.dt = 1.0 / frequency
    conf.no_sim_per_mpc = int(round(conf.dt_mpc / conf.dt))
    conf.no_sim_per_step = int(round(conf.step_dur / conf.dt))

    return PybulletWrapper(sim_rate=frequency, num_solver_iterations=50,
                           num_sub_steps=8, render=True)


def main():
    # NOTE: ros2 run skips the __main__ block, so all setup happens here.
    simulator = _setup_simulator()

    # Crowd MUST be created before the robot: spawning the cylinder bodies
    # changes the MuJoCo model, which forces a recompile that resets the
    # simulation state; doing it after the robot would wipe the robot's pose.
    crowd = CrowdEnv(simulator)

    # Robot: the new Talos.__init__ builds the TSID wrapper (self.stack), spawns
    # the MuJoCo robot (self.robot) in its home posture, and attaches the onboard
    # depth camera + ROS publishers.  No walking controller is created here.
    robot = Talos(simulator)
    robot.set_obstacle_geoms(crowd.geom_names)

    # Hold the spawn CoM: the WBC keeps the double-support home posture, so the
    # robot settles into its stance and simply stands still while perceiving.
    robot.stack.setComRefState(robot.stack.comState().value().copy())

    print("TALOS steht. Streame Tiefenbilder auf /camera/depth/image_rect ...")

    t_publish = 0.0
    try:
        while simulator.isRunning() and rclpy.ok():
            t = simulator.simTime()

            # move the crowd every physics step so the motion is smooth
            crowd.update(t)

            # advance physics + whole-body control (holds the standing posture)
            simulator.step()
            robot.update()

            # publish at ~30 Hz
            if t - t_publish > 1.0 / 30.0:
                t_publish = t
                robot.publish()              # joint states + TF (RViz)
                robot.publish_depth_image()  # depth image (32FC1)
                robot.update_obstacles()     # (x, y, r, vx, vy) per cylinder
                robot.publish_obstacle_markers(crowd.ground_truth())
                rclpy.spin_once(robot._node, timeout_sec=0.0)

    finally:
        print("Schließe Simulator und fahre ROS 2 Node herunter...")
        simulator.disconnect()
        if rclpy.ok():
            robot._node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
