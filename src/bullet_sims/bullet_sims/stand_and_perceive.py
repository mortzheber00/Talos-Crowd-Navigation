"""
talos standing + depth perception

A stripped-down sibling of walking.py: it re-uses the exact same simulator
init and robot loading, but drops the whole walking pipeline (TSID controller
loop, LIP-MPC, foot-step planner, contact state machine, force filtering,
matplotlib plots).

The robot simply stands in its home posture (held by the joint position
servos).  A crowd of 5 scripted cylinder agents (CrowdEnv) moves in front of
the head camera; the onboard depth image is streamed to ROS 2 on
/camera/depth/image_raw, alongside the usual joint states / TF so the robot
stays visible in RViz, and per-cylinder 2D obstacle estimates (x, y, r, vx,
vy) are extracted from aligned depth + segmentation renders for ORCA.
"""

import numpy as np
import mujoco

# simulator
from simulator.mujoco_wrapper import MujocoWrapper as PybulletWrapper

# whole-body controller (only used to obtain the pinocchio model)
from bullet_sims.tsid_wrapper import TSIDWrapper

# robot configs
import bullet_sims.talos_conf as conf

from bullet_sims.talos import Talos
from bullet_sims.crowd_env import CrowdEnv

# ROS
import rclpy


def spawn_static_box(simulator, position=(1.7, 0.0, 1.2),
                     half_size=0.15, rgba=(0.9, 0.1, 0.1, 1.0)):
    """Spawn a static (worldbody-fixed) box into the MuJoCo scene.

    The box has no joint, so it is immovable, and contype/conaffinity 0 makes it
    collision-inert (it never disturbs the robot) while still being rendered by
    the depth camera.  Default pose: ~1.7 m in front of the robot at head height.
    """
    body = simulator._spec.worldbody.add_body(
        name='perception_target', pos=list(position))
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[half_size, half_size, half_size],
        rgba=list(rgba),
        contype=0, conaffinity=0,
    )
    # topology changed -> recompile (current state is preserved)
    simulator._dirty = True
    simulator._ensure_compiled()


def main():
    tsid_conf = conf

    # Home posture = the natural standing pose
    # Roboter startet flach auf dem Boden (Gelenke auf 0)
    q_home = tsid_conf.q_home.copy()
    q_init = np.zeros_like(q_home)      
    q_init[:3] = q_home[:3]
    q_init[3:7] = np.array([0.0, 0.0, 0.0, 1.0])

    # TSID wrapper only to get the pinocchio model; no controller runs here.
    try:
        tw = TSIDWrapper(tsid_conf)
    except Exception as e:
        print('Failed to create TSIDWrapper:', e)
        return
    pin_model = tw.robot.model()

    # Simulator (1 kHz, same as walking.py: tsid_conf.dt = 0.001)
    sim_freq = int(1.0 / tsid_conf.dt)
    simulator = PybulletWrapper(sim_rate=sim_freq)

    # Robot: floating base, held upright in the home posture by the position
    # servos.  Talos.__init__ attaches the depth camera and creates depth_pub.
    robot = Talos(
        simulator,
        tsid_conf.urdf,
        pin_model,
        q=q_init,
        basePosition=q_init[:3],
        baseQuaternion=q_init[3:7],
        verbose=True,
        useFixedBase=False,
    )
    robot.stack = tw

    # Hold the home posture (position control is enabled by Robot.__init__).
    # robot.setActuatedJointPositions(q_home)

    # Spawn the crowd of 5 scripted cylinder agents in front of the camera.
    crowd = CrowdEnv(simulator)
    robot.set_obstacle_geoms(crowd.geom_names)
    # spawning recompiled the model -> re-command the home posture.
    robot.setActuatedJointPositions(q_home)

    print("TALOS steht. Streame Tiefenbilder auf /camera/depth/image_raw ...")

    t_publish = 0.0
    try:
        while simulator.isRunning() and rclpy.ok():

            t = simulator.simTime()

            # # keep commanding the home posture so the robot stays standing
            # robot.setActuatedJointPositions(q_home)

            robot.update()

            # get pinocchio-style states
            q = robot.q()
            v = robot.v()

            # update TSID and solve for torques/accelerations
            try:
                tau, dv = tw.update(q, v, t, do_solve=True)
                # store TSID base state for publishing
                robot.tsid_base_T, robot.tsid_base_v = tw.baseState()
            except Exception as e:
                print('TSID update error:', e)
                tau = np.zeros(robot.numActuatedJoints())
                dv = np.zeros(robot._v.shape)

            # command to the robot (torques)
            robot.setActuatedJointTorques(tau)

            # publish at ~30 Hz
            t = simulator.simTime()
            if t - t_publish > 1.0 / 30.0:
                t_publish = t
                robot.publish()              # joint states + TF (RViz)
                robot.publish_depth_image()  # depth image (32FC1)
                robot.update_obstacles()     # (x, y, r, vx, vy) per cylinder
                robot.publish_obstacle_markers(crowd.ground_truth())
                rclpy.spin_once(robot.ros_node, timeout_sec=0.0)

            # move the crowd every physics step so the motion is smooth
            crowd.update(simulator.simTime())
            simulator.step()


    finally:
        print("Schließe Simulator und fahre ROS 2 Node herunter...")
        simulator.disconnect()
        if rclpy.ok():
            robot.ros_node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    rclpy.init()
    main()
