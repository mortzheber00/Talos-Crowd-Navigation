import numpy as np
from numpy import nan
from numpy.linalg import norm as norm
import matplotlib.pyplot as plt

# pinocchio
import pinocchio as pin

# simulator
import pybullet as pb
from simulator.pybullet_wrapper import PybulletWrapper
from simulator.robot import Robot

# robot and controller
from bullet_sims.tsid_wrapper import TSIDWrapper
import bullet_sims.config as conf

# ROS
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, TransformStamped

################################################################################
# settings
################################################################################

DO_PLOT = True

################################################################################
# Robot
################################################################################

class Talos(Robot, Node):
    def __init__(self, simulator, urdf, model, q=None, basePosition=None, baseQuaternion=None, verbose=True, useFixedBase=False):
        Node.__init__(self, 'talos_standing_node')

        if basePosition is None:
            basePosition = np.zeros(3)
        if baseQuaternion is None:
            baseQuaternion = np.array([0.0, 0.0, 0.0, 1.0])

        # initialize simulator/robot base
        super().__init__(
            simulator,
            filename=urdf,
            model=model,
            basePosition=basePosition,
            baseQuationerion=baseQuaternion,
            q=q,
            useFixedBase=useFixedBase,
            verbose=verbose,
        )

        # ROS publishers
        self.joint_state_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Store TSID base state for publishing
        self.tsid_base_T = None
        self.tsid_base_v = None

    def update(self):
        super().update()

    def publish(self):
        stamp = self.get_clock().now().to_msg()

        js = JointState()
        js.header.stamp = stamp
        js.name = self.actuatedJointNames()
        js.position = self.actuatedJointPosition().tolist()
        js.velocity = self.actuatedJointVelocity().tolist()
        js.effort = self._tau_cmd.tolist()
        self.joint_state_pub.publish(js)

        # publish base transform from TSID solution
        if self.tsid_base_T is not None:
            world_pos = self.tsid_base_T.translation
            world_orient = pin.Quaternion(self.tsid_base_T.rotation)
        else:
            # fallback to pybullet if TSID not available
            world_pos, world_orient = self.baseWorldPose()
            world_orient = pin.Quaternion(world_orient)

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'world'
        t.child_frame_id = self.baseName()
        t.transform.translation.x = float(world_pos[0])
        t.transform.translation.y = float(world_pos[1])
        t.transform.translation.z = float(world_pos[2])
        t.transform.rotation.x = float(world_orient.x)
        t.transform.rotation.y = float(world_orient.y)
        t.transform.rotation.z = float(world_orient.z)
        t.transform.rotation.w = float(world_orient.w)
        self.tf_broadcaster.sendTransform(t)

################################################################################
# main
################################################################################

def main(): 
    rclpy.init()

    print('[standing] config module file:', conf.__file__)
    print('[standing] loaded q_home shape:', conf.q_home.shape)
    print('[standing] loaded q_home[:10]:', np.round(conf.q_home[:10], 4))
    print('[standing] loaded q_home z:', float(conf.q_home[2]))

    # The robot starts from zero base position and zero joint angles, while
    # TSID still uses q_home as the standing target.
    tsid_conf = conf
    q_home = tsid_conf.q_home.copy()
    q_init = np.zeros_like(q_home)      
    q_init[:3] = q_home[:3]
    q_init[3:7] = np.array([0.0, 0.0, 0.0, 1.0])

    # instantiate TSID wrapper with the exact same initial state
    try:
        tw = TSIDWrapper(tsid_conf)
    except Exception as e:
        print('Failed to create TSIDWrapper:', e)
        return

    # simulator
    simulator = PybulletWrapper(sim_rate=1000)

    # create robot using the pinocchio model from TSID
    pin_model = tw.robot.model()
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

    print('[standing] URDF:', tsid_conf.urdf)
    print('[standing] q_home base xyz:', q_home[:3])
    print('[standing] q_init base xyz:', q_init[:3])
    print('[standing] basePosition passed to Robot:', q_init[:3])
    print('[standing] baseQuaternion passed to Robot:', q_init[3:7])
    print('[standing] robot base name:', robot.baseName())

    t_publish = 0.0
    debug_counter = 0

    try:
        while simulator.isRunning():
            # elapsed time
            t = simulator.simTime()

            # update robot state from simulator
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

            if debug_counter % 100 == 0:
                pb_base_pos, pb_base_quat = robot.baseWorldPose()
                print(f'[standing] t={t:.3f}s')
                print(f'[standing]   PB base pos : {np.round(pb_base_pos, 4)}')
                print(f'[standing]   PB base quat: {np.round(pb_base_quat, 4)}')
                if robot.tsid_base_T is not None:
                    tsid_base_pos = robot.tsid_base_T.translation
                    tsid_base_quat = pin.Quaternion(robot.tsid_base_T.rotation)
                    print(f'[standing]   TSID base pos : {np.round(tsid_base_pos, 4)}')
                    print(f'[standing]   TSID base quat: {np.round([tsid_base_quat.x, tsid_base_quat.y, tsid_base_quat.z, tsid_base_quat.w], 4)}')
                    print(f'[standing]   base pos diff  : {np.round(pb_base_pos - tsid_base_pos, 4)}')
                print(f'[standing]   |tau|={norm(tau):.4f} |dv|={norm(dv):.4f}')

            # command to the robot (torques)
            robot.setActuatedJointTorques(tau)

            # step simulation
            simulator.step()
            simulator.debug()

            # publish to ROS at ~30Hz
            if t - t_publish > 1.0 / 30.0:
                t_publish = t
                rclpy.spin_once(robot, timeout_sec=0.0)
                robot.publish()

            debug_counter += 1
    finally:
        simulator.disconnect()
        robot.destroy_node()
        rclpy.shutdown()
    
if __name__ == '__main__': 
    main()
    
