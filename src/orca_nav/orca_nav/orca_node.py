"""orca_node.py
ROS 2 node around ``RobotORCA``: the robot side of the crowd-navigation loop,
communicating over topics instead of direct variable passing.

Subscribes:
  * ``odom`` (nav_msgs/Odometry) -- the robot's ground-plane state. NOTE: this
    project's odometry twist is expressed in the WORLD frame (see
    ``Talos.publish``), which is exactly the frame ORCA plans in.
  * ``<obstacle_topic>`` (crowd_nav_interfaces/ObstacleArray) -- the dynamic
    obstacles to avoid. Point it at ``obstacles/ground_truth`` (simulation
    truth) or ``obstacles/perceived`` (onboard depth-camera estimates); when
    run standalone the choice is a parameter / remap, not an if-branch.
  * ``goal_pose`` (geometry_msgs/PoseStamped) -- navigation goal, e.g. from
    RViz's "2D Goal Pose" tool.

Publishes (on every odom message, so the command rate follows the odom rate):
  * ``cmd_vel`` (geometry_msgs/TwistStamped) -- the collision-free velocity
    command; the reactive footstep planner turns this into footsteps.
  * ``cmd_vel_pref`` (geometry_msgs/TwistStamped) -- the goal-direction
    preferred velocity before ORCA (debug/introspection only).

Can be constructed in-process (the walking loop adds it to an executor) or run
standalone::

    ros2 run orca_nav orca_planner --ros-args -p goal_x:=4.0 -p goal_y:=0.0 \
        -p obstacle_topic:=obstacles/perceived
"""

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry

from crowd_nav_interfaces.msg import ObstacleArray
from orca_nav.robot_orca import RobotORCA


class ORCAVelocityNode(Node):
    """Publishes the ORCA velocity command computed from topic inputs."""

    def __init__(self, goal=(0.0, 0.0), obstacle_topic='obstacles/ground_truth',
                 radius=0.35, max_speed=0.4, desired_speed=0.25,
                 safety_margin=0.2, node_name='orca_planner', **orca_kwargs):
        """
        Args:
            goal (array-like, 2): initial navigation goal in the world plane [m]
                (updated at runtime via the ``goal_pose`` topic).
            obstacle_topic (str): ObstacleArray topic to avoid.
            radius, max_speed, desired_speed, safety_margin: see ``RobotORCA``.
            node_name (str): ROS node name.
            **orca_kwargs: forwarded to ``RobotORCA`` (goal_tol, brake_radius,
                responsibility, params).

        All constructor values only act as defaults: the matching ROS
        parameters (``goal_x``, ``goal_y``, ``obstacle_topic``, ``radius``,
        ``max_speed``, ``desired_speed``, ``safety_margin``) override them at
        launch, so the standalone node is fully configurable via --ros-args.
        """
        super().__init__(node_name)

        self.declare_parameter('goal_x', float(goal[0]))
        self.declare_parameter('goal_y', float(goal[1]))
        self.declare_parameter('obstacle_topic', obstacle_topic)
        self.declare_parameter('radius', float(radius))
        self.declare_parameter('max_speed', float(max_speed))
        self.declare_parameter('desired_speed', float(desired_speed))
        self.declare_parameter('safety_margin', float(safety_margin))

        goal = (self.get_parameter('goal_x').value,
                self.get_parameter('goal_y').value)
        obstacle_topic = self.get_parameter('obstacle_topic').value

        self.ctrl = RobotORCA(
            goal,
            radius=self.get_parameter('radius').value,
            max_speed=self.get_parameter('max_speed').value,
            desired_speed=self.get_parameter('desired_speed').value,
            safety_margin=self.get_parameter('safety_margin').value,
            **orca_kwargs)

        # latest obstacle snapshot as (pos(2), vel(2), radius) tuples -- the
        # exact neighbour format RobotORCA.compute_velocity expects
        self._obstacles = []

        self.create_subscription(Odometry, 'odom', self._odom_cb, 10)
        self.create_subscription(ObstacleArray, obstacle_topic,
                                 self._obstacles_cb, 10)
        self.create_subscription(PoseStamped, 'goal_pose', self._goal_cb, 10)

        self._cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 10)
        self._pref_pub = self.create_publisher(TwistStamped, 'cmd_vel_pref', 10)

        self.get_logger().info(
            f"ORCA planner: goal=({goal[0]:.2f}, {goal[1]:.2f}), "
            f"obstacles from '{obstacle_topic}'")

    # ------------------------------------------------------------------ #
    # subscriptions
    # ------------------------------------------------------------------ #

    def _obstacles_cb(self, msg):
        # stale tracks (occluded, still holding the last estimate) are kept:
        # an unseen pedestrian still occupies space
        self._obstacles = [
            (np.array([o.x, o.y]), np.array([o.vx, o.vy]), float(o.radius))
            for o in msg.obstacles]

    def _goal_cb(self, msg):
        goal = (msg.pose.position.x, msg.pose.position.y)
        self.ctrl.set_goal(goal)
        self.get_logger().info(f"new goal: ({goal[0]:.2f}, {goal[1]:.2f})")

    def _odom_cb(self, msg):
        position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        # world-frame twist by this project's convention (see module docstring)
        velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])

        v_pref = self.ctrl.preferred_velocity(position)
        v_cmd = self.ctrl.compute_velocity(position, velocity, self._obstacles)

        self._publish_twist(self._cmd_pub, msg.header.stamp, v_cmd)
        self._publish_twist(self._pref_pub, msg.header.stamp, v_pref)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _publish_twist(self, pub, stamp, v):
        msg = TwistStamped()
        msg.header.stamp = stamp        # inherit the odom stamp it answers to
        msg.header.frame_id = 'world'
        msg.twist.linear.x = float(v[0])
        msg.twist.linear.y = float(v[1])
        pub.publish(msg)


def main(args=None):
    """Standalone entry point (``ros2 run orca_nav orca_planner``)."""
    rclpy.init(args=args)
    node = ORCAVelocityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
