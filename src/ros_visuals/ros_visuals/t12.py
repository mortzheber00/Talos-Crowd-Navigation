import rclpy
from rclpy.node import Node
import pinocchio as pin
import numpy as np
from tf2_ros import TransformBroadcaster 
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point

class CagePublisher(Node):
    def __init__(self):
        super().__init__('cage_publisher')
        self.tf_broadcaster = TransformBroadcaster(self)  

        # Start translation and rotaion of the cage
        self.R_center = pin.utils.rotate('z', np.pi/2) @ pin.utils.rotate('x', np.pi/4) @ pin.utils.rotate('y', np.pi/2)
        self.p_center = np.array([1.0, 0.0, 1.0])

        # Rotationsmatrizen für 90 Grad um die weltlichen Koordinatenachsen
        A_x = pin.utils.rotate('x', np.pi/2)  # 90° um x-Achse
        A_y = pin.utils.rotate('y', np.pi/2)  # 90° um y-Achse
        A_z = pin.utils.rotate('z', np.pi/2)  # 90° um z-Achse

        # Array that describes the cage
        self.cage_elements = [
            pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.0])),    # origin
            pin.SE3(np.eye(3), np.array([-0.5, -0.4, -0.4])), # Ecke 1
            pin.SE3(A_x @ A_x @ A_x @ A_z @ A_z @ np.eye(3), np.array([0.5, -0.4, -0.4])), # Ecke 2
            pin.SE3(A_x @ A_z @ np.eye(3), np.array([0.5, 0.2, -0.4])), # Ecke 3
            pin.SE3(A_x @ np.eye(3), np.array([-0.5, 0.2, -0.4])), # Ecke 4
            pin.SE3(A_x @ A_x @ A_x @ np.eye(3), np.array([-0.5, -0.4, 0.4])), # Ecke 5
            pin.SE3(A_z @ A_y @ np.eye(3), np.array([0.5, -0.4, 0.4])), # Ecke 6
            pin.SE3(A_x @ A_z @ A_z @ A_z @ np.eye(3), np.array([-0.5, 0.2, 0.4])), # Ecke 7
            pin.SE3(A_y @ A_x @ A_z @ A_z @ A_z @ np.eye(3), np.array([0.5, 0.2, 0.4])), # Ecke 8
        ]

        # Variables for movement of the cage
        self.dt = 0.05
        self.I = np.eye(3) 
        self.omega = np.array([0, 0, 0.05])
        self.v = np.array([0.02, 0.0, 0.0]) # Bewegt sich mit 0.02 m/s in x-Richtung
        
        self.timer = self.create_timer(self.dt, self.update_cage)

        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        self.marker_array_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.twist_pub_corner = self.create_publisher(TwistStamped, 'twist_corner', 10)
        self.twist_pub_world_lecture = self.create_publisher(TwistStamped, 'twist_world_lecture', 10)
        self.twist_pub_world_pin = self.create_publisher(TwistStamped, 'twist_world_pin', 10)
        self.twist_pub_world = self.create_publisher(TwistStamped, 'twist_world', 10)
        self.twist_pub_corner2_lecture = self.create_publisher(TwistStamped, 'twist_corner2_lecture', 10)
        self.twist_pub_corner2_pin = self.create_publisher(TwistStamped, 'twist_corner2_pin', 10)
        self._marker_batch = []
        self._cycle_stamp = None

    @staticmethod
    def _skew(v):
        return np.array([
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ])

    def transform_twist_lecture(self, transform_ab, twist_b):
        # Motion order in Pinocchio is [linear; angular].
        R = transform_ab.rotation
        p = transform_ab.translation
        v_b = twist_b.linear
        w_b = twist_b.angular

        w_a = R @ w_b
        v_a = R @ v_b + self._skew(p) @ w_a
        return pin.Motion(np.concatenate([v_a, w_a]))

    def transform_twist_action(self, transform_ab, twist_b):
        action_matrix = transform_ab.toActionMatrix()
        return pin.Motion(action_matrix @ twist_b.vector)

    def publish_twist_stamped(self, publisher, frame_id, twist):
        msg = TwistStamped()
        msg.header.stamp = self._cycle_stamp
        msg.header.frame_id = frame_id
        msg.twist.linear.x = float(twist.linear[0])
        msg.twist.linear.y = float(twist.linear[1])
        msg.twist.linear.z = float(twist.linear[2])
        msg.twist.angular.x = float(twist.angular[0])
        msg.twist.angular.y = float(twist.angular[1])
        msg.twist.angular.z = float(twist.angular[2])
        publisher.publish(msg)

    def publish_twist_markers(self, frame_id, twist, base_marker_id, linear_color, angular_color, scale=0.35):
        self.publish_vector_marker(
            start=np.zeros(3),
            end=scale * twist.linear,
            frame_id=frame_id,
            marker_id=base_marker_id,
            color=linear_color,
        )
        self.publish_vector_marker(
            start=np.zeros(3),
            end=scale * twist.angular,
            frame_id=frame_id,
            marker_id=base_marker_id + 1,
            color=angular_color,
        )

    # exp6 Version:
    def update_cage(self):
        self._marker_batch = []
        self._cycle_stamp = self.get_clock().now().to_msg()
        
        # Transform v into local coordinate frame, otherwise we see a circular trajectory
        v_local = self.R_center.T @ self.v
        
        nu = np.concatenate([v_local, self.omega]) 
        
        relative_transform = pin.exp6(nu * self.dt)
        
        current_pose = pin.SE3(self.R_center, self.p_center)
        new_pose = current_pose * relative_transform 
        
        self.R_center = new_pose.rotation
        self.p_center = new_pose.translation
        self.publish_cage()

        # 1) Define a non-zero local twist in corner_8 and publish it.
        c_V = pin.Motion(np.array([0.5, 0.0, 0.0, 1.0, 0.0, 0.0]))
        self.publish_twist_stamped(self.twist_pub_corner, 'corner_8', c_V)
        self.publish_twist_markers('corner_8', c_V, 10, (0.95, 0.2, 0.2), (0.5, 0.55, 0.2))

        T_world_corner8 = new_pose * self.cage_elements[8]

        # 2) Transform corner_8 twist to world using lecture implementation and action matrix.
        w_V_lecture = self.transform_twist_lecture(T_world_corner8, c_V)
        w_V_pin = self.transform_twist_action(T_world_corner8, c_V)
        self.publish_twist_stamped(self.twist_pub_world_lecture, 'world', w_V_lecture)
        self.publish_twist_stamped(self.twist_pub_world_pin, 'world', w_V_pin)
        # Same colors as corner_8 to show they belong together
        self.publish_twist_markers('world', w_V_lecture, 20, (0.95, 0.2, 0.2), (0.5, 0.55, 0.2))
        self.publish_twist_markers('world', w_V_pin, 30, (0.0, 0.0, 0.8), (0.0, 0.0, 0.2))

        # 3) Define a world twist and transform it into corner_2.
        w_V = pin.Motion(np.array([0.25, -0.15, 0.4, 0.4, 0.2, -0.1]))
        self.publish_twist_stamped(self.twist_pub_world, 'world', w_V)
        self.publish_twist_markers('world', w_V, 40, (0.0, 0.95, 0.2), (0.0, 0.95, 0.2))

        T_world_corner2 = new_pose * self.cage_elements[2]
        T_corner2_world = T_world_corner2.inverse()
        c2_V_lecture = self.transform_twist_lecture(T_corner2_world, w_V)
        c2_V_pin = self.transform_twist_action(T_corner2_world, w_V)
        self.publish_twist_stamped(self.twist_pub_corner2_lecture, 'corner_2', c2_V_lecture)
        self.publish_twist_stamped(self.twist_pub_corner2_pin, 'corner_2', c2_V_pin)
        self.publish_twist_markers('corner_2', c2_V_lecture, 50, (0.2, 0.9, 0.9), (0.3, 0.6, 1.0))
        self.publish_twist_markers('corner_2', c2_V_pin, 60, (0.05, 0.6, 0.6), (0.9, 0.2, 0.6))
        self.flush_marker_batch()


    def publish_cage(self):
        transforms = []
        now = self._cycle_stamp

        # WORLD to CENTER Transformation
        t_world_center = TransformStamped()
        t_world_center.header.stamp = now
        t_world_center.header.frame_id = 'world'
        t_world_center.child_frame_id = 'center_frame'
        
        t_world_center.transform.translation.x = self.p_center[0]
        t_world_center.transform.translation.y = self.p_center[1]
        t_world_center.transform.translation.z = self.p_center[2]
        
        quat_center = pin.Quaternion(self.R_center)
        t_world_center.transform.rotation.x = quat_center.x
        t_world_center.transform.rotation.y = quat_center.y
        t_world_center.transform.rotation.z = quat_center.z
        t_world_center.transform.rotation.w = quat_center.w
        
        transforms.append(t_world_center)

        # Transformations CENTER to CORNERS
        for i, se3 in enumerate(self.cage_elements):
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'center_frame'
            t.child_frame_id = f'corner_{i}'
            
            t.transform.translation.x = se3.translation[0]
            t.transform.translation.y = se3.translation[1]
            t.transform.translation.z = se3.translation[2]
            
            quat = pin.Quaternion(se3.rotation)
            t.transform.rotation.x = quat.x
            t.transform.rotation.y = quat.y
            t.transform.rotation.z = quat.z
            t.transform.rotation.w = quat.w
            
            transforms.append(t)
        
        self.tf_broadcaster.sendTransform(transforms)
        self.get_logger().info('published cage!')         

        
    def flush_marker_batch(self):
        marker_array = MarkerArray()
        marker_array.markers = self._marker_batch
        self.marker_array_pub.publish(marker_array)

    def publish_vector_marker(self, end, frame_id, marker_id, start=None, color=(1.0, 0.0, 0.0)):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self._cycle_stamp
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = marker_id
        
        # Wenn kein Startpunkt übergeben wurde, starten wir im lokalen Ursprung [0, 0, 0]
        if start is None:
            start = np.zeros(3)
            
        marker.points = [
            Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
            Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))
        ]

        # Shaft diameter, head diameter, head length
        marker.scale.x, marker.scale.y, marker.scale.z = 0.02, 0.04, 0.06
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0

        self._marker_batch.append(marker)

def main():
    rclpy.init()
    node = CagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()