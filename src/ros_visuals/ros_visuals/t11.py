import rclpy
from rclpy.node import Node
import pinocchio as pin
import numpy as np
from tf2_ros import TransformBroadcaster 
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import Marker
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
     
    # # lecture method
    # def update_cage(self):
    #     # Linear Integration 
    #     self.p_center = self.p_center + self.v * self.dt

    #     # angular Integration
    #     w = self.omega
    #     S = np.array([
    #         [0, -w[2], w[1]],
    #         [w[2], 0, -w[0]],
    #         [-w[1], w[0], 0]
    #     ])
        
    #     # R_neu = R_alt * (I + dt/2 * S) * (I - dt/2 * S)^-1
    #     A = (self.I + (self.dt/2) * S) @ np.linalg.inv(self.I - (self.dt/2) * S)
    #     self.R_center = self.R_center @ A
        
    #     self.publish_cage()

    # exp6 Version:
    def update_cage(self):
        
        # Transform v into local coordinate frame, otherwise we see a circular trajectory
        v_local = self.R_center.T @ self.v
        
        nu = np.concatenate([v_local, self.omega]) 
        
        relative_transform = pin.exp6(nu * self.dt)
        
        current_pose = pin.SE3(self.R_center, self.p_center)
        new_pose = current_pose * relative_transform 
        
        self.R_center = new_pose.rotation
        self.p_center = new_pose.translation
        self.publish_cage()

        # send vector marker
        p_vector = np.array([0.5, 0.0, 0.0]) # Vector p
        self.publish_vector_marker(end=p_vector, frame_id = 'corner_8', marker_id = 0, color = (0.7, 0.0, 0.4))

        # transform matrices from corner to world:
        T_center_corner = self.cage_elements[8] 
        T_world_corner = new_pose * T_center_corner

        # Transformiert den Punkt in die Welt
        w_p_tip = T_world_corner.act(p_vector) 
        w_p_start = T_world_corner.translation

        self.publish_vector_marker(end=w_p_tip, frame_id = 'world', start=w_p_start, marker_id = 1, color = (0.2, 0.4, 0.4))



    def publish_cage(self):
        transforms = []
        now = self.get_clock().now().to_msg()

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

        
    def publish_vector_marker(self, end, frame_id, marker_id, start=None, color=(1.0, 0.0, 0.0)):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        # marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.id = marker_id
        
        # Wenn kein Startpunkt übergeben wurde, starten wir im lokalen Ursprung [0, 0, 0]
        if start is None:
            start = np.zeros(3)
            
        marker.points = [
            Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
            Point(x=float(end[0]), y=float(end[1]), z=float(end[2]))
        ]

        # --- ÄNDERUNG HIER für Cube ---
        marker.type = Marker.CUBE 
        marker.action = Marker.ADD
        marker.id = marker_id
        
        # Ein Cube braucht eine Pose (Position + Orientation)
        # Wir setzen den Würfel einfach an das Ende des Vektors (die Spitze)
        marker.pose.position.x = float(end[0])
        marker.pose.position.y = float(end[1])
        marker.pose.position.z = float(end[2])
        
        # Die Orientierung lassen wir auf Default (Identity)
        marker.pose.orientation.w = 1.0 
        
        # Skalierung (ein kleiner Würfel)
        marker.scale.x, marker.scale.y, marker.scale.z = 1, 1, 1

        # Scale & Color
        marker.scale.x, marker.scale.y, marker.scale.z = 0.02, 0.04, 0.06
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        
        self.marker_pub.publish(marker)

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