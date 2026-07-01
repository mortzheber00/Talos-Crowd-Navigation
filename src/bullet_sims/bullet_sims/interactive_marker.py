#!/usr/bin/env python3

"""
Interactive Marker with 6 DOF control (3 translation axes + 3 rotation axes).
This script creates an interactive marker at the robot's hand position for Cartesian
control target tracking. The marker can be moved and rotated in RVIZ to set
desired end-effector poses.
"""

import sys
import rclpy
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import TransformListener, Buffer
from interactive_markers import InteractiveMarkerServer
from visualization_msgs.msg import InteractiveMarker
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped

# Globals set in main()
rclpy_node = None
pose_pub = None


def processFeedback(feedback):
    """Process feedback from interactive marker interactions and publish PoseStamped."""
    global rclpy_node, pose_pub
    p = feedback.pose.position
    o = feedback.pose.orientation
    print(f'{feedback.marker_name} feedback:')
    print(f'  Position: x={p.x:.4f}, y={p.y:.4f}, z={p.z:.4f}')
    print(f'  Orientation: x={o.x:.4f}, y={o.y:.4f}, z={o.z:.4f}, w={o.w:.4f}')
    print()

    # Publish PoseStamped to goal topic if publisher is available
    if pose_pub is not None and rclpy_node is not None:
        ps = PoseStamped()
        ps.header.stamp = rclpy_node.get_clock().now().to_msg()
        # Use the marker's frame (marker is in base_link)
        ps.header.frame_id = feedback.header.frame_id if feedback.header.frame_id else 'base_link'
        ps.pose = feedback.pose
        pose_pub.publish(ps)


def get_end_effector_pose(tf_buffer, target_frame='arm_right_7_link', source_frame='base_link', timeout=2.0):
    """
    Get the end-effector pose from TF listener.
    
    Args:
        tf_buffer: TF2 buffer to use for lookups
        target_frame: Frame to lookup (end-effector frame)
        source_frame: Reference frame (base_link)
        timeout: Timeout for TF lookup in seconds
    
    Returns:
        geometry_msgs.Pose: The pose of the target frame relative to source frame
    """
    # Look up the transform and return None if unavailable.
    if not tf_buffer.can_transform(
        source_frame,
        target_frame,
        Time(),
        timeout=Duration(seconds=timeout),
    ):
        return None

    transform = tf_buffer.lookup_transform(
        source_frame,
        target_frame,
        Time(seconds=0),  # Use latest available transform
        timeout=Duration(seconds=timeout)
    )

    # Convert to Pose message
    pose = Pose()
    pose.position.x = transform.transform.translation.x
    pose.position.y = transform.transform.translation.y
    pose.position.z = transform.transform.translation.z
    pose.orientation = transform.transform.rotation

    print(f"Found {target_frame} at position: ({pose.position.x:.4f}, {pose.position.y:.4f}, {pose.position.z:.4f})")
    return pose


def make6DOFMarker(server, name, initial_pose):
    """
    Create a 6 DOF interactive marker with controls for translation and rotation.
    
    Args:
        server: InteractiveMarkerServer instance
        name: Name of the marker
        initial_pose: geometry_msgs.Pose object for initial position and orientation
    """
    # Create the interactive marker
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = 'base_link'
    int_marker.name = name
    int_marker.description = '6-DOF Control\n(Move + Rotate)'
    
    # Set initial pose from parameter
    int_marker.pose = initial_pose

    # Create a cube marker for visualization
    box_marker = Marker()
    box_marker.type = Marker.CUBE
    box_marker.scale.x = 0.15
    box_marker.scale.y = 0.15
    box_marker.scale.z = 0.15
    box_marker.color.r = 0.5
    box_marker.color.g = 0.5
    box_marker.color.b = 0.5
    box_marker.color.a = 1.0

    # Create a non-interactive control which contains the box
    box_control = InteractiveMarkerControl()
    box_control.always_visible = True
    box_control.markers.append(box_marker)
    int_marker.controls.append(box_control)

    # Create controls for translation along X, Y, Z axes
    # X-axis (red arrow)
    move_x_control = InteractiveMarkerControl()
    move_x_control.name = 'move_x'
    move_x_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    move_x_control.orientation.w = 1.0
    move_x_control.orientation.x = 1.0
    move_x_control.orientation.y = 0.0
    move_x_control.orientation.z = 0.0
    int_marker.controls.append(move_x_control)

    # Y-axis (green arrow)
    move_y_control = InteractiveMarkerControl()
    move_y_control.name = 'move_y'
    move_y_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    move_y_control.orientation.w = 1.0
    move_y_control.orientation.x = 0.0
    move_y_control.orientation.y = 1.0
    move_y_control.orientation.z = 0.0
    int_marker.controls.append(move_y_control)

    # Z-axis (blue arrow)
    move_z_control = InteractiveMarkerControl()
    move_z_control.name = 'move_z'
    move_z_control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    move_z_control.orientation.w = 1.0
    move_z_control.orientation.x = 0.0
    move_z_control.orientation.y = 0.0
    move_z_control.orientation.z = 1.0
    int_marker.controls.append(move_z_control)

    # Create controls for rotation around X, Y, Z axes
    # Rotate around X-axis
    rotate_x_control = InteractiveMarkerControl()
    rotate_x_control.name = 'rotate_x'
    rotate_x_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    rotate_x_control.orientation.w = 1.0
    rotate_x_control.orientation.x = 1.0
    rotate_x_control.orientation.y = 0.0
    rotate_x_control.orientation.z = 0.0
    int_marker.controls.append(rotate_x_control)

    # Rotate around Y-axis
    rotate_y_control = InteractiveMarkerControl()
    rotate_y_control.name = 'rotate_y'
    rotate_y_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    rotate_y_control.orientation.w = 1.0
    rotate_y_control.orientation.x = 0.0
    rotate_y_control.orientation.y = 1.0
    rotate_y_control.orientation.z = 0.0
    int_marker.controls.append(rotate_y_control)

    # Rotate around Z-axis
    rotate_z_control = InteractiveMarkerControl()
    rotate_z_control.name = 'rotate_z'
    rotate_z_control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    rotate_z_control.orientation.w = 1.0
    rotate_z_control.orientation.x = 0.0
    rotate_z_control.orientation.y = 0.0
    rotate_z_control.orientation.z = 1.0
    int_marker.controls.append(rotate_z_control)

    # Add the interactive marker to the server
    server.insert(int_marker, feedback_callback=processFeedback)


def main():
    import time
    rclpy.init(args=sys.argv)
    node = rclpy.create_node('interactive_marker_6dof')
    # set globals for callbacks
    global rclpy_node, pose_pub
    rclpy_node = node
    pose_pub = node.create_publisher(PoseStamped, 'goal_pose', 10)

    # Create TF2 listener to get hand position
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    # Wait and retry to get the end-effector pose, spinning the node so TF callbacks run
    hand_pose = None
    max_retries = 60
    retry_count = 0

    print("Waiting for TF transforms to be available...")
    while retry_count < max_retries:
        # Let the node process incoming TF messages
        rclpy.spin_once(node, timeout_sec=0.5)
        hand_pose = get_end_effector_pose(tf_buffer, 'arm_right_7_link', 'base_link', timeout=0.2)
        if hand_pose is not None:
            break
        retry_count += 1
        print(f"Attempt {retry_count}: Waiting for transforms...")

    # If we still don't have a pose, use default
    if hand_pose is None:
        print("Could not retrieve hand TF in time; using default position at base_link origin.")
        hand_pose = Pose()
        hand_pose.position.x = 0.0
        hand_pose.position.y = 0.0
        hand_pose.position.z = 0.0
        hand_pose.orientation.x = 0.0
        hand_pose.orientation.y = 0.0
        hand_pose.orientation.z = 0.0
        hand_pose.orientation.w = 1.0

    # Create an interactive marker server
    server = InteractiveMarkerServer(node, 'cartesian_target_marker')

    # Create a 6 DOF interactive marker at hand position
    make6DOFMarker(server, 'target_pose', hand_pose)

    # Commit changes and send to all clients
    server.applyChanges()

    print("Interactive marker 'target_pose' created at robot hand position.")
    print("Open RVIZ and look for the 'cartesian_target_marker' in the Displays panel.")
    print("You should see a cube with arrow controls for translation and rotation.")

    rclpy.spin(node)
    server.shutdown()


if __name__ == '__main__':
    main()
