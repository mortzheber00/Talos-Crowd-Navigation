import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share_ros_visuals = get_package_share_directory('ros_visuals')
    pkg_share_talos = get_package_share_directory('talos_description')

    # URDF file used by Pinocchio and robot_state_publisher
    urdf_file = os.path.join(pkg_share_talos, 'robots', 'talos_reduced_no_hands.urdf')
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()

    # RViz config without the InteractiveMarkers display
    rviz_config = os.path.join(pkg_share_ros_visuals, 'config', 'talos_no_markers.rviz')

    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}]
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )

    return LaunchDescription([
        rsp_node,
        rviz_node,
    ])