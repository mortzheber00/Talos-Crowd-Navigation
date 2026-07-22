"""Launch file for depth perception pipeline + visualization.

Starts:
  1. The perception_bias_demo simulation (static cylinder, depth + joint states,
     and the front-surface-bias Marker topics)
  2. depth_image_proc to convert depth → point cloud
  3. RViz, pre-configured with a Marker display per bias-demo topic (ground
     truth, raw/corrected centroid, surface points, correction vector) so each
     can be toggled independently for the presentation -- see
     perception_bias_demo.rviz in ros_visuals/config.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # Depth image processing (converts depth image to XYZ point cloud)
    depth_image_proc_launch = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('bullet_sims'),
            'launch',
            'depth_image_proc.launch.py',
        ])
    )

    # The stand_and_perceive simulation script
    perception_bias_demo = Node(
        package='bullet_sims',
        executable='perception_bias_demo',
        output='screen',
        name='perception_bias_demo',
    )

    # robot_state_publisher + RViz, same pattern as
    # ros_visuals/talos_rviz_no_hands.launch.py but with the bias-demo Marker
    # displays pre-added (talos_no_markers.rviz has no display for those
    # topics, so they'd silently not show up even though they publish fine).
    pkg_share_ros_visuals = get_package_share_directory('ros_visuals')
    pkg_share_talos = get_package_share_directory('talos_description')
    urdf_file = os.path.join(pkg_share_talos, 'robots', 'talos_reduced_no_hands.urdf')
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()
    rviz_config = os.path.join(pkg_share_ros_visuals, 'config', 'perception_bias_demo.rviz')

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
        arguments=['-d', rviz_config],
    )

    return LaunchDescription([
        perception_bias_demo,
        depth_image_proc_launch,
        rsp_node,
        rviz_node,
    ])
