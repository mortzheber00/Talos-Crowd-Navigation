"""Launch file for depth perception pipeline + visualization.

Starts:
  1. The stand_and_perceive simulation (publishes depth + joint states)
  2. depth_image_proc to convert depth → point cloud
  3. RViz with TALOS no-hands visualization
"""

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

    # RViz visualization
    ros_visuals_launch = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('ros_visuals'),
            'launch',
            'talos_rviz_no_hands.launch.py',
        ])
    )

    # The stand_and_perceive simulation script
    stand_and_perceive_node = Node(
        package='bullet_sims',
        executable='walk_and_perceive',
        output='screen',
        name='walk_and_perceive',
    )

    return LaunchDescription([
        stand_and_perceive_node,
        depth_image_proc_launch,
        ros_visuals_launch,
    ])
