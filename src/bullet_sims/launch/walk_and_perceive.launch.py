"""Launch file for depth perception pipeline + visualization.

Starts:
  1. The stand_and_perceive simulation (publishes depth + joint states)
  2. depth_image_proc to convert depth → point cloud
  3. RViz with TALOS no-hands visualization
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    # High-res front camera + point cloud: off by default (RViz visualization
    # only, not used by obstacle perception -- see walking.py's FOV_CAMERA).
    fov_camera = LaunchConfiguration('fov_camera')
    declare_fov_camera = DeclareLaunchArgument('fov_camera', default_value='false')

    # Depth image processing (converts depth image to XYZ point cloud). Only
    # useful while the walking node's front camera is enabled.
    depth_image_proc_launch = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('bullet_sims'),
            'launch',
            'depth_image_proc.launch.py',
        ]),
        condition=IfCondition(fov_camera),
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
    walking_node = Node(
        package='bullet_sims',
        executable='walking',
        output='screen',
        name='walking',
        parameters=[{'fov_camera': fov_camera}],
    )

    return LaunchDescription([
        declare_fov_camera,
        walking_node,
        depth_image_proc_launch,
        ros_visuals_launch,
    ])
