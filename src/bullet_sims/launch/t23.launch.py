import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share_talos = get_package_share_directory('talos_description')
    ros_visuals_share = get_package_share_directory('ros_visuals')
    rviz_config = os.path.join(ros_visuals_share, 'config', 'talos_t2_base_link.rviz')

    urdf_file = os.path.join(pkg_share_talos, 'robots', 'talos_reduced.urdf')
    with open(urdf_file, 'r') as f:
        robot_description_content = f.read()

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}],
    )

    t23_node = Node(
        package='bullet_sims',
        executable='t23',
        name='t23',
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )

    return LaunchDescription([
        robot_state_publisher_node,
        t23_node,
        TimerAction(period=1.0, actions=[rviz_node]),
    ])
