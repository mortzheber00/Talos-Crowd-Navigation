import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('ros_visuals')
    rviz_config = os.path.join(package_share, 'config', 'ros_visuals.rviz')

    cage_publisher = Node(
        package='ros_visuals',
        executable='t11',
        name='cage_publisher',
        output='screen',
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    return LaunchDescription([
        cage_publisher,
        rviz,
    ])