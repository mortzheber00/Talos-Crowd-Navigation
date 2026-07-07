from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch depth_image_proc's point_cloud_xyz node for depth -> XYZ cloud.

    Wiring facts that actually matter (verified against depth_image_proc /
    image_transport 5.1.7 on Jazzy):

      * The node's raw input topics are ``image_rect`` and ``camera_info``,
        the output is ``points``.
      * It subscribes *lazily*: the inputs are only connected once something
        subscribes to ``points`` (that is why the node "wakes up" when you run
        ``ros2 topic hz /camera/depth/points``).
      * Image <-> CameraInfo are paired with an **ExactTime** synchronizer, so
        both headers must carry the byte-identical ``stamp`` (the Talos
        publisher guarantees this by stamping both from one Time value).
      * ``camera_info`` is **derived from the image topic's namespace** by
        image_transport -- remapping ``image_rect`` to ``/camera/depth/image_rect``
        already makes it subscribe ``/camera/depth/camera_info`` automatically.
        The explicit ``camera_info`` remap below is therefore only for clarity;
        the *image* remap is the one that counts, and the CameraInfo topic MUST
        be the sibling ``<image_ns>/camera_info``.

    There is intentionally no ``exact_sync``/slop or subscription-QoS argument:
    point_cloud_xyz exposes neither. Its subscriptions request BEST_EFFORT,
    which a RELIABLE publisher (the Talos default) satisfies -- so no QoS
    override is needed on the input side.
    """
    pointcloud_topic = LaunchConfiguration('pointcloud_topic')
    depth_image_topic = LaunchConfiguration('depth_image_topic')
    depth_info_topic = LaunchConfiguration('depth_info_topic')
    queue_size = LaunchConfiguration('queue_size')

    declared_args = [
        # NOTE: must match Talos.publish_depth_image(). The publisher uses
        # /camera/depth/image_rect, and its CameraInfo sibling.
        DeclareLaunchArgument('depth_image_topic', default_value='/camera/depth/image_rect'),
        DeclareLaunchArgument('depth_info_topic', default_value='/camera/depth/camera_info'),
        DeclareLaunchArgument('pointcloud_topic', default_value='/camera/depth/points'),
        # Bigger queue tolerates timing jitter between the two inputs.
        DeclareLaunchArgument('queue_size', default_value='10'),
    ]

    pointcloud_node = Node(
        package='depth_image_proc',
        executable='point_cloud_xyz_node',
        name='point_cloud_xyz',
        output='screen',
        parameters=[{
            'queue_size': queue_size,
            # Make the /points output RELIABLE so RViz (reliable by default)
            # reliably receives it. Harmless for best-effort consumers.
            'qos_overrides./points.publisher.reliability': 'reliable',
        }],
        remappings=[
            ('image_rect', depth_image_topic),
            ('camera_info', depth_info_topic),   # redundant with derivation, kept explicit
            ('points', pointcloud_topic),
        ],
    )

    return LaunchDescription(declared_args + [pointcloud_node])
