"""Front-surface-bias demo: one static cylinder, one camera, two centroids.

A minimal sibling of stand_and_perceive.py, stripped down to the single effect
it is meant to explain: a depth camera only ever sees the near half of a
cylindrical obstacle, so the *raw* centroid of the visible points sits biased
forward, roughly ``2r/pi`` in front of the true axis (see the derivation in
Talos._front_arc_estimate, talos.py). This script spawns exactly one
non-moving cylinder in front of TALOS, re-uses that same static method to
compute the bias-corrected centroid, and publishes both plus the raw
segmented points and the ground-truth cylinder as separate RViz Marker
topics -- each individually toggle-able in the RViz Displays panel -- so the
effect can be shown live without any of the multi-camera fusion, EMA
smoothing or radius clamping that the real update_obstacles() pipeline adds
on top for a moving crowd.

Nothing here is added to the Talos class: this script only reads its public
state (sim handle, perception camera list/intrinsics) and calls the existing
static helper.

RViz setup: add one "Marker" display per topic below, then toggle them
independently via their checkboxes for the presentation.
    perception_bias_demo/ground_truth          -- true cylinder (translucent blue)
    perception_bias_demo/centroid_ground_truth -- true center point (small, blue)
    perception_bias_demo/surface_points        -- raw segmented depth points
    perception_bias_demo/centroid_raw          -- naive (biased) centroid, red
    perception_bias_demo/centroid_corrected    -- corrected centroid, green
    perception_bias_demo/correction_vector     -- arrow from raw -> corrected
"""

import rclpy
import numpy as np
import mujoco

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

# simulator
from simulator.mujoco_wrapper import MujocoWrapper as PybulletWrapper

# robot configs
import bullet_sims.talos_conf as conf

from bullet_sims.talos import Talos

# --- the one demo cylinder: edit these to move it around for the talk ---
CYL_XY = (3.0, 0.0)        # [m] world-frame position, robot spawns facing +X
CYL_RADIUS = 0.35          # [m] larger radius -> more visible 2r/pi bias
CYL_HALF_HEIGHT = 0.875    # [m] -> 1.75 m tall, human-scale
CYL_RGBA = (0.85, 0.55, 0.15, 1.0)   # only seen in the MuJoCo viewer, not RViz


def _spawn_static_cylinder(simulator, xy, radius, half_height, rgba,
                            name='bias_demo_cyl'):
    """Add one fixed (no joint, no mocap -- it simply never moves) cylinder
    body to the scene. Must run BEFORE Talos(simulator): spawning a body
    forces a model recompile, which resets the sim state and would wipe the
    robot's pose if it already existed (see CrowdEnv's constructor note)."""
    body = simulator._spec.worldbody.add_body(
        name=name, pos=[xy[0], xy[1], half_height])
    geom_name = f'{name}_geom'
    body.add_geom(
        name=geom_name,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[radius, half_height, 0.0],
        rgba=list(rgba),
        contype=0, conaffinity=0,
    )
    simulator._dirty = True
    simulator._ensure_compiled()
    return geom_name


def _front_camera(robot):
    """The perception camera looking along the robot's +X (its yaw=0 surround
    cam if the 4-camera rig came up, else the single RGB-D fallback)."""
    return robot._percep_cams[0] if robot._percep_cams else 'rgbd_depth_cam'


def _segmented_points(robot, camera, geom_id):
    """One camera's world-frame points belonging to ``geom_id``: the same
    per-camera unprojection update_obstacles() does (talos.py), just for a
    single fixed camera/geom instead of the full multi-camera/multi-track
    loop. Returns (points_xyz (N,3) or None, camera_xy (2,))."""
    depth, seg = robot.sim.get_depth_and_seg_image(invalid_to=np.nan, camera=camera)
    cam_pos, R_wc = robot.sim.get_camera_pose(camera=camera)
    R_w_opt = R_wc @ np.diag([1.0, -1.0, -1.0])   # MuJoCo cam -> ROS optical axes

    K = robot._percep_K[camera]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    valid = np.isfinite(depth)
    geom_pixels = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    mask = (seg[..., 0] == geom_id) & geom_pixels & valid
    if not mask.any():
        return None, cam_pos[:2].copy()

    v_pix, u_pix = np.nonzero(mask)
    Z = depth[mask]
    X = (u_pix - cx) * Z / fx
    Y = (v_pix - cy) * Z / fy
    pts_w = (R_w_opt @ np.vstack((X, Y, Z))).T + cam_pos
    return pts_w, cam_pos[:2].copy()


def _marker(frame_id, stamp, ns, mtype, rgba, scale):
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = ns
    m.id = 0
    m.type = mtype
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    m.scale.x, m.scale.y, m.scale.z = scale
    m.color.r, m.color.g, m.color.b, m.color.a = rgba
    return m


def _publish_ground_truth(pub, stamp):
    m = _marker('world', stamp, 'ground_truth', Marker.CYLINDER,
                (0.15, 0.55, 1.0, 0.5),
                (2.0 * CYL_RADIUS, 2.0 * CYL_RADIUS, 2.0 * CYL_HALF_HEIGHT))
    m.pose.position.x, m.pose.position.y = CYL_XY
    m.pose.position.z = CYL_HALF_HEIGHT
    pub.publish(m)


def _publish_centroid(pub, stamp, ns, xy, rgba, diameter=0.05):
    m = _marker('world', stamp, ns, Marker.SPHERE, rgba,
                (diameter, diameter, diameter))
    m.pose.position.x, m.pose.position.y = float(xy[0]), float(xy[1])
    m.pose.position.z = CYL_HALF_HEIGHT
    pub.publish(m)


def _publish_surface_points(pub, stamp, pts_xyz, max_points=1500):
    m = _marker('world', stamp, 'surface_points', Marker.POINTS,
                (1.0, 0.6, 0.1, 0.8), (0.008, 0.008, 0.0))
    stride = max(1, len(pts_xyz) // max_points)
    for p in pts_xyz[::stride]:
        m.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
    pub.publish(m)


def _publish_correction_vector(pub, stamp, raw_xy, corrected_xy):
    m = _marker('world', stamp, 'correction_vector', Marker.ARROW,
                (1.0, 0.85, 0.0, 0.9), (0.025, 0.05, 0.08))
    z = CYL_HALF_HEIGHT
    m.points = [
        Point(x=float(raw_xy[0]), y=float(raw_xy[1]), z=z),
        Point(x=float(corrected_xy[0]), y=float(corrected_xy[1]), z=z),
    ]
    pub.publish(m)


def _setup_simulator():
    if not rclpy.ok():
        rclpy.init()

    frequency = 300
    conf.dt = 1.0 / frequency
    conf.no_sim_per_mpc = int(round(conf.dt_mpc / conf.dt))
    conf.no_sim_per_step = int(round(conf.step_dur / conf.dt))

    return PybulletWrapper(sim_rate=frequency, num_solver_iterations=50,
                           num_sub_steps=8, render=True,
                           collision_boxes=conf.sole_collision_boxes)


def main():
    simulator = _setup_simulator()

    # Cylinder MUST be spawned before Talos: see _spawn_static_cylinder's note.
    geom_name = _spawn_static_cylinder(
        simulator, CYL_XY, CYL_RADIUS, CYL_HALF_HEIGHT, CYL_RGBA)

    robot = Talos(simulator)
    robot.stack.setComRefState(robot.stack.comState().value().copy())

    node = rclpy.create_node('perception_bias_demo')
    pub_gt = node.create_publisher(Marker, 'perception_bias_demo/ground_truth', 10)
    pub_gt_centroid = node.create_publisher(
        Marker, 'perception_bias_demo/centroid_ground_truth', 10)
    pub_pts = node.create_publisher(Marker, 'perception_bias_demo/surface_points', 10)
    pub_raw = node.create_publisher(Marker, 'perception_bias_demo/centroid_raw', 10)
    pub_corr = node.create_publisher(Marker, 'perception_bias_demo/centroid_corrected', 10)
    pub_vec = node.create_publisher(Marker, 'perception_bias_demo/correction_vector', 10)

    camera = _front_camera(robot)
    print(f"TALOS steht. Zylinder bei {CYL_XY} (r={CYL_RADIUS} m), "
          f"beobachtet von Kamera '{camera}'. "
          f"Front-surface-bias Marker auf 'perception_bias_demo/*'.")

    t_publish = 0.0
    try:
        while simulator.isRunning() and rclpy.ok():
            t = simulator.simTime()
            simulator.step()
            robot.update()

            if t - t_publish > 1.0 / 30.0:
                t_publish = t
                stamp = node.get_clock().now().to_msg()
                robot.publish()             # joint states + TF, so the robot stays visible
                robot.publish_depth_image()  # feeds depth_image_proc's point_cloud_xyz node

                geom_id = robot.sim.getGeomId(geom_name)  # ids don't survive recompiles
                pts_xyz, cam_xy = _segmented_points(robot, camera, geom_id)

                _publish_ground_truth(pub_gt, stamp)
                _publish_centroid(pub_gt_centroid, stamp, 'centroid_ground_truth',
                                  CYL_XY, (0.15, 0.55, 1.0, 1.0), diameter=0.05)
                if pts_xyz is not None and len(pts_xyz) >= robot.MIN_OBSTACLE_PIXELS:
                    xy = pts_xyz[:, :2]
                    raw_centroid = xy.mean(axis=0)
                    corrected_centroid, _r = Talos._front_arc_estimate(xy, cam_xy)

                    _publish_surface_points(pub_pts, stamp, pts_xyz)
                    _publish_centroid(pub_raw, stamp, 'centroid_raw',
                                      raw_centroid, (0.9, 0.15, 0.15, 0.9))
                    _publish_centroid(pub_corr, stamp, 'centroid_corrected',
                                      corrected_centroid, (0.15, 0.85, 0.3, 0.9))
                    _publish_correction_vector(pub_vec, stamp, raw_centroid,
                                               corrected_centroid)

                rclpy.spin_once(node, timeout_sec=0.0)
                rclpy.spin_once(robot._node, timeout_sec=0.0)

    finally:
        print("Schließe Simulator und fahre ROS 2 Node herunter...")
        simulator.disconnect()
        if rclpy.ok():
            node.destroy_node()
            robot._node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
