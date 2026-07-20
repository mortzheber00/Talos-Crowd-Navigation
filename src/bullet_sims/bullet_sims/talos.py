from collections import deque

import numpy as np
import pinocchio as pin
import mujoco

# simulator
from simulator.robot import Robot
# whole-body controller
from bullet_sims.tsid_wrapper import TSIDWrapper
# robot configs
import bullet_sims.talos_conf as conf
from bullet_sims.footstep_planner import Side

# ROS visualizations
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from nav_msgs.msg import Odometry
from geometry_msgs.msg import WrenchStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import JointState, Image, CameraInfo
from std_msgs.msg import Header
from crowd_nav_interfaces.msg import Obstacle, ObstacleArray

class Talos:
    """Talos robot
    combines wbc with pybullet, functions to read and set
    sensor values.
    """
    def __init__(self, simulator):
        self.conf = conf
        self.sim = simulator

        # setup tsid wrapper for whole body controller
        # stack = stack of tasks in TSID framework
        self.stack = TSIDWrapper(self.conf)

        # spawn robot in simulation
        self.robot = Robot(
            self.sim,
            self.conf.urdf,
            self.stack.model,
            basePosition=[0, 0, 1.1],
            baseQuationerion=[0, 0, 0, 1],
            useFixedBase=False,
            verbose=True,
        )

        ########################################################################
        # state
        ########################################################################
        self.support_foot = Side.RIGHT
        self.swing_foot = Side.LEFT

        ########################################################################
        # estimators
        ########################################################################
        self.zmp = None
        self.dcm = None

        ########################################################################
        # trajectory recording for the in-sim overlay (executed footsteps +
        # CoM trail); bounded so the viewer's user_scn geom budget holds
        ########################################################################
        self._footstep_history = deque(maxlen=24)
        self._com_trail = deque(maxlen=150)

        ########################################################################
        # sensors
        ########################################################################
        self._ft_joint = {
            self.conf.rf_frame_name: "leg_right_6_joint",
            self.conf.lf_frame_name: "leg_left_6_joint",
        }
        for ankle_joint in self._ft_joint.values():
            self.sim.enableJointForceTorqueSensor(
                self.robot.id(), self.robot.jointNameIndexMap()[ankle_joint], True
            )
        ########################################################################
        # visualizations
        ########################################################################
        self._node = rclpy.create_node("walking_control_node")
        self._joint_names = list(self.robot.actuatedJointNames())
        self._tf_broadcaster = TransformBroadcaster(self._node)

        ########################################################################
        # onboard depth camera
        ########################################################################
        # Attach a MuJoCo camera to the robot's real RGB-D sensor frame and set
        # up the ROS publisher.  'rgbd_optical_frame' already follows the ROS
        # optical convention (+Z forward, +X right, +Y down), so the rendered
        # depth image *and* its frame_id line up directly with TF / RViz.
        self.depth_frame_id = 'rgbd_optical_frame'
        self._has_depth_cam = False

        # Camera intrinsic parameters
        self.depth_width = 640
        self.depth_height = 480
        self.depth_fovy = 58.0  # vertical field of view in degrees

        # Publishers for depth image and camera info
        self.depth_pub = self._node.create_publisher(
            Image, '/camera/depth/image_rect', 10
        )
        self.camera_info_pub = self._node.create_publisher(
            CameraInfo, '/camera/depth/camera_info', 10
        )

        # Compute camera intrinsic matrix once at startup
        self._compute_camera_intrinsics()

        # Mount the MuJoCo camera so it *coincides* with the URDF's existing
        # rgbd_optical_frame (published statically by robot_state_publisher as
        # head_2_link -> rgbd_link -> rgbd_optical_frame). We do NOT broadcast
        # this transform ourselves -- doing so would give rgbd_optical_frame a
        # second parent and break the TF tree ("Could not transform ... world").
        #
        # URDF geometry (talos_reduced_no_hands):
        #   rgbd_joint         head_2_link -> rgbd_link           xyz=(0.066,0,0.1982) rpy=0
        #   rgbd_optical_joint rgbd_link  -> rgbd_optical_frame   xyz=0 rpy=(-90,0,-90) deg
        # => optical frame origin in head_2_link = (0.066, 0, 0.1982), and its
        #    +Z (view axis) points along head_2_link +X (straight ahead).
        # A MuJoCo camera looks down its own -Z, so the quaternion below aligns
        # cam -Z with optical +Z (it equals addCameraToLink's "look forward"
        # default; see the derivation for [0.5,-0.5,-0.5,0.5] xyzw).
        self.camera_parent_link = 'head_2_link'
        self.camera_pos = [0.066, 0.0, 0.1982]              # = URDF rgbd_link origin
        self.camera_orient = [0.5, -0.5, -0.5, 0.5]         # xyzw: cam frame == rgbd_optical_frame

        if hasattr(self.sim, 'addCameraToLink'):
            try:
                self.sim.addCameraToLink(
                    self.camera_parent_link,
                    name='rgbd_depth_cam',
                    pos=self.camera_pos,
                    orientation=self.camera_orient,
                    fovy=self.depth_fovy, width=self.depth_width, height=self.depth_height,
                )
                self._has_depth_cam = True
                print(f'[Talos] Tiefenkamera an {self.camera_parent_link} ausgerichtet auf {self.depth_frame_id}')
            except Exception as e:
                print(f'[Talos] Fehler: Tiefenkamera konnte nicht angehängt werden: {e}')
                self._has_depth_cam = False
        else:
            self._has_depth_cam = False

        ########################################################################
        # surround perception cameras (obstacle tracking only, no ROS topic)
        ########################################################################
        # Four wide-angle depth+seg cameras yawed 0/90/180/270 deg give full
        # 360 deg coverage: fovy=90 at 4:3 -> ~106 deg horizontal each, vs the
        # 90 deg spacing. Each camera sits at the RGB-D sensor height, pushed
        # 0.15 m outward along its own view direction -- inside that radius
        # the head mesh swallows the view entirely (verified empirically: a
        # camera at the head center renders 0 pedestrian pixels). The ~0.15 m
        # mount spread is negligible against pedestrian distances, so
        # multi-camera observations of the same pedestrian still merge as if
        # seen by a single omnidirectional camera. Lower resolution than the
        # RGB-D cam keeps the 4x render cost at the perception rate affordable.
        self._percep_cams = []           # camera names used by update_obstacles
        self._percep_K = {}              # per-camera intrinsics
        percep_fovy, percep_w, percep_h = 90.0, 320, 240
        percep_radial = 0.15             # [m] mount offset along the view dir
        if self._has_depth_cam:
            try:
                for k in range(4):
                    name = f'percep_cam_{k}'
                    yaw = k * np.pi / 2.0
                    self.sim.addCameraToLink(
                        self.camera_parent_link, name=name,
                        pos=[percep_radial * np.cos(yaw),
                             percep_radial * np.sin(yaw),
                             self.camera_pos[2]],
                        orientation=None, yaw=yaw,
                        fovy=percep_fovy, width=percep_w, height=percep_h,
                    )
                    self._percep_cams.append(name)
                    self._percep_K[name] = self._pinhole_K(
                        percep_fovy, percep_w, percep_h)
                print(f'[Talos] {len(self._percep_cams)} Rundumsicht-Kameras '
                      f'an {self.camera_parent_link} montiert')
            except Exception as e:
                print(f'[Talos] Fehler: Rundumsicht-Kameras nicht verfügbar, '
                      f'nutze nur die Frontkamera: {e}')
                self._percep_cams = ['rgbd_depth_cam']
                self._percep_K = {'rgbd_depth_cam': self.K}

        ########################################################################
        # 2D obstacle tracking (segmentation-based, for ORCA)
        ########################################################################
        # Tracks are keyed by geom NAME, not id: model recompiles renumber the
        # ids, names stay stable.
        self._obstacle_geom_names = []   # set via set_obstacle_geoms()
        self._obstacles = {}             # name -> dict(id,x,y,r,vx,vy,stale,t)
        self._obs_ema_alpha = 0.15       # velocity EMA smoothing factor
        self._obs_pos_alpha = 0.5        # center-position EMA smoothing factor
        self._obs_last_t = None          # sim time of the previous update
        self.MIN_OBSTACLE_PIXELS = 30    # below this a mask counts as occluded
        self.OBSTACLE_STALE_TIMEOUT = 1.0  # [s] drop tracks unseen longer than this

        self._joint_pub = self._node.create_publisher(JointState, 'joint_states', 10)
        self._odom_pub = self._node.create_publisher(Odometry, 'odom', 10)
        # Center-of-mass state for the local planner (ORCA), on its own topic.
        # ORCA must plan from the CoM, not the base_link odom above: the pelvis
        # sways/bobs at the gait frequency, and feeding that oscillating base
        # velocity into ORCA's reciprocal avoidance makes v_cmd flip and the
        # robot back up near a crowd. Published at the control rate (see
        # publish_com_odom), not the 30 Hz sensor rate, so v_cmd stays fresh.
        self._com_odom_pub = self._node.create_publisher(Odometry, 'odom_com', 10)
        self._zmp_pub = self._node.create_publisher(MarkerArray, 'zmp', 10)
        self._wrench_rf_pub = self._node.create_publisher(WrenchStamped, 'wrench_rf', 10)
        self._wrench_lf_pub = self._node.create_publisher(WrenchStamped, 'wrench_lf', 10)
        # obstacle-estimate overlay for RViz (used by publish_obstacle_markers)
        self.marker_pub = self._node.create_publisher(MarkerArray, 'obstacle_markers', 10)
        # obstacle estimates as DATA for downstream planners (ORCA); the marker
        # topic above is only a visualization of the same estimates
        self.obstacle_pub = self._node.create_publisher(
            ObstacleArray, 'obstacles/perceived', 10)
           
    def update(self):
        """updates the robot
        """
        t = self.sim.simTime()
        dt = self.sim.stepTime()

        self.robot.update()

        # update the estimators
        self._update_zmp_estimate()
        self._update_dcm_estimate()
        
        # update wbc and send back to pybullet
        self._solve(t, dt)
        
    def setSupportFoot(self, side):
        """sets the the support foot of the robot on given side
        """
        self.support_foot = side

        if side == Side.RIGHT:
            self.stack.add_contact_RF()
            self.stack.remove_motion_RF()
        elif side == Side.LEFT:
            self.stack.add_contact_LF()
            self.stack.remove_motion_LF()
        else: 
            raise ValueError("Invalid side for Support foot: {}".format(side))


    def setSwingFoot(self, side):
        """sets the swing foot of the robot on given side
        """
        self.swing_foot = side
        
        if side == Side.RIGHT:
            self.stack.remove_contact_RF()
            self.stack.add_motion_RF()
        elif side == Side.LEFT:
            self.stack.remove_contact_LF()
            self.stack.add_motion_LF()
        else:
            raise ValueError("Invalid side for swing foot: {}".format(side))
        
    def updateSwingFootRef(self, T_swing_w, V_swing_w, A_swing_w):
        """updates the swing foot motion reference
        """

        if self.swing_foot == Side.RIGHT:
            self.stack.set_RF_pose_ref(T_swing_w, V_swing_w, A_swing_w)
        elif self.swing_foot == Side.LEFT:
            self.stack.set_LF_pose_ref(T_swing_w, V_swing_w, A_swing_w)
        else:
            raise ValueError("Invalid side for swing foot: {}".format(self.swing_foot))


    def swingFootPose(self):
        """return the pose of the current swing foot
        """
        if self.swing_foot == Side.RIGHT:
            return self.stack.get_placement_RF()
        elif self.swing_foot == Side.LEFT:
            return self.stack.get_placement_LF()
        else:
            raise ValueError("Invalid side for swing foot: {}".format(self.swing_foot))

    def supportFootPose(self):
        """return the pose of the current support foot
        """
        if self.support_foot == Side.RIGHT:
            return self.stack.get_placement_RF()
        elif self.support_foot == Side.LEFT:
            return self.stack.get_placement_LF()
        else:
            raise ValueError("Invalid side for support foot: {}".format(self.support_foot))

    ############################################################################
    # trajectory recording (for the in-sim debug overlay)
    ############################################################################

    def record_footstep(self, footstep):
        """Remember an executed/committed footstep for the viewer overlay.
        Call once per step transition."""
        self._footstep_history.append(footstep)

    def record_com(self, com_xyz, min_dist=0.05):
        """Append a CoM sample to the trail if it moved at least min_dist [m]
        since the last recorded point. Call every tick (cheap)."""
        p = np.asarray(com_xyz, dtype=float)[:3].copy()
        if (not self._com_trail
                or np.hypot(*(p[:2] - self._com_trail[-1][:2])) >= min_dist):
            self._com_trail.append(p)

    @property
    def footstep_history(self):
        """Executed footsteps, oldest first (bounded)."""
        return list(self._footstep_history)

    @property
    def com_trail(self):
        """Recorded CoM positions, oldest first (bounded, decimated)."""
        return list(self._com_trail)

    def publish(self):
        # Einheitlichen Zeitstempel generieren. Wird gespeichert, damit
        # publish_depth_image() und publish_obstacle_markers() im selben Zyklus
        # EXAKT denselben Stempel nutzen: traegt die Punktwolke einen (auch nur
        # ms) juengeren Stempel als der letzte TF, verweigert RViz den Lookup
        # mit einer ExtrapolationException ("Could not transform ... to world").
        now = self._node.get_clock().now().to_msg()
        self._last_pub_stamp = now

        # joint state
        msg = JointState()
        msg.header.stamp = now
        msg.name = self._joint_names
        msg.position = self.robot.actuatedJointPosition().tolist()
        msg.velocity = self.robot.actuatedJointVelocity().tolist()
        self._joint_pub.publish(msg)

        # odometry (base pose + velocity)
        T_b_w, V_b_w = self.stack.baseState()
        xyzquat = pin.SE3ToXYZQUAT(T_b_w)
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = xyzquat[0]
        odom.pose.pose.position.y = xyzquat[1]
        odom.pose.pose.position.z = xyzquat[2]
        odom.pose.pose.orientation.x = xyzquat[3]
        odom.pose.pose.orientation.y = xyzquat[4]
        odom.pose.pose.orientation.z = xyzquat[5]
        odom.pose.pose.orientation.w = xyzquat[6]
        odom.twist.twist.linear.x = V_b_w.linear[0]
        odom.twist.twist.linear.y = V_b_w.linear[1]
        odom.twist.twist.linear.z = V_b_w.linear[2]
        odom.twist.twist.angular.x = V_b_w.angular[0]
        odom.twist.twist.angular.y = V_b_w.angular[1]
        odom.twist.twist.angular.z = V_b_w.angular[2]
        self._odom_pub.publish(odom)
        # world -> base_link TF: gleiche Pose (TSID baseState) und gleicher
        # Stempel wie die Odometry -- ein zweiter Broadcast mit eigenem
        # get_clock()-Aufruf wuerde die Stempel wieder auseinanderziehen.
        self._tf_broadcaster.sendTransform(self._odomToTransform(odom))

        # feet wrenches
        wr_rf = self.stack.get_wrench_RF(self.stack.sol)
        wr_lf = self.stack.get_wrench_LF(self.stack.sol)
        self._publish_wrench(self._wrench_rf_pub, now, self.conf.rf_frame_name, wr_rf)
        self._publish_wrench(self._wrench_lf_pub, now, self.conf.lf_frame_name, wr_lf)

        # zmp and dcm markers
        markers = MarkerArray()
        if self.zmp is not None:
            markers.markers.append(self._point_marker(now, 0, self.zmp, [1.0, 0.0, 0.0, 1.0]))
        if self.dcm is not None:
            markers.markers.append(self._point_marker(now, 1, self.dcm, [0.0, 0.0, 1.0, 1.0]))
        self._zmp_pub.publish(markers)

    def publish_com_odom(self):
        """Publish the ground-plane CoM state (position + world-frame velocity)
        as Odometry on ``odom_com`` -- the state the ORCA planner should use.

        Kept separate from publish() so the walking loop can call it at the
        control rate: the ORCA node recomputes v_cmd on every message here, so a
        high rate keeps the command fresh (the 30 Hz sensor-publish rate makes
        the robot hesitate near a crowd). It is cheap (no render), unlike the
        rest of publish()."""
        com = self.sim.computeComPosition(self.robot.id())
        vcom = self.sim.computeComVelocity(self.robot.id())
        odom = Odometry()
        odom.header.stamp = self._node.get_clock().now().to_msg()
        odom.header.frame_id = "world"
        odom.child_frame_id = "com"
        odom.pose.pose.position.x = float(com[0])
        odom.pose.pose.position.y = float(com[1])
        odom.pose.pose.position.z = float(com[2])
        odom.pose.pose.orientation.w = 1.0
        odom.twist.twist.linear.x = float(vcom[0])
        odom.twist.twist.linear.y = float(vcom[1])
        odom.twist.twist.linear.z = float(vcom[2])
        self._com_odom_pub.publish(odom)

    @staticmethod
    def _pinhole_K(fovy_deg, width, height):
        """Pinhole intrinsics from the vertical field of view:
        fy = (height / 2) / tan(fovy_rad / 2), square pixels (fx = fy),
        principal point at the image center."""
        fy = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
        return np.array([
            [fy, 0.0, width / 2.0],
            [0.0, fy, height / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @staticmethod
    def _front_arc_estimate(xy, cam_xy):
        """Estimate a cylinder's ground-plane (center, radius) from the front
        arc a *single* camera sees.

        ``xy`` are ground-projected surface points from ONE camera (Nx2) and
        ``cam_xy`` is THAT camera's mount point. A camera only sees the front
        arc, whose centroid sits ~2r/pi in front of the true axis along the
        viewing ray; estimate r from the lateral chord (immune to that bias),
        then push the centroid back onto the axis. Returns ``(center(2),
        r_radius)`` where ``r_radius`` already covers the points actually seen.

        Must be called per camera: merging points from cameras at different
        yaw angles violates the single-front-arc assumption (the combined set
        spans a wider arc than any one viewpoint), which places ``center`` off
        the axis and makes the coverage distance below explode -- the source
        of the radius inflation seen at camera seams.
        """
        centroid = xy.mean(axis=0)
        ray = centroid - cam_xy
        ray /= max(np.linalg.norm(ray), 1e-9)
        lateral = xy @ np.array([-ray[1], ray[0]])
        # robust chord + coverage via percentiles, not raw min/max: when an
        # occluder splits the mask into a crescent/fragment, a few stray points
        # sit far from the true axis and a raw max would blow the radius up.
        r_est = 0.5 * (np.percentile(lateral, 97.5) - np.percentile(lateral, 2.5))
        center = centroid + (2.0 / np.pi) * r_est * ray
        # within one front arc the farthest point is ~r from the axis, so this
        # stays bounded (unlike the same measure on a merged multi-view cloud)
        cover = float(np.percentile(np.linalg.norm(xy - center, axis=1), 97.5))
        return center, max(r_est, cover)

    def _compute_camera_intrinsics(self):
        """Intrinsic (K), projection (P) and distortion (D) parameters of the
        front RGB-D camera, for the ROS CameraInfo publisher."""
        self.K = self._pinhole_K(self.depth_fovy, self.depth_width,
                                 self.depth_height)
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]

        # Projection matrix P (3x4) for monocular camera
        self.P = np.array([
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ], dtype=np.float64)

        # Distortion coefficients (empty for ideal camera)
        self.D = np.array([], dtype=np.float64)

    def get_depth_image(self):
        """Return the onboard camera's metric depth as an (H, W) float32 numpy
        array (values in metres).

        Pixels that hit nothing are returned as NaN (not 0.0): for a 32FC1
        depth image depth_image_proc treats every *finite* value as a valid 3D
        point, so 0.0 would spawn a wall of bogus points at the camera origin.
        NaN is the REP-118 "no return" sentinel and is dropped from the cloud.
        """
        return self.sim.get_depth_image(invalid_to=np.nan)

    def publish_depth_image(self):
        """Publish the depth image and its CameraInfo under ONE identical stamp.

        depth_image_proc's point_cloud_xyz node pairs image + camera_info with
        an *ExactTime* synchronizer, so both headers must carry the byte-for-byte
        same ``stamp`` and ``frame_id``.  We guarantee that by rendering first,
        taking a single ``Time``, and sharing the very same header object between
        both messages -- there is no room for a sub-microsecond drift.
        """
        if not self._has_depth_cam:
            return

        # Render first: a slow or failed render must never sit between the two
        # get_clock() reads and desynchronise the stamps.
        depth = np.ascontiguousarray(self.get_depth_image(), dtype=np.float32)
        h, w = int(depth.shape[0]), int(depth.shape[1])

        # ONE Time -> ONE header, shared verbatim by image and camera_info.
        # Reuse the stamp of the publish() call from this cycle (TF + joint
        # states): the resulting point cloud then matches the TF tree exactly,
        # so RViz's lookup is an exact hit instead of a future-extrapolation.
        # Fallback to now() only if publish() has not run yet.
        header = Header()
        header.stamp = getattr(self, '_last_pub_stamp', None) \
            or self._node.get_clock().now().to_msg()
        header.frame_id = self.depth_frame_id

        img_msg = Image()
        img_msg.header = header
        img_msg.height = h
        img_msg.width = w
        img_msg.encoding = '32FC1'          # float32 metres
        img_msg.is_bigendian = 0
        img_msg.step = w * 4                 # 4 bytes / float32 pixel
        img_msg.data = depth.tobytes()

        info_msg = CameraInfo()
        info_msg.header = header             # same object => identical stamp+frame_id
        info_msg.height = h
        info_msg.width = w
        info_msg.distortion_model = 'plumb_bob'
        info_msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]   # ideal pinhole, 5 zeros for plumb_bob
        info_msg.k = self.K.flatten().tolist()
        info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info_msg.p = self.P.flatten().tolist()

        # Ordering is irrelevant for ExactTime (each input has its own queue);
        # info-then-image just mirrors the usual camera-driver convention.
        self.camera_info_pub.publish(info_msg)
        self.depth_pub.publish(img_msg)

    def _create_marker(self, marker_id, position, color, scale, frame_id, ns, timestamp):
        """Helper function to create a sphere marker"""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = timestamp
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        return marker

    ############################################################################
    # 2D obstacle perception (depth + segmentation) for ORCA
    ############################################################################

    def set_obstacle_geoms(self, names):
        """Register the geom names of the dynamic obstacles (e.g.
        ``CrowdEnv.geom_names``) that update_obstacles() should track."""
        self._obstacle_geom_names = list(names)

    def obstacles2d(self):
        """Current 2D obstacle estimates for a local planner (ORCA):
        list of ``{id, x, y, r, vx, vy, stale}`` in WORLD coordinates.
        ``stale`` marks tracks that were occluded/out of view on the last
        update and still carry their previous estimate."""
        return [dict(v) for v in self._obstacles.values()]

    def update_obstacles(self):
        """Extract (x, y, r, vx, vy) per tracked cylinder from one aligned
        depth + segmentation render of every surround perception camera.
        Call at the perception rate (~30 Hz).

        Pipeline (per camera): segmentation mask per geom id -> pinhole
        unprojection with that camera's K -> transform to world. The cameras
        share one mount point, so the merged per-geom point sets behave like
        the view of a single omnidirectional camera -> ground-plane centroid +
        radius -> per-track finite-difference velocity smoothed with an EMA.
        """
        if not self._percep_cams or not self._obstacle_geom_names:
            return self._obstacles

        t_now = self.sim.simTime()
        dt = None if self._obs_last_t is None else (t_now - self._obs_last_t)
        self._obs_last_t = t_now

        geom_ids = {name: self.sim.getGeomId(name)   # per frame: survives recompiles
                    for name in self._obstacle_geom_names}

        # --- gather world-frame ground points per geom from every camera ---
        # Keep each camera's contribution SEPARATE (points + that camera's own
        # mount point + FOV-clip flag): the front-arc estimate below is only
        # valid per single viewpoint, so points from cameras at different yaw
        # angles must not be pooled before estimation.
        pts = {name: [] for name in geom_ids}    # per-camera world-xy arrays
        cams_xy = {name: [] for name in geom_ids}  # that camera's mount (ground)
        clip = {name: [] for name in geom_ids}   # FOV-clipped in that camera?
        for cam in self._percep_cams:
            depth, seg = self.sim.get_depth_and_seg_image(invalid_to=np.nan,
                                                          camera=cam)
            cam_pos, R_wc = self.sim.get_camera_pose(camera=cam)
            # MuJoCo cameras look along local -Z with +Y up; the pinhole
            # equations below use the ROS optical convention (+Z forward,
            # +Y down), so flip the Y and Z axes of the camera rotation.
            R_w_opt = R_wc @ np.diag([1.0, -1.0, -1.0])

            K = self._percep_K[cam]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            valid = np.isfinite(depth)
            geom_pixels = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)

            for name, gid in geom_ids.items():
                mask = (seg[..., 0] == gid) & geom_pixels & valid
                if not mask.any():
                    continue
                v_pix, u_pix = np.nonzero(mask)      # rows = v, cols = u
                Z = depth[mask]                      # eye-space z: pinhole-ready
                X = (u_pix - cx) * Z / fx            # optical frame, +X right
                Y = (v_pix - cy) * Z / fy            # +Y down
                pts_w = (R_w_opt @ np.vstack((X, Y, Z))).T + cam_pos
                pts[name].append(pts_w[:, :2])       # ground-plane projection
                cams_xy[name].append(cam_pos[:2].copy())
                clip[name].append(u_pix.min() == 0
                                  or u_pix.max() == depth.shape[1] - 1)

        # --- per-geom estimation on the merged point sets ---
        for name, gid in geom_ids.items():
            prev = self._obstacles.get(name)

            if sum(len(p) for p in pts[name]) < self.MIN_OBSTACLE_PIXELS:
                # occluded: hold the last estimate briefly, then drop the
                # track -- a frozen estimate fed to ORCA forever acts as a
                # phantom obstacle blocking the robot's path. While held,
                # treat it as static: extrapolating the last velocity places
                # the ORCA half-plane where the agent no longer is.
                if prev is not None:
                    if t_now - prev['t'] > self.OBSTACLE_STALE_TIMEOUT:
                        del self._obstacles[name]
                    else:
                        prev['stale'] = True
                        prev['vx'] = prev['vy'] = 0.0
                continue

            # Estimate (center, radius) PER camera, where the single-front-arc
            # model holds, then fuse -- never pool points across viewpoints
            # first (that inflates the radius at camera seams). Two per-view
            # filters keep a bad view from driving the fit:
            #  * too-thin sliver (< the per-camera minimum) -> skip;
            #  * FOV border-clipped arc -> the chord is cut off, so its center
            #    and radius are unreliable; skip it as long as some clean
            #    (unclipped, non-sliver) view remains. This is the seam case
            #    clip=[True, False]: use only the camera that sees the whole arc.
            # If nothing clean survives (all slivers or all clipped), fall back
            # to the pooled points so a split-but-visible track still gets an
            # estimate.
            has_clean = any((not cl) and len(p) >= self.MIN_OBSTACLE_PIXELS
                            for p, cl in zip(pts[name], clip[name]))
            centers, radii, weights = [], [], []
            for p, cxy, cl in zip(pts[name], cams_xy[name], clip[name]):
                if len(p) < self.MIN_OBSTACLE_PIXELS:
                    continue
                if has_clean and cl:
                    continue
                c, r = self._front_arc_estimate(p, cxy)
                centers.append(c)
                radii.append(r)
                weights.append(len(p))
            if centers:
                w = np.asarray(weights, dtype=float)
                w /= w.sum()
                center = (np.asarray(centers) * w[:, None]).sum(axis=0)
                # each per-view radius is bounded ~r, so the max across views
                # is safe (never under-estimates) without inflating
                r_cover = max(radii)
            else:
                xy = np.vstack(pts[name])
                center, r_cover = self._front_arc_estimate(
                    xy, cams_xy[name][int(np.argmax([len(p) for p in pts[name]]))])
            # 15% safety padding on top of whatever the points actually cover
            # (the percentile coverage above trims the outermost points, so this
            # keeps a healthy margin over the true radius for a well-seen agent)
            radius = 1.15 * r_cover
            # A pedestrian is a rigid body: its true radius is constant, so any
            # large frame-to-frame change is measurement error. Two guards:
            #  * if every contributing mask touches its left/right image border
            #    (adjacent cameras overlap at the seams, so a genuinely
            #    complete arc is never border-clipped in ALL views) the chord is
            #    cut off and can only under-estimate: hold, don't shrink;
            #  * otherwise clamp the change to +/-1% per frame. Symmetric on
            #    purpose: a passing occluder that leaves a thin sliver can make
            #    the single-frame estimate collapse OR spike (an occluder
            #    fragment's stray points survive even a robust percentile), and
            #    neither should move the radius before the mask drops below
            #    MIN_OBSTACLE_PIXELS (-> stale hold). Legitimate change (an agent
            #    resolving as it nears) is gradual and tracks fine at 1%/frame.
            if prev is not None:
                if all(clip[name]):
                    radius = max(radius, prev['r'])
                else:
                    radius = float(np.clip(radius,
                                           0.99 * prev['r'], 1.01 * prev['r']))

            if prev is None or dt is None or dt <= 0.0:
                vx = vy = 0.0                        # first sighting: EMA starts next frame
            else:
                # Smooth the center estimate itself: pixel quantisation plus
                # the swaying camera mount put ~1 cm of frame-to-frame noise on
                # the raw center, which the finite difference below would blow
                # up into ~0.3 m/s velocity noise -- enough to make the ORCA
                # half-planes (and with them v_cmd) flip sides erratically.
                # Skip the blend for re-emerging stale tracks, whose position
                # may genuinely have jumped.
                if not prev['stale']:
                    b = self._obs_pos_alpha
                    center = b * center + (1.0 - b) * np.array([prev['x'],
                                                                prev['y']])
                v_raw = (center - np.array([prev['x'], prev['y']])) / dt
                # position jumps when a track re-emerges from behind an
                # occluder; clamp to human-plausible speeds before smoothing
                v_raw = np.clip(v_raw, -3.0, 3.0)
                a = self._obs_ema_alpha
                vx = a * v_raw[0] + (1.0 - a) * prev['vx']
                vy = a * v_raw[1] + (1.0 - a) * prev['vy']

            self._obstacles[name] = dict(
                id=gid, x=float(center[0]), y=float(center[1]),
                r=float(radius), vx=float(vx), vy=float(vy),
                stale=False, t=t_now)

        return self._obstacles

    def publish_obstacles(self):
        """Publish the current depth-camera obstacle estimates (the state kept
        by update_obstacles()) as an ObstacleArray on ``obstacles/perceived``.
        Call after update_obstacles(); planners subscribe to this instead of
        polling obstacles2d()."""
        msg = ObstacleArray()
        msg.header.stamp = getattr(self, '_last_pub_stamp', None) \
            or self._node.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        for name, obs in sorted(self._obstacles.items()):
            o = Obstacle()
            o.name = name
            o.id = int(obs['id'])
            o.x = float(obs['x'])
            o.y = float(obs['y'])
            o.radius = float(obs['r'])
            o.vx = float(obs['vx'])
            o.vy = float(obs['vy'])
            o.stale = bool(obs['stale'])
            msg.obstacles.append(o)
        self.obstacle_pub.publish(msg)

    def publish_obstacle_markers(self, ground_truth=None):
        """RViz overlay of the obstacle estimates: cyan cylinders at the
        estimated (x, y, r), optionally grey ghost cylinders at the ground
        truth (``CrowdEnv.ground_truth()``) for visual validation."""
        if not self._obstacles:
            return
        stamp = getattr(self, '_last_pub_stamp', None) \
            or self._node.get_clock().now().to_msg()
        height = 1.75
        marker_array = MarkerArray()
        for i, (name, obs) in enumerate(sorted(self._obstacles.items())):
            m = self._create_marker(
                i, [obs['x'], obs['y'], height / 2.0],
                [0.0, 0.9, 0.9, 0.4 if obs['stale'] else 0.8],
                1.0, 'world', 'obstacle_estimates', stamp)
            m.type = Marker.CYLINDER
            m.scale.x = m.scale.y = 2.0 * obs['r']
            m.scale.z = height
            marker_array.markers.append(m)
        if ground_truth:
            for i, (name, gt) in enumerate(sorted(ground_truth.items())):
                m = self._create_marker(
                    i, [gt['x'], gt['y'], height / 2.0],
                    [0.7, 0.7, 0.7, 0.3], 1.0, 'world',
                    'obstacle_ground_truth', stamp)
                m.type = Marker.CYLINDER
                m.scale.x = m.scale.y = 2.0 * gt['r']
                m.scale.z = height
                marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)

    ############################################################################
    # private funcitons
    ############################################################################

    def _odomToTransform(self, odom):
        tf = TransformStamped()
        tf.header.stamp = odom.header.stamp
        tf.header.frame_id = odom.header.frame_id
        tf.child_frame_id = odom.child_frame_id
        tf.transform.translation.x = odom.pose.pose.position.x
        tf.transform.translation.y = odom.pose.pose.position.y
        tf.transform.translation.z = odom.pose.pose.position.z
        tf.transform.rotation = odom.pose.pose.orientation
        return tf

    def _publish_wrench(self, pub, stamp, frame_id, wrench):
        msg = WrenchStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.wrench.force.x = wrench[0]
        msg.wrench.force.y = wrench[1]
        msg.wrench.force.z = wrench[2]
        msg.wrench.torque.x = wrench[3]
        msg.wrench.torque.y = wrench[4]
        msg.wrench.torque.z = wrench[5]
        pub.publish(msg)

    def _point_marker(self, stamp, marker_id, position, rgba):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = stamp
        marker.ns = "ground_reference_points"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba
        return marker

    def _solve(self, t, dt):
        # get the current state
        q = self.robot.q()
        v = self.robot.v()
        
        # solve the whole body qp
        tau, dv = self.stack.update(q, v, t)
        self.robot.setActuatedJointTorques(tau)

    def _foot_wrench_world(self, frame_name):
        """reads the foot FT sensor (mounted on the ankle joint) and returns
        (force, torque) expressed about the world origin
        """
        # the sole_fix joint carries no force; read the ankle joint instead
        joint_id = self.robot.jointNameIndexMap()[self._ft_joint[frame_name]]
        # PyBullet reports the constraint reaction wrench, negate to get the contact wrench
        wrench_local = -np.array(self.sim.getJointState(self.robot.id(), joint_id)[2])
        ls = self.sim.getLinkState(self.robot.id(), joint_id, computeForwardKinematics=True)
        H_w_ankle = pin.XYZQUATToSE3(np.array(list(ls[0]) + list(ls[1])))
        # act() transfers the moment to the world origin, so the downstream ZMP
        # formula uses d=0 (the ankle height is absorbed here)
        wrench_w = H_w_ankle.act(pin.Force(wrench_local))
        return wrench_w.linear, wrench_w.angular

    def _zmp_single_foot(self, force, torque, f_min=1.0):
        """ZMP under a single foot, given its world-frame force/torque at the sole plane (d=0)
        """
        if force[2] < f_min:
            return np.zeros(3)
        p_x = (-torque[1]) / force[2]
        p_y = (torque[0]) / force[2]
        return np.array([p_x, p_y, 0.0])

    def _update_zmp_estimate(self):
        """update the estimated zmp position
        """
        f_r, tau_r = self._foot_wrench_world(self.conf.rf_frame_name)
        f_l, tau_l = self._foot_wrench_world(self.conf.lf_frame_name)

        zmp_r = self._zmp_single_foot(f_r, tau_r)
        zmp_l = self._zmp_single_foot(f_l, tau_l)

        f_z_total = f_r[2] + f_l[2]
        if f_z_total < 10.0:
            self.zmp = np.zeros(3)
        else:
            self.zmp = (zmp_r * f_r[2] + zmp_l * f_l[2]) / f_z_total

    def _update_dcm_estimate(self):
        """update the estimated capture point / DCM based on current center of mass state
        """
        com_state = self.stack.comState()
        com_pos = com_state.value()
        com_vel = com_state.derivative()

        omega = np.sqrt(9.81 / com_pos[2])
        self.dcm = np.array([
            com_pos[0] + com_vel[0] / omega,
            com_pos[1] + com_vel[1] / omega,
            0.0,
        ])