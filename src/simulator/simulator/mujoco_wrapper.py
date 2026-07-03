"""mujoco_wrapper
MuJoCo simulation wrapper.

Drop-in replacement for :class:`simulator.pybullet_wrapper.PybulletWrapper`.
It exposes the same public API but is backed by MuJoCo instead of PyBullet.

Design notes
------------
* MuJoCo compiles a *static* model (``MjModel``) up front and then steps a
  ``MjData``.  PyBullet on the other hand builds the world incrementally.  To
  bridge the two we keep an ``mjSpec`` around, add bodies/actuators/sensors to
  it and (re)compile lazily whenever the spec changed.
* PyBullet uses quaternions in ``[x, y, z, w]`` order, MuJoCo uses
  ``[w, x, y, z]``.  Every quaternion crossing the public API is converted here
  so that consumers keep seeing PyBullet's ``[x, y, z, w]`` convention.
* PyBullet identifies bodies/links/joints with per-body integer indices
  (``-1`` == base link).  We keep a small registry per loaded body that maps
  those local indices onto MuJoCo's global name spaces.

Only the subset of the PyBullet API exercised by the Talos sim+control path is
implemented faithfully; debug-draw helpers are best-effort / no-ops.
"""

import os
import re
import tempfile
import time

import numpy as np
import mujoco

try:  # optional, only needed for render=True
    import mujoco.viewer as mj_viewer
except Exception:  # pragma: no cover
    mj_viewer = None


# ---------------------------------------------------------------------------
# quaternion / transform helpers (engine independent, PyBullet [x,y,z,w] order)
# ---------------------------------------------------------------------------

def _xyzw_to_wxyz(q):
    q = np.asarray(q, dtype=float).ravel()
    return np.array([q[3], q[0], q[1], q[2]])


def _wxyz_to_xyzw(q):
    q = np.asarray(q, dtype=float).ravel()
    return np.array([q[1], q[2], q[3], q[0]])


def getMatrixFromQuaternion(q_xyzw):
    """Return a length-9 row-major rotation matrix (matches pb.getMatrixFromQuaternion)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, _xyzw_to_wxyz(q_xyzw))
    return R


def invertTransform(pos, quat_xyzw):
    """Inverse of a rigid transform. Returns (pos, quat) in PyBullet convention."""
    pos = np.asarray(pos, dtype=float).ravel()
    q_w = _xyzw_to_wxyz(quat_xyzw)
    inv_pos = np.zeros(3)
    inv_quat = np.zeros(4)
    mujoco.mju_negPose(inv_pos, inv_quat, pos, q_w)
    return inv_pos, _wxyz_to_xyzw(inv_quat)


def multiplyTransforms(posA, quatA, posB, quatB):
    """Compose two rigid transforms A * B. Returns (pos, quat) in PyBullet convention."""
    posA = np.asarray(posA, dtype=float).ravel()
    posB = np.asarray(posB, dtype=float).ravel()
    out_pos = np.zeros(3)
    out_quat = np.zeros(4)
    mujoco.mju_mulPose(out_pos, out_quat,
                       posA, _xyzw_to_wxyz(quatA),
                       posB, _xyzw_to_wxyz(quatB))
    return out_pos, _wxyz_to_xyzw(out_quat)


def getQuaternionFromEuler(euler_xyz):
    """Roll/pitch/yaw (extrinsic XYZ) -> quaternion [x,y,z,w] (matches PyBullet)."""
    r, p, y = np.asarray(euler_xyz, dtype=float).ravel()
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
    cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
    x = sr * cp * cy - cr * sp * sy
    yy = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return np.array([x, yy, z, w])


def getEulerFromQuaternion(q_xyzw):
    """Quaternion [x,y,z,w] -> roll/pitch/yaw (matches PyBullet)."""
    x, y, z, w = np.asarray(q_xyzw, dtype=float).ravel()
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw])


# PyBullet constant aliases (so callers relying on pb.* flags keep working)
JOINT_REVOLUTE = 0
JOINT_PRISMATIC = 1
JOINT_SPHERICAL = 2
JOINT_PLANAR = 3
JOINT_FIXED = 4
LINK_FRAME = 1
WORLD_FRAME = 2
GEOM_SPHERE = 2
GEOM_BOX = 3
GEOM_MESH = 5
GEOM_PLANE = 6
URDF_USE_INERTIA_FROM_FILE = 0
URDF_MERGE_FIXED_LINKS = 0


class _BodyRecord:
    """Bookkeeping for one loaded articulated body (PyBullet-style local indexing)."""

    def __init__(self):
        self.root_body = None          # mujoco body name of the base link
        self.has_free_base = False     # True if a floating-base freejoint was added
        self.joint_names = []          # local joint index i -> mujoco joint name
        self.link_bodies = []          # local link index i -> mujoco body name
        self.actuated_names = []       # subset of joint_names that are actuated


class Floor:
    """Ground plane, kept for API compatibility with the PyBullet wrapper."""

    def __init__(self, wrapper=None, basePosition=[0, 0, 0], baseRPY=[0, 0, 0]):
        self._wrapper = wrapper
        self._id = -1  # geom is part of worldbody

    def id(self):
        return self._id

    def changeFriction(self, lateralFriction=1.0, spinningFriction=1.0):
        if self._wrapper is not None:
            self._wrapper._floor_friction = [lateralFriction, spinningFriction, spinningFriction]

    def changeImpedance(self, stiffness, damping):
        # MuJoCo models contact compliance through solref/solimp; left as a no-op
        # so the closed-loop behaviour can be tuned in the model instead.
        pass


class MujocoWrapper:
    """MuJoCo-backed re-implementation of :class:`PybulletWrapper`."""

    def __init__(self,
                 render=True,
                 sim_rate=1000,
                 gravity=np.array([0, 0, -9.81]),
                 num_solver_iterations=None,
                 num_sub_steps=None):

        self._render = bool(render)
        self._sim_rate = sim_rate
        self._sim_time_step = 1.0 / self._sim_rate
        self._gravity = np.asarray(gravity, dtype=float)

        self._running = True
        self._sim_count = 0
        self._sim_time = 0.0
        self._start_time = time.time()
        self._time_warp = 1.0
        self._floor_friction = [1.0, 0.005, 0.0001]

        # model building state
        self._spec = mujoco.MjSpec()
        self._spec.option.timestep = self._sim_time_step
        self._spec.option.gravity = self._gravity
        if num_solver_iterations is not None:
            self._spec.option.iterations = int(num_solver_iterations)

        # ground plane
        self._spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            friction=self._floor_friction)

        self._bodies = {}          # body_id -> _BodyRecord
        self._next_body_id = 0
        self._mesh_link_dir = None  # symlink farm giving meshes unique basenames
        self._model = None
        self._data = None
        self._dirty = True         # spec changed -> needs (re)compile
        self._viewer = None

        self._floor = Floor(self)

    # ------------------------------------------------------------------ #
    # compilation / lifecycle
    # ------------------------------------------------------------------ #

    def _ensure_compiled(self):
        if self._model is None:
            self._model = self._spec.compile()
            self._data = mujoco.MjData(self._model)
            mujoco.mj_forward(self._model, self._data)
            self._apply_pending_state()
            if self._render:
                self._open_viewer()
            self._dirty = False
        elif self._dirty:
            # recompile in place, keeping the current state where possible
            self._model, self._data = self._spec.recompile(self._model, self._data)
            self._dirty = False

    def _apply_pending_state(self):
        # placeholder hook: initial poses are applied through resetJointState /
        # resetBasePose once the model exists.
        pass

    def _open_viewer(self):
        if mj_viewer is None:
            self._render = False
            return
        self._viewer = mj_viewer.launch_passive(self._model, self._data)

    def setRealTime(self):
        self._sim_time_step = 1.0 / 240.0
        self._spec.option.timestep = self._sim_time_step
        self._dirty = True

    def isRunning(self):
        if self._viewer is not None:
            return self._running and self._viewer.is_running()
        return self._running

    def isRendering(self):
        return self._render

    def configVisuals(self, **kwargs):
        # No equivalent needed for the MuJoCo viewer.
        pass

    def stepTime(self):
        return self._sim_time_step

    def stepFreq(self):
        return self._sim_rate

    def simTime(self):
        return self._sim_time

    def simCount(self):
        return self._sim_count

    def step(self):
        self._ensure_compiled()
        mujoco.mj_step(self._model, self._data)
        self._sim_count += 1
        self._sim_time = self._sim_count * self._sim_time_step
        if self._viewer is not None:
            self._viewer.sync()
            time.sleep(self._sim_time_step * self._time_warp)

    def debug(self):
        # Passive viewer does not expose keyboard events the way PyBullet does.
        pass

    def disconnect(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._running = False

    def resetSimulation(self):
        if self._model is not None:
            mujoco.mj_resetData(self._model, self._data)
            mujoco.mj_forward(self._model, self._data)

    def gravity(self):
        return self._gravity

    def floor(self):
        return self._floor

    # ------------------------------------------------------------------ #
    # internal index mapping helpers
    # ------------------------------------------------------------------ #

    def _rec(self, body_id):
        return self._bodies[body_id]

    def _jnt_id(self, body_id, local_idx):
        name = self._rec(body_id).joint_names[local_idx]
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _body_mjid(self, body_id, link_idx):
        rec = self._rec(body_id)
        if link_idx == -1:
            name = rec.root_body
        else:
            name = rec.link_bodies[link_idx]
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _qpos_adr(self, body_id, local_idx):
        jid = self._jnt_id(body_id, local_idx)
        return self._model.jnt_qposadr[jid]

    def _qvel_adr(self, body_id, local_idx):
        jid = self._jnt_id(body_id, local_idx)
        return self._model.jnt_dofadr[jid]

    def _actuator_id(self, prefix, body_id, local_idx):
        name = prefix + self._rec(body_id).joint_names[local_idx]
        return mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _free_base_adr(self, body_id):
        """Return (qposadr, dofadr) of the floating-base freejoint, or None."""
        rec = self._rec(body_id)
        if not rec.has_free_base:
            return None
        mjid = self._body_mjid(body_id, -1)
        jid = int(self._model.body_jntadr[mjid])  # root body's first joint
        assert self._model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
        return int(self._model.jnt_qposadr[jid]), int(self._model.jnt_dofadr[jid])

    # ------------------------------------------------------------------ #
    # body creation
    # ------------------------------------------------------------------ #

    def _resolve_mesh_path(self, ref, urdf_dir):
        """Resolve a URDF mesh ``filename`` (possibly package://) to an abs path."""
        if ref.startswith("package://"):
            rel = ref[len("package://"):]
            pkg = rel.split("/", 1)[0]
            cur = urdf_dir
            while cur and cur != os.path.dirname(cur):
                if os.path.basename(cur) == pkg:
                    return os.path.join(os.path.dirname(cur), rel)
                cur = os.path.dirname(cur)
            # package root not found on disk: best-effort strip of the prefix
            return os.path.join(urdf_dir, rel.split("/", 1)[1] if "/" in rel else rel)
        if os.path.isabs(ref):
            return ref
        return os.path.join(urdf_dir, ref)

    def _load_urdf_spec(self, filename):
        """Read a URDF and return an mjSpec.

        MuJoCo names every mesh by the *basename* of its file, so two distinct
        meshes that happen to share a basename (e.g. ``torso/base_link.STL`` and
        ``gripper/base_link.STL``) silently collapse into one -- which shows up
        as, for instance, giant grippers.  To avoid that we build a symlink farm
        in which every distinct mesh file gets a unique flattened name, and point
        the URDF at it.
        """
        xml = open(filename).read()
        urdf_dir = os.path.dirname(os.path.abspath(filename))

        if self._mesh_link_dir is None:
            self._mesh_link_dir = tempfile.mkdtemp(prefix="mjw_meshes_")
        link_dir = self._mesh_link_dir

        refs = set(re.findall(r'filename="([^"]+\.[sS][tT][lL])"', xml))
        for ref in refs:
            src = self._resolve_mesh_path(ref, urdf_dir)
            unique = re.sub(r"[^A-Za-z0-9._-]", "_", ref.replace("package://", ""))
            dst = os.path.join(link_dir, unique)
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    import shutil
                    shutil.copyfile(src, dst)
            xml = xml.replace(f'"{ref}"', f'"{unique}"')

        inject = (f'<mujoco><compiler meshdir="{link_dir}" balanceinertia="true" '
                  f'discardvisual="false"/></mujoco>')
        xml = re.sub(r"(<robot[^>]*>)", r"\1\n  " + inject, xml, count=1)
        return mujoco.MjSpec.from_string(xml)

    def createBodyFromUrdf(self, filename, position, orientation=[0., 0., 0., 1.],
                           use_fixed_base=True, flags=None, *args, **kwargs):
        """Load a URDF, add a floating-base joint (if requested), position/torque
        actuators and per-link force/torque sensors, then merge into the world.
        Returns an integer body id (PyBullet-compatible)."""
        if isinstance(position, np.ndarray):
            position = position.ravel().tolist()
        if isinstance(orientation, np.ndarray):
            orientation = orientation.ravel().tolist()

        robot_spec = self._load_urdf_spec(filename)
        root = robot_spec.worldbody.bodies[0]
        root.pos = position
        root.quat = _xyzw_to_wxyz(orientation)

        rec = _BodyRecord()
        rec.root_body = root.name

        if not use_fixed_base:
            root.add_freejoint()
            rec.has_free_base = True

        hinge_joints = [j for j in robot_spec.joints
                        if j.type in (mujoco.mjtJoint.mjJNT_HINGE,
                                      mujoco.mjtJoint.mjJNT_SLIDE)]
        rec.joint_names = [j.name for j in hinge_joints]
        rec.actuated_names = list(rec.joint_names)

        # position + torque actuator per actuated joint
        for jn in rec.joint_names:
            robot_spec.add_actuator(
                name="mot_" + jn, target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gear=[1, 0, 0, 0, 0, 0])
            robot_spec.add_actuator(
                name="pos_" + jn, target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gainprm=[1000, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                biasprm=[0, -1000, 0, 0, 0, 0, 0, 0, 0, 0])

        # link bodies (child body of each joint), and F/T sensors on each
        for j in hinge_joints:
            child = j.parent  # body owning the joint
            rec.link_bodies.append(child.name)
            site_name = "ft_" + child.name
            child.add_site(name=site_name)
            robot_spec.add_sensor(
                name="force_" + child.name, type=mujoco.mjtSensor.mjSENS_FORCE,
                objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)
            robot_spec.add_sensor(
                name="torque_" + child.name, type=mujoco.mjtSensor.mjSENS_TORQUE,
                objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site_name)

        # merge the robot spec into the world spec
        frame = self._spec.worldbody.add_frame()
        frame.attach_body(root, "", "")

        body_id = self._next_body_id
        self._next_body_id += 1
        self._bodies[body_id] = rec
        self._model = None  # force full recompile (topology changed)
        self._dirty = True
        self._ensure_compiled()

        # apply the initial base pose to the freshly compiled data
        if not use_fixed_base:
            self.resetBasePose(body_id, position, orientation)
        return body_id

    def createPrimitiveBody(self, *args, **kwargs):
        raise NotImplementedError("createPrimitiveBody is not needed for the Talos path")

    def createBodyFromMesh(self, *args, **kwargs):
        raise NotImplementedError("createBodyFromMesh is not needed for the Talos path")

    # ------------------------------------------------------------------ #
    # model introspection (PyBullet getX equivalents)
    # ------------------------------------------------------------------ #

    def getNumJoints(self, body_id):
        return len(self._rec(body_id).joint_names)

    def numJoints(self, body_id):
        return self.getNumJoints(body_id)

    def getBodyInfo(self, body_id):
        rec = self._rec(body_id)
        # (baseLinkName, bodyName) as *bytes* to mirror PyBullet
        return (rec.root_body.encode(), rec.root_body.encode())

    def getJointInfo(self, body_id, joint_index):
        """Return a PyBullet-shaped joint-info tuple (only fields used downstream)."""
        rec = self._rec(body_id)
        jname = rec.joint_names[joint_index]
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        child_body = rec.link_bodies[joint_index]
        jtype = JOINT_REVOLUTE
        lower, upper = self._model.jnt_range[jid]
        limited = bool(self._model.jnt_limited[jid])
        if not limited:
            lower, upper = 0.0, -1.0
        axis = self._model.jnt_axis[jid]
        info = [0] * 17
        info[0] = joint_index
        info[1] = jname.encode()
        info[2] = jtype
        info[3] = int(self._model.jnt_qposadr[jid])
        info[4] = int(self._model.jnt_dofadr[jid])
        info[5] = 0
        info[6] = 0.0            # joint damping
        info[7] = 0.0            # joint friction
        info[8] = float(lower)   # lower limit
        info[9] = float(upper)   # upper limit
        info[10] = 0.0           # max force
        info[11] = 0.0           # max velocity
        info[12] = child_body.encode()
        info[13] = tuple(float(a) for a in axis)
        info[14] = (0.0, 0.0, 0.0)
        info[15] = (0.0, 0.0, 0.0, 1.0)
        info[16] = -1
        return tuple(info)

    def getDynamicsInfo(self, body_id, link_index):
        """Return a PyBullet-shaped dynamics-info tuple (mass + inertial frame)."""
        mjid = self._body_mjid(body_id, link_index)
        mass = float(self._model.body_mass[mjid])
        ipos = self._model.body_ipos[mjid]
        iquat = self._model.body_iquat[mjid]  # wxyz
        info = [0] * 12
        info[0] = mass
        info[1] = 1.0                                   # lateral friction
        info[2] = tuple(float(x) for x in self._model.body_inertia[mjid])
        info[3] = tuple(float(x) for x in ipos)         # local inertial pos
        info[4] = tuple(float(x) for x in _wxyz_to_xyzw(iquat))  # local inertial orn
        info[5] = 0.0
        info[6] = 0.0
        info[7] = 0.0
        info[8] = -1.0
        info[9] = -1.0
        info[10] = -1.0
        info[11] = 0
        return tuple(info)

    def enableJointForceTorqueSensor(self, body_id, joint_index, enable=True):
        # Sensors are added at model-build time, so this is a no-op flag.
        return

    def getJointState(self, body_id, joint_index, physicsClientId=0):
        """Return (position, velocity, reactionForces(6), appliedTorque)."""
        self._ensure_compiled()
        qadr = self._qpos_adr(body_id, joint_index)
        vadr = self._qvel_adr(body_id, joint_index)
        pos = float(self._data.qpos[qadr])
        vel = float(self._data.qvel[vadr])
        wrench = self._link_ft_sensor(body_id, joint_index)
        return (pos, vel, tuple(wrench), 0.0)

    def _link_ft_sensor(self, body_id, joint_index):
        child = self._rec(body_id).link_bodies[joint_index]
        fadr = self._sensor_adr("force_" + child)
        tadr = self._sensor_adr("torque_" + child)
        if fadr is None or tadr is None:
            return np.zeros(6)
        f = self._data.sensordata[fadr:fadr + 3]
        t = self._data.sensordata[tadr:tadr + 3]
        return np.concatenate([f, t])

    def _sensor_adr(self, name):
        sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            return None
        return int(self._model.sensor_adr[sid])

    def applyExternalForce(self, body_id, link_index, force, position, frame):
        self._ensure_compiled()
        mjid = self._body_mjid(body_id, link_index)
        force = np.asarray(force, dtype=float).ravel()
        pos = np.asarray(position, dtype=float).ravel()
        torque = np.zeros(3)
        mujoco.mj_applyFT(self._model, self._data, force, torque, pos, mjid,
                          self._data.qfrc_applied)

    # ------------------------------------------------------------------ #
    # cartesian / base state
    # ------------------------------------------------------------------ #

    def resetBasePose(self, body_id, pos, orient=[0, 0, 0, 1]):
        self._ensure_compiled()
        rec = self._rec(body_id)
        if isinstance(pos, np.ndarray):
            pos = pos.ravel().tolist()
        if isinstance(orient, np.ndarray):
            orient = orient.ravel().tolist()
        adr = self._free_base_adr(body_id)
        if adr is not None:
            qadr = adr[0]
            self._data.qpos[qadr:qadr + 3] = pos
            self._data.qpos[qadr + 3:qadr + 7] = _xyzw_to_wxyz(orient)
            mujoco.mj_forward(self._model, self._data)

    def resetBaseVelocity(self, body_id, linear, angular):
        self._ensure_compiled()
        adr = self._free_base_adr(body_id)
        if adr is not None:
            dadr = adr[1]
            self._data.qvel[dadr:dadr + 3] = np.asarray(linear).ravel()
            self._data.qvel[dadr + 3:dadr + 6] = np.asarray(angular).ravel()
            mujoco.mj_forward(self._model, self._data)

    def _com_pos(self, mjid):
        # PyBullet's getBasePositionAndOrientation / getLinkState[0] report the
        # *inertial* (CoM) frame, which in MuJoCo is xipos/ximat (not xpos/xquat).
        return np.array(self._data.xipos[mjid])

    def _com_orn(self, mjid):
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self._data.ximat[mjid].ravel())
        return _wxyz_to_xyzw(quat)

    def baseWorldPosition(self, body_id):
        self._ensure_compiled()
        return self._com_pos(self._body_mjid(body_id, -1))

    def baseWorldOrientation(self, body_id):
        self._ensure_compiled()
        return self._com_orn(self._body_mjid(body_id, -1))

    def baseWorldPose(self, body_id):
        return self.baseWorldPosition(body_id), self.baseWorldOrientation(body_id)

    def _body_cvel_world(self, mjid):
        """Return (linear, angular) world-frame velocity of a body's frame origin."""
        self._ensure_compiled()
        vel = np.zeros(6)
        # flg_local=0 -> world frame; result is [angular(3), linear(3)] at body origin
        mujoco.mj_objectVelocity(self._model, self._data,
                                 mujoco.mjtObj.mjOBJ_BODY, mjid, vel, 0)
        return vel[3:6].copy(), vel[0:3].copy()

    def baseWorldLinearVelocity(self, body_id):
        mjid = self._body_mjid(body_id, -1)
        return self._body_cvel_world(mjid)[0]

    def baseWorldAngularVelocity(self, body_id):
        mjid = self._body_mjid(body_id, -1)
        return self._body_cvel_world(mjid)[1]

    def baseWorldVelocity(self, body_id):
        lin, ang = self._body_cvel_world(self._body_mjid(body_id, -1))
        return np.concatenate([lin, ang])

    def baseMass(self, body_id):
        return self.linkMass(body_id, -1)

    # ------------------------------------------------------------------ #
    # link state
    # ------------------------------------------------------------------ #

    def _link_pos(self, body_id, link_id):
        if link_id == -1:
            return self.baseWorldPosition(body_id)
        return self._com_pos(self._body_mjid(body_id, link_id))

    def _link_orn(self, body_id, link_id):
        if link_id == -1:
            return self.baseWorldOrientation(body_id)
        return self._com_orn(self._body_mjid(body_id, link_id))

    def linkWorldPositions(self, body_id, link_ids):
        self._ensure_compiled()
        if isinstance(link_ids, int):
            return self._link_pos(body_id, link_ids)
        return np.asarray([self._link_pos(body_id, i) for i in link_ids])

    def linkWorldOrientation(self, body_id, link_ids):
        self._ensure_compiled()
        if isinstance(link_ids, int):
            return self._link_orn(body_id, link_ids)
        return np.asarray([self._link_orn(body_id, i) for i in link_ids])

    def _link_linvel(self, body_id, link_id):
        if link_id == -1:
            return self.baseWorldLinearVelocity(body_id)
        return self._body_cvel_world(self._body_mjid(body_id, link_id))[0]

    def _link_angvel(self, body_id, link_id):
        if link_id == -1:
            return self.baseWorldAngularVelocity(body_id)
        return self._body_cvel_world(self._body_mjid(body_id, link_id))[1]

    def linkWorldLinearVelocity(self, body_id, link_ids):
        self._ensure_compiled()
        if isinstance(link_ids, int):
            return self._link_linvel(body_id, link_ids)
        return np.asarray([self._link_linvel(body_id, i) for i in link_ids])

    def linkWorldAngularVelocity(self, body_id, link_ids):
        self._ensure_compiled()
        if isinstance(link_ids, int):
            return self._link_angvel(body_id, link_ids)
        return np.asarray([self._link_angvel(body_id, i) for i in link_ids])

    def linkWorldVelocity(self, body_id, link_ids):
        self._ensure_compiled()
        if isinstance(link_ids, int):
            return np.concatenate([self._link_linvel(body_id, link_ids),
                                   self._link_angvel(body_id, link_ids)])
        out = []
        for i in link_ids:
            out.append(np.concatenate([self._link_linvel(body_id, i),
                                       self._link_angvel(body_id, i)]))
        return np.asarray(out)

    def linkWorldPose(self, body_id, link_ids):
        return self.linkWorldPositions(body_id, link_ids), \
            self.linkWorldOrientation(body_id, link_ids)

    def linkMass(self, body_id, link_ids):
        self._ensure_compiled()
        if isinstance(link_ids, int):
            return float(self._model.body_mass[self._body_mjid(body_id, link_ids)])
        return np.array([float(self._model.body_mass[self._body_mjid(body_id, i)])
                         for i in link_ids])

    # ------------------------------------------------------------------ #
    # joint state
    # ------------------------------------------------------------------ #

    def jointNames(self, body_id, joint_ids):
        rec = self._rec(body_id)
        return [rec.joint_names[i].encode() for i in joint_ids]

    def jointPosition(self, body_id, joint_ids=None):
        self._ensure_compiled()
        if joint_ids is None:
            joint_ids = range(self.getNumJoints(body_id))
        if isinstance(joint_ids, int):
            return float(self._data.qpos[self._qpos_adr(body_id, joint_ids)])
        return np.array([self._data.qpos[self._qpos_adr(body_id, i)] for i in joint_ids])

    def jointVelocity(self, body_id, joint_ids=None):
        self._ensure_compiled()
        if joint_ids is None:
            joint_ids = range(self.getNumJoints(body_id))
        if isinstance(joint_ids, int):
            return float(self._data.qvel[self._qvel_adr(body_id, joint_ids)])
        return np.array([self._data.qvel[self._qvel_adr(body_id, i)] for i in joint_ids])

    def jointDamping(self, body_id, joint_ids=None, joint_damping=0):
        # Joint damping is baked into the compiled model; runtime change is a no-op.
        return

    def resetJointState(self, body_id, joint_ids=None, positions=None, velocites=None):
        self._ensure_compiled()
        if isinstance(joint_ids, int):
            joint_ids = [joint_ids]
            positions = [positions]
            velocites = [velocites] if velocites is not None else None
        if joint_ids is None:
            joint_ids = list(range(self.getNumJoints(body_id)))
        if positions is None:
            positions = [0.0] * len(joint_ids)
        for k, j in enumerate(joint_ids):
            self._data.qpos[self._qpos_adr(body_id, j)] = positions[k]
            vadr = self._qvel_adr(body_id, j)
            self._data.qvel[vadr] = velocites[k] if velocites is not None else 0.0
        mujoco.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------ #
    # control
    # ------------------------------------------------------------------ #

    def enableJointPositionControlMode(self, body_id, joint_ids=None):
        self._ensure_compiled()
        if joint_ids is None:
            joint_ids = range(self.getNumJoints(body_id))
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        # disable torque actuators, hold current position on the position ones
        for j in joint_ids:
            self._set_ctrl("mot_", body_id, j, 0.0)
            self._set_ctrl("pos_", body_id, j,
                           self._data.qpos[self._qpos_adr(body_id, j)])

    def enableJointTorqueControlMode(self, body_id, joint_ids=None, friction=0):
        self._ensure_compiled()
        if joint_ids is None:
            joint_ids = range(self.getNumJoints(body_id))
        elif isinstance(joint_ids, int):
            joint_ids = [joint_ids]
        # zero the position actuators so only motor torques act
        for j in joint_ids:
            aid = self._actuator_id("pos_", body_id, j)
            self._data.ctrl[aid] = 0.0
            # neutralise the position servo gains so it stops fighting the motors
            self._model.actuator_gainprm[aid, 0] = 0.0
            self._model.actuator_biasprm[aid, 1] = 0.0

    def _set_ctrl(self, prefix, body_id, local_idx, value):
        aid = self._actuator_id(prefix, body_id, local_idx)
        self._data.ctrl[aid] = value

    def jointPositionControl(self, body_id, joint_ids, q_target, qP_target=None,
                             max_torque=10000):
        self._ensure_compiled()
        if isinstance(joint_ids, int):
            self._set_ctrl("pos_", body_id, joint_ids, q_target)
            return
        if isinstance(q_target, np.ndarray):
            q_target = q_target.ravel().tolist()
        for k, j in enumerate(joint_ids):
            self._set_ctrl("pos_", body_id, j, q_target[k])

    def jointTorqueControl(self, body_id, joint_ids, torques):
        self._ensure_compiled()
        if isinstance(joint_ids, int):
            self._set_ctrl("mot_", body_id, joint_ids, torques)
            return
        if isinstance(torques, np.ndarray):
            torques = torques.ravel().tolist()
        for k, j in enumerate(joint_ids):
            self._set_ctrl("mot_", body_id, j, torques[k])

    def computeInverseKinematic(self, body_id, link_id, pos, orn=None):
        raise NotImplementedError(
            "IK is provided by pinocchio/TSID in this project, not the simulator")

    # ------------------------------------------------------------------ #
    # debug draw (best effort / no-ops so callers keep working)
    # ------------------------------------------------------------------ #

    def addUserDebugText(self, *args, **kwargs):
        return -1

    def addLinkDebugFrame(self, *args, **kwargs):
        return []

    def addUserDebugLine(self, *args, **kwargs):
        return -1

    def addGlobalDebugLine(self, *args, **kwargs):
        return -1

    def addGlobalDebugRectancle(self, *args, **kwargs):
        return [-1, -1, -1, -1]

    def addGlobalDebugTrajectory(self, X, Y, Z, color=None, line_ids=None, **kwargs):
        return line_ids if line_ids is not None else []

    def removeDebugItem(self, item_id):
        return

    def addSphereMarker(self, position=[0, 0, 0], orientation=[0, 0, 0, 1],
                        radius=0.02, color=[1, 0, 0, 1]):
        # Visual-only marker; not represented as a body in this backend.
        return -1

    # ------------------------------------------------------------------ #
    # utility
    # ------------------------------------------------------------------ #

    def save(self, filename=None):
        self._ensure_compiled()
        return (self._data.qpos.copy(), self._data.qvel.copy())

    def load(self, state):
        self._ensure_compiled()
        qpos, qvel = state
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        mujoco.mj_forward(self._model, self._data)


if __name__ == '__main__':
    sim = MujocoWrapper(render=False)
    for _ in range(100):
        sim.step()
    print("stepped", sim.simCount(), "sim_time", sim.simTime())
