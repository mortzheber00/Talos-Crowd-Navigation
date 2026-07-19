"""pedestrian.py
A MuJoCo pedestrian crowd simulator driven by ORCA.

Each pedestrian is a **kinematic disk**: a MuJoCo *mocap* body carrying a
vertical cylinder geom (radius = the ORCA disk radius). Mocap bodies are moved
by directly writing their pose each tick and are *not* affected by physics
(they never fall, gravity ignores them), which is exactly what we want for a
scripted/reactive crowd. Their motion each tick comes from the ORCA solver in
``orca.py``: every pedestrian steers toward its own goal while reciprocally
avoiding the other pedestrians and any external agents (e.g. the robot).

Typical use (standalone crowd, no robot)::

    sim = MujocoWrapper(sim_rate=..., render=True)
    crowd = PedestrianSimulator(sim)
    scenarios.crossing(crowd)          # or add pedestrians manually
    while sim.isRunning():
        crowd.update(sim.stepTime())   # move the disks via ORCA
        sim.step()

Use with the robot: create the ``PedestrianSimulator`` and add pedestrians
**before** the robot is created (adding bodies changes the model topology and
forces a recompile, which resets simulation state). Then each tick call
``crowd.update(dt, external_agents=[(robot_xy, robot_vxy, robot_radius)])`` and
feed ``crowd.states()`` into the robot's own ORCA layer.

Coordinates are the MuJoCo world frame; ORCA works in the (x, y) ground plane.
"""

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover - mujoco is provided by the sim env
    mujoco = None

from orca_nav.orca import orca_new_velocity, ORCAParams


class Pedestrian:
    """State for a single disk pedestrian."""

    def __init__(self, ped_id, position, goal, radius, max_speed,
                 height, color, on_arrival):
        self.id = ped_id
        self.position = np.asarray(position, dtype=float)[:2].copy()
        self.velocity = np.zeros(2)
        self.goal = np.asarray(goal, dtype=float)[:2].copy()
        self.start = self.position.copy()
        self.radius = float(radius)
        self.max_speed = float(max_speed)
        self.height = float(height)
        self.color = list(color)
        self.on_arrival = on_arrival          # 'stop' | 'reverse' | 'remove'
        self.active = True                    # steered + counted as neighbour
        self.name = f"ped_{ped_id}"
        self._mocap_id = -1                   # resolved after compile

    def z_center(self):
        return 0.5 * self.height


class PedestrianSimulator:
    """Spawns and drives a crowd of ORCA disks inside a MuJoCo wrapper."""

    def __init__(self, simulator,
                 params=None,
                 radius=0.3,
                 height=1.7,
                 max_speed=1.3,
                 responsibility=0.3,
                 goal_tol=0.3,
                 pref_perturb=0.0,
                 personal_space=0.1,
                 contact=False,
                 seed=0):
        """
        Args:
            simulator: the MujocoWrapper instance.
            params (ORCAParams): ORCA parameters (defaults are reasonable).
            radius, height, max_speed: defaults for pedestrians (per-ped override
                available in ``add``).
            responsibility (float): ORCA reciprocity share, 0.5 = mutual.
            goal_tol (float): [m] distance at which a pedestrian counts as arrived.
            pref_perturb (float): [m/s] uniform noise added to preferred velocity
                each tick; a small value (~0.05) helps break symmetric deadlocks.
            personal_space (float): [m] extra clearance ORCA keeps between disks
                on top of their radii (planning radius = radius + personal_space/2
                per agent). The rendered disk and collision metrics still use the
                true radius, so this just gives the crowd breathing room.
            contact (bool): if True the disks physically collide with the robot
                (not with each other or the floor). Default False = "ghost" disks,
                so a mis-avoidance never destabilises the robot; collisions are
                still measurable geometrically via ``min_separation``.
            seed (int): RNG seed for the perturbation.
        """
        if mujoco is None:
            raise ImportError("mujoco is required for PedestrianSimulator")
        self.sim = simulator
        self.params = params or ORCAParams()
        self.def_radius = radius
        self.def_height = height
        self.def_max_speed = max_speed
        self.responsibility = responsibility
        self.goal_tol = goal_tol
        self.pref_perturb = pref_perturb
        self.personal_space = personal_space
        self.contact = contact
        self._rng = np.random.default_rng(seed)

        self.peds = []
        self._model_token = None              # detect recompiles to refresh ids

        # sanity: the wrapper must expose the MjSpec-based internals we need
        for attr in ("_spec", "_ensure_compiled"):
            if not hasattr(self.sim, attr):
                raise AttributeError(
                    f"simulator is missing '{attr}'; PedestrianSimulator needs the "
                    "MjSpec-based MujocoWrapper")

    # ------------------------------------------------------------------ #
    # building the crowd
    # ------------------------------------------------------------------ #

    def add(self, position, goal, radius=None, max_speed=None,
            color=None, height=None, on_arrival="stop"):
        """Register one pedestrian and add its mocap disk body to the model.

        Must be called before the model is first compiled (i.e. before the robot
        is created / before the first ``sim.step``). Returns the Pedestrian.
        """
        ped_id = len(self.peds)
        radius = self.def_radius if radius is None else radius
        max_speed = self.def_max_speed if max_speed is None else max_speed
        height = self.def_height if height is None else height
        if color is None:
            # spread hues so pedestrians are visually distinguishable
            hue = (0.13 * ped_id) % 1.0
            color = _hsv_to_rgba(hue, 0.65, 0.95)

        ped = Pedestrian(ped_id, position, goal, radius, max_speed,
                         height, color, on_arrival)
        self.peds.append(ped)

        # add a mocap body with a vertical cylinder geom to the world spec
        body = self.sim._spec.worldbody.add_body(name=ped.name)
        body.mocap = True
        body.pos = [ped.position[0], ped.position[1], ped.z_center()]
        if self.contact:
            # collide with the robot only (robot: contype=2, conaffinity=1;
            # floor: contype=1, conaffinity=2). ped contype=1/conaffinity=2 ->
            #   ped-robot: (1&1)|(2&2) != 0 -> collide
            #   ped-floor: (1&2)|(1&2) == 0 -> no ; ped-ped likewise no
            contype, conaffinity = 1, 2
        else:
            contype, conaffinity = 0, 0   # visual only ("ghost")
        body.add_geom(
            name=f"{ped.name}_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[ped.radius, ped.z_center(), 0.0],
            rgba=ped.color,
            contype=contype,
            conaffinity=conaffinity,
        )

        # force the (re)compile to pick up the new body on next ensure/step
        self.sim._model = None
        if hasattr(self.sim, "_dirty"):
            self.sim._dirty = True
        return ped

    # ------------------------------------------------------------------ #
    # per-tick update
    # ------------------------------------------------------------------ #

    def _refresh_ids(self):
        """Resolve each pedestrian's mocap index against the current model.
        Re-runs whenever the compiled model object has changed (e.g. after the
        robot was added and triggered a recompile)."""
        self.sim._ensure_compiled()
        m = self.sim._model
        if self._model_token is id(m):
            return
        for ped in self.peds:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ped.name)
            ped._mocap_id = int(m.body_mocapid[bid]) if bid >= 0 else -1
        self._model_token = id(m)

    def update(self, dt, external_agents=()):
        """Advance the crowd by ``dt`` seconds.

        Args:
            dt (float): timestep [s] (e.g. ``sim.stepTime()``).
            external_agents: iterable of (position2, velocity2, radius) for agents
                that are not pedestrians but should be avoided (e.g. the robot).
        """
        self._refresh_ids()
        half = 0.5 * self.personal_space
        # planning radii include half the personal-space buffer per agent, so any
        # pair keeps ~personal_space of clearance beyond their true radii
        ext = [(np.asarray(p, float)[:2], np.asarray(v, float)[:2], float(r) + half)
               for (p, v, r) in external_agents]

        # 1) compute every active pedestrian's new velocity from a common snapshot
        new_vel = {}
        for ped in self.peds:
            if not ped.active:
                continue
            neighbors = [(o.position, o.velocity, o.radius + half)
                         for o in self.peds if o is not ped and o.active]
            neighbors.extend(ext)
            pref = self._preferred_velocity(ped)
            new_vel[ped.id] = orca_new_velocity(
                ped.position, ped.velocity, ped.radius + half, ped.max_speed,
                pref, neighbors, self.params, self.responsibility)

        # 2) integrate and push poses into MuJoCo
        m, d = self.sim._model, self.sim._data
        for ped in self.peds:
            if ped.active and ped.id in new_vel:
                ped.velocity = new_vel[ped.id]
                ped.position = ped.position + ped.velocity * dt
                self._check_arrival(ped)
            if ped._mocap_id >= 0:
                d.mocap_pos[ped._mocap_id] = [ped.position[0], ped.position[1],
                                              ped.z_center()]
                d.mocap_quat[ped._mocap_id] = _yaw_quat_wxyz(ped.velocity)

    def _preferred_velocity(self, ped):
        to_goal = ped.goal - ped.position
        dist = np.hypot(to_goal[0], to_goal[1])
        if dist < 1e-6:
            return np.zeros(2)
        # linear slow-down inside one "braking" radius so disks settle on goals
        brake = max(self.goal_tol, ped.max_speed * 0.5)
        speed = ped.max_speed * min(1.0, dist / brake)
        v = to_goal / dist * speed
        if self.pref_perturb > 0.0:
            v = v + self._rng.uniform(-self.pref_perturb, self.pref_perturb, 2)
        return v

    def _check_arrival(self, ped):
        if np.hypot(*(ped.goal - ped.position)) > self.goal_tol:
            return
        if ped.on_arrival == "reverse":
            ped.goal, ped.start = ped.start.copy(), ped.goal.copy()
        elif ped.on_arrival == "remove":
            ped.active = False
            ped.velocity = np.zeros(2)
        else:  # 'stop'
            ped.active = False
            ped.velocity = np.zeros(2)

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #

    def states(self, include_inactive=True):
        """Ground-truth crowd state for the robot's ORCA layer:
        list of (position(2), velocity(2), radius)."""
        return [(p.position.copy(), p.velocity.copy(), p.radius)
                for p in self.peds if include_inactive or p.active]

    def num_active(self):
        return sum(1 for p in self.peds if p.active)

    def all_arrived(self):
        return all(not p.active for p in self.peds)

    def min_separation(self, external_agents=()):
        """Smallest surface-to-surface gap in the crowd right now (negative =
        overlap). Useful as a safety/collision metric in tests."""
        pts = [(p.position, p.radius) for p in self.peds]
        pts += [(np.asarray(p, float)[:2], float(r)) for (p, _, r) in external_agents]
        m = np.inf
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                gap = np.hypot(*(pts[i][0] - pts[j][0])) - pts[i][1] - pts[j][1]
                m = min(m, gap)
        return m


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _yaw_quat_wxyz(velocity):
    """Quaternion (w, x, y, z) that yaws the disk to face its velocity."""
    if np.hypot(velocity[0], velocity[1]) < 1e-6:
        return [1.0, 0.0, 0.0, 0.0]
    yaw = np.arctan2(velocity[1], velocity[0])
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def _hsv_to_rgba(h, s, v, a=1.0):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t),
               (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return [r, g, b, a]
