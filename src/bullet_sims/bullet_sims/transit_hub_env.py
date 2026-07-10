"""Das dynamische Nadelöhr — ein modernes Transit-Terminal für ORCA-Stresstests.

Programmatically built MuJoCo scene (pure ``mujoco.MjSpec``, no XML files):

* visual ambience: gradient skybox, tiled floor texture, strong shadow-casting
  sun light plus a soft fill light,
* static set pieces: a 4-pillar slalom and a glass-look bottleneck (two
  diagonal walls narrowing the field to a ~1.8 m passage),
* 5 dynamic capsule agents on a ``freejoint`` each, with a red "nose" sphere
  marking their facing direction — 3 approach the robot head-on through the
  bottleneck, 2 cross its path at 90 degrees.

Drop-in replacement for :class:`bullet_sims.crowd_env.CrowdEnv`::

    env = TransitHubEnv(simulator)            # builds + recompiles the scene
    robot.set_obstacle_geoms(env.geom_names)  # agent capsules for perception
    ...
    while running:
        env.update(simulator.simTime())       # once per physics step
        simulator.step()

The agents carry a freejoint (as required for later ORCA integration) but are
driven kinematically: ``update`` writes their freejoint qpos/qvel directly, so
they never topple and follow the scripted paths exactly.  Like the CrowdEnv
cylinders they are non-colliding (``contype=0``/``conaffinity=0``) yet fully
visible to the depth and segmentation cameras.  The static structures *do*
collide with the robot (same collision groups as the floor), so a failed
avoidance manoeuvre has consequences.
"""

import numpy as np
import mujoco


def _yaw_quat_wxyz(yaw):
    """Quaternion (MuJoCo wxyz order) for a rotation of ``yaw`` about world z."""
    return [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]


class TransitHubEnv:
    """Builds the transit-terminal scene into a MujocoWrapper via MjSpec and
    animates the 5 crowd agents along scripted 2D paths."""

    N = 5
    RADIUS = 0.25          # m, roughly a human torso
    HALF_HEIGHT = 0.875    # m -> 1.75 m tall, human-like (capsule total height)
    # vivid, well-separated colors for the segmentation camera; deliberately no
    # pure red so the red nose marker stands out on every agent
    COLORS = [
        (1.00, 0.55, 0.10, 1.0),   # orange
        (0.10, 0.85, 0.20, 1.0),   # green
        (0.15, 0.40, 0.95, 1.0),   # blue
        (0.95, 0.85, 0.10, 1.0),   # yellow
        (0.85, 0.20, 0.85, 1.0),   # magenta
    ]
    NOSE_RGBA = (0.95, 0.05, 0.05, 1.0)

    # bottleneck layout: the two glass walls pinch the field at GATE_X down to
    # a passage of GATE_WIDTH between their inner ends
    GATE_X = 8.5
    GATE_WIDTH = 1.8
    WALL_LENGTH = 4.0      # m
    WALL_HALF_HEIGHT = 1.25
    WALL_ANGLE = np.radians(30.0)  # opening angle of the funnel vs. the y-axis

    PILLAR_RADIUS = 0.30
    PILLAR_HALF_HEIGHT = 1.5
    # staggered left/right of the robot's main path (y = 0, walking along +x)
    PILLAR_XY = [(2.0, -0.95), (3.5, 0.95), (5.0, -0.95), (6.5, 0.95)]

    def __init__(self, simulator, paths=None):
        """
        Args:
            simulator: MujocoWrapper instance (scene still extendable)
            paths: optional list of N callables ``t -> (x, y)`` [m]; defaults
                to the head-on / crossing choreography described in the module
                docstring.
        """
        self.sim = simulator
        self.body_names = [f'hub_agent_{i}' for i in range(self.N)]
        self.geom_names = [f'hub_agent_{i}_geom' for i in range(self.N)]
        self.static_geom_names = []

        self._paths = paths if paths is not None else self._default_paths()
        if len(self._paths) != self.N:
            raise ValueError(f'expected {self.N} paths, got {len(self._paths)}')
        self._t = 0.0
        # last known heading per agent, used when the path velocity vanishes
        # (turning points); initialised from the path direction at t=0+
        self._heading = [self._path_heading(i, 0.0, fallback=np.pi)
                         for i in range(self.N)]

        spec = simulator._spec
        self._build_ambience(spec)
        self._build_static(spec)
        self._build_agents(spec)

        # same flow as CrowdEnv: mark dirty and recompile in place
        simulator._dirty = True
        simulator._ensure_compiled()

    # ------------------------------------------------------------------ #
    # scene construction
    # ------------------------------------------------------------------ #

    def _build_ambience(self, spec):
        """Skybox gradient, tiled floor material, sun + fill lighting."""
        # skybox: deep night blue at the zenith fading into a light horizon grey
        spec.add_texture(
            name='hub_skybox',
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.06, 0.10, 0.24],   # zenith
            rgb2=[0.72, 0.76, 0.82],   # horizon
            width=512, height=3072)

        # floor: light tiles with dark grout lines -> strong, regular contrast
        # edges for the depth camera
        spec.add_texture(
            name='hub_floor_tex',
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.78, 0.80, 0.84],
            rgb2=[0.62, 0.66, 0.72],
            mark=mujoco.mjtMark.mjMARK_EDGE,
            markrgb=[0.35, 0.38, 0.42],  # soft grout lines (hard ones moire at distance)
            width=512, height=512)
        floor_mat = spec.add_material(
            name='hub_floor_mat',
            texrepeat=[0.5, 0.5],   # with texuniform: 1 m tiles
            texuniform=True,
            reflectance=0.15,
            specular=0.3,
            shininess=0.4)
        floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'hub_floor_tex'
        for g in spec.geoms:
            if g.name == 'floor':
                g.material = 'hub_floor_mat'
                g.rgba = [1.0, 1.0, 1.0, 1.0]  # untinted texture
                # render the plane as a finite 30x30 m terminal platform (the
                # collision plane stays infinite regardless of the size)
                g.size = [15.0, 15.0, 1.0]
                break

        # the directional-light shadow volume scales with stat.extent; the
        # auto-computed extent (~13 m) leaves the volume edge in the middle of
        # the platform, which shows up as a speckled shadow-acne band on the
        # far floor. Enlarging the extent pushes the edge past the platform.
        # (Also grows the render far plane: zfar = extent * vis.map.zfar.)
        spec.stat.extent = 45.0

        # glass material for the bottleneck walls (futuristic look)
        spec.add_material(
            name='hub_glass_mat',
            reflectance=0.4,
            specular=0.9,
            shininess=0.9)

        # main sun: strong, low-angle directional light with soft shadows
        spec.worldbody.add_light(
            name='hub_sun',
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.75, 0.73, 0.68],
            specular=[0.3, 0.3, 0.3],
            pos=[0.0, 0.0, 12.0],
            dir=[0.35, -0.5, -1.0],
            castshadow=True)
        # soft fill from the opposite side so shadowed faces stay readable
        spec.worldbody.add_light(
            name='hub_fill',
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
            diffuse=[0.18, 0.19, 0.22],
            specular=[0.0, 0.0, 0.0],
            pos=[0.0, 0.0, 8.0],
            dir=[-0.4, 0.5, -1.0],
            castshadow=False)
        # the wrapper's default "sun" would double-expose the scene and cast a
        # second, conflicting shadow -> demote it to a shadowless helper
        for light in spec.lights:
            if light.name == 'sun':
                light.diffuse = [0.25, 0.25, 0.25]
                light.castshadow = False

        # high-res shadow map for soft, clean shadow edges; dim headlight so
        # the directional lighting dominates
        spec.visual.quality.shadowsize = 8192
        spec.visual.headlight.diffuse = [0.25, 0.25, 0.25]
        spec.visual.headlight.ambient = [0.12, 0.12, 0.12]

    def _add_static_geom(self, spec, name, **kwargs):
        """Static scenery geom, colliding with the robot like the floor does
        (floor: contype=1/conaffinity=2, robot geoms: contype=2/conaffinity=1)."""
        geom = spec.worldbody.add_geom(
            name=name, contype=1, conaffinity=2, **kwargs)
        self.static_geom_names.append(name)
        return geom

    def _build_static(self, spec):
        """Pillar slalom + the glass bottleneck walls."""
        for k, (x, y) in enumerate(self.PILLAR_XY):
            self._add_static_geom(
                spec, f'static_pillar_{k + 1}',
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[self.PILLAR_RADIUS, self.PILLAR_HALF_HEIGHT, 0.0],
                pos=[x, y, self.PILLAR_HALF_HEIGHT],
                rgba=[0.25, 0.25, 0.28, 1.0])   # matte dark grey

        # two thin diagonal walls funnelling everything into the gate: each
        # wall's inner end sits at (GATE_X, +-GATE_WIDTH/2) and the wall runs
        # backwards-outwards from there (towards the robot's spawn side)
        half_gap = self.GATE_WIDTH / 2.0
        for k, side in enumerate((+1.0, -1.0)):
            yaw = side * (np.pi - self.WALL_ANGLE)   # long axis direction
            direction = np.array([np.cos(yaw), np.sin(yaw)])
            inner_end = np.array([self.GATE_X, side * half_gap])
            center = inner_end + direction * (self.WALL_LENGTH / 2.0)
            self._add_static_geom(
                spec, f'static_wall_{k + 1}',
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.WALL_LENGTH / 2.0, 0.06, self.WALL_HALF_HEIGHT],
                pos=[center[0], center[1], self.WALL_HALF_HEIGHT],
                quat=_yaw_quat_wxyz(yaw),
                material='hub_glass_mat',
                rgba=[0.2, 0.6, 0.8, 0.4])       # transparent glass look

    def _build_agents(self, spec):
        """5 free-floating capsule agents with a red nose marking the front."""
        for i, bname in enumerate(self.body_names):
            x0, y0 = self._paths[i](0.0)
            body = spec.worldbody.add_body(
                name=bname,
                pos=[x0, y0, self.HALF_HEIGHT],
                quat=_yaw_quat_wxyz(self._heading[i]))
            body.add_freejoint(name=f'{bname}_free')
            body.add_geom(
                name=self.geom_names[i],
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=[self.RADIUS, self.HALF_HEIGHT - self.RADIUS, 0.0],
                rgba=list(self.COLORS[i % len(self.COLORS)]),
                contype=0, conaffinity=0)
            # nose: small sphere on the local +x face at head height, shows the
            # facing direction at a glance in the viewer
            body.add_geom(
                name=f'{bname}_nose',
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.06, 0.0, 0.0],
                pos=[self.RADIUS + 0.02, 0.0, 0.45],
                rgba=list(self.NOSE_RGBA),
                contype=0, conaffinity=0)

    # ------------------------------------------------------------------ #
    # scripted choreography
    # ------------------------------------------------------------------ #

    def _default_paths(self):
        """3 agents shuttle head-on through the bottleneck, 2 cross at 90 deg."""
        def shuttle(p0, p1, period, phase=0.0):
            # smooth back-and-forth between p0 (at t=0, phase=0) and p1 with
            # zero velocity at both turning points
            p0 = np.asarray(p0, dtype=float)
            p1 = np.asarray(p1, dtype=float)
            w = 2.0 * np.pi / period
            def path(t):
                s = 0.5 * (1.0 - np.cos(w * t + phase))
                return tuple(p0 + (p1 - p0) * s)
            return path

        lane = self.GATE_WIDTH / 2.0 - self.RADIUS - 0.20  # stay inside the gate
        return [
            # head-on: spawn behind the bottleneck, walk through the gate
            # towards the robot and back
            shuttle((11.5, -lane), (5.0, -lane), 26.0),
            shuttle((12.0, 0.0), (4.5, 0.0), 30.0),
            shuttle((11.0, lane), (5.5, lane), 22.0),
            # crossers: cut the robot's path at 90 degrees, between the pillars
            shuttle((2.75, -3.0), (2.75, 3.0), 16.0),
            shuttle((5.75, 3.0), (5.75, -3.0), 18.0),
        ]

    def _path_heading(self, i, t, fallback=0.0, dt=1e-3):
        xp, yp = self._paths[i](t + dt)
        xm, ym = self._paths[i](t - dt)
        vx, vy = (xp - xm) / (2.0 * dt), (yp - ym) / (2.0 * dt)
        if np.hypot(vx, vy) < 1e-4:
            return fallback
        return float(np.arctan2(vy, vx))

    # ------------------------------------------------------------------ #
    # runtime interface (CrowdEnv-compatible)
    # ------------------------------------------------------------------ #

    def update(self, t):
        """Write the agents' freejoint state for sim time ``t``. Call once per
        physics step. Names are re-resolved on every call, so the method
        survives model recompiles (which renumber bodies and joints)."""
        self._t = float(t)
        m, d = self.sim._model, self.sim._data
        for i, bname in enumerate(self.body_names):
            x, y = self._paths[i](self._t)
            xp, yp = self._paths[i](self._t + 1e-3)
            xm, ym = self._paths[i](self._t - 1e-3)
            vx, vy = (xp - xm) / 2e-3, (yp - ym) / 2e-3
            if np.hypot(vx, vy) >= 1e-4:
                self._heading[i] = float(np.arctan2(vy, vx))

            jid = int(m.body_jntadr[m.body(bname).id])   # the freejoint
            qadr = int(m.jnt_qposadr[jid])
            dadr = int(m.jnt_dofadr[jid])
            d.qpos[qadr:qadr + 3] = (x, y, self.HALF_HEIGHT)
            d.qpos[qadr + 3:qadr + 7] = _yaw_quat_wxyz(self._heading[i])
            d.qvel[dadr:dadr + 3] = (vx, vy, 0.0)
            d.qvel[dadr + 3:dadr + 6] = 0.0

    def ground_truth(self, dt=1e-3):
        """Exact state per agent for validating the perception pipeline:
        ``{geom_name: {'x', 'y', 'r', 'vx', 'vy'}}`` with the velocity from a
        central finite difference of the path at the last ``update`` time."""
        out = {}
        for i, gname in enumerate(self.geom_names):
            x, y = self._paths[i](self._t)
            xp, yp = self._paths[i](self._t + dt)
            xm, ym = self._paths[i](self._t - dt)
            out[gname] = dict(x=x, y=y, r=self.RADIUS,
                              vx=(xp - xm) / (2.0 * dt),
                              vy=(yp - ym) / (2.0 * dt))
        return out
