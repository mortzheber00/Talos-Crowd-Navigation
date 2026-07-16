"""Kinematically scripted "crowd" of cylinder agents for the MuJoCo scene.

The cylinders stand in for humans during local-navigation experiments (ORCA).
They are spawned as mocap bodies: no joints, no dynamics — their pose is
written directly via ``data.mocap_pos`` every physics step, which gives
perfectly scripted motion at zero physics cost. With ``contype=0`` /
``conaffinity=0`` they never collide with the robot but remain fully visible
to the depth and segmentation cameras.
"""

import numpy as np
import mujoco

from simulator import simulator


class CrowdEnv:
    """Spawns N cylinder agents into a MujocoWrapper scene and moves them
    along scripted 2D paths.

    Usage::

        crowd = CrowdEnv(simulator)          # spawns + recompiles the model
        robot.set_obstacle_geoms(crowd.geom_names)
        ...
        while running:
            crowd.update(simulator.simTime())   # once per physics step
            simulator.step()
    """

    N = 5
    RADIUS = 0.25          # m, roughly a human torso
    HALF_HEIGHT = 0.875    # m -> 1.75 m tall, human-like
    COLORS = [
        (0.9, 0.2, 0.2, 1.0),
        (0.2, 0.8, 0.2, 1.0),
        (0.2, 0.4, 0.9, 1.0),
        (0.9, 0.8, 0.1, 1.0),
        (0.8, 0.3, 0.8, 1.0),
    ]

    def __init__(self, simulator, paths=None):
        """
        Args:
            simulator: MujocoWrapper instance (scene is still extendable)
            paths: optional list of N callables ``t -> (x, y)`` [m]; defaults
                to a mix of linear back-and-forth and circular paths that stay
                inside the head camera's field of view.
        """
        self.sim = simulator
        self.body_names = [f'crowd_cyl_{i}' for i in range(self.N)]
        self.geom_names = [f'crowd_cyl_{i}_geom' for i in range(self.N)]
        self._paths = paths if paths is not None else self._default_paths()
        if len(self._paths) != self.N:
            raise ValueError(f'expected {self.N} paths, got {len(self._paths)}')
        self._t = 0.0

        for i, (bname, gname) in enumerate(zip(self.body_names, self.geom_names)):
            x0, y0 = self._paths[i](0.0)
            body = simulator._spec.worldbody.add_body(
                name=bname, pos=[x0, y0, self.HALF_HEIGHT])
            body.mocap = True
            body.add_geom(
                name=gname,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[self.RADIUS, self.HALF_HEIGHT, 0.0],
                rgba=list(self.COLORS[i % len(self.COLORS)]),
                contype=0, conaffinity=0,
            )

        spec = simulator._spec
        self._build_ambience(spec)

        # same flow as spawn_static_box: mark dirty and recompile in place
        simulator._dirty = True
        simulator._ensure_compiled()

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

    def _default_paths(self):
        # Keep the agents 2-5 m in front of the robot (camera looks along
        # world +X at spawn) and inside the ~58 deg vertical FOV.
        def linear(cx, cy, ax, ay, period, phase=0.0):
            w = 2.0 * np.pi / period
            return lambda t: (cx + ax * np.sin(w * t + phase),
                              cy + ay * np.sin(w * t + phase))

        def circle(cx, cy, r, period, phase=0.0):
            w = 2.0 * np.pi / period
            return lambda t: (cx + r * np.cos(w * t + phase),
                              cy + r * np.sin(w * t + phase))

        return [
            linear(3.0, -1.5, 0.0, 1.2, 12.0),
            linear(4.0, 1.5, 0.0, -1.2, 10.0),
            circle(3.5, 0.0, 1.0, 16.0),
            linear(2.5, 0.5, 0.8, 0.0, 14.0),
            circle(4.5, -0.5, 0.7, 12.0, np.pi),
        ]

    def update(self, t):
        """Write the mocap poses for sim time ``t``. Call once per physics
        step. Names are re-resolved on every call, so the method survives
        model recompiles (which renumber body/mocap ids)."""
        self._t = float(t)
        m, d = self.sim._model, self.sim._data
        for i, bname in enumerate(self.body_names):
            mid = m.body_mocapid[m.body(bname).id]
            x, y = self._paths[i](self._t)
            d.mocap_pos[mid] = (x, y, self.HALF_HEIGHT)

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
