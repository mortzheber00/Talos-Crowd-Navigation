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

        # same flow as spawn_static_box: mark dirty and recompile in place
        simulator._dirty = True
        simulator._ensure_compiled()

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
