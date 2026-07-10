"""debug_viz.py
Lightweight debug drawing for the MuJoCo passive viewer.

The MujocoWrapper's addUserDebugLine / addSphereMarker are no-ops, so for live
debug graphics we draw directly into the passive viewer's overlay scene
(``viewer.user_scn``). Call the draw functions each tick *before* ``sim.step()``
(which syncs the viewer). Everything is a no-op when there is no viewer (headless
runs), so it is safe to leave the calls in.
"""

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover - mujoco is provided by the sim env
    mujoco = None


def _viewer(simulator):
    return getattr(simulator, "_viewer", None)


def draw_scene(simulator, arrows=(), points=(), width=0.02):
    """Draw debug arrows and point/sphere markers in the viewer overlay.

    Args:
        simulator: the MujocoWrapper.
        arrows: iterable of (from_xyz, to_xyz, rgba) tuples.
        points: iterable of (pos_xyz, radius, rgba) tuples (spheres).
        width: arrow shaft width [m].

    Clears the previous overlay each call, so pass *everything* you want visible
    this frame in one call. No-op without a viewer.
    """
    viewer = _viewer(simulator)
    if viewer is None or mujoco is None:
        return
    scn = viewer.user_scn
    scn.ngeom = 0

    for (pos, radius, rgba) in points:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], dtype=float),
            np.asarray(pos, dtype=float), np.eye(3).flatten(),
            np.asarray(rgba, dtype=np.float32))
        scn.ngeom += 1

    for (frm, to, rgba) in arrows:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9),
            np.asarray(rgba, dtype=np.float32))
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_ARROW, width,
            np.asarray(frm, dtype=float), np.asarray(to, dtype=float))
        scn.ngeom += 1


def draw_arrows(simulator, arrows, width=0.02):
    """Back-compat wrapper: draw only arrows."""
    draw_scene(simulator, arrows=arrows, width=width)


def point_marker(pos_xy, z=0.1, radius=0.1, rgba=(1.0, 0.2, 0.2, 1.0)):
    """Build one (pos_xyz, radius, rgba) sphere-marker tuple from a 2D position."""
    return (np.array([pos_xy[0], pos_xy[1], z], dtype=float), float(radius),
            np.asarray(rgba, dtype=np.float32))


def velocity_arrow(origin_xy, vel_xy, z=1.0, scale=3.0,
                   rgba=(0.1, 0.9, 0.1, 1.0), min_len=1e-3):
    """Build one (from, to, rgba) arrow tuple for a 2D velocity drawn at height z.

    Returns None if the velocity is ~zero (nothing to draw). ``scale`` stretches
    the arrow so slow speeds are still visible.
    """
    v = np.asarray(vel_xy, dtype=float)[:2]
    if np.hypot(*v) < min_len:
        return None
    frm = np.array([origin_xy[0], origin_xy[1], z])
    to = frm + scale * np.array([v[0], v[1], 0.0])
    return (frm, to, rgba)


def draw_velocity(simulator, origin_xy, vel_xy, z=1.0, scale=3.0,
                  rgba=(0.1, 0.9, 0.1, 1.0), width=0.02):
    """Convenience: draw a single velocity arrow (clears other overlay arrows)."""
    a = velocity_arrow(origin_xy, vel_xy, z=z, scale=scale, rgba=rgba)
    draw_arrows(simulator, [a] if a is not None else [], width=width)
