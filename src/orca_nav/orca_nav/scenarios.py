"""scenarios.py
Named, repeatable crowd setups. Each function takes a PedestrianSimulator and
populates it with pedestrians (positions + goals). They return the simulator so
calls can be chained/inspected.

All scenarios are laid out around the world origin so a robot walking along +x
(as in walking.py) meets the crowd. Distances are in metres.
"""

import numpy as np


def head_on(crowd, gap=6.0, offset=0.2):
    """Two pedestrians walking straight at each other (slightly offset so the
    perfectly-symmetric ORCA deadlock is avoided)."""
    crowd.add(position=[-gap / 2,  0.0],    goal=[gap / 2,  offset], on_arrival="reverse")
    crowd.add(position=[ gap / 2,  offset], goal=[-gap / 2, 0.0],    on_arrival="reverse")
    return crowd


def crossing(crowd, n=6, span=6.0, x_lo=1.0, x_hi=3.5, speed=1.2):
    """A perpendicular stream cutting across the robot's +x route.

    ``n`` pedestrians walk along +/-y (alternating direction so it reads as a
    two-way crossing) at ``n`` x-stations spread over the robot's path between
    ``x_lo`` and ``x_hi``. Centred on the route rather than on the origin so the
    robot meets the flow mid-walk instead of at its own spawn -- the old
    origin-centred layout let the robot clear the intersection before the crowd
    arrived.
    """
    xs = np.linspace(x_lo, x_hi, n)
    for k, x in enumerate(xs):
        y0, y1 = (span / 2, -span / 2) if k % 2 == 0 else (-span / 2, span / 2)
        crowd.add(position=[x, y0], goal=[x, y1],
                  max_speed=speed, on_arrival="reverse")
    return crowd


def circle(crowd, n=8, radius=4.0, jitter=0.15):
    """N pedestrians on a circle, each heading to the opposite point. A hard,
    near-symmetric case; small jitter keeps it from locking up."""
    rng = np.random.default_rng(0)
    for k in range(n):
        a = 2 * np.pi * k / n
        p = np.array([radius * np.cos(a), radius * np.sin(a)])
        p = p + rng.uniform(-jitter, jitter, 2)
        crowd.add(position=p.tolist(), goal=(-p).tolist(), on_arrival="reverse")
    return crowd


def corridor(crowd, n_each=3, length=8.0, width=1.6, fast=1.3, slow=0.5):
    """'HBF Platform': two opposing files along the robot's axis with MIXED
    walking speeds.

    Half the pedestrians walk faster than the robot can (``fast``, commuters
    catching a train) and half stroll at roughly its own achievable pace
    (``slow``), so the robot has to both be overtaken by the quick ones and
    thread around the slow ones. The speed is alternated per lane and flipped
    between the two directions so each file is a mix, not a fast lane and a slow
    lane.
    """
    ys = np.linspace(-width / 2, width / 2, n_each)
    for k, y in enumerate(ys):
        spd_fwd = fast if k % 2 == 0 else slow   # file walking +x (with the robot)
        spd_bwd = slow if k % 2 == 0 else fast   # file walking -x (against it)
        crowd.add(position=[-length / 2, y], goal=[length / 2, y],
                  max_speed=spd_fwd, on_arrival="reverse")
        crowd.add(position=[length / 2, -y], goal=[-length / 2, -y],
                  max_speed=spd_bwd, on_arrival="reverse")
    return crowd


def random_crowd(crowd, n=12, extent=5.0, min_start_gap=1.0, seed=0,
                 clear_radius=0.0, clear_centre=(0.0, 0.0),
                 goal_share=0.0, shared_goal=None):
    """'Marienplatz': N pedestrians with random, non-overlapping starts.

    ``clear_radius`` keeps starts out of a disk around ``clear_centre`` (the
    robot's spawn). Without it a pedestrian can materialise on top of the robot,
    which scores as a collision before the run has begun -- a spawn artefact, not
    an avoidance failure.

    ``goal_share`` is the fraction of pedestrians sent to ``shared_goal`` (the
    robot's own goal, jittered) instead of a random one. With a purely random
    goal most pedestrians drift off *behind* the robot and it walks to an empty
    target -- a de-facto second control. Routing a share of the crowd to the same
    goal makes them converge on the robot's destination like a crowd crossing a
    busy square, so encounters near the goal are guaranteed rather than left to
    chance.
    """
    rng = np.random.default_rng(seed)
    centre = np.asarray(clear_centre, dtype=float)

    # A dense request can be geometrically impossible at the requested spacing
    # (48 pedestrians 1 m apart do not fit in a 6x6 box). Rather than silently
    # returning a smaller crowd -- which would quietly turn a density sweep into
    # a sweep of "however many happened to fit" -- relax the spacing until they
    # do fit, and say so. The floor is one pedestrian diameter, below which they
    # would start already overlapping.
    floor = 2.0 * getattr(crowd, "def_radius", 0.3)
    gap = float(min_start_gap)
    starts = []
    while True:
        starts, tries = [], 0
        while len(starts) < n and tries < 20000:
            tries += 1
            c = rng.uniform(-extent, extent, 2)
            if clear_radius > 0.0 and np.hypot(*(c - centre)) < clear_radius:
                continue
            if all(np.hypot(*(c - s)) >= gap for s in starts):
                starts.append(c)
        if len(starts) >= n or gap <= floor + 1e-9:
            break
        gap = max(floor, 0.85 * gap)
        print(f"[scenarios] {n} pedestrians do not fit at {min_start_gap:.2f} m "
              f"spacing in a {2*extent:.1f} m box; relaxing to {gap:.2f} m")
    if len(starts) < n:
        print(f"[scenarios] WARNING: placed only {len(starts)}/{n} pedestrians "
              f"at the {floor:.2f} m floor -- the box is saturated")

    # Send the first ``n_shared`` pedestrians to the shared goal. They are drawn
    # in start order, which is itself random, so this is an unbiased subset.
    n_shared = int(round(goal_share * len(starts))) if shared_goal is not None else 0
    shared = np.asarray(shared_goal, dtype=float) if shared_goal is not None else None
    for i, s in enumerate(starts):
        if i < n_shared:
            g = shared + rng.uniform(-0.6, 0.6, 2)   # cluster, not a single point
        else:
            g = rng.uniform(-extent, extent, 2)
        crowd.add(position=s.tolist(), goal=g.tolist(), on_arrival="reverse")
    return crowd


def empty(crowd):
    """No pedestrians at all: an isolated gait test with the crowd/ORCA layer
    still wired up, so v_cmd comes only from the goal."""
    return crowd


# convenience registry so a demo/CLI can pick by name
REGISTRY = {
    "empty": empty,
    "head_on": head_on,
    "crossing": crossing,
    "circle": circle,
    "corridor": corridor,
    "random": random_crowd,
}

# audience-friendly labels for slides/plots (the registry keys stay stable so
# the sweeps and CLI keep working).
DISPLAY_NAMES = {
    "empty": "Empty",
    "head_on": "Head-on",
    "crossing": "Crossing",
    "circle": "Circle",
    "corridor": "HBF Platform",
    "random": "Marienplatz",
}


def build(crowd, name, **kwargs):
    """Populate ``crowd`` with the named scenario."""
    if name not in REGISTRY:
        raise KeyError(f"unknown scenario '{name}'. options: {list(REGISTRY)}")
    return REGISTRY[name](crowd, **kwargs)
