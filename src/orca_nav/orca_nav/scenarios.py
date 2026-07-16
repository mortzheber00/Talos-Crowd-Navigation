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


def crossing(crowd, gap=6.0):
    """Two streams crossing at 90 degrees through the origin."""
    crowd.add(position=[-gap / 2,  0.0], goal=[gap / 2, 0.0],  on_arrival="reverse")
    crowd.add(position=[ 0.0, -gap / 2], goal=[0.0, gap / 2],  on_arrival="reverse")
    crowd.add(position=[-gap / 2,  1.0], goal=[gap / 2, 1.0],  on_arrival="reverse")
    crowd.add(position=[ 1.0, -gap / 2], goal=[1.0, gap / 2],  on_arrival="reverse")
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


def corridor(crowd, n_each=3, length=8.0, width=1.6, speed=1.2):
    """Two opposing files in a corridor of the given width — a bidirectional
    flow that ORCA typically resolves into lanes."""
    ys = np.linspace(-width / 2, width / 2, n_each)
    for y in ys:
        crowd.add(position=[-length / 2, y], goal=[length / 2, y],
                  max_speed=speed, on_arrival="reverse")
        crowd.add(position=[length / 2, -y], goal=[-length / 2, -y],
                  max_speed=speed, on_arrival="reverse")
    return crowd


def random_crowd(crowd, n=12, extent=5.0, min_start_gap=1.0, seed=0):
    """N pedestrians with random, non-overlapping starts and random goals."""
    rng = np.random.default_rng(seed)
    starts = []
    tries = 0
    while len(starts) < n and tries < 10000:
        tries += 1
        c = rng.uniform(-extent, extent, 2)
        if all(np.hypot(*(c - s)) >= min_start_gap for s in starts):
            starts.append(c)
    for s in starts:
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


def build(crowd, name, **kwargs):
    """Populate ``crowd`` with the named scenario."""
    if name not in REGISTRY:
        raise KeyError(f"unknown scenario '{name}'. options: {list(REGISTRY)}")
    return REGISTRY[name](crowd, **kwargs)
