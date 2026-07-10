"""orca.py
Self-contained 2D ORCA (Optimal Reciprocal Collision Avoidance) solver.

This is a NumPy port of the reference RVO2 algorithm (van den Berg et al.,
"Reciprocal n-body Collision Avoidance", 2011): for one agent it builds one
ORCA half-plane per neighbour and then solves a small 2D linear program for the
new velocity closest to the preferred velocity that lies inside every half-plane
and inside the max-speed disk.

No external dependencies beyond NumPy, so it needs no compilation and runs
anywhere the rest of the project runs. The single reusable entry point is
``orca_new_velocity`` — it is called once per agent per tick. The pedestrian
simulator calls it for every pedestrian; the robot layer can call the exact
same function for the robot.

Everything is 2D and in the world ground plane (x, y), SI units.
"""

import numpy as np

_EPS = 1e-5


class ORCAParams:
    """Tunable ORCA parameters (shared by all agents unless overridden)."""

    def __init__(self,
                 neighbor_dist=4.0,     # [m] ignore neighbours farther than this
                 max_neighbors=10,      # cap on neighbours considered
                 time_horizon=2.5,      # [s] look-ahead for agent-agent avoidance
                 time_step=0.05):       # [s] planner tick (used for the overlap case)
        self.neighbor_dist = neighbor_dist
        self.max_neighbors = max_neighbors
        self.time_horizon = time_horizon
        self.time_step = time_step


# ---------------------------------------------------------------------------
# small 2D vector helpers
# ---------------------------------------------------------------------------

def _det(a, b):
    """2D cross product / determinant: a.x*b.y - a.y*b.x."""
    return a[0] * b[1] - a[1] * b[0]


def _normalize(v):
    n = np.hypot(v[0], v[1])
    if n < _EPS:
        return np.zeros(2)
    return v / n


class _Line:
    """A directed line. Feasible half-plane = velocities left of it:
    det(direction, v - point) >= 0."""
    __slots__ = ("point", "direction")

    def __init__(self, point=None, direction=None):
        self.point = np.zeros(2) if point is None else point
        self.direction = np.zeros(2) if direction is None else direction


# ---------------------------------------------------------------------------
# linear program (RVO2 linearProgram1/2/3)
# ---------------------------------------------------------------------------

def _linear_program1(lines, line_no, radius, opt_velocity, direction_opt):
    """Optimize along a single constraint line. Returns (ok, result)."""
    line = lines[line_no]
    dot = line.point @ line.direction
    discriminant = dot * dot + radius * radius - (line.point @ line.point)
    if discriminant < 0.0:
        # max speed disk does not intersect this line -> infeasible
        return False, None

    sqrt_disc = np.sqrt(discriminant)
    t_left = -dot - sqrt_disc
    t_right = -dot + sqrt_disc

    for i in range(line_no):
        denominator = _det(line.direction, lines[i].direction)
        numerator = _det(lines[i].direction, line.point - lines[i].point)

        if abs(denominator) <= _EPS:
            # lines nearly parallel
            if numerator < 0.0:
                return False, None
            continue

        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)
        if t_left > t_right:
            return False, None

    if direction_opt:
        if opt_velocity @ line.direction > 0.0:
            result = line.point + t_right * line.direction
        else:
            result = line.point + t_left * line.direction
    else:
        t = line.direction @ (opt_velocity - line.point)
        if t < t_left:
            result = line.point + t_left * line.direction
        elif t > t_right:
            result = line.point + t_right * line.direction
        else:
            result = line.point + t * line.direction
    return True, result


def _linear_program2(lines, radius, opt_velocity, direction_opt):
    """Optimize over all constraint lines. Returns (fail_index, result).
    fail_index == len(lines) means fully feasible."""
    if direction_opt:
        result = opt_velocity * radius
    elif (opt_velocity @ opt_velocity) > radius * radius:
        result = _normalize(opt_velocity) * radius
    else:
        result = opt_velocity.copy()

    for i in range(len(lines)):
        if _det(lines[i].direction, lines[i].point - result) > 0.0:
            # current result is on the wrong side of constraint i
            temp = result.copy()
            ok, result = _linear_program1(lines, i, radius, opt_velocity, direction_opt)
            if not ok:
                return i, temp
    return len(lines), result


def _linear_program3(lines, num_obst_lines, begin_line, radius, result):
    """Handle the densely-packed (infeasible) case by relaxing constraints
    uniformly — pushes the result to the 'least bad' velocity."""
    distance = 0.0
    for i in range(begin_line, len(lines)):
        if _det(lines[i].direction, lines[i].point - result) > distance:
            proj_lines = [lines[k] for k in range(num_obst_lines)]
            for j in range(num_obst_lines, i):
                line = _Line()
                determinant = _det(lines[i].direction, lines[j].direction)
                if abs(determinant) <= _EPS:
                    if lines[i].direction @ lines[j].direction > 0.0:
                        continue  # same direction, line j redundant
                    line.point = 0.5 * (lines[i].point + lines[j].point)
                else:
                    line.point = lines[i].point + (
                        _det(lines[j].direction, lines[i].point - lines[j].point)
                        / determinant) * lines[i].direction
                line.direction = _normalize(lines[j].direction - lines[i].direction)
                proj_lines.append(line)

            temp = result.copy()
            new_opt = np.array([-lines[i].direction[1], lines[i].direction[0]])
            fail, result = _linear_program2(proj_lines, radius, new_opt, True)
            if fail < len(proj_lines):
                # should not happen; keep previous best
                result = temp
            distance = _det(lines[i].direction, lines[i].point - result)
    return result


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def orca_new_velocity(position, velocity, radius, max_speed, pref_velocity,
                      neighbors, params, responsibility=0.5):
    """Compute one agent's new, collision-free velocity for this tick.

    Args:
        position (array-like, 2): agent position in the world plane [m].
        velocity (array-like, 2): agent current velocity [m/s].
        radius (float): agent disk radius [m].
        max_speed (float): agent max speed [m/s].
        pref_velocity (array-like, 2): desired velocity toward the goal [m/s].
        neighbors (list): iterable of (pos(2), vel(2), radius) for every other
            agent/obstacle to avoid. Positions/velocities in the same world frame.
        params (ORCAParams): shared parameters.
        responsibility (float): share of the avoidance this agent takes.
            0.5 = reciprocal (both agents share, the ORCA default). Use 1.0 to
            make this agent take full responsibility (e.g. a pedestrian yielding
            entirely to the robot).

    Returns:
        np.ndarray shape (2,): the new velocity, magnitude <= max_speed.
    """
    position = np.asarray(position, dtype=float)[:2]
    velocity = np.asarray(velocity, dtype=float)[:2]
    pref_velocity = np.asarray(pref_velocity, dtype=float)[:2]

    inv_time_horizon = 1.0 / params.time_horizon
    inv_time_step = 1.0 / params.time_step

    # --- select the closest neighbours within range ---
    cand = []
    for (n_pos, n_vel, n_rad) in neighbors:
        n_pos = np.asarray(n_pos, dtype=float)[:2]
        d2 = (n_pos - position) @ (n_pos - position)
        if d2 <= params.neighbor_dist * params.neighbor_dist:
            cand.append((d2, n_pos, np.asarray(n_vel, dtype=float)[:2], float(n_rad)))
    cand.sort(key=lambda c: c[0])
    cand = cand[:params.max_neighbors]

    lines = []
    for (_, n_pos, n_vel, n_rad) in cand:
        rel_position = n_pos - position
        rel_velocity = velocity - n_vel
        dist_sq = rel_position @ rel_position
        combined_radius = radius + n_rad
        combined_radius_sq = combined_radius * combined_radius

        line = _Line()

        if dist_sq > combined_radius_sq:
            # no collision yet: apply the velocity obstacle over the time horizon
            w = rel_velocity - inv_time_horizon * rel_position
            w_len_sq = w @ w
            dot1 = w @ rel_position

            if dot1 < 0.0 and dot1 * dot1 > combined_radius_sq * w_len_sq:
                # project on the cut-off circle
                w_len = np.sqrt(w_len_sq)
                unit_w = w / w_len
                line.direction = np.array([unit_w[1], -unit_w[0]])
                u = (combined_radius * inv_time_horizon - w_len) * unit_w
            else:
                # project on one of the legs of the cone
                leg = np.sqrt(dist_sq - combined_radius_sq)
                if _det(rel_position, w) > 0.0:
                    # left leg
                    line.direction = np.array([
                        rel_position[0] * leg - rel_position[1] * combined_radius,
                        rel_position[0] * combined_radius + rel_position[1] * leg,
                    ]) / dist_sq
                else:
                    # right leg
                    line.direction = -np.array([
                        rel_position[0] * leg + rel_position[1] * combined_radius,
                        -rel_position[0] * combined_radius + rel_position[1] * leg,
                    ]) / dist_sq
                dot2 = rel_velocity @ line.direction
                u = dot2 * line.direction - rel_velocity
        else:
            # already overlapping: resolve within one time step
            w = rel_velocity - inv_time_step * rel_position
            w_len = np.sqrt(w @ w)
            unit_w = w / w_len if w_len > _EPS else np.zeros(2)
            line.direction = np.array([unit_w[1], -unit_w[0]])
            u = (combined_radius * inv_time_step - w_len) * unit_w

        line.point = velocity + responsibility * u
        lines.append(line)

    fail, result = _linear_program2(lines, max_speed, pref_velocity, False)
    if fail < len(lines):
        result = _linear_program3(lines, 0, fail, max_speed, result)
    return result
