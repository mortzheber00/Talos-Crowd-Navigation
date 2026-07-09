import numpy as np 
from scipy.optimize import minimizet 


def orca_half_plane(ego, other, tau, dt, weight=0.5):
    """
    weight: responsibility share assigned to ego.
            0.5 = standard reciprocal split (both agents ORCA-compliant).
            1.0 = ego takes full responsibility (non-reciprocal neighbors,
                  e.g. SFM-driven pedestrians, or static obstacle pseudo-agents).
    """
    rel_pos = other["position"] - ego["position"]
    rel_vel = ego["velocity"] - other["velocity"]
    combined_radius = ego["radius"] + other["radius"]
    dist_sq = float(rel_pos @ rel_pos)
    combined_radius_sq = combined_radius ** 2
 
    if dist_sq > combined_radius_sq:
        # Not currently overlapping -> truncated-cone VO, horizon tau.
        w = rel_vel - rel_pos / tau
        w_len_sq = float(w @ w)
        dot = float(w @ rel_pos)
 
        if dot < 0.0 and dot ** 2 > combined_radius_sq * w_len_sq:
            # Closest to the cap (cut-off circle).
            w_len = np.sqrt(w_len_sq)
            unit_w = w / w_len if w_len > 1e-9 else np.array([1.0, 0.0])
            u = (combined_radius / tau - w_len) * unit_w
            normal = unit_w
        else:
            # Closest to one of the two cone legs.
            leg = np.sqrt(max(dist_sq - combined_radius_sq, 0.0))
            if cross(rel_pos, w) > 0.0:
                direction = np.array([
                    rel_pos[0] * leg - rel_pos[1] * combined_radius,
                    rel_pos[0] * combined_radius + rel_pos[1] * leg,
                ]) / dist_sq
            else:
                direction = -np.array([
                    rel_pos[0] * leg + rel_pos[1] * combined_radius,
                    -rel_pos[0] * combined_radius + rel_pos[1] * leg,
                ]) / dist_sq
            u = (rel_vel @ direction) * direction - rel_vel
            normal = perp(direction)
            if normal @ u < 0.0:
                normal = -normal
    else:
        # Already overlapping -> use dt instead of tau, more urgent correction.
        w = rel_vel - rel_pos / dt
        w_len = np.linalg.norm(w)
        unit_w = w / w_len if w_len > 1e-9 else np.array([1.0, 0.0])
        u = (combined_radius / dt - w_len) * unit_w
        normal = unit_w
 
    point = ego["velocity"] + weight * u
    return point, normal


def solve_new_velocity(ego, half_planes):
    pref = ego["pref_velocity"]
    v0 = pref.copy()
 
    def objective(v):
        d = v - pref
        return 0.5 * float(d @ d)
 
    def objective_grad(v):
        return v - pref
 
    constraints = [
        {"type": "ineq",
         "fun": (lambda v, p=point, n=normal: float(n @ (v - p))),
         "jac": (lambda v, n=normal: n)}
        for point, normal in half_planes
    ]
 
    result = minimize(objective, v0, jac=objective_grad,
                       constraints=constraints, method="SLSQP")
 
    v_new = result.x if result.success else ego["velocity"]  # fallback: hold velocity
    speed = np.linalg.norm(v_new)
    if speed > ego["max_speed"]:
        v_new = v_new / speed * ego["max_speed"]
    return v_new


def nearby(ego, candidates, cutoff):
    return [a for a in candidates
            if np.linalg.norm(a["position"] - ego["position"]) <= cutoff]

 
def perp(v):
    """90-degree CCW rotation."""
    return np.array([-v[1], v[0]])
 
 
def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])



def orca_step(
    ego,
    pedestrians, 
    obstacle_agents, 
    tau, 
    dt, 
    neighbor_cutoff, 
    pedestrian_weight=1.0
    ): 
    """
    Returns new velocity for robot 
    """

    neighbors = nearby(ego, pedestrians, neighbor_cutoff)
    obstacles = nearby(ego, obstacle_agents, neighbor_cutoff)

    half_planes = [orca_half_plane(ego, ped, tau, dt, weight=pedestrian_weight)
                   for ped in neighbors]
    half_planes += [orca_half_plane(ego, obs, tau, dt, weight=1.0)
                     for obs in obstacles]
 
    return solve_new_velocity(ego, half_planes)


# ==========================================================================
# Testing 
# ==========================================================================
 
def make_agent(position, velocity, radius, max_speed=1.0, pref_velocity=None):
    return {
        "position": np.asarray(position, dtype=float),
        "velocity": np.asarray(velocity, dtype=float),
        "radius": radius,
        "max_speed": max_speed,
        "pref_velocity": (np.asarray(pref_velocity, dtype=float)
                          if pref_velocity is not None else None),
    }

def sample_obstacle_polygon(vertices, spacing):
    """Densely sample points along a closed polygon's edges."""
    points = []
    n = len(vertices)
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        edge_len = np.linalg.norm(b - a)
        steps = max(int(np.ceil(edge_len / spacing)), 1)
        for s in range(steps):
            points.append(a + (b - a) * (s / steps))
    return points
 
 
def obstacle_pseudo_agents(sample_points):
    return [make_agent(p, [0.0, 0.0], radius=1e-3, max_speed=0.0)
            for p in sample_points]

if __name__ == "__main__":
    robot = make_agent([0.0, 0.0], [0.0, 0.0], radius=0.3,
                        max_speed=1.0, pref_velocity=[1.0, 0.0])
 
    pedestrians = [
        make_agent([3.0, 0.05], [-0.8, 0.0], radius=0.35),
        make_agent([2.0, -1.5], [0.0, 0.9], radius=0.35),
    ]
 
    wall = np.array([[1.5, -2.0], [1.5, 2.0], [1.7, 2.0], [1.7, -2.0]])
    obstacle_agents = obstacle_pseudo_agents(sample_obstacle_polygon(wall, spacing=0.25))
 
    v_new = orca_step(robot, pedestrians, obstacle_agents,
                       tau=2.0, dt=0.1, neighbor_cutoff=5.0,
                       pedestrian_weight=1.0)
 
    print(f"pref_velocity : {robot['pref_velocity']}")
    print(f"new_velocity  : {v_new}")

