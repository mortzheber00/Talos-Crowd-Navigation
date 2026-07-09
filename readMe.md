## Install Instructions
The Docker Container provided by the lecture was used, so dependencies should match.

```bash
docker pull ghcr.io/frustiger/tum-ics-ros:jazzy-legged-robotics-devel-vscode
```

The workspace should be recreated like this:

```bash
mkdir -p /workspaces/legged_robots/legged_robots_ws/src
# copy the delivered src folder into /workspaces/legged_robots/legged_robots_ws/src
cd /workspaces/legged_robots/legged_robots_ws
source /opt/ros/rolling/setup.bash
colcon build
source install/setup.bash
```

Important: run every script and launch file from the workspace root, not from inside a package directory. The code loads URDFs and other assets using paths such as `src/talos_description/...`, so the workspace layout must stay intact.

## Main Run Commands

Using a different workspace structure needs adjustment of the rviz2 commands with the correct cwd!

### For Tutorial 1:

Launch Rviz2 config: 

```bash
rviz2 -d /workspaces/legged_robots/legged_robots_ws/src/ros_visuals/config/ros_visuals.rviz
```

```bash
ros2 run ros_visuals t11
ros2 run ros_visuals t12
ros2 run ros_visuals t13
```

### For Tutorial 2:
RViz with the Talos model:

```bash
ros2 launch ros_visuals talos_rviz.launch.py
```

Individual tutorial scripts:

```bash
ros2 run bullet_sims t21
ros2 run bullet_sims t22
ros2 run bullet_sims t23
```
Launch Rviz + t23:

```bash
ros2 launch bullet_sims t23.launch.py 
```

### Tutorial 3:

Individual tutorial scripts:

```bash
rviz2 -d /workspaces/legged_robots/legged_robots_ws/src/ros_visuals/config/talos.rviz 
ros2 run bullet_sims t31  
ros2 run bullet_sims interactive_marker
```

Interactive marker + robot model + RViz:

```bash
ros2 launch bullet_sims interactive_marker.launch.py
```

### For Tutorial 4 & 5:
RViz with the Talos-no-hands model:

```bash
ros2 launch ros_visuals talos_rviz_no_hands.launch.py
```

Individual tutorial scripts:

```bash
ros2 run bullet_sims t41
ros2 run bullet_sims t42
ros2 run bullet_sims t43
ros2 run bullet_sims t51
ros2 run bullet_sims t52
```
## Questions Tutorial 5:

### Which ones of the ground reference points can exist outside the supporting polygon?
The ZMP is restricted to the support polygon. The CP and the CMP however, can exist outside the support polygon. 

### Which modality holds higher pushing forces, torque or position hardware interface?
The position hardware interface holds higher pushing forces because it rigidly enforces the commanded joint angles by applying maximum torque to the joints.

### Are the torque and position control modalities equivalent with the proposed method? if not, why?
No, they are not equivalent with the proposed method. Using the virtual position control scheme, the TSID solver becomes "blind" to the real world because it computes the robot's state by integrating its own theoretical acceleration (dv) rather than using the physical joint measurements (q,v). Consequently, the simulator's hardware interface rigidly enforces these purely theoretical joint angles, turning the robot into a stiff block that cannot react to external forces.


### For Tutorial 6 & 7:
RViz with the Talos-no-hands model:

```bash
ros2 launch ros_visuals talos_rviz_no_hands.launch.py
```

Individual tutorial scripts:

Since there is no ROS functionality for the drake simulations, just open them up in src/drake_sim/drake_sim and execute each as regular pythonfile.

```bash
ros2 run bullet_sims walking
```

## Questions Tutorial 6:

### What is the meaning of the term S*sum(h) ?
Answer: The term S*sum(h) in the cost function represents a penalty on the total time taken for the trajectory. 
By minimizing this term, this leads to a solution that reaches the goal state in less time, ergo faster trajectories.
The constant S is a weight determinig the importance of the time penalty compared to the other costs.

## Why the robot falls:

The robot completes a single step but collapses thereafter, likely because the PyBullet physics engine struggles to resolve the impact forces due to the "stomping" motion 
during contact transitions, leading to unstable constraint handling and numerical collapse.

## Crowd Navigation (Final Project)

An ORCA-driven pedestrian crowd simulated in MuJoCo. Each pedestrian is a
kinematic disk (a mocap cylinder) that steers toward its goal while reciprocally
avoiding the others via ORCA. Code lives in `src/bullet_sims/bullet_sims/crowd/`
(`orca.py`, `pedestrian.py`, `scenarios.py`) with a standalone viewer demo in
`crowd_demo.py`.

### Run the crowd demo

```bash
cd /workspaces/crowd_navigation/talos_crowd_navigation
colcon build --packages-select bullet_sims
source install/setup.bash

ros2 run bullet_sims crowd_demo <scenario> [max_seconds]
```

Scenarios: `head_on`, `crossing` (default), `circle`, `corridor`, `random`.
Example: `ros2 run bullet_sims crowd_demo random 60`.

### Adjustable parameters

**Crowd-wide defaults — `PedestrianSimulator(...)` in `crowd_demo.py`:**

| Parameter | Default | Meaning |
|---|---|---|
| `radius` | `0.3` | Disk radius [m] — physical size *and* ORCA collision size. |
| `max_speed` | `1.3` | Top walking speed [m/s]. |
| `height` | `1.7` | Cylinder height [m], visual only (no effect on avoidance). |
| `personal_space` | `0.1` | Extra clearance [m] kept beyond the radii. Larger = more spread-out crowd. |
| `goal_tol` | `0.3` | Distance [m] at which a pedestrian counts as "arrived". |
| `pref_perturb` | `0.0` | Random noise [m/s] on preferred velocity; ~`0.05` breaks symmetric deadlocks. |
| `responsibility` | `0.5` | Share of avoidance each agent takes (`0.5` = reciprocal; `1.0` = yield fully). |
| `contact` | `False` | `True` = disks physically collide with the robot instead of being "ghosts". |
| `seed` | `0` | RNG seed for the perturbation (reproducibility). |

**ORCA solver tuning — `ORCAParams(...)`, passed as `params=`:**

| Parameter | Default | Meaning |
|---|---|---|
| `neighbor_dist` | `4.0` | How far [m] an agent looks for others to avoid. |
| `max_neighbors` | `10` | Cap on neighbors considered per agent. |
| `time_horizon` | `2.5` | Look-ahead [s] for collisions. Higher = more anticipatory, wider berths. |
| `time_step` | `0.05` | Internal planning tick [s]. |

**Per-pedestrian overrides — `crowd.add(...)`:** `radius`, `max_speed`, `height`,
`color`, and `on_arrival` (`"reverse"` = loop back and forth, `"stop"` = halt,
`"remove"` = deactivate).

**Scenario shape — `scenarios.*` functions:** e.g. `crossing(crowd, gap=6.0)`,
`circle(crowd, n=8, radius=4.0, jitter=0.15)`,
`corridor(crowd, n_each=3, length=8.0, width=1.6, speed=1.2)`,
`random_crowd(crowd, n=12, extent=5.0)`.

**Simulation rate — `MujocoWrapper(...)` in `crowd_demo.py`:** `sim_rate` [Hz] and
`num_sub_steps` (physics substeps per control step).

The two parameters that most change the feel of the crowd are `radius` /
`personal_space` (how much space pedestrians demand) and `time_horizon` (how far
ahead they plan — high looks polite and anticipatory, low looks last-second).

## Group members final project:
Lara Heidtmann , Florian Bohnes, Sebastián Serna, Moritz Heber, Nicolas Guenther

## Notes

- If the interactive marker starts in `base_link` once in a while, rerun the launch file. The marker node waits for TF, but startup timing can still vary slightly on slower machines.
- If the robot model does not appear in RViz, the workspace was likely not built from the root or the `robot_description` did not get loaded.