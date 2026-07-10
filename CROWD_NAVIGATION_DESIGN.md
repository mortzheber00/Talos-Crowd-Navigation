# Crowd Navigation for Talos — Integration Design

**Goal:** make the Talos humanoid walk through a crowd of simulated pedestrians, using
ORCA (Optimal Reciprocal Collision Avoidance) to stay collision-free, on top of the
existing MuJoCo LIP‑MPC + TSID walking controller.

This document is the plan for the crowd-navigation part of the project. It maps ORCA
onto the walking stack we already have, identifies the hard parts, and breaks the work
into testable milestones.

---

## 1. What we already have (the walking stack)

The pipeline in `bullet_sims` (entry point `walking.py`) is a textbook layered biped
controller:

1. **Footstep plan** — `FootStepPlanner.planLine(T_support, side, no_steps)` produces a
   *fixed, precomputed* straight line of alternating `FootStep`s (pose + footprint + side).
2. **ZMP reference** — `generate_zmp_reference(plan, samples_per_step)` turns that plan
   into a piecewise-constant ZMP trajectory (one constant segment per footstep center).
3. **LIP‑MPC** — `LIPMPC.buildSolveOCP(x_k, ZMP_ref_window, terminal_idx)` is a receding-
   horizon QP (Drake) that tracks the ZMP reference, keeps the commanded ZMP inside the
   support foot (box constraint from footprint half-sizes), and applies a terminal
   capture-point (DCM) constraint. Horizon = `no_steps_per_horizon` steps (~2).
4. **LIP interpolator** — integrates the LIP at sim rate with the held MPC ZMP command and
   emits CoM position/velocity/acceleration references.
5. **Swing foot** — `SwingFootTrajectory(T0, T1, step_dur)` is a BPoly spline from the
   current swing foot to the next footstep target, with an arch height.
6. **Whole-body control** — `Talos` + `TSIDWrapper` track the CoM ref + swing-foot ref +
   contact constraints and output joint torques to MuJoCo.

**Main loop cadence** (`walking.py`): a pre-walk CoM sway, then every `no_sim_per_mpc`
sim steps re-solve the MPC, every `no_sim_per_step` sim steps advance to the next footstep
in `plan` and build a new swing spline, and every sim step integrate the LIP + step the
sim + run WBC.

**Key limitation for us:** the whole footstep plan is computed once, up front, as a
straight line. Navigation requires the *opposite*: footsteps generated reactively, one or
two at a time, in response to where the pedestrians are right now.

---

## 2. The core challenge

ORCA and a walking humanoid live in different worlds:

- **ORCA is a 2D holonomic velocity planner.** It treats each agent as a disk with a
  position, a current velocity, a preferred velocity, and a radius, and returns a new
  velocity that is collision-free for the next short horizon, assuming every agent can
  change velocity *instantly*.
- **A biped is not holonomic and not instantaneous.** It moves by discrete footsteps of
  fixed duration (`step_dur = 0.8 s`), it is dynamically committed ~1–2 steps ahead
  (the MPC preview horizon), it can only shift laterally by alternating feet, and abrupt
  velocity or heading changes can make it fall.

So the integration is fundamentally a **timescale + representation bridge**: convert
ORCA's continuous 2D velocity command into a *feasible, rate-limited stream of footsteps*
that the existing LIP‑MPC/WBC can execute without losing balance, and do it fast enough
to react to a moving crowd.

This is exactly the "join ORCA with humanoid control" difficulty. Everything below is
about building that bridge cleanly.

---

## 3. Proposed architecture (three layers, different rates)

```
   goal ──► [ ORCA layer ]  ~10-20 Hz   -> desired 2D body velocity  v_cmd = (vx, vy)  (+ heading)
                │  (robot as a disk among pedestrian disks; reciprocal avoidance)
                ▼
        [ Footstep generator ]  once per step (~1.25 Hz)  -> next FootStep pose (x,y,yaw,side)
                │  (velocity -> footstep placement, clamped & rate-limited; rolling buffer)
                ▼
        [ LIP-MPC + swing + TSID ]  MPC ~10 Hz, sim ~300 Hz  (EXISTING, mostly unchanged)
                │
                ▼
             MuJoCo  (Talos + pedestrians)
```

The bottom layer is what we already have. The two new layers are the ORCA planner and the
velocity→footstep generator. The footstep generator is where most of the real engineering
sits.

---

## 4. Component design

### 4.1 Pedestrian simulation (MuJoCo)

The MuJoCo wrapper builds the scene programmatically via `mujoco.MjSpec` and can add
bodies/geoms/freejoints (see `objects.py`'s `Cube`/`Ball`, and the `add_geom`/
`add_freejoint` calls in `mujoco_wrapper.py`). Pedestrians should be:

- **Represented** as simple vertical capsules/cylinders (radius ~0.25–0.3 m, height ~1.7 m)
  added as bodies with a freejoint, or as kinematic bodies whose pose we set directly each
  step. Kinematic (position-controlled) is simplest and avoids contact dynamics fighting
  the walker — set their `qpos` every step from the pedestrian simulator; do not let the
  physics push them.
- **Driven** by their *own* ORCA instance so they avoid each other and the robot
  reciprocally (this is what makes a "crowd" rather than moving obstacles). Each pedestrian
  gets a goal and a preferred speed; some can be scripted (cross traffic, head-on, static)
  for repeatable test scenarios.

Add a small `crowd/pedestrian.py` that owns pedestrian state and a `crowd/scenarios.py`
that defines named test setups (head-on, crossing, random crowd, narrow corridor).

### 4.2 ORCA layer

- **Library choice.** The standard is **Python‑RVO2** (Cython bindings to the RVO2/ORCA C++
  library). It is fast and battle-tested but must be built from source in the container.
  If the build is painful, a **from-scratch ORCA** (~200–300 lines: build velocity-obstacle
  half-planes per neighbor, solve the small 2D linear program) is very doable and gives us
  full control over the reciprocity coefficient and the robot's special treatment. Wrap
  whichever we pick behind our own `crowd/orca.py` interface so the rest of the code
  doesn't care.
- **Robot as an agent.** Insert the robot into the ORCA world as a disk at the current CoM
  ground projection, with velocity = measured/commanded body velocity, radius = robot
  footprint radius **plus a safety margin** (see 4.6). Preferred velocity
  `v_pref = clamp(normalize(goal - pos) * v_desired, v_max)`.
- **Output.** `v_cmd = orca.compute_velocity(robot_state, pedestrian_states)` — a 2D
  collision-free velocity for the robot for the next planning tick. Optionally also a
  desired heading (default: face `v_cmd`).
- **State source.** For the first version, feed ORCA **ground-truth** pedestrian positions
  and velocities from the simulator. Perception from the depth camera
  (`stand_and_perceive.py` already attaches a head depth camera) is a later stretch goal.

### 4.3 Velocity → footstep generator (the bridge — most important)

Replace the up-front `planLine` with a **reactive, receding-horizon** planner that emits
one footstep at a time. Add `crowd/reactive_footstep_planner.py` exposing something like:

```python
class ReactiveFootStepPlanner:
    def next_step(self, support_pose, support_side, v_cmd, yaw_cmd) -> FootStep: ...
```

Placement model (velocity-based, per step of duration `T = step_dur`):

- **Nominal forward displacement** from the commanded velocity: `d = v_cmd * T`, expressed
  in the *current support-foot heading frame*.
- **Lateral alternation:** add the mandatory sideways offset `±(2 * step_size_y)` so feet
  don't cross, sign depending on which foot swings next.
- **Heading:** turn the footstep frame toward `yaw_cmd`, limited to a max yaw change per
  step (e.g. ≤ 15–20°).
- **Clamp everything** to feasible ranges: `|step_x| ≤ step_x_max`,
  lateral within `[min_width, max_width]`, `|Δyaw| ≤ dyaw_max`. These limits are the
  kinematic/stability envelope; they are what keeps ORCA from commanding a step that
  topples the robot.

Rolling buffer: keep a short list `plan = [committed_step, ..., horizon_steps]`. Each step
transition, pop the executed step and append **one freshly generated** step from the
latest `v_cmd`. Only the steps already inside the MPC preview horizon are "committed"; the
tail is always re-derived, so the robot reacts within ~1 step of latency.

### 4.4 Changes to `walking.py` main loop

The current loop precomputes a global `ZMP_ref` array and indexes it with `k`. Reactive
walking needs the ZMP window rebuilt from the *current* footstep buffer each MPC update:

1. Remove the fixed `planLine` + global `ZMP_ref`. Instead maintain the rolling `plan`
   buffer (4.3).
2. At each **step transition** block: call `v_cmd = orca.compute_velocity(...)`, generate
   the next footstep, push it into `plan`, set swing/support foot, build the swing spline
   (as today).
3. At each **MPC update** block: build the ZMP reference window *for the current horizon*
   from the committed footsteps in `plan` (reuse `generate_zmp_reference` on that slice),
   then `mpc.buildSolveOCP(...)` as today.
4. Advance the ORCA/pedestrian world each planning tick and push pedestrian poses into
   MuJoCo (4.1).
5. Add a **goal-reached** termination and a **stop gait** (feet together, `v_cmd → 0`).

Everything below the footstep buffer — LIP interpolation, swing spline, TSID — stays as is.

### 4.5 Turning / heading support (a real gap today)

Right now the stack is axis-aligned:

- `planLine` keeps the initial rotation for every step.
- `SwingFootTrajectory.evaluate` returns `pin.SE3(np.eye(3), pos)` — **identity rotation**,
  so the swing foot's yaw is never tracked.
- The MPC ZMP box constraint assumes an axis-aligned foot.

To follow arbitrary ORCA directions we must add yaw:

- Give each `FootStep` a real yaw in its `pose.rotation`.
- Extend `SwingFootTrajectory` to interpolate yaw from `T0` to `T1` (slerp or a scalar-yaw
  spline) and return it in the SE3.
- For the ZMP box, either rotate the box into the foot frame or keep the axis-aligned box
  as an approximation valid for small per-step yaw (acceptable given the `dyaw_max` clamp).

A cheaper interim option: **omnidirectional walking without turning the torso** — allow
`v_cmd` to have a lateral component and place feet sideways/diagonally while keeping a
fixed heading. This avoids the yaw plumbing for a first milestone, at the cost of the robot
"strafing" rather than turning to face its travel direction.

### 4.6 Robustness & the timescale bridge

These details are what make the coupling actually stable:

- **Safety margin on the robot's ORCA radius.** Because the robot is committed ~1 step
  ahead, inflate its disk by roughly `v_max * step_dur` so ORCA leaves room for the lag.
- **Rate-limit `v_cmd`** before it hits the footstep generator (limit acceleration and
  yaw rate) so a sudden ORCA swerve becomes a smooth change in footstep placement.
- **QP-infeasible fallback.** `LIPMPC.buildSolveOCP` currently *raises* `RuntimeError` if
  the QP fails. Under aggressive avoidance this can happen. Change it to fall back to a
  safe recovery (e.g. command a "stop / step in place" footstep and reduce speed) instead
  of crashing the run.
- **Emergency stop.** If a pedestrian enters an inner safety radius, force `v_pref = 0`
  (ORCA will still try to slide away) and let the robot halt with feet together.
- **Deadlock handling.** Reciprocal ORCA can deadlock in tight head-on cases; a small
  preferred-velocity perturbation or a right-of-way bias resolves most of these.

---

## 5. New / changed files (checklist)

New (suggest a `crowd/` module inside `bullet_sims`, or a new package `crowd_nav`):

- `crowd/orca.py` — ORCA wrapper (Python‑RVO2 or custom), single `compute_velocity` API.
- `crowd/pedestrian.py` — pedestrian state + MuJoCo body management.
- `crowd/scenarios.py` — named test scenarios (head-on, crossing, corridor, random).
- `crowd/reactive_footstep_planner.py` — velocity→footstep generator + rolling buffer.
- `crowd/crowd_navigation.py` — new main entry point (a reactive copy of `walking.py`).

Changed:

- `foot_trajectory.py` — add yaw interpolation to `SwingFootTrajectory`.
- `lip_mpc.py` — graceful QP-infeasibility fallback; optionally rotate the ZMP box.
- `footstep_planner.py` — reuse `FootStep`; add a `nextStep`-style generator (or keep the
  new planner separate and import `FootStep`).
- `talos_conf.py` — new params: `v_max`, `v_desired`, `step_x_max`, `width_min/max`,
  `dyaw_max`, `robot_radius`, `safety_margin`, pedestrian radii, ORCA time horizon.
- Register the new entry point in `setup.py`/`setup.cfg` so `ros2 run bullet_sims
  crowd_navigation` works.

---

## 6. Phased roadmap (each phase independently testable)

**Phase 0 — Reactive walking, no crowd.**
Replace `planLine` with the rolling footstep buffer driven by a *constant* `v_cmd`.
Success: robot walks straight/at an angle indefinitely from a velocity command, matching
today's stability. This de-risks the biggest refactor (buffer + per-MPC ZMP window) before
ORCA enters.

**Phase 1 — Velocity command follows a goal.**
Add `v_pref` toward a fixed goal (no pedestrians). Robot walks to a point and stops.
Adds the goal logic, stop gait, and (if pursued) turning.

**Phase 2 — Pedestrians in the scene.**
Spawn kinematic pedestrian capsules moving on scripted straight lines. Robot still ignores
them. Success: pedestrians render and move; ground-truth states are readable each tick.

**Phase 3 — ORCA on the robot only.**
Robot runs ORCA against the (scripted) pedestrians as moving obstacles. Success: robot
avoids a single crossing pedestrian, then a head-on pedestrian, without falling.

**Phase 4 — Reciprocal crowd.**
Pedestrians run their own ORCA (avoid each other and the robot). Success: robot crosses a
small random crowd to its goal; measure collisions, min distance, time-to-goal.

**Phase 5 — Hardening + (stretch) perception.**
Tune safety margins/rate limits, add deadlock handling, run the scenario suite. Stretch:
replace ground-truth pedestrian states with depth-camera detections.

---

## 7. Verification per phase

- **Balance/quantitative:** reuse the existing logging in `walking.py` (CoM ref vs
  MuJoCo truth, ZMP ref vs estimated, DCM). The robot must not fall; ZMP must stay in the
  support polygon.
- **Navigation metrics:** number of collisions (0 is the bar), minimum robot–pedestrian
  distance, path length vs straight-line, time-to-goal, and per-step QP feasibility rate.
- **Scenario suite:** head-on, 90° crossing, overtaking, static obstacle, narrow corridor,
  random N-pedestrian crowd. Log each as a repeatable seed.

---

## 8. Key risks

- **Footstep feasibility vs avoidance aggressiveness** — the central tension. Mitigate with
  the clamps (4.3), rate limiting and safety margin (4.6), and the QP fallback.
- **Python‑RVO2 build in the container** — mitigate with the from-scratch ORCA fallback.
- **Turning plumbing** (4.5) touches three files — start with the omnidirectional
  no-turn interim if schedule is tight.
- **Reciprocal deadlocks** in dense/head-on cases — perturbation / right-of-way bias.

---

## 9. Suggested starting parameters

`v_desired ≈ 0.2 m/s`, `v_max ≈ 0.3 m/s` (Talos is slow), `step_dur = 0.8 s` (unchanged),
`step_x_max ≈ 0.30 m`, lateral width `≈ 0.16–0.24 m`, `dyaw_max ≈ 15°/step`,
`robot_radius ≈ 0.35 m`, `safety_margin ≈ v_max * step_dur ≈ 0.25 m`, pedestrian
`radius ≈ 0.25 m`, ORCA `time_horizon ≈ 2–3 s`, `neighbor_dist ≈ 4 m`. Tune from there.

---

### One-line summary

Keep the LIP‑MPC/TSID walker intact; insert two layers on top — ORCA (2D velocity
avoidance) and a **clamped, rate-limited, receding-horizon footstep generator** that turns
ORCA's velocity into feasible footsteps — and swap the offline `planLine` for a rolling,
per-step-replanned footstep buffer. The footstep generator is the bridge, and getting its
feasibility clamps and safety margins right is the crux of the project.
