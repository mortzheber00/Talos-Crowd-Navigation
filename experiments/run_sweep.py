#!/usr/bin/env python3
"""run_sweep.py
Batch runner for the crowd-navigation evaluation.

Builds a list of run configurations from a named experiment, launches them as
independent ``ros2 run bullet_sims walking`` processes (headless, several at a
time), and collects the per-run JSON summaries into one CSV.

    python3 experiments/run_sweep.py --list
    python3 experiments/run_sweep.py qc_scenarios --workers 6
    python3 experiments/run_sweep.py qd_responsibility --dry-run

Each run is a separate process on purpose: the walking stack keeps state in
module globals and in a ROS node, and one crashed or wedged run must not take
the sweep with it.

**ROS domain isolation.** Every worker gets its own ``ROS_DOMAIN_ID``. Without
it, concurrent runs share a DDS domain, so one run's ORCA node receives another
run's ``/odom`` and both robots are steered by a crowd they cannot see. The
results look plausible and are nonsense, so this is not optional.
"""

import argparse
import csv
import itertools
import json
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "experiments", "results")

# ROS_DOMAIN_ID must stay in [0, 101]; 0 is the default everything else uses, so
# workers start above it to avoid colliding with an interactive session.
DOMAIN_BASE = 40

# Seeds: the same set everywhere, so any two experiments are paired samples.
SEEDS = list(range(10))


# ---------------------------------------------------------------------------
# experiment definitions
# ---------------------------------------------------------------------------
#
# A run config is a dict of ROS parameters. Anything omitted keeps the default
# in walking.py, which is the interactive demo's configuration.

def _base(**kw):
    cfg = dict(scenario="random", n_peds=8, pedestrian_sensor_data=False,
               robot_responsibility=0.2, ped_responsibility=0.3,
               safety_margin=0.2, crowd_avoids_robot=True, crowd_extent=5.0,
               walk_duration=45.0, render=False)
    cfg.update(kw)
    return cfg


def qc_scenarios():
    """Q-C: does it work, per scenario, against an empty-world baseline.

    'empty' is the control: the same stack with the crowd layer wired up but no
    pedestrians, so any drop in success or path efficiency elsewhere is
    attributable to the crowd rather than to the walker.
    """
    runs = []
    for scen in ("empty", "head_on", "crossing", "corridor", "circle", "random"):
        # 'empty' has no crowd, so the seed reaches nothing and every run is
        # bit-identical (verified). Three are kept as a determinism check
        # rather than ten as a statistic.
        seeds = SEEDS[:3] if scen == "empty" else SEEDS
        for seed in seeds:
            runs.append(_base(scenario=scen, seed=seed,
                              run_id=f"qc_{scen}_s{seed}"))
    return runs


def qd_responsibility():
    """Q-D: how much of the avoidance must the robot take on itself?

    Sweeps the robot's ORCA reciprocity share against its safety margin. The
    pedestrians' share is held fixed: the question is what the *robot* should
    assume, given a crowd it does not control.
    """
    runs = []
    for resp, margin in itertools.product((0.0, 0.2, 0.5, 1.0), (0.1, 0.2, 0.4)):
        for seed in SEEDS:
            runs.append(_base(robot_responsibility=resp, safety_margin=margin,
                              seed=seed,
                              run_id=f"qd_r{resp}_m{margin}_s{seed}"))
    return runs


def qe_density():
    """Q-E: where does it break? Crowd size at the best Q-D setting.

    ``robot_responsibility``/``safety_margin`` are placeholders here -- rerun
    with the winner from Q-D (see --set) once that is known.
    """
    runs = []
    for n in (0, 2, 4, 8, 12, 16, 24):
        for seed in SEEDS:
            runs.append(_base(n_peds=n, seed=seed, run_id=f"qe_n{n}_s{seed}"))
    return runs


def qf_perception():
    """Q-F: what does replacing ground truth with the onboard depth camera cost?

    Paired: identical scenario and seed, only the obstacle topic differs, so the
    comparison is within-pair and the crowd layout variance cancels.
    """
    runs = []
    for sensor in (False, True):
        for scen in ("head_on", "crossing", "random"):
            for seed in SEEDS:
                runs.append(_base(scenario=scen, seed=seed,
                                  pedestrian_sensor_data=sensor,
                                  run_id=f"qf_{'perc' if sensor else 'gt'}"
                                         f"_{scen}_s{seed}"))
    return runs


def qab_timing():
    """Q-A + Q-B: compute budget and command tracking.

    Both are read off a handful of long, information-dense runs rather than a
    sweep: timing is a property of the implementation, not of the scenario, and
    the velocity-tracking analysis needs runs where ORCA actually commanded
    sustained avoidance. Perception is included because the camera path is the
    one plausible way to blow the real-time budget.
    """
    runs = []
    for scen in ("random", "crossing"):
        for sensor in (False, True):
            for seed in SEEDS[:5]:
                runs.append(_base(scenario=scen, n_peds=12, seed=seed,
                                  pedestrian_sensor_data=sensor,
                                  run_id=f"qab_{scen}_"
                                         f"{'perc' if sensor else 'gt'}_s{seed}"))
    return runs


def qb2_speed():
    """Q-B2: should ORCA plan at a speed the biped can actually walk?

    The footstep planner clamps a step to ``1.4 * step_size_x`` per ``step_dur``,
    about 0.44 m/s. The demo asks ORCA for 1.2 m/s, so the command saturates the
    clamp permanently: only its direction survives, and the lateral avoidance
    component is scaled down along with the forward one. If that matters, lowering
    desired_speed to the achievable range should improve clearance and command
    tracking at no cost in time-to-goal -- because the extra commanded speed was
    never delivered in the first place.

    max_speed is kept a constant 1.25x above desired_speed so ORCA retains the
    same relative headroom to manoeuvre at every setpoint.
    """
    runs = []
    for v in (0.3, 0.45, 0.6, 1.2):
        for scen in ("random", "crossing"):
            for seed in SEEDS:
                runs.append(_base(scenario=scen, seed=seed, n_peds=8,
                                  desired_speed=v, max_speed=round(1.25 * v, 3),
                                  run_id=f"qb2_v{v}_{scen}_s{seed}"))
    return runs


def qa_solo():
    """Q-A, measured properly: timing on an otherwise idle machine.

    The other experiments run 10 simulations at once to get through the sweep,
    which is fine for every metric except this one -- ten processes contending
    for 13 cores inflate every wall-clock number, and a real-time claim measured
    under 10x self-inflicted load is worthless. Run this one with
    ``--workers 1``; it is the only timing result that should be quoted.
    """
    runs = []
    for sensor in (False, True):
        for seed in (0, 1):
            runs.append(_base(scenario="random", n_peds=12, seed=seed,
                              pedestrian_sensor_data=sensor,
                              walk_duration=30.0,
                              run_id=f"qasolo_{'perc' if sensor else 'gt'}_s{seed}"))
    return runs


def qs_stress():
    """Stage A: find a crowd hard enough that the outcome is not saturated.

    The existing sweeps sit at random x8, where essentially everything succeeds,
    so a 4x5 reciprocity plane measured there is 100% in 19 of 20 cells and can
    show no gradient. This calibrates a harder operating point before spending
    200 runs on it.

    Two levers. ``n_peds`` is the obvious one; ``crowd_extent`` is the effective
    one -- it is the half-width of the box starts and goals are drawn in, so
    shrinking it puts the same number of pedestrians into a smaller area and,
    more importantly, puts more of them across the robot's corridor rather than
    wandering the periphery.

    Target: a setting with roughly 20-40% collisions. Harder than that and the
    plane saturates at the other end, which is equally uninformative.
    """
    runs = []
    for extent in (5.0, 3.0):
        for n in (8, 16, 24, 32, 48):
            for seed in SEEDS:
                runs.append(_base(n_peds=n, crowd_extent=extent, seed=seed,
                                  run_id=f"qs_e{extent}_n{n}_s{seed}"))
    return runs


def qh_hard():
    """Q-H rerun on a crowd dense enough that the plane is not saturated.

    Q-H at random x8 was 100% in 19 of 20 cells -- no gradient to read. The
    stress calibration (qs_stress) showed why adding bodies alone does not fix
    that: from 8 to 32 pedestrians the collision rate stays at zero and only the
    effort rises (time to goal 10 -> 23 s, path ratio up to 1.7). Crowding costs
    time, not safety, while the crowd still yields. Failures appear only at 48.

    So the hard point is 48 pedestrians, and the gradient has to come from
    crossing that with the cooperation axis -- density alone cannot produce it.
    Same 10x10 m extent as every other sweep, so only the crowd size changed.
    """
    runs = []
    for rr in (0.0, 0.2, 0.5, 1.0):
        for pr in (0.0, 0.1, 0.2, 0.3, 0.5):
            for seed in SEEDS:
                runs.append(_base(scenario="random", n_peds=48, crowd_extent=5.0,
                                  seed=seed, robot_responsibility=rr,
                                  ped_responsibility=pr,
                                  run_id=f"qhh_r{rr}_p{pr}_s{seed}"))
    return runs


def qh_reciprocity():
    """Q-H: the reciprocity plane -- robot share x crowd share.

    Q-D swept the robot's responsibility against its safety margin and found
    clearance inert. Q-G2 swept the crowd's responsibility alone. This crosses
    the two: robot_responsibility x ped_responsibility on the random x8 scene.

    The question is whether ORCA's reciprocity intuition holds on this coupled
    system -- in the reference algorithm a pair's shares sum to ~1.0 for a fully
    resolved encounter, and the shipped default sums to only 0.2 + 0.3 = 0.5. If
    the sum is what matters, the failure frontier should fall along an
    anti-diagonal (robot + ped = const); if only the crowd's share matters (as
    Q-G/Q-G2 hint), the frontier should be horizontal.

    random x8, 10 seeds -> 200 runs.
    """
    runs = []
    for rr in (0.0, 0.2, 0.5, 1.0):
        for pr in (0.0, 0.1, 0.2, 0.3, 0.5):
            for seed in SEEDS:
                runs.append(_base(scenario="random", n_peds=8, seed=seed,
                                  robot_responsibility=rr, ped_responsibility=pr,
                                  run_id=f"qh_r{rr}_p{pr}_s{seed}"))
    return runs


def qg2_ped_responsibility():
    """Q-G2: how much crowd cooperation does the robot minimally need?

    Q-G showed the two endpoints -- a crowd that fully yields (100% success) and
    a crowd blind to the robot (0%). This fills the middle: the robot stays
    visible to the pedestrians, but each pedestrian takes a smaller share of the
    avoidance (``ped_responsibility`` from 0.5 down to 0.0). It is the dose-
    response curve between Q-G's extremes.

    Confound to state honestly: ped_responsibility applies to every neighbour a
    pedestrian has, so lowering it also makes the pedestrians less willing to
    avoid *each other*, not only the robot. The reading is therefore "a less
    accommodating crowd overall", not a clean robot-only isolation. The clean
    version needs a per-neighbour responsibility (a code change), and is only
    worth it if this curve is interesting.

    Note the default sums oddly: robot 0.2 + ped 0.3 = 0.5, where reciprocal ORCA
    would put the pair's shares near 1.0. The crowd is over-compensating for a
    robot that under-commits, and this sweep makes that visible.
    """
    runs = []
    for pr in (0.0, 0.1, 0.2, 0.3, 0.5):
        for scen, n in (("head_on", 2), ("crossing", 4), ("random", 8)):
            for seed in SEEDS:
                runs.append(_base(scenario=scen, n_peds=n, seed=seed,
                                  ped_responsibility=pr,
                                  run_id=f"qg2_p{pr}_{scen}{n}_s{seed}"))
    return runs


def qg_cooperation():
    """Q-G: is the robot avoiding the crowd, or is the crowd avoiding the robot?

    In the shipped configuration every pedestrian treats the robot as an ORCA
    neighbour, using the simulator's exact robot state -- a perfectly cooperative
    crowd with perfect information about the robot. A 100% success rate under
    those conditions does not establish that the robot avoids anyone, because the
    crowd would part around a fencepost too.

    This turns the pedestrians blind to the robot, so every metre of clearance
    has to come from the robot's own ORCA layer. The gap between the two
    conditions is the robot's actual contribution -- and the cooperative number
    is the one to distrust when reporting results.

    Run at two densities so the effect is not read off a single crowd size.
    """
    runs = []
    for coop in (True, False):
        for scen, n in (("head_on", 2), ("crossing", 4), ("random", 8),
                        ("random", 16)):
            for seed in SEEDS:
                runs.append(_base(scenario=scen, n_peds=n, seed=seed,
                                  crowd_avoids_robot=coop,
                                  run_id=f"qg_{'coop' if coop else 'blind'}"
                                         f"_{scen}{n}_s{seed}"))
    return runs


EXPERIMENTS = {
    "qab_timing": qab_timing,
    "qa_solo": qa_solo,
    "qb2_speed": qb2_speed,
    "qg_cooperation": qg_cooperation,
    "qg2_ped_responsibility": qg2_ped_responsibility,
    "qh_reciprocity": qh_reciprocity,
    "qs_stress": qs_stress,
    "qh_hard": qh_hard,
    "qc_scenarios": qc_scenarios,
    "qd_responsibility": qd_responsibility,
    "qe_density": qe_density,
    "qf_perception": qf_perception,
}


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------

def _cmd(cfg, out_dir):
    args = ["ros2", "run", "bullet_sims", "walking", "--ros-args"]
    for k, v in cfg.items():
        if isinstance(v, bool):
            v = "true" if v else "false"
        args += ["-p", f"{k}:={v}"]
    args += ["-p", f"out_dir:={out_dir}"]
    return args


def _run_one(cfg, out_dir, domains, log_dir, timeout):
    run_id = cfg["run_id"]
    done = os.path.join(out_dir, f"{run_id}.json")
    if os.path.exists(done):
        return run_id, "cached"

    # Take a domain out of the pool for the lifetime of this run. Deriving it
    # from the run's index instead would be wrong: the pool hands the next job
    # to whichever worker frees up, which is not necessarily the one whose
    # domain the index-based scheme would reuse.
    domain_id = domains.get()
    env = dict(os.environ)
    env["ROS_DOMAIN_ID"] = str(domain_id)
    # a stray discovery packet from an interactive session on the default domain
    # would still be a cross-talk source; keep everything on loopback
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["MUJOCO_GL"] = env.get("MUJOCO_GL", "egl")

    log_path = os.path.join(log_dir, f"{run_id}.log")
    t0 = time.time()
    try:
        with open(log_path, "w") as log:
            try:
                p = subprocess.run(_cmd(cfg, out_dir), env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout,
                                   cwd=REPO)
                status = "ok" if p.returncode == 0 else f"exit{p.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
    finally:
        domains.put(domain_id)
    if not os.path.exists(done):
        status += "+nojson"
    return run_id, f"{status} ({time.time() - t0:.0f}s)"


def collect(out_dir, csv_path):
    """Flatten every <run>.json in ``out_dir`` into one CSV."""
    rows = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(out_dir, name)) as f:
            s = json.load(f)
        flat = {}
        for k, v in s.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            flat[f"{k}.{k2}.{k3}"] = v3
                    else:
                        flat[f"{k}.{k2}"] = v2
            else:
                flat[k] = v
        rows.append(flat)
    if not rows:
        return 0
    cols = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", nargs="?", help="name of the experiment to run")
    ap.add_argument("--list", action="store_true", help="list experiments and exit")
    ap.add_argument("--workers", type=int, default=6,
                    help="concurrent runs (each is ~1 core)")
    ap.add_argument("--timeout", type=float, default=900,
                    help="[s] wall-clock limit per run")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the run list and exit")
    ap.add_argument("--set", action="append", default=[],
                    help="override a parameter for every run, e.g. "
                         "--set safety_margin=0.3 (repeatable)")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N configs (pilot / smoke test)")
    args = ap.parse_args()

    if args.list or not args.experiment:
        for name, fn in EXPERIMENTS.items():
            print(f"{name:22s} {len(fn()):4d} runs  "
                  f"{(fn.__doc__ or '').strip().splitlines()[0]}")
        return 0

    if args.experiment not in EXPERIMENTS:
        print(f"unknown experiment '{args.experiment}'; "
              f"options: {list(EXPERIMENTS)}", file=sys.stderr)
        return 2

    runs = EXPERIMENTS[args.experiment]()
    if args.limit:
        runs = runs[:args.limit]
    # --set overrides. The value has to be coerced to the type the default
    # already has: ROS 2 parameters are strictly typed, and "20" on the command
    # line declares an INTEGER, which a DOUBLE parameter then rejects.
    for ov in args.set:
        k, _, v = ov.partition("=")
        for cfg in runs:
            cur = cfg.get(k)
            if isinstance(cur, bool):
                cfg[k] = v.lower() in ("1", "true", "yes")
            elif isinstance(cur, float):
                cfg[k] = float(v)
            elif isinstance(cur, int):
                cfg[k] = int(v)
            else:
                cfg[k] = v
    if args.set:
        # the run_id encodes the config, so an override has to reach it too --
        # otherwise an overridden sweep silently reuses the cached JSON of the
        # un-overridden one
        tag = "_" + "_".join(s.replace("=", "").replace(".", "")
                             for s in args.set)
        for cfg in runs:
            cfg["run_id"] += tag

    out_dir = args.out or os.path.join(RESULTS, args.experiment)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    if args.dry_run:
        for cfg in runs:
            print(" ".join(_cmd(cfg, out_dir)))
        print(f"\n{len(runs)} runs -> {out_dir}")
        return 0

    print(f"[sweep] {args.experiment}: {len(runs)} runs, "
          f"{args.workers} workers -> {out_dir}", flush=True)
    t0 = time.time()
    done = 0
    domains = queue.Queue()
    for w in range(args.workers):
        domains.put(DOMAIN_BASE + w)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_run_one, cfg, out_dir, domains, log_dir,
                             args.timeout)
                   for cfg in runs]
        for fut in futures:
            run_id, status = fut.result()
            done += 1
            print(f"[sweep] {done}/{len(runs)}  {run_id}: {status}", flush=True)

    csv_path = os.path.join(RESULTS, f"{os.path.basename(out_dir.rstrip('/'))}.csv")
    n = collect(out_dir, csv_path)
    print(f"[sweep] finished in {(time.time() - t0) / 60:.1f} min; "
          f"{n} rows -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
