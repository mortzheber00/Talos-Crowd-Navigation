#!/usr/bin/env python3
"""make_ensemble.py
The "800 runs at once" montage: every run's robot trajectory overlaid in one
top-down view, coloured by outcome, animated on a shared clock.

Reads every ``<run>.npz`` in the sweep result dirs (paired with its ``<run>.json``
for the outcome), resamples each robot path onto one common time grid, and draws
them together — faint full paths as ghosting underneath, bright moving heads on
top. Successes bundle into the goal; the perception and blind-crowd failures
spray outward, so the picture carries both the scale of the evaluation and its
headline finding.

    python3 experiments/make_ensemble.py
    python3 experiments/make_ensemble.py --T 20 --fps 15 --poster

Outcome classes (from the run JSON):
    collision-free success  green   reached goal, never collided, never fell
    collided                red     robot-pedestrian surface gap went negative
    fell                    violet  CoM dropped below 0.70 m
    did not reach           amber   ran out of time without colliding (e.g.
                                    perception fleeing a phantom obstacle)
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# outcome -> (colour, label, z). Kept off the categorical viz palette on purpose:
# these are status colours, not series.
CLASSES = {
    "success":  ("#16a34a", "collision-free"),
    "collided": ("#dc2626", "collided"),
    "fell":     ("#7c3aed", "fell"),
    "timeout":  ("#d97706", "did not reach"),
}
INK = "#12171b"
INK2 = "#5b666e"
GRID = "#dde2e5"


def _classify(s):
    if s.get("fell"):
        return "fell"
    if s.get("collided"):
        return "collided"
    if s.get("success_collision_free"):
        return "success"
    return "timeout"


def _load(dirs):
    """Every (robot_xy, time, class) with a JSON outcome, across the given dirs."""
    runs = []
    for d in dirs:
        for jp in glob.glob(os.path.join(d, "*.json")):
            npz = jp[:-5] + ".npz"
            if not os.path.exists(npz):
                continue
            try:
                s = json.load(open(jp))
                z = np.load(npz)
            except Exception:
                continue
            rx = z["ROBOT_XY"]
            t = z["TIME"]
            ok = np.isfinite(t) & np.isfinite(rx[:, 0])
            if ok.sum() < 5:
                continue
            runs.append(dict(t=t[ok], xy=rx[ok], cls=_classify(s),
                             goal=z["GOAL"]))
    return runs


def _resample(run, t_grid):
    """Robot position at each frame time; holds the last pose once the run ends
    (a successful robot that reached the goal early simply rests there)."""
    t, xy = run["t"], run["xy"]
    idx = np.searchsorted(t, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(t) - 1)
    return xy[idx]


def _limits(paths, goal, qlo=1, qhi=99, pad=0.5):
    """Robust axis box: percentiles so one runaway failure can't blow out the
    frame, but always including start and goal."""
    allxy = np.concatenate(paths, axis=0)
    xlo, xhi = np.percentile(allxy[:, 0], [qlo, qhi])
    ylo, yhi = np.percentile(allxy[:, 1], [qlo, qhi])
    xlo = min(xlo, 0.0, goal[0]); xhi = max(xhi, 0.0, goal[0])
    ylo = min(ylo, 0.0, goal[1]); yhi = max(yhi, 0.0, goal[1])
    return xlo - pad, xhi + pad, ylo - pad, yhi + pad


def build(runs, T, fps, out, poster=False, title=None):
    t_grid = np.linspace(0.0, T, max(2, int(round(T * fps))))
    paths = [_resample(r, t_grid) for r in runs]          # each (F, 2)
    classes = [r["cls"] for r in runs]
    goal = runs[0]["goal"]

    counts = {k: classes.count(k) for k in CLASSES}
    order = ["success", "timeout", "collided", "fell"]     # draw good first
    n = len(runs)

    fig, ax = plt.subplots(figsize=(8, 6.6))
    xlo, xhi, ylo, yhi = _limits(paths, goal)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfcfc")
    ax.grid(True, color=GRID, alpha=0.6, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)

    # ghosting: every full path, very faint, coloured by outcome
    for cls in order:
        segs = [p for p, c in zip(paths, classes) if c == cls]
        if not segs:
            continue
        lc = LineCollection(segs, colors=CLASSES[cls][0], linewidths=0.6,
                            alpha=0.05 if cls == "success" else 0.10, zorder=1)
        ax.add_collection(lc)

    # start + goal. The goal label sits below the star: the successful heads pile
    # up on the goal, so a label beside it would be buried under green.
    ax.plot(0, 0, "o", ms=9, mfc="#ffffff", mec=INK, mew=1.4, zorder=7)
    ax.annotate("start", (0, 0), textcoords="offset points", xytext=(-4, 10),
                ha="right", fontsize=9, color=INK2, zorder=7)
    ax.plot(goal[0], goal[1], "*", ms=20, color="#cc3b3a", mec="#fff", mew=1.0,
            zorder=7)
    ax.annotate("goal", (goal[0], goal[1]), textcoords="offset points",
                xytext=(0, -20), ha="center", fontsize=9, color="#cc3b3a",
                fontweight="bold", zorder=7)

    # animated heads, one scatter per class (clean legend + per-class alpha)
    P = np.stack(paths, axis=0)            # (N, F, 2)
    cls_arr = np.array(classes)
    scatters = {}
    for cls in order:
        m = cls_arr == cls
        if not m.any():
            continue
        sc = ax.scatter(P[m, 0, 0], P[m, 0, 1], s=14, c=CLASSES[cls][0],
                        alpha=0.55 if cls == "success" else 0.8,
                        edgecolors="none", zorder=4 if cls == "success" else 5,
                        label=f"{CLASSES[cls][1]}  ({counts[cls]})")
        scatters[cls] = (sc, m)

    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, markerscale=1.6,
              title=f"{n} runs", title_fontsize=10)
    time_text = ax.text(0.985, 0.02, "", transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=12, color=INK, family="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffffcc",
                                  ec=GRID, lw=0.6))
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", color=INK)
    fig.tight_layout()

    def upd(f):
        for cls, (sc, m) in scatters.items():
            sc.set_offsets(P[m, f, :])
        time_text.set_text(f"t = {t_grid[f]:4.1f} s")
        return []

    if poster:
        upd(len(t_grid) - 1)
        out_png = out.replace(".gif", "_poster.png")
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        print(f"  -> {os.path.relpath(out_png, HERE)}")

    anim = FuncAnimation(fig, upd, frames=len(t_grid), blit=False,
                         interval=1000.0 / fps)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    print(f"  -> {os.path.relpath(out, HERE)}  "
          f"({os.path.getsize(out)/1e6:.1f} MB, {n} runs, {len(t_grid)} frames)")
    print("     " + "  ".join(f"{CLASSES[k][1]}={counts[k]}" for k in order))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="result subdirs to include (default: all sweeps)")
    ap.add_argument("--T", type=float, default=20.0, help="clip length [s]")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "gifs",
                                                   "ensemble_all.gif"))
    ap.add_argument("--poster", action="store_true",
                    help="also save the final-frame still (the 'fan')")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.dirs:
        dirs = [os.path.join(RESULTS, d) for d in args.dirs]
    else:
        dirs = [d for d in glob.glob(os.path.join(RESULTS, "*"))
                if os.path.isdir(d) and os.path.basename(d) != "gif_runs"]

    runs = _load(dirs)
    if not runs:
        print("no runs with paired npz+json found")
        return 1
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"[ensemble] {len(runs)} runs from {len(dirs)} dirs, "
          f"T={args.T}s @ {args.fps} fps")
    build(runs, args.T, args.fps, args.out, poster=args.poster, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
