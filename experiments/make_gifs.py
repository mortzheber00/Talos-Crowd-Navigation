#!/usr/bin/env python3
"""make_gifs.py
Animated top-down GIFs of the robot walking through each crowd scenario, for the
presentation slides.

Reads the time-series NPZ written by a batch run (``ROBOT_XY``, ``PED_XY``,
``TIME``, ``GOAL``, ``PED_RADII``) and renders one GIF per scenario, plus a 2x3
grid of all six.

**Synchronised by construction.** Every GIF is rendered on the *same* time grid
``[0, max_time]`` at the same fps, so all six have an identical frame count and
loop period. Dropped side by side in a slide and started together, they stay in
lockstep — the robots all set off at t = 0 and the animations all wrap at once.
For that to work the source runs must all cover ``max_time`` (run them with
``early_stop:=false -p walk_duration:=<max_time>``); a run that stopped early
would freeze on its last frame instead.

    python3 experiments/make_gifs.py                     # from results/gif_runs
    python3 experiments/make_gifs.py --src <dir> --fps 12
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))

SCEN_ORDER = ["empty", "head_on", "crossing", "corridor", "circle", "random"]
SCEN_TITLE = {"empty": "Empty (control)", "head_on": "Head-on",
              "crossing": "Crossing", "corridor": "Corridor",
              "circle": "Circle", "random": "Random crowd"}

ROBOT = "#0d7a4f"       # green
ROBOT_EDGE = "#064f33"
GOAL = "#cc3b3a"        # red
PED = "#5b666e"         # slate
PED_FILL = "#aab3ba"
INK = "#12171b"
INK2 = "#5b666e"
GRID = "#dde2e5"
ROBOT_R = 0.30          # physical robot radius [m], for the drawn body

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "text.color": INK,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
})


def _load(src):
    """Load every gif_<scenario>.npz in ``src`` into an ordered dict."""
    runs = {}
    for scen in SCEN_ORDER:
        hits = glob.glob(os.path.join(src, f"gif_{scen}.npz"))
        if not hits:
            hits = glob.glob(os.path.join(src, f"*{scen}*.npz"))
        if hits:
            runs[scen] = np.load(hits[0])
    return runs


def _n_peds(d):
    """Real pedestrian count. The logger allocates ``max(n_peds, 1)`` columns,
    so the empty scenario carries one all-NaN column that must not be drawn."""
    px = d["PED_XY"]
    if px.size == 0:
        return 0
    return int(np.sum(np.any(np.isfinite(px[..., 0]), axis=0)))


def _time_grid(runs, fps):
    """Common frame times [0, max_time] shared by every scenario."""
    max_t = 0.0
    for d in runs.values():
        t = d["TIME"]
        t = t[np.isfinite(t)]
        if len(t):
            max_t = max(max_t, float(t.max()))
    n = max(2, int(round(max_t * fps)))
    return np.linspace(0.0, max_t, n), max_t


def _sample_indices(t_series, t_grid):
    """For each frame time, the index of the nearest earlier valid sample.
    Clamps past the end (a short run holds its last pose)."""
    valid = np.flatnonzero(np.isfinite(t_series))
    tv = t_series[valid]
    idx = np.searchsorted(tv, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(valid) - 1)
    return valid[idx]


def _extent(d, pad=0.6):
    """Axis limits covering the robot and every pedestrian, with a margin."""
    xs, ys = [], []
    rx = d["ROBOT_XY"]
    ok = np.isfinite(rx[:, 0])
    xs.append(rx[ok, 0]); ys.append(rx[ok, 1])
    px = d["PED_XY"]
    if px.size:
        okp = np.isfinite(px[..., 0])
        xs.append(px[..., 0][okp].ravel()); ys.append(px[..., 1][okp].ravel())
    gx, gy = d["GOAL"]
    xs.append(np.array([gx])); ys.append(np.array([gy]))
    x = np.concatenate(xs); y = np.concatenate(ys)
    return (x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad)


def _draw_static(ax, d, xlim, ylim, title):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, alpha=0.7, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left",
                     color=INK, pad=6)
    gx, gy = d["GOAL"]
    ax.plot(gx, gy, "*", ms=20, color=GOAL, mec="#fff", mew=0.8, zorder=6)


def _make_artists(ax, n_peds, ped_radii):
    """Trails and body patches; returned so the frame updater can move them."""
    robot_trail, = ax.plot([], [], "-", color=ROBOT, lw=2.2, alpha=0.55, zorder=3)
    ped_trails = [ax.plot([], [], "-", color=PED, lw=1.1, alpha=0.35, zorder=2)[0]
                  for _ in range(n_peds)]
    ped_bodies = []
    for k in range(n_peds):
        c = Circle((0, 0), ped_radii[k] if k < len(ped_radii) else 0.3,
                   facecolor=PED_FILL, edgecolor=PED, lw=1.0, alpha=0.9, zorder=4)
        ax.add_patch(c)
        ped_bodies.append(c)
    robot_body = Circle((0, 0), ROBOT_R, facecolor=ROBOT, edgecolor=ROBOT_EDGE,
                        lw=1.4, zorder=5)
    ax.add_patch(robot_body)
    time_text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top",
                        ha="left", fontsize=11, color=INK,
                        family="monospace",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#ffffffcc",
                                  ec=GRID, lw=0.6))
    return dict(robot_trail=robot_trail, robot_body=robot_body,
                ped_trails=ped_trails, ped_bodies=ped_bodies, time_text=time_text)


TRAIL = 45   # samples of history drawn behind each body


def _update(A, d, ridx, t_grid, frame):
    i = ridx[frame]
    lo = max(0, i - TRAIL)
    rx = d["ROBOT_XY"]
    A["robot_trail"].set_data(rx[lo:i + 1, 0], rx[lo:i + 1, 1])
    A["robot_body"].center = (rx[i, 0], rx[i, 1])
    px = d["PED_XY"]
    for k, (tr, bd) in enumerate(zip(A["ped_trails"], A["ped_bodies"])):
        tr.set_data(px[lo:i + 1, k, 0], px[lo:i + 1, k, 1])
        bd.center = (px[i, k, 0], px[i, k, 1])
    A["time_text"].set_text(f"t = {t_grid[frame]:4.1f} s")


def make_single(scen, d, t_grid, fps, out_dir):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    xlim = _extent(d)[:2]; ylim = _extent(d)[2:]
    _draw_static(ax, d, xlim, ylim, SCEN_TITLE.get(scen, scen))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    n_peds = _n_peds(d)
    A = _make_artists(ax, n_peds, d["PED_RADII"])
    ridx = _sample_indices(d["TIME"], t_grid)
    fig.tight_layout()

    def upd(f):
        _update(A, d, ridx, t_grid, f)
        return []

    anim = FuncAnimation(fig, upd, frames=len(t_grid), blit=False,
                         interval=1000.0 / fps)
    out = os.path.join(out_dir, f"nav_{scen}.gif")
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    size = os.path.getsize(out) / 1e6
    print(f"  -> {os.path.relpath(out, HERE)}  ({size:.1f} MB, "
          f"{len(t_grid)} frames)")


def make_grid(runs, t_grid, fps, out_dir):
    scens = [s for s in SCEN_ORDER if s in runs]
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 9))
    axs = axs.ravel()
    per = []
    for ax, scen in zip(axs, scens):
        d = runs[scen]
        xlim = _extent(d)[:2]; ylim = _extent(d)[2:]
        _draw_static(ax, d, xlim, ylim, SCEN_TITLE.get(scen, scen))
        n_peds = _n_peds(d)
        A = _make_artists(ax, n_peds, d["PED_RADII"])
        per.append((d, A, _sample_indices(d["TIME"], t_grid)))
    for ax in axs[len(scens):]:
        ax.axis("off")
    fig.suptitle("Talos navigating six crowd scenarios  (synchronised)",
                 x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    def upd(f):
        for d, A, ridx in per:
            _update(A, d, ridx, t_grid, f)
        return []

    anim = FuncAnimation(fig, upd, frames=len(t_grid), blit=False,
                         interval=1000.0 / fps)
    out = os.path.join(out_dir, "nav_grid.gif")
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    size = os.path.getsize(out) / 1e6
    print(f"  -> {os.path.relpath(out, HERE)}  ({size:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=os.path.join(HERE, "results", "gif_runs"))
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "gifs"))
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--grid", action="store_true", help="also render the 2x3 grid")
    args = ap.parse_args()

    runs = _load(args.src)
    if not runs:
        print(f"no gif_*.npz in {args.src} -- run the scenarios first "
              f"(early_stop:=false, matched walk_duration)")
        return 1
    os.makedirs(args.out, exist_ok=True)
    t_grid, max_t = _time_grid(runs, args.fps)
    print(f"[gifs] {len(runs)} scenarios, common duration {max_t:.1f}s, "
          f"{len(t_grid)} frames @ {args.fps} fps -> {args.out}")
    for scen, d in runs.items():
        make_single(scen, d, t_grid, args.fps, args.out)
    if args.grid:
        make_grid(runs, t_grid, args.fps, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
