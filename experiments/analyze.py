#!/usr/bin/env python3
"""analyze.py
Turns the sweep CSVs in ``experiments/results/`` into the figures and tables for
the evaluation slides.

    python3 experiments/analyze.py            # everything that has data
    python3 experiments/analyze.py qa qc      # only these

Figures land in ``experiments/figures/`` as PNG (slides) and PDF (vector).

Chart conventions, applied throughout:
  * one y-axis per panel -- never a second scale on a twin axis, which makes two
    unrelated series look comparable.
  * a fixed categorical hue order, assigned by entity, so a series keeps its
    colour across figures.
  * every series is identified by a legend or a direct label, never by colour
    alone.
"""

import json
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "figures")

# Validated categorical palette (light surface), assigned in fixed order.
C = ["#2a78d6",  # 1 blue
     "#008300",  # 2 green
     "#e87ba4",  # 3 magenta
     "#eda100",  # 4 yellow
     "#1baf7a",  # 5 aqua
     "#eb6834",  # 6 orange
     "#4a3aa7",  # 7 violet
     "#e34948"]  # 8 red
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d8d7d2"
BAD = "#e34948"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelcolor": INK2,
    "axes.edgecolor": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "grid.color": GRID,
    "grid.alpha": 0.6,
    "legend.frameon": False,
})


def _save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}.png")


def _load(name):
    path = os.path.join(RESULTS, f"{name}.csv")
    if not os.path.exists(path):
        print(f"[skip] {name}: no {path}")
        return None
    df = pd.read_csv(path)
    print(f"[load] {name}: {len(df)} runs")
    return df


def _rate(x):
    """Success rate with a Wilson 95% interval -- the right interval for a
    proportion from ~10 runs, where the textbook normal one can run past 0/1."""
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    p = float(np.mean(x))
    z = 1.96
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    # The Wilson interval is shifted toward 0.5 and so need not contain p itself
    # (at p = 1 it lies entirely below it). Drawn as an error bar around p that
    # would be a negative whisker, so widen to include the estimate.
    return p, min(lo, p), max(hi, p)


def _bar_ci(ax, labels, rates, los, his, colors, ylabel):
    x = np.arange(len(labels))
    ax.bar(x, rates, color=colors, width=0.62, zorder=2)
    ax.errorbar(x, rates, yerr=[np.array(rates) - np.array(los),
                                np.array(his) - np.array(rates)],
                fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4, zorder=3)
    for xi, r in zip(x, rates):
        ax.text(xi, r + 0.03, f"{100 * r:.0f}%", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", zorder=0)


# ---------------------------------------------------------------------------
# Q-A: compute budget
# ---------------------------------------------------------------------------

def qa():
    # Prefer the uncontended measurement. The sweep runs 10 simulations at once,
    # which inflates every wall-clock number; qa_solo is the same configuration
    # run one at a time and is the only timing worth quoting.
    df = _load("qa_solo")
    contended = False
    if df is None:
        df = _load("qab_timing")
        contended = True
    if df is None:
        return
    if contended:
        print("  WARNING: falling back to the contended sweep timings "
              "(10 concurrent runs) -- run qa_solo with --workers 1")

    # Each layer runs at its own rate, so "is it real-time" is per-layer:
    # the budget is one period of that layer, not one control tick.
    layers = [
        ("orca",       "ORCA solve",        1 / 30.0),
        ("footstep",   "Footstep planner",  0.8),
        ("mpc",        "LIP-MPC (Drake QP)", 0.1),
        ("wbc",        "TSID whole-body",   1 / 300.0),
        ("sim",        "MuJoCo step",       1 / 300.0),
        ("perception", "Depth perception",  1 / 30.0),
    ]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ys, labels = [], []
    for i, (key, label, budget) in enumerate(layers):
        col = f"timing.{key}.p95_ms"
        if col not in df or df[col].isna().all():
            continue
        vals = df[col].dropna()
        mean = df[f"timing.{key}.mean_ms"].dropna().mean()
        p95 = vals.mean()
        mx = df[f"timing.{key}.max_ms"].dropna().mean()
        y = len(ys)
        ys.append(y)
        labels.append(f"{label}\n({1e3 * budget:.0f} ms budget)")
        frac = p95 / (1e3 * budget)
        color = BAD if frac > 1.0 else C[0]
        ax.barh(y, p95, height=0.5, color=color, zorder=2)
        ax.plot([mean], [y], "o", ms=6, color="#ffffff",
                markeredgecolor=INK, markeredgewidth=1.2, zorder=4)
        ax.plot([1e3 * budget], [y], "|", ms=26, color=INK, mew=2.0, zorder=5)
        ax.text(p95 * 1.04, y, f"p95 {p95:.1f} ms  ({100 * frac:.0f}% of budget)",
                va="center", fontsize=9,
                color=BAD if frac > 1.0 else INK)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(0.02, 400)
    ax.set_xlabel("wall-clock time per invocation [ms, log scale]")
    ax.set_title("Per-layer compute cost against its own real-time budget\n"
                 "bar = p95, white dot = mean, tick = budget (one period at that "
                 "layer's rate)"
                 + ("\nCONTENDED measurement -- 10 concurrent runs, upper bound"
                    if contended else "\nmeasured one run at a time on an "
                    "otherwise idle machine"), loc="left")
    ax.grid(axis="x", zorder=0)
    # identity without relying on colour
    ax.text(0.99, 0.02, "red = over budget", transform=ax.transAxes,
            ha="right", fontsize=9, color=BAD)
    _save(fig, "qa_timing_budget")

    # table for the slide
    rows = []
    for key, label, budget in layers:
        col = f"timing.{key}.p95_ms"
        if col not in df or df[col].isna().all():
            continue
        rows.append(dict(layer=label, rate_hz=round(1 / budget, 1),
                         budget_ms=round(1e3 * budget, 2),
                         mean_ms=round(df[f"timing.{key}.mean_ms"].mean(), 2),
                         p95_ms=round(df[col].mean(), 2),
                         max_ms=round(df[f"timing.{key}.max_ms"].mean(), 2),
                         pct_budget=round(100 * df[col].mean() / (1e3 * budget), 1)))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RESULTS, "table_qa_timing.csv"), index=False)
    print(t.to_string(index=False))


# ---------------------------------------------------------------------------
# Q-B: does the footstep bridge execute the ORCA command?
# ---------------------------------------------------------------------------

def qb():
    df = _load("qab_timing")
    if df is None:
        return

    # panel 1: one representative run, commanded vs achieved speed
    npz_files = sorted(glob.glob(os.path.join(RESULTS, "qab_timing", "qab_random_gt_*.npz")))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4),
                            gridspec_kw=dict(width_ratios=[1.6, 1]))

    if npz_files:
        d = np.load(npz_files[0])
        t = d["TIME"]
        ok = np.isfinite(t)
        step = 0.8
        w = max(1, int(round(step / np.nanmedian(np.diff(t[ok])))))
        k = np.ones(w) / w

        def sm(a):
            return np.convolve(a, k, mode="same")

        cmd = np.linalg.norm(d["VCMD"][ok], axis=1)
        ach = np.linalg.norm(d["COM_VEL_PB"][ok, :2], axis=1)
        ax = axs[0]
        ax.plot(t[ok], sm(cmd), lw=2, color=C[0], label="commanded  |v_cmd|  (ORCA)")
        ax.plot(t[ok], sm(ach), lw=2, color=C[1], label="achieved  |v_CoM|  (robot)")
        # the kinematic ceiling the footstep planner imposes
        ceiling = 1.4 * 0.25 / 0.8
        ax.axhline(ceiling, color=INK2, ls="--", lw=1.2)
        # below the line and hard left: the commanded trace crosses this height
        # repeatedly, so anything sitting on top of the line collides with it
        ax.text(t[ok][0], ceiling - 0.03,
                f"footstep clamp = 1.4·step_x/step_dur = {ceiling:.2f} m/s",
                ha="left", va="top", fontsize=9, color=INK2)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("speed [m/s]")
        ax.set_title("ORCA asks for more speed than the biped can walk\n"
                     "(one run, smoothed over one step period)", loc="left")
        ax.grid(True)
        ax.legend(fontsize=9)

    # panel 2: gain distribution over all runs
    ax = axs[1]
    gains = df["vel_tracking.speed_gain"].dropna()
    lat = df["vel_tracking.lat_gain"].dropna()
    parts = [g for g in (gains.values, lat.values) if len(g)]
    labels = [n for n, g in (("forward\nspeed", gains), ("lateral\n(avoidance)", lat))
              if len(g)]
    bp = ax.boxplot(parts, labels=labels, widths=0.5, patch_artist=True,
                    medianprops=dict(color=INK, lw=2))
    for patch, c in zip(bp["boxes"], (C[0], C[3])):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)
        patch.set_edgecolor(INK2)
    ax.axhline(1.0, color=INK2, ls="--", lw=1.2)
    ax.text(0.02, 1.0, " 1.0 = command fully executed",
            transform=ax.get_yaxis_transform(), fontsize=9, color=INK2,
            va="bottom", ha="left")
    # medians labelled beside the box rather than on it, so the number does not
    # sit on the median line it describes
    for i, g in enumerate(parts):
        ax.annotate(f"{np.median(g):.2f}", (i + 1.29, np.median(g)),
                    fontsize=9, color=INK, va="center", ha="left")
    ax.set_ylabel("achieved / commanded")
    ax.set_title(f"Fraction of the command actually executed\n"
                 f"({len(df)} runs)", loc="left")
    ax.grid(axis="y")
    fig.tight_layout()
    _save(fig, "qb_command_tracking")

    print(f"  forward gain  median {gains.median():.3f}")
    if len(lat):
        print(f"  lateral gain  median {lat.median():.3f}")
    lag = df["vel_tracking.lat_lag_s"].dropna()
    if len(lag):
        print(f"  lateral lag   median {lag.median():.2f} s "
              f"({lag.median() / 0.8:.1f} steps)")


# ---------------------------------------------------------------------------
# Q-B2: matching ORCA's speed to the walker
# ---------------------------------------------------------------------------

def qb2():
    df = _load("qb2_speed")
    if df is None:
        return
    g = df.groupby("desired_speed")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    speeds = sorted(df["desired_speed"].unique())

    # collision-free success
    rates, los, his = [], [], []
    for v in speeds:
        p, lo, hi = _rate(g.get_group(v)["success_collision_free"])
        rates.append(p); los.append(lo); his.append(hi)
    _bar_ci(axs[0], [f"{v}" for v in speeds], rates, los, his,
            [C[0]] * len(speeds), "collision-free success")
    axs[0].set_xlabel("ORCA desired_speed [m/s]")
    axs[0].set_title("Outcome", loc="left")

    # clearance
    ax = axs[1]
    med = [g.get_group(v)["min_sep_m"].median() for v in speeds]
    q1 = [g.get_group(v)["min_sep_m"].quantile(0.25) for v in speeds]
    q3 = [g.get_group(v)["min_sep_m"].quantile(0.75) for v in speeds]
    ax.plot(speeds, med, "-o", color=C[0], lw=2, ms=7)
    ax.fill_between(speeds, q1, q3, color=C[0], alpha=0.18)
    ax.axhline(0, color=BAD, ls="--", lw=1.2)
    ax.text(speeds[-1], 0.01, "contact", ha="right", va="bottom",
            fontsize=9, color=BAD)
    ax.set_xlabel("ORCA desired_speed [m/s]")
    ax.set_ylabel("min. clearance [m]")
    ax.set_title("Safety  (median, IQR band)", loc="left")
    ax.grid(True)

    # time to goal
    ax = axs[2]
    med = [g.get_group(v)["t_goal_s"].median() for v in speeds]
    q1 = [g.get_group(v)["t_goal_s"].quantile(0.25) for v in speeds]
    q3 = [g.get_group(v)["t_goal_s"].quantile(0.75) for v in speeds]
    ax.plot(speeds, med, "-o", color=C[1], lw=2, ms=7)
    ax.fill_between(speeds, q1, q3, color=C[1], alpha=0.18)
    ax.set_xlabel("ORCA desired_speed [m/s]")
    ax.set_ylabel("time to goal [s]")
    ax.set_title("Cost  (median, IQR band)", loc="left")
    ax.grid(True)

    fig.suptitle("Q-B2  Asking ORCA for a speed the walker cannot deliver",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "qb2_speed_matching")

    t = g.agg(success=("success", "mean"),
              collision_free=("success_collision_free", "mean"),
              min_sep=("min_sep_m", "median"),
              t_goal=("t_goal_s", "median"),
              path_ratio=("path_ratio", "median"),
              speed_gain=("vel_tracking.speed_gain", "median"),
              ach_speed=("vel_tracking.ach_speed_mean", "median")).round(3)
    t.to_csv(os.path.join(RESULTS, "table_qb2_speed.csv"))
    print(t.to_string())


# ---------------------------------------------------------------------------
# Q-C: per-scenario validation
# ---------------------------------------------------------------------------

SCEN_ORDER = ["empty", "head_on", "crossing", "corridor", "circle", "random"]


def qc():
    df = _load("qc_scenarios")
    if df is None:
        return
    scens = [s for s in SCEN_ORDER if s in set(df["scenario"])]
    g = df.groupby("scenario")

    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.4))

    rates, los, his = [], [], []
    for s in scens:
        p, lo, hi = _rate(g.get_group(s)["success_collision_free"])
        rates.append(p); los.append(lo); his.append(hi)
    # 'empty' is the control, so it is drawn in a different slot than the
    # crowd scenarios rather than reading as just another condition
    colors = [C[6] if s == "empty" else C[0] for s in scens]
    _bar_ci(axs[0], scens, rates, los, his, colors,
            "collision-free success")
    axs[0].axhline(0.8, color=BAD, ls="--", lw=1.4, zorder=4)
    axs[0].text(-0.45, 0.815, "project bar: 80%", ha="left", va="bottom",
                fontsize=9, color=BAD, zorder=5)
    axs[0].tick_params(axis="x", rotation=30)
    axs[0].set_title("Outcome  (Wilson 95% CI)", loc="left")

    ax = axs[1]
    data = [g.get_group(s)["min_sep_m"].dropna().values for s in scens]
    keep = [(s, d, c) for s, d, c in zip(scens, data, colors) if len(d)]
    bp = ax.boxplot([d for _, d, _ in keep], labels=[s for s, _, _ in keep],
                    widths=0.55, patch_artist=True,
                    medianprops=dict(color=INK, lw=2))
    for patch, (_, _, c) in zip(bp["boxes"], keep):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor(INK2)
    ax.axhline(0, color=BAD, ls="--", lw=1.4)
    ax.text(len(keep) + 0.4, 0.01, "contact", ha="right", va="bottom",
            fontsize=9, color=BAD)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("min. clearance over the run [m]")
    ax.set_title("Safety", loc="left")
    ax.grid(axis="y")

    ax = axs[2]
    data = [g.get_group(s)["path_ratio"].dropna().values for s in scens]
    keep = [(s, d, c) for s, d, c in zip(scens, data, colors) if len(d)]
    bp = ax.boxplot([d for _, d, _ in keep], labels=[s for s, _, _ in keep],
                    widths=0.55, patch_artist=True,
                    medianprops=dict(color=INK, lw=2))
    for patch, (_, _, c) in zip(bp["boxes"], keep):
        patch.set_facecolor(c); patch.set_alpha(0.85); patch.set_edgecolor(INK2)
    ax.axhline(1.0, color=INK2, ls="--", lw=1.2)
    ax.text(0.55, 1.0, " straight line", ha="left", va="bottom",
            fontsize=9, color=INK2)
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("path length / straight-line distance")
    ax.set_title("Detour cost", loc="left")
    ax.grid(axis="y")

    fig.suptitle("Q-C  Navigation performance per scenario "
                 "(violet = empty-world control)",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "qc_scenarios")

    t = g.agg(n=("success", "size"),
              success=("success", "mean"),
              collision_free=("success_collision_free", "mean"),
              fell=("fell", "mean"),
              collided=("collided", "mean"),
              min_sep=("min_sep_m", "median"),
              t_goal=("t_goal_s", "median"),
              path_ratio=("path_ratio", "median"),
              zmp_rmse=("zmp_rmse_m", "median")).reindex(scens).round(3)
    t.to_csv(os.path.join(RESULTS, "table_qc_scenarios.csv"))
    print(t.to_string())


def qc_traj():
    """Top-down trajectory small multiples -- the qualitative companion to Q-C."""
    fig, axs = plt.subplots(2, 3, figsize=(13, 8))
    for ax, scen in zip(axs.ravel(), SCEN_ORDER):
        files = sorted(glob.glob(os.path.join(RESULTS, "qc_scenarios",
                                              f"qc_{scen}_s*.npz")))
        if not files:
            ax.axis("off")
            continue
        d = np.load(files[0])
        for j in range(d["PED_XY"].shape[1]):
            ax.plot(d["PED_XY"][:, j, 0], d["PED_XY"][:, j, 1], "-", lw=1.0,
                    color=INK2, alpha=0.45,
                    label="pedestrians" if j == 0 else None)
        ax.plot(d["ROBOT_XY"][:, 0], d["ROBOT_XY"][:, 1], "-", lw=2.6,
                color=C[0], label="robot", zorder=3)
        v = np.flatnonzero(np.isfinite(d["ROBOT_XY"][:, 0]))
        if len(v):
            ax.plot(*d["ROBOT_XY"][v[0]], "o", ms=8, color=C[1],
                    label="start", zorder=4)
        ax.plot(d["GOAL"][0], d["GOAL"][1], "*", ms=15, color=BAD,
                label="goal", zorder=4)
        ax.set_title(scen, loc="left")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    axs.ravel()[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Q-C  Top-down trajectories, one representative run per scenario",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "qc_trajectories")


# ---------------------------------------------------------------------------
# Q-D: who yields?
# ---------------------------------------------------------------------------

def qd():
    df = _load("qd_responsibility")
    if df is None:
        return
    resps = sorted(df["robot_responsibility"].unique())
    margins = sorted(df["safety_margin"].unique())

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))

    # heatmap of collision-free success: sequential, one hue, light -> dark
    ax = axs[0]
    M = np.zeros((len(margins), len(resps)))
    for i, m in enumerate(margins):
        for j, r in enumerate(resps):
            sel = df[(df.safety_margin == m) & (df.robot_responsibility == r)]
            M[i, j] = sel["success_collision_free"].mean() if len(sel) else np.nan
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, origin="lower", aspect="auto")
    ax.set_xticks(range(len(resps))); ax.set_xticklabels(resps)
    ax.set_yticks(range(len(margins))); ax.set_yticklabels(margins)
    ax.set_xlabel("robot responsibility"); ax.set_ylabel("safety margin [m]")
    for i in range(len(margins)):
        for j in range(len(resps)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{100 * M[i, j]:.0f}%", ha="center", va="center",
                        fontsize=10,
                        color="#ffffff" if M[i, j] > 0.55 else INK)
    ax.set_title("Collision-free success rate", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, label="rate")

    # clearance vs responsibility, one line per margin
    ax = axs[1]
    for k, m in enumerate(margins):
        sub = df[df.safety_margin == m].groupby("robot_responsibility")["min_sep_m"]
        vals = [sub.median().get(r, np.nan) for r in resps]
        # The three lines sit on top of each other here, so direct labels would
        # only collide; the legend carries identity for all three panels.
        ax.plot(resps, vals, "-o", color=C[k], lw=2, ms=7, label=f"margin {m} m")
    ax.axhline(0, color=BAD, ls="--", lw=1.2)
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.02, 0.45)
    ax.set_xlabel("robot responsibility")
    ax.set_ylabel("min. clearance [m]  (median)")
    # The flatness is the finding, so the axis is held from 0 rather than
    # auto-scaled to the 1.5 mm the three lines actually span.
    ax.set_title("Safety — flat at 0.35 m, whatever we set", loc="left")
    ax.grid(True); ax.legend(fontsize=9, loc="lower left")

    # Where the parameters DO have an effect. Plotting clearance against
    # time-to-goal would be the natural trade-off panel, but clearance spans
    # 0.0015 m across the whole sweep -- on its own axis that noise fills the
    # panel and reads as a trend. Efficiency is the axis that actually moves.
    ax = axs[2]
    for k, m in enumerate(margins):
        sub = df[df.safety_margin == m].groupby("robot_responsibility")["t_goal_s"]
        vals = [sub.median().get(r, np.nan) for r in resps]
        ax.plot(resps, vals, "-o", color=C[k], lw=2, ms=7, label=f"margin {m} m")
    ax.set_xlim(-0.05, 1.08)
    ax.set_xlabel("robot responsibility")
    ax.set_ylabel("time to goal [s]  (median)")
    ax.set_title("Efficiency  — the only thing these knobs change", loc="left")
    ax.grid(True); ax.legend(fontsize=9)

    fig.suptitle("Q-D  How much of the avoidance should the robot take on itself?",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "qd_responsibility")

    t = df.groupby(["robot_responsibility", "safety_margin"]).agg(
        n=("success", "size"), success=("success", "mean"),
        collision_free=("success_collision_free", "mean"),
        fell=("fell", "mean"), min_sep=("min_sep_m", "median"),
        t_goal=("t_goal_s", "median"), path_ratio=("path_ratio", "median")).round(3)
    t.to_csv(os.path.join(RESULTS, "table_qd_responsibility.csv"))
    print(t.to_string())
    best = t.sort_values(["collision_free", "min_sep"], ascending=False).head(3)
    print("\n  best settings by collision-free rate, then clearance:")
    print(best.to_string())


# ---------------------------------------------------------------------------
# Q-E: density
# ---------------------------------------------------------------------------

def qe():
    df = _load("qe_density")
    if df is None:
        return
    ns = sorted(df["n_peds"].unique())
    g = df.groupby("n_peds")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))

    rates, los, his = [], [], []
    for n in ns:
        p, lo, hi = _rate(g.get_group(n)["success_collision_free"])
        rates.append(p); los.append(lo); his.append(hi)
    ax = axs[0]
    ax.plot(ns, rates, "-o", color=C[0], lw=2, ms=7)
    ax.fill_between(ns, los, his, color=C[0], alpha=0.18)
    ax.axhline(0.8, color=BAD, ls="--", lw=1.4)
    ax.text(ns[-1], 0.815, "project bar: 80%", ha="right", fontsize=9, color=BAD)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("pedestrians"); ax.set_ylabel("collision-free success")
    ax.set_title("Where it breaks  (Wilson 95% band)", loc="left")
    ax.grid(True)

    ax = axs[1]
    med = [g.get_group(n)["min_sep_m"].median() for n in ns]
    q1 = [g.get_group(n)["min_sep_m"].quantile(0.25) for n in ns]
    q3 = [g.get_group(n)["min_sep_m"].quantile(0.75) for n in ns]
    ax.plot(ns, med, "-o", color=C[0], lw=2, ms=7)
    ax.fill_between(ns, q1, q3, color=C[0], alpha=0.18)
    ax.axhline(0, color=BAD, ls="--", lw=1.2)
    ax.set_xlabel("pedestrians"); ax.set_ylabel("min. clearance [m]")
    ax.set_title("Safety", loc="left")
    ax.grid(True)

    ax = axs[2]
    for key, c, lbl in (("fell", C[7], "fell"),
                        ("collided", C[3], "collided"),
                        ("reached_goal", C[1], "reached goal")):
        ax.plot(ns, [g.get_group(n)[key].mean() for n in ns], "-o",
                color=c, lw=2, ms=6, label=lbl)
        ax.text(ns[-1] + 0.3, g.get_group(ns[-1])[key].mean(), lbl,
                fontsize=9, color=INK, va="center")
    ax.set_ylim(-0.05, 1.15)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "50%", "100%"])
    ax.set_xlabel("pedestrians"); ax.set_ylabel("fraction of runs")
    ax.set_title("Failure modes", loc="left")
    ax.grid(True); ax.legend(fontsize=9, loc="center left")

    fig.suptitle("Q-E  Performance against crowd density",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "qe_density")

    t = g.agg(n=("success", "size"), success=("success", "mean"),
              collision_free=("success_collision_free", "mean"),
              fell=("fell", "mean"), collided=("collided", "mean"),
              min_sep=("min_sep_m", "median"), t_goal=("t_goal_s", "median"),
              path_ratio=("path_ratio", "median")).round(3)
    t.to_csv(os.path.join(RESULTS, "table_qe_density.csv"))
    print(t.to_string())


# ---------------------------------------------------------------------------
# Q-F: perception vs ground truth
# ---------------------------------------------------------------------------

def qf():
    df = _load("qf_perception")
    if df is None:
        return
    df["mode"] = np.where(df["sensor"], "depth camera", "ground truth")
    scens = [s for s in SCEN_ORDER if s in set(df["scenario"])]
    modes = ["ground truth", "depth camera"]
    colors = {"ground truth": C[0], "depth camera": C[3]}

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))

    # grouped bars: identity by legend, values direct-labelled
    ax = axs[0]
    x = np.arange(len(scens))
    w = 0.36
    for i, mode in enumerate(modes):
        rates, errs_lo, errs_hi = [], [], []
        for s in scens:
            sel = df[(df.scenario == s) & (df["mode"] == mode)]
            p, lo, hi = _rate(sel["success_collision_free"])
            rates.append(p); errs_lo.append(p - lo); errs_hi.append(hi - p)
        pos = x + (i - 0.5) * w
        ax.bar(pos, rates, width=w * 0.92, color=colors[mode], label=mode, zorder=2)
        ax.errorbar(pos, rates, yerr=[errs_lo, errs_hi], fmt="none",
                    ecolor=INK2, elinewidth=1.1, capsize=3, zorder=3)
        for xi, r in zip(pos, rates):
            ax.text(xi, r + 0.03, f"{100 * r:.0f}", ha="center", fontsize=8,
                    color=INK)
    ax.set_xticks(x); ax.set_xticklabels(scens)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "50%", "100%"])
    ax.set_ylabel("collision-free success")
    ax.set_title("Outcome", loc="left")
    ax.grid(axis="y", zorder=0)
    # above the plot: the ground-truth bars reach 100%, leaving no free corner
    ax.legend(fontsize=9, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.06))

    # paired clearance: each line is one (scenario, seed) pair
    ax = axs[1]
    piv = df.pivot_table(index=["scenario", "seed"], columns="mode",
                         values="min_sep_m")
    piv = piv.dropna()
    for (scen, seed), row in piv.iterrows():
        ax.plot([0, 1], [row["ground truth"], row["depth camera"]], "-",
                lw=1.0, color=INK2, alpha=0.45, zorder=2)
    ax.plot(np.zeros(len(piv)), piv["ground truth"], "o", ms=7,
            color=C[0], zorder=3, label="ground truth")
    ax.plot(np.ones(len(piv)), piv["depth camera"], "o", ms=7,
            color=C[3], zorder=3, label="depth camera")
    ax.axhline(0, color=BAD, ls="--", lw=1.2)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1]); ax.set_xticklabels(modes)
    ax.set_ylabel("min. clearance [m]")
    d = (piv["depth camera"] - piv["ground truth"])
    ax.set_title(f"Paired by scenario+seed\nmedian change "
                 f"{d.median():+.3f} m over {len(piv)} pairs", loc="left")
    ax.grid(axis="y")

    # Why it fails. The camera tracks the whole crowd (100% of pedestrians in
    # every scenario), so the interesting quantity is not how many it sees but
    # how big it thinks they are: the radius estimator has no upper bound and
    # ratchets a 0.30 m pedestrian up to several metres.
    ax = axs[2]
    sel = df[df["mode"] == "depth camera"]
    if len(sel) and "perc_rmed_m" in sel:
        x = np.arange(len(scens))
        med = [sel[sel.scenario == s]["perc_rmed_m"].median() for s in scens]
        mx = [sel[sel.scenario == s]["perc_rmax_m"].median() for s in scens]
        ax.bar(x - 0.19, med, width=0.34, color=C[3], zorder=2,
               label="typical (median over run)")
        ax.bar(x + 0.19, mx, width=0.34, color=C[5], zorder=2,
               label="worst in run")
        for xi, v in zip(x - 0.19, med):
            if np.isfinite(v):
                ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=9, color=INK)
        for xi, v in zip(x + 0.19, mx):
            if np.isfinite(v):
                ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=9, color=INK)
        ax.axhline(0.3, color=INK2, ls="--", lw=1.4)
        ax.text(-0.45, 0.45, "true pedestrian radius 0.30 m", ha="left",
                fontsize=9, color=INK2)
    ax.set_xticks(range(len(scens))); ax.set_xticklabels(scens)
    ax.set_ylabel("perceived radius [m]")
    ax.set_title("Why: it sees everyone, 10–30× too big", loc="left")
    ax.grid(axis="y", zorder=0)
    ax.legend(fontsize=9, loc="upper left")

    fig.suptitle("Q-F  What replacing ground truth with the onboard depth camera costs",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "qf_perception")

    t = df.groupby(["scenario", "mode"]).agg(
        n=("success", "size"), success=("success", "mean"),
        collision_free=("success_collision_free", "mean"),
        fell=("fell", "mean"), collided=("collided", "mean"),
        min_sep=("min_sep_m", "median"), t_goal=("t_goal_s", "median"),
        tracks=("mean_n_tracks", "median")).round(3)
    t.to_csv(os.path.join(RESULTS, "table_qf_perception.csv"))
    print(t.to_string())

    # paired significance: Wilcoxon signed-rank, the right test for paired,
    # small-n, not-necessarily-normal samples
    try:
        from scipy.stats import wilcoxon
        if len(piv) >= 6:
            stat, p = wilcoxon(piv["ground truth"], piv["depth camera"])
            print(f"\n  Wilcoxon signed-rank on paired clearance: "
                  f"W={stat:.1f}, p={p:.4f}  (n={len(piv)} pairs)")
    except Exception as e:
        print(f"  (wilcoxon skipped: {e})")


# ---------------------------------------------------------------------------
# Q-G: whose avoidance is it?
# ---------------------------------------------------------------------------

def qg():
    df = _load("qg_cooperation")
    if df is None:
        return
    df["crowd"] = np.where(df["crowd_avoids_robot"], "cooperative crowd",
                           "crowd blind to robot")
    df["cond"] = df["scenario"] + " ×" + df["n_peds"].astype(str)
    conds = list(dict.fromkeys(df.sort_values(["scenario", "n_peds"])["cond"]))
    modes = ["cooperative crowd", "crowd blind to robot"]
    colors = {"cooperative crowd": C[0], "crowd blind to robot": C[3]}

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axs[0]
    x = np.arange(len(conds))
    w = 0.36
    for i, mode in enumerate(modes):
        rates, lo_e, hi_e = [], [], []
        for c in conds:
            sel = df[(df.cond == c) & (df["crowd"] == mode)]
            p, lo, hi = _rate(sel["success_collision_free"])
            rates.append(p); lo_e.append(p - lo); hi_e.append(hi - p)
        pos = x + (i - 0.5) * w
        ax.bar(pos, rates, width=w * 0.92, color=colors[mode], label=mode, zorder=2)
        ax.errorbar(pos, rates, yerr=[lo_e, hi_e], fmt="none", ecolor=INK2,
                    elinewidth=1.1, capsize=3, zorder=3)
        for xi, r in zip(pos, rates):
            ax.text(xi, r + 0.03, f"{100 * r:.0f}", ha="center", fontsize=8,
                    color=INK)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=20)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "50%", "100%"])
    ax.set_ylabel("collision-free success")
    ax.set_title("Outcome", loc="left")
    ax.grid(axis="y", zorder=0)
    # above the plot: the cooperative bars run to 100%, so every in-axes corner
    # is occupied and a legend inside would sit on the data
    ax.legend(fontsize=9, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.06))

    ax = axs[1]
    piv = df.pivot_table(index=["scenario", "n_peds", "seed"], columns="crowd",
                         values="min_sep_m").dropna()
    for _, row in piv.iterrows():
        ax.plot([0, 1], [row["cooperative crowd"], row["crowd blind to robot"]],
                "-", lw=1.0, color=INK2, alpha=0.4, zorder=2)
    ax.plot(np.zeros(len(piv)), piv["cooperative crowd"], "o", ms=7,
            color=C[0], zorder=3)
    ax.plot(np.ones(len(piv)), piv["crowd blind to robot"], "o", ms=7,
            color=C[3], zorder=3)
    ax.axhline(0, color=BAD, ls="--", lw=1.2)
    ax.text(1.3, 0.005, "contact", ha="right", va="bottom", fontsize=9, color=BAD)
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["cooperative", "blind"])
    ax.set_ylabel("min. clearance [m]")
    d = piv["crowd blind to robot"] - piv["cooperative crowd"]
    ax.set_title(f"Paired clearance\nmedian change {d.median():+.3f} m "
                 f"({len(piv)} pairs)", loc="left")
    ax.grid(axis="y")

    ax = axs[2]
    for i, mode in enumerate(modes):
        vals = [df[(df.cond == c) & (df["crowd"] == mode)]["path_ratio"].median()
                for c in conds]
        ax.plot(x, vals, "-o", color=colors[mode], lw=2, ms=7, label=mode)
    ax.axhline(1.0, color=INK2, ls="--", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(conds, rotation=20)
    ax.set_ylabel("path length / straight line")
    ax.set_title("Detour the robot had to make", loc="left")
    ax.grid(True); ax.legend(fontsize=9)

    fig.suptitle("Q-G  Is the robot avoiding the crowd, or the crowd avoiding "
                 "the robot?", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "qg_cooperation")

    t = df.groupby(["scenario", "n_peds", "crowd"]).agg(
        n=("success", "size"), success=("success", "mean"),
        collision_free=("success_collision_free", "mean"),
        fell=("fell", "mean"), collided=("collided", "mean"),
        min_sep=("min_sep_m", "median"), t_goal=("t_goal_s", "median"),
        path_ratio=("path_ratio", "median")).round(3)
    t.to_csv(os.path.join(RESULTS, "table_qg_cooperation.csv"))
    print(t.to_string())
    try:
        from scipy.stats import wilcoxon
        if len(piv) >= 6:
            stat, p = wilcoxon(piv["cooperative crowd"],
                               piv["crowd blind to robot"])
            print(f"\n  Wilcoxon signed-rank on paired clearance: "
                  f"W={stat:.1f}, p={p:.4g}  (n={len(piv)} pairs)")
    except Exception as e:
        print(f"  (wilcoxon skipped: {e})")


ALL = dict(qa=qa, qb=qb, qb2=qb2, qc=qc, qc_traj=qc_traj, qd=qd, qe=qe, qf=qf,
           qg=qg)


def main():
    want = sys.argv[1:] or list(ALL)
    for name in want:
        if name not in ALL:
            print(f"unknown analysis '{name}'; options: {list(ALL)}")
            continue
        print(f"\n=== {name} ===")
        try:
            ALL[name]()
        except Exception as e:
            import traceback
            print(f"[fail] {name}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
