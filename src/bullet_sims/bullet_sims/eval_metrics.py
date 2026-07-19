"""eval_metrics.py
Turns one run's log buffers into (a) a summary dict of scalar metrics and (b) a
compressed archive of the raw time series, so a batch of runs can be aggregated
offline without re-simulating.

Written for the evaluation of the crowd-navigation stack; nothing here feeds
back into the controller. The summary is the unit of analysis: one JSON object
per run, one row per run in the sweep tables.

Conventions:
  * separations are surface-to-surface gaps [m]; negative = interpenetration.
  * a run is a SUCCESS if the robot reached the goal and never fell.
    Collisions are reported separately, so "collision-free success rate"
    (the project's stated bar) is success AND no collision.
"""

import json
import os

import numpy as np


# CoM height thresholds [m]. Nominal walking height is ~0.87.
COM_Z_FALL = 0.70        # below this the robot is going down
COM_Z_HEALTHY = 0.75     # gait diagnostics only trust samples above this


def _first_true_time(mask, time):
    """Time of the first True in ``mask``, or NaN."""
    idx = np.flatnonzero(mask)
    return float(time[idx[0]]) if idx.size else float('nan')


def _count_excursions(mask):
    """Number of separate contiguous True runs (e.g. distinct collisions)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return 0
    return int(np.count_nonzero(mask[1:] & ~mask[:-1]) + mask[0])


def _path_length(xy):
    """Arc length of a 2D path, NaN samples dropped."""
    ok = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
    p = xy[ok]
    if len(p) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def _timing_stats(samples):
    """mean/p50/p95/max of a list of durations [s], reported in milliseconds."""
    a = np.asarray(samples, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0, mean_ms=float('nan'), p50_ms=float('nan'),
                    p95_ms=float('nan'), max_ms=float('nan'))
    return dict(n=int(a.size),
                mean_ms=float(1e3 * a.mean()),
                p50_ms=float(1e3 * np.percentile(a, 50)),
                p95_ms=float(1e3 * np.percentile(a, 95)),
                max_ms=float(1e3 * a.max()))


def _velocity_tracking(L, goal_tol):
    """Q-B: does the footstep bridge actually execute the ORCA command?

    Compares the commanded velocity ``v_cmd`` against the CoM velocity the robot
    achieves. Both are world-frame 2D. Three things are worth separating:

      * gain      -- |v_achieved| / |v_cmd|, how much of the commanded speed
                     survives the bridge.
      * lag       -- the shift [s] maximising the cross-correlation of the two
                     speed signals; the bridge is committed ~1 step ahead, so a
                     lag near ``step_dur`` is the expected cost of the timescale
                     bridge rather than a bug.
      * lateral   -- the same, restricted to the component perpendicular to the
                     goal direction, i.e. the avoidance part of the command. A
                     bridge that tracks forward motion but drops the lateral
                     command would look fine overall and still not avoid anyone.

    The CoM sways laterally at the step frequency (~+-0.3 m/s), which swamps the
    per-tick signal, so everything here is computed on a moving average over one
    step period.
    """
    t = L.TIME
    ok = np.isfinite(t) & np.isfinite(L.VCMD[:, 0]) & np.isfinite(L.COM_VEL_PB[:, 0])
    ok &= L.COM_POS_PB[:, 2] > COM_Z_HEALTHY
    if ok.sum() < 50:
        return dict(n=int(ok.sum()))

    dt = float(np.nanmedian(np.diff(t)))
    win = max(1, int(round(L.STEP_DUR / dt)))

    def smooth(a):
        k = np.ones(win) / win
        return np.convolve(a, k, mode='same')

    v_cmd = L.VCMD[ok]
    v_ach = L.COM_VEL_PB[ok, :2]

    # only while the robot is actually being asked to move
    moving = np.linalg.norm(v_cmd, axis=1) > 0.05
    if moving.sum() < 50:
        return dict(n=int(moving.sum()))

    sp_cmd = smooth(np.linalg.norm(v_cmd, axis=1))
    sp_ach = smooth(np.linalg.norm(v_ach, axis=1))

    # lateral = perpendicular to the instantaneous goal direction
    to_goal = L.GOAL[None, :] - L.ROBOT_XY[ok]
    n = np.linalg.norm(to_goal, axis=1, keepdims=True)
    fwd = np.divide(to_goal, np.where(n > 1e-6, n, 1.0))
    left = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
    lat_cmd = smooth(np.einsum('ij,ij->i', v_cmd, left))
    lat_ach = smooth(np.einsum('ij,ij->i', v_ach, left))

    def gain(a, b, sel):
        num, den = np.abs(b[sel]).mean(), np.abs(a[sel]).mean()
        return float(num / den) if den > 1e-6 else float('nan')

    def corr(a, b, sel):
        a, b = a[sel], b[sel]
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    def best_lag(a, b, sel):
        """Shift [s] of b relative to a that maximises correlation (b lags a)."""
        a, b = a[sel] - a[sel].mean(), b[sel] - b[sel].mean()
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float('nan')
        max_shift = int(round(2.0 / dt))
        best, best_c = 0, -np.inf
        for s in range(0, max_shift):
            if len(a) - s < 50:
                break
            c = np.corrcoef(a[:len(a) - s], b[s:])[0, 1]
            if np.isfinite(c) and c > best_c:
                best_c, best = c, s
        return float(best * dt)

    lat_cmd_rms = float(np.sqrt(np.mean(lat_cmd[moving] ** 2)))
    # A run where ORCA never asked for avoidance says nothing about whether the
    # bridge executes it, and the ratio degenerates (small over small): an empty
    # world produced lat_gain = 24 off a 5 mm command. Report NaN instead.
    lat_meaningful = lat_cmd_rms > 0.05

    return dict(
        n=int(moving.sum()),
        cmd_speed_mean=float(np.abs(sp_cmd[moving]).mean()),
        ach_speed_mean=float(np.abs(sp_ach[moving]).mean()),
        speed_gain=gain(sp_cmd, sp_ach, moving),
        speed_corr=corr(sp_cmd, sp_ach, moving),
        speed_lag_s=best_lag(sp_cmd, sp_ach, moving),
        lat_gain=gain(lat_cmd, lat_ach, moving) if lat_meaningful else float('nan'),
        lat_corr=corr(lat_cmd, lat_ach, moving) if lat_meaningful else float('nan'),
        lat_lag_s=(best_lag(lat_cmd, lat_ach, moving) if lat_meaningful
                   else float('nan')),
        lat_cmd_rms=lat_cmd_rms,
    )


def summarize(L, meta, goal_tol=0.3):
    """Reduce one run's logs to a flat dict of scalar metrics.

    Args:
        L: the log namespace filled by ``walking._run_loop``.
        meta (dict): the run's configuration (scenario, seed, params, ...),
            copied into the summary so a sweep table needs nothing else.
        goal_tol (float): [m] distance counting as "goal reached".
    """
    t = L.TIME
    fin = np.isfinite(t)
    t_end = float(np.nanmax(t[fin])) if fin.any() else float('nan')

    com_z = L.COM_POS_PB[:, 2]
    fell = bool(np.nanmin(np.where(np.isfinite(com_z), com_z, np.inf)) < COM_Z_FALL)
    t_fall = _first_true_time(np.isfinite(com_z) & (com_z < COM_Z_FALL), t)

    reached = np.isfinite(L.DIST_GOAL) & (L.DIST_GOAL <= goal_tol)
    t_goal = _first_true_time(reached, t)
    reached_goal = bool(np.isfinite(t_goal))
    # a goal reached only *after* going down does not count
    if reached_goal and fell and np.isfinite(t_fall) and t_fall < t_goal:
        reached_goal = False
        t_goal = float('nan')

    # separations: ground truth is the safety metric; the perceived one says
    # what the robot believed it had (they differ only when perception is on)
    sep = L.MIN_SEP
    sep_ok = np.isfinite(sep)
    # ignore samples after a fall: a collapsed robot drifting into a pedestrian
    # is not an avoidance failure
    if np.isfinite(t_fall):
        sep_ok &= t < t_fall
    collided = bool((sep[sep_ok] < 0).any()) if sep_ok.any() else False

    horizon = t_goal if reached_goal else t_end
    upto = fin & (t <= horizon) if np.isfinite(horizon) else fin
    path_len = _path_length(L.ROBOT_XY[upto])
    start = L.ROBOT_XY[np.flatnonzero(np.isfinite(L.ROBOT_XY[:, 0]))[0]] \
        if np.isfinite(L.ROBOT_XY[:, 0]).any() else np.array([np.nan, np.nan])
    straight = float(np.hypot(*(L.GOAL - start)))

    # ZMP tracking, healthy samples only: the gait-stability side of the story
    healthy = np.isfinite(com_z) & (com_z > COM_Z_HEALTHY)
    zok = healthy & np.isfinite(L.ZMP_EST[:, 0]) & np.isfinite(L.ZMP_REF_LOG[:, 0])
    zmp_err = (np.linalg.norm(L.ZMP_REF_LOG[zok] - L.ZMP_EST[zok], axis=1)
               if zok.any() else np.array([np.nan]))

    out = dict(meta)
    out.update(
        # --- outcome ---
        success=bool(reached_goal and not fell),
        success_collision_free=bool(reached_goal and not fell and not collided),
        reached_goal=reached_goal,
        fell=fell,
        collided=collided,
        n_collisions=_count_excursions(sep_ok & (sep < 0)),
        t_goal_s=t_goal,
        t_fall_s=t_fall,
        t_end_s=t_end,
        # --- safety ---
        min_sep_m=float(np.nanmin(sep[sep_ok])) if sep_ok.any() else float('nan'),
        min_sep_p5_m=(float(np.percentile(sep[sep_ok], 5))
                      if sep_ok.any() else float('nan')),
        mean_sep_m=float(np.nanmean(sep[sep_ok])) if sep_ok.any() else float('nan'),
        frac_time_below_0m2=(float(np.mean(sep[sep_ok] < 0.2))
                             if sep_ok.any() else float('nan')),
        # --- efficiency ---
        path_len_m=path_len,
        straight_len_m=straight,
        path_ratio=float(path_len / straight) if straight > 1e-6 else float('nan'),
        final_dist_goal_m=(float(L.DIST_GOAL[upto][-1]) if upto.any()
                           else float('nan')),
        mean_speed_mps=(float(path_len / horizon)
                        if np.isfinite(horizon) and horizon > 0 else float('nan')),
        # --- gait quality ---
        zmp_rmse_m=float(np.sqrt(np.mean(zmp_err ** 2))),
        zmp_max_m=float(np.nanmax(zmp_err)),
        frac_healthy=float(np.mean(healthy[fin])) if fin.any() else float('nan'),
        # --- perception (only meaningful when the camera path is on) ---
        mean_n_tracks=float(np.nanmean(L.N_TRACKS[fin])) if fin.any() else float('nan'),
        # The believed clearance and the largest believed radius. Together they
        # say whether a perception run failed because the robot could not see
        # the crowd, or because it saw it wrongly: tracks present and correct in
        # number but carrying a runaway radius drive ORCA into its overlap
        # branch, which commands a full-speed escape from an obstacle that is
        # not there.
        min_sep_perc_m=(float(np.nanmin(L.MIN_SEP_PERC[fin]))
                        if fin.any() and np.isfinite(L.MIN_SEP_PERC[fin]).any()
                        else float('nan')),
        med_sep_perc_m=(float(np.nanmedian(L.MIN_SEP_PERC[fin]))
                        if fin.any() and np.isfinite(L.MIN_SEP_PERC[fin]).any()
                        else float('nan')),
        perc_rmax_m=(float(np.nanmax(L.PERC_RMAX[fin]))
                     if fin.any() and np.isfinite(L.PERC_RMAX[fin]).any()
                     else float('nan')),
        perc_rmed_m=(float(np.nanmedian(L.PERC_RMAX[fin]))
                     if fin.any() and np.isfinite(L.PERC_RMAX[fin]).any()
                     else float('nan')),
        n_peds_true=int(L.PED_XY.shape[1]),
    )
    out['timing'] = {k: _timing_stats(v) for k, v in L.TIMERS.items()}
    out['vel_tracking'] = _velocity_tracking(L, goal_tol)
    return out


def dump(L, meta, out_dir, run_id, goal_tol=0.3, save_series=True):
    """Write ``<run_id>.json`` (summary) and ``<run_id>.npz`` (time series).

    Returns the summary dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    summary = summarize(L, meta, goal_tol=goal_tol)

    with open(os.path.join(out_dir, f"{run_id}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    if save_series:
        series = dict(
            TIME=L.TIME, ROBOT_XY=L.ROBOT_XY, PED_XY=L.PED_XY,
            MIN_SEP=L.MIN_SEP, MIN_SEP_PERC=L.MIN_SEP_PERC,
            DIST_GOAL=L.DIST_GOAL, VCMD=L.VCMD, VPREF=L.VPREF,
            COM_VEL_PB=L.COM_VEL_PB, COM_POS_PB=L.COM_POS_PB,
            ZMP_REF=L.ZMP_REF_LOG, ZMP_EST=L.ZMP_EST, DCM_EST=L.DCM_EST,
            N_TRACKS=L.N_TRACKS, PERC_RMAX=L.PERC_RMAX,
            GOAL=L.GOAL, PED_RADII=L.PED_RADII,
        )
        np.savez_compressed(os.path.join(out_dir, f"{run_id}.npz"), **series)

    return summary
