"""reactive_planner_viz.py
Standalone test + visualisation of the ReactiveFootStepPlanner (no MuJoCo,
no Drake, no TSID needed).

Feeds command sequences (straight, pure rotation, arcs, lateral, and an
adversarial crossing command) to the planner, executes the footsteps
kinematically, checks the stability invariants numerically and plots the
resulting footstep patterns top-down.

Run:  ros2 run bullet_sims reactive_planner_viz
      (or python3 -m bullet_sims.reactive_planner_viz)

Invariants checked after every step (exit code 1 on violation):
  * no-crossing: the left foot stays >= conf.min_foot_sep left of the right
    foot in the heading frame;
  * per-step foot yaw delta <= conf.max_dyaw_per_step;
  * step duration within [conf.step_dur_min, conf.step_dur_max] and a
    multiple of conf.dt_mpc.
"""

import os
import sys

import numpy as np
import pinocchio as pin

import bullet_sims.talos_conf as conf
from bullet_sims.footstep_planner import Side
from bullet_sims.reactive_footstep_planner import (
    ReactiveFootStepPlanner, wrap_angle, yaw_of, _e_left)


################################################################################
# command scenarios: fn(step_idx, heading) -> (v_world (2,), omega)
################################################################################

def _fwd(heading, speed):
    """World-frame velocity pointing along the current heading."""
    return speed * np.array([np.cos(heading), np.sin(heading)])


SCENARIOS = {
    # constant forward walk
    'straight': lambda k, psi: (np.array([0.3, 0.0]), 0.0),
    # spin in place (v ~ 0, omega != 0)
    'pure_rotation': lambda k, psi: (np.zeros(2), 0.4),
    # forward along the heading while turning -> circular arc
    'arc_left': lambda k, psi: (_fwd(psi, 0.25), 0.3),
    'arc_right': lambda k, psi: (_fwd(psi, 0.25), -0.3),
    # pure sidestep (fixed heading)
    'lateral': lambda k, psi: (np.array([0.0, 0.2]), 0.0),
    # adversarial: strong lateral command toward the planted foot, flipping
    # every step, plus an opposing turn -> demands crossing steps
    'adversarial_cross': lambda k, psi: (
        np.array([0.1, 0.5 * (-1.0) ** k]), 0.4 * (-1.0) ** (k + 1)),
}


################################################################################
# kinematic rollout + invariant checks
################################################################################

def simulate(cmd_fn, n_steps=14):
    """Roll the planner forward, executing each committed step kinematically.

    Returns a dict with the footstep list, per-step durations, headings and
    the (left, right) foot positions after every step.
    """
    # both feet straddle the origin; the RIGHT foot is the initial support
    feet = {
        Side.LEFT: pin.SE3(np.eye(3), np.array([0.0, conf.step_size_y, 0.0])),
        Side.RIGHT: pin.SE3(np.eye(3), np.array([0.0, -conf.step_size_y, 0.0])),
    }
    goal = (1e6, 0.0)     # far away so at_goal never triggers
    planner = ReactiveFootStepPlanner(
        conf, goal, feet[Side.RIGHT],
        conf.no_mpc_samples_per_step, conf.no_mpc_samples_per_horizon)

    support_side = Side.RIGHT
    support = planner.make_footstep(feet[support_side], support_side)

    out = {'steps': [], 'durations': [], 'headings': [], 'feet': []}
    for k in range(n_steps):
        v, om = cmd_fn(k, planner.heading)
        planner.regenerate(support, v, omega_cmd=om)
        fs = planner.swing_target()

        feet[fs.side] = fs.pose
        out['steps'].append(fs)
        out['durations'].append(planner.step_duration())
        out['headings'].append(planner.heading)
        out['feet'].append((feet[Side.LEFT].copy(), feet[Side.RIGHT].copy()))

        support = fs   # the new foot becomes the support of the next step
    return out


def check_invariants(result, label=''):
    """Numeric stability checks on a simulate() result. Returns list of errors."""
    errors = []
    prev_yaws = {}
    for k, (T_l, T_r) in enumerate(result['feet']):
        fs = result['steps'][k]
        p_l = np.asarray(T_l.translation)[:2]
        p_r = np.asarray(T_r.translation)[:2]
        yaw_l, yaw_r = yaw_of(T_l), yaw_of(T_r)

        # mean heading of the two feet (circular mean, wrap-safe)
        psi = np.arctan2(np.sin(yaw_l) + np.sin(yaw_r),
                         np.cos(yaw_l) + np.cos(yaw_r))
        sep = float(_e_left(psi) @ (p_l - p_r))
        if sep < conf.min_foot_sep - 1e-6:
            errors.append(f'{label} step {k}: feet cross/too close '
                          f'(sep={sep:.3f} < {conf.min_foot_sep})')

        # per-step yaw delta of the swing foot
        yaw_new = yaw_of(fs.pose)
        if fs.side in prev_yaws:
            dyaw = abs(wrap_angle(yaw_new - prev_yaws[fs.side]))
            # the same foot rests for two steps -> up to 2x the per-step clamp
            if dyaw > 2.0 * conf.max_dyaw_per_step + 1e-9:
                errors.append(f'{label} step {k}: foot yaw delta too large '
                              f'({dyaw:.3f} rad)')
        prev_yaws[fs.side] = yaw_new

        # step duration bounds + dt_mpc quantisation
        dur = result['durations'][k]
        if not (conf.step_dur_min - 1e-9 <= dur <= conf.step_dur_max + 1e-9):
            errors.append(f'{label} step {k}: duration {dur:.2f}s out of '
                          f'[{conf.step_dur_min}, {conf.step_dur_max}]')
        if abs(dur / conf.dt_mpc - round(dur / conf.dt_mpc)) > 1e-9:
            errors.append(f'{label} step {k}: duration {dur}s not a multiple '
                          f'of dt_mpc={conf.dt_mpc}')
    return errors


################################################################################
# plotting
################################################################################

def _foot_polygon(pose):
    """World-frame (4,2) corner array of the foot rectangle for a pose."""
    corners = np.array([[conf.lfxp, conf.lfyp],
                        [conf.lfxp, -conf.lfyn],
                        [-conf.lfxn, -conf.lfyn],
                        [-conf.lfxn, conf.lfyp]])
    R = np.asarray(pose.rotation)[:2, :2]
    p = np.asarray(pose.translation)[:2]
    return corners @ R.T + p


def plot_scenarios(results, out_path=None, show=False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    n = len(results)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axs = np.atleast_1d(axs).ravel()

    for ax, (name, res) in zip(axs, results.items()):
        steps = res['steps']
        n_steps = len(steps)
        for k, fs in enumerate(steps):
            alpha = 0.25 + 0.7 * (k + 1) / n_steps
            color = 'tab:blue' if fs.side == Side.LEFT else 'tab:red'
            ax.add_patch(Polygon(_foot_polygon(fs.pose), closed=True,
                                 facecolor=color, edgecolor='k',
                                 lw=0.5, alpha=alpha))
            # yaw quiver at the foot centre
            yaw = yaw_of(fs.pose)
            p = np.asarray(fs.pose.translation)[:2]
            ax.quiver(p[0], p[1], np.cos(yaw), np.sin(yaw),
                      scale=25, width=0.004, color='k', alpha=0.6)
        # base path = midpoint of the two feet after every step
        mid = np.array([(f[0].translation[:2] + f[1].translation[:2]) / 2.0
                        for f in res['feet']])
        ax.plot(mid[:, 0], mid[:, 1], '-', color='tab:green', lw=1.5,
                label='base path')
        durs = res['durations']
        ax.set_title(f'{name}\nstep durations {min(durs):.1f}-{max(durs):.1f} s '
                     f'(blue=L, red=R)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    for ax in axs[n:]:
        ax.axis('off')
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f'saved footstep plots to {out_path}')
    if show:
        plt.show()
    else:
        plt.close(fig)


################################################################################
# main
################################################################################

def main(args=None):
    results = {}
    all_errors = []
    for name, cmd_fn in SCENARIOS.items():
        res = simulate(cmd_fn)
        results[name] = res
        errors = check_invariants(res, label=name)
        all_errors += errors
        status = 'PASS' if not errors else 'FAIL'
        durs = res['durations']
        print(f'[{status}] {name:18s} steps={len(res["steps"])}  '
              f'dur {min(durs):.1f}-{max(durs):.1f} s')
        for e in errors:
            print('   ', e)

    plots_dir = os.environ.get(
        'WALKING_PLOTS_DIR',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'T7'))
    plot_scenarios(results,
                   out_path=os.path.join(plots_dir, 'reactive_planner_test.png'),
                   show=bool(os.environ.get('WALKING_SHOW_PLOTS')))

    if all_errors:
        print(f'{len(all_errors)} invariant violation(s)')
        sys.exit(1)
    print('all scenarios passed')


if __name__ == '__main__':
    main()
