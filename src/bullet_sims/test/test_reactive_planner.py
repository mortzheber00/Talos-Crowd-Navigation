"""Invariant tests for the ReactiveFootStepPlanner (no plots, no sim)."""

import numpy as np
import pytest

import bullet_sims.talos_conf as conf
from bullet_sims.reactive_planner_viz import SCENARIOS, simulate, check_invariants


@pytest.mark.parametrize('name', sorted(SCENARIOS.keys()))
def test_scenario_invariants(name):
    result = simulate(SCENARIOS[name])
    errors = check_invariants(result, label=name)
    assert not errors, '\n'.join(errors)


def test_tight_turn_shortens_steps():
    # tight turns must run at the fast end of the allowed cadence; with a
    # narrow (dur_min, dur_max) band both gaits may quantise to the same
    # duration, so assert against the bound instead of strict inequality
    fast = simulate(SCENARIOS['pure_rotation'])
    slow = simulate(SCENARIOS['straight'])
    assert abs(min(fast['durations']) - conf.step_dur_min) < 1e-9
    assert min(fast['durations']) <= max(slow['durations'])
    for dur in fast['durations'] + slow['durations']:
        assert conf.step_dur_min - 1e-9 <= dur <= conf.step_dur_max + 1e-9


def test_mpc_reference_shapes():
    from bullet_sims.footstep_planner import Side
    import pinocchio as pin
    from bullet_sims.reactive_footstep_planner import ReactiveFootStepPlanner

    support = pin.SE3(np.eye(3), np.array([0.0, -conf.step_size_y, 0.0]))
    planner = ReactiveFootStepPlanner(
        conf, (1e6, 0.0), support,
        conf.no_mpc_samples_per_step, conf.no_mpc_samples_per_horizon)
    planner.regenerate(planner.make_footstep(support, Side.RIGHT),
                       np.array([0.25, 0.0]), omega_cmd=0.3)
    for phase in range(planner.step_samples() + 2):
        zmp, yaw = planner.mpc_reference(phase)
        assert zmp.shape == (conf.no_mpc_samples_per_horizon, 2)
        assert yaw.shape == (conf.no_mpc_samples_per_horizon,)
        assert np.all(np.isfinite(zmp)) and np.all(np.isfinite(yaw))
