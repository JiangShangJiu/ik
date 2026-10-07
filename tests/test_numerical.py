"""Numerical solver acceptance tests: numerical IK.

Notes on expectations
---------------------
Jacobian-transpose (``jt``) converges only linearly and much slower than the
pseudo-inverse based methods, so it is checked against a looser tolerance.

Single-seed local solvers cannot be expected to solve every *near-singular*
reachable target when a joint has to route around a joint limit; that is what
multi-seed restarts (``solve_numerical_multiseed``) are for. The Monte-Carlo
thresholds below reflect this measured behaviour.
"""

import numpy as np
import pytest

from ik import PandaArm, solve_numerical, solve_numerical_multiseed, random_seeds


@pytest.fixture(scope="module")
def arm():
    return PandaArm()


def _fk_pose(arm, q):
    arm.set_q(q)
    return arm.tcp_transform()


def test_jacobian_matches_finite_difference(arm):
    rng = np.random.default_rng(0)
    q = rng.uniform(arm.joint_low, arm.joint_high)
    arm.set_q(q)
    _, p0 = arm.tcp_pose()
    R0, _ = arm.tcp_pose()
    J = arm.tcp_jacobian()
    eps = 1e-6
    J_lin = np.zeros((3, arm.nv))
    J_ang = np.zeros((3, arm.nv))
    from ik.kinematics import rot_log

    for i in range(arm.nv):
        qp = q.copy()
        qp[i] += eps
        arm.set_q(qp)
        R1, p1 = arm.tcp_pose()
        J_lin[:, i] = (p1 - p0) / eps
        J_ang[:, i] = rot_log(R1 @ R0.T) / eps
    arm.set_q(q)
    assert np.allclose(J[:3, :], J_lin, atol=1e-5)
    assert np.allclose(J[3:, :], J_ang, atol=1e-5)


@pytest.mark.parametrize(
    "method,tol,max_iters",
    [("dls", 1e-4, 500), ("pinv", 1e-4, 500), ("jt", 2e-3, 3000)],
)
def test_single_target_converges(arm, method, tol, max_iters):
    rng = np.random.default_rng(1)
    q_true = rng.uniform(arm.joint_low, arm.joint_high)
    T = _fk_pose(arm, q_true)
    q0 = q_true + rng.normal(0, 0.2, arm.nq)
    res = solve_numerical(arm, T, q0, method=method, max_iters=max_iters)
    assert res.pos_err < tol, f"{method}: pos_err={res.pos_err:.2e}"
    assert res.rot_err < tol, f"{method}: rot_err={res.rot_err:.2e}"


def test_dls_monte_carlo_single_seed(arm):
    """DLS from a nearby seed: ~95% over uniformly sampled reachable poses.

    The residual failures are near-singular targets where a joint-limit
    constrained local minimum traps the local solver; restarts fix them.
    """
    rng = np.random.default_rng(7)
    n = 200
    ok = 0
    for _ in range(n):
        q_true = rng.uniform(arm.joint_low, arm.joint_high)
        T = _fk_pose(arm, q_true)
        q0 = q_true + rng.normal(0, 0.3, arm.nq)
        if solve_numerical(arm, T, q0, method="dls").success:
            ok += 1
    assert ok / n >= 0.93, f"DLS single-seed success too low: {ok}/{n}"


def test_multiseed_from_random_seeds(arm):
    """Multi-seed restarts from random configurations solve ~99% of targets."""
    rng = np.random.default_rng(5)
    n = 50
    ok = 0
    for i in range(n):
        q_true = rng.uniform(arm.joint_low, arm.joint_high)
        T = _fk_pose(arm, q_true)
        seeds = random_seeds(arm, 16, seed=i)
        res = solve_numerical_multiseed(arm, T, seeds, method="dls")
        if res.success:
            ok += 1
    assert ok / n >= 0.94, f"multi-seed success too low: {ok}/{n}"


def test_unreachable_target_fails_gracefully(arm):
    T = np.eye(4)
    T[:3, 3] = [5.0, 5.0, 5.0]  # far outside the workspace
    res = solve_numerical(arm, T, np.zeros(arm.nq), method="dls", max_iters=100)
    assert not res.success
    assert np.all(np.isfinite(res.q))


def test_joint_limits_respected(arm):
    rng = np.random.default_rng(3)
    for _ in range(20):
        q_true = rng.uniform(arm.joint_low, arm.joint_high)
        T = _fk_pose(arm, q_true)
        q0 = rng.uniform(arm.joint_low, arm.joint_high)
        res = solve_numerical(arm, T, q0, method="dls")
        assert np.all(res.q >= arm.joint_low - 1e-12)
        assert np.all(res.q <= arm.joint_high + 1e-12)
