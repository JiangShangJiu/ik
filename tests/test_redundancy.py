"""Redundancy resolution acceptance tests: null-space redundancy resolution."""

import numpy as np
import pytest

from ik import (
    PandaArm,
    limit_avoidance,
    manipulability,
    nullspace_projector,
    nullspace_step,
    pose_error,
    posture,
    solve_numerical,
    solve_redundant,
)
from ik.redundancy import _limit_gradient


@pytest.fixture(scope="module")
def arm():
    return PandaArm()


def test_nullspace_projector_annihilates_task(arm):
    rng = np.random.default_rng(0)
    q = rng.uniform(arm.joint_low, arm.joint_high)
    arm.set_q(q)
    J = arm.tcp_jacobian()
    N = nullspace_projector(J, lam=1e-6)
    # For a well-conditioned J, J @ N should be ~0.
    assert np.linalg.norm(J @ N) < 1e-3


def test_nullspace_step_preserves_pose(arm):
    """A small null-space step must not change the TCP pose (to first order)."""
    rng = np.random.default_rng(1)
    q = rng.uniform(arm.joint_low, arm.joint_high)
    arm.set_q(q)
    R0, p0 = arm.tcp_pose()

    task = posture(np.zeros(arm.nq), weight=1.0)
    dq = nullspace_step(arm, q, task, lam=1e-6)
    assert np.linalg.norm(dq) > 1e-6  # non-trivial null-space direction

    scale = 1e-3
    arm.set_q(q + scale * dq)
    R1, p1 = arm.tcp_pose()
    e = pose_error(R0, p0, R1, p1)
    assert np.linalg.norm(e) < 1e-3, f"null-space step moved the target: {e}"


def test_posture_secondary_improves_posture(arm):
    rng = np.random.default_rng(2)
    q_des = rng.uniform(arm.joint_low, arm.joint_high)
    q_true = rng.uniform(arm.joint_low, arm.joint_high)
    arm.set_q(q_true)
    T = arm.tcp_transform()
    q0 = rng.uniform(arm.joint_low, arm.joint_high)

    res_plain = solve_numerical(arm, T, q0, method="dls")
    res_posture = solve_redundant(
        arm, T, q0, (posture(q_des, weight=2.0),)
    )
    assert res_posture.success
    assert np.linalg.norm(res_posture.q - q_des) <= np.linalg.norm(
        res_plain.q - q_des
    ) + 1e-6


def test_limit_gradient_direction(arm):
    lo, hi = arm.joint_low, arm.joint_high
    q = (lo + hi) / 2
    # At the lower limit the gradient should push upward.
    q_low = lo + 1e-6
    g = _limit_gradient(q_low, lo, hi, margin=0.35)
    assert g[0] > 0
    # At the upper limit the gradient should push downward.
    q_high = hi - 1e-6
    g = _limit_gradient(q_high, lo, hi, margin=0.35)
    assert g[0] < 0


def test_solve_redundant_converges_with_secondaries(arm):
    rng = np.random.default_rng(4)
    tasks = (
        limit_avoidance(margin=0.35, weight=0.2),
        manipulability(weight=0.1),
    )
    ok = 0
    n = 30
    for _ in range(n):
        q_true = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q_true)
        T = arm.tcp_transform()
        q0 = q_true + rng.normal(0, 0.3, arm.nq)
        res = solve_redundant(arm, T, q0, tasks)
        if res.success:
            ok += 1
        assert np.all(res.q >= arm.joint_low - 1e-12)
        assert np.all(res.q <= arm.joint_high + 1e-12)
    assert ok / n >= 0.9, f"redundant solve success too low: {ok}/{n}"
