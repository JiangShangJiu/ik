"""Semi-analytic solver acceptance tests: arm-angle semi-analytic IK."""

import numpy as np
import pytest

from ik import (
    PANDA_MDH,
    PandaArm,
    analytic_seeds,
    elbow_circle,
    shoulder_solve,
    solve_analytic,
    solve_analytic_best,
    wrist_branches,
)
from ik.kinematics import foot_of_perpendicular, line_intersection, mdh_transform


@pytest.fixture(scope="module")
def arm():
    return PandaArm()


def test_wrist_branches_roundtrip():
    """Both wrist branches must reproduce the same R4^7."""
    from ik.analytic import _R04, _R03  # noqa: F401

    def Rx(a):
        return mdh_transform(a, 0, 0, 0)[:3, :3]

    def Rz(a):
        return mdh_transform(0, 0, 0, a)[:3, :3]

    rng = np.random.default_rng(0)
    for _ in range(50):
        q5, q6, q7 = rng.uniform(-3, 3), rng.uniform(0.2, 3.0), rng.uniform(-3, 3)
        R = Rx(-np.pi / 2) @ Rz(q5) @ Rx(np.pi / 2) @ Rz(q6) @ Rx(np.pi / 2) @ Rz(q7)
        for (a5, a6, a7) in wrist_branches(R):
            Rb = Rx(-np.pi / 2) @ Rz(a5) @ Rx(np.pi / 2) @ Rz(a6) @ Rx(np.pi / 2) @ Rz(a7)
            assert np.allclose(Rb, R, atol=1e-9)


def test_shoulder_solve_recovers_true_q(arm):
    """Given the true elbow centre and joint-4 axis, recover q1..q3."""
    S = arm.geometry().shoulder_center
    rng = np.random.default_rng(1)
    ok = 0
    for _ in range(20):
        q = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q)
        a, ax = arm.world_lines()
        E = foot_of_perpendicular(S, a[3], ax[3])
        A4 = ax[3] / np.linalg.norm(ax[3])
        cands = shoulder_solve(E, A4, S)
        hit = False
        for (q1, q2, q3) in cands:
            # verify by forward kinematics of the arm geometry
            arm.set_q([q1, q2, q3, q[3], q[4], q[5], q[6]])
            a2, ax2 = arm.world_lines()
            E2 = foot_of_perpendicular(S, a2[3], ax2[3])
            A42 = ax2[3] / np.linalg.norm(ax2[3])
            if np.linalg.norm(E2 - E) < 1e-6 and np.linalg.norm(A42 - A4) < 1e-6:
                hit = True
        if hit:
            ok += 1
    assert ok == 20


def test_elbow_circle_contains_true_elbow(arm):
    S = arm.geometry().shoulder_center
    geom = arm.geometry()
    rng = np.random.default_rng(2)
    for _ in range(10):
        q = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q)
        a, ax = arm.world_lines()
        E_true = foot_of_perpendicular(S, a[3], ax[3])
        W = line_intersection(a[4], ax[4], a[5], ax[5])
        # find the arm angle of the true elbow
        res = elbow_circle(W, S, 0.0, geom)
        assert res is not None
        _, n, h, v, rho = res
        d = E_true - n
        psi = np.arctan2(d @ v, d @ h)
        res2 = elbow_circle(W, S, psi, geom)
        assert np.allclose(res2[0], E_true, atol=1e-9)


def test_analytic_seeds_reach_target(arm):
    """At least one seed, after polishing, must reach a random target."""
    rng = np.random.default_rng(5)
    n = 15
    ok = 0
    for _ in range(n):
        q_true = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q_true)
        T = arm.tcp_transform()
        best = solve_analytic_best(arm, T)
        if best is not None and best.success:
            ok += 1
    assert ok == n, f"only {ok}/{n} targets solved"


def test_analytic_multiple_solutions(arm):
    """A generic reachable pose should admit several distinct solutions."""
    rng = np.random.default_rng(7)
    q_true = rng.uniform(arm.joint_low, arm.joint_high)
    arm.set_q(q_true)
    T = arm.tcp_transform()
    sols = solve_analytic(arm, T)
    assert len(sols) >= 4, f"only {len(sols)} solutions found"
    for s in sols:
        arm.set_q(s.q)
        R, p = arm.tcp_pose()
        assert np.linalg.norm(p - T[:3, 3]) < 1e-3
        assert np.linalg.norm(R - T[:3, :3]) < 1e-3


def test_analytic_solutions_within_limits(arm):
    rng = np.random.default_rng(9)
    for _ in range(10):
        q_true = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q_true)
        T = arm.tcp_transform()
        for s in solve_analytic(arm, T):
            assert np.all(s.q >= arm.joint_low - 1e-9)
            assert np.all(s.q <= arm.joint_high + 1e-9)


def test_analytic_monte_carlo(arm):
    rng = np.random.default_rng(11)
    n = 40
    ok = 0
    for _ in range(n):
        q_true = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q_true)
        T = arm.tcp_transform()
        if solve_analytic_best(arm, T) is not None:
            ok += 1
    assert ok / n >= 0.95, f"analytic success too low: {ok}/{n}"
