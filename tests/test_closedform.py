"""Closed-form solvers for the two new Pieper stories (UR5e, Lite 6)."""

import numpy as np
import pytest

from ik import Lite6Solver, Ur5eSolver, load_robot


@pytest.fixture(scope="module")
def arms():
    return {name: load_robot(name) for name in ("ur5e", "lite6")}


def test_ur5e_recovers_known_solution(arms):
    arm = arms["ur5e"]
    solver = Ur5eSolver(arm)
    rng = np.random.default_rng(0)
    for _ in range(25):
        q_true = rng.uniform(arm.joint_low * 0.8, arm.joint_high * 0.8)
        arm.set_q(q_true)
        T = arm.tcp_transform()
        sols = solver.solve(T)
        assert sols, "no solution found"
        assert max(s.pos_err for s in sols) < 1e-9
        assert max(s.rot_err for s in sols) < 1e-9
        assert any(
            np.linalg.norm(np.mod(s.q - q_true + np.pi, 2 * np.pi) - np.pi) < 1e-4
            for s in sols
        )


def test_ur5e_residual_is_machine_precision(arms):
    arm = arms["ur5e"]
    solver = Ur5eSolver(arm)
    rng = np.random.default_rng(1)
    for _ in range(15):
        arm.set_q(rng.uniform(arm.joint_low * 0.7, arm.joint_high * 0.7))
        for s in solver.solve(arm.tcp_transform()):
            assert s.pos_err < 1e-10 and s.rot_err < 1e-9


def test_lite6_orientation_is_exact(arms):
    arm = arms["lite6"]
    solver = Lite6Solver(arm)
    rng = np.random.default_rng(2)
    for _ in range(15):
        q_true = rng.uniform(arm.joint_low * 0.8, arm.joint_high * 0.8)
        arm.set_q(q_true)
        T = arm.tcp_transform()
        sols = solver.solve(T, tol_pos=1e-4)
        assert sols, "no solution found"
        # orientation is exact; position is floored by the model's CAD rounding
        assert max(s.rot_err for s in sols) < 1e-9
        assert max(s.pos_err for s in sols) < 1e-4


def test_lite6_wrist_is_spherical(arms):
    from ik.structure import analyze_structure, wrist_offset
    arm = arms["lite6"]
    st = analyze_structure(arm, n=6)
    kinds = {t.kind for t in st["triples"]}
    assert "spherical" in kinds  # J4-J6 concur
    # last axis passes through the wrist centre -> zero wrist offset
    a, ax = arm.world_lines()
    assert wrist_offset(a, ax) < 1e-6
