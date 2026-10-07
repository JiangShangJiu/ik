"""Multi-robot and S-R-S acceptance tests: multi-robot wrapper, structure, and S-R-S closed form.

Three representative arms back the survey:

* ``ur5e``   -- 6R, Pieper *parallel axes* (joints 2,3,4), no spherical wrist;
* ``iiwa14`` -- 7R, true S-R-S (spherical shoulder AND wrist) => pure arm-angle
  closed form with *zero* iterations;
* ``panda``  -- 7R, spherical shoulder but a 0.088 m wrist offset, so the
  analytic seed must be polished numerically.
"""

import numpy as np
import pytest

from ik import REGISTRY, SrsSolver, analyze_structure, load_robot


@pytest.fixture(scope="module")
def iiwa():
    return load_robot("iiwa14")


@pytest.fixture(scope="module")
def srs(iiwa):
    return SrsSolver(iiwa, n_psi=12)


# -- registry / wrapper ------------------------------------------------------ #
def test_registry_has_three_arms():
    assert set(REGISTRY) == {"panda", "ur5e", "iiwa14", "lite6"}
    assert REGISTRY["ur5e"].dof == 6
    assert REGISTRY["lite6"].dof == 6
    assert REGISTRY["iiwa14"].dof == 7
    assert REGISTRY["panda"].dof == 7


@pytest.mark.parametrize("key", ["panda", "ur5e", "iiwa14", "lite6"])
def test_load_robot_roundtrip(key):
    arm = load_robot(key)
    assert arm.n == REGISTRY[key].dof
    q = 0.5 * (arm.joint_low + arm.joint_high)
    arm.set_q(q)
    assert np.allclose(arm.qpos(), q)
    R, p = arm.tcp_pose()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)   # orthonormal
    assert abs(np.linalg.det(R) - 1.0) < 1e-9


def test_jacobian_matches_finite_difference():
    arm = load_robot("ur5e")
    rng = np.random.default_rng(0)
    q = rng.uniform(arm.joint_low * 0.5, arm.joint_high * 0.5)
    arm.set_q(q)
    J = arm.tcp_jacobian()[:, :6]
    _, p0 = arm.tcp_pose()
    eps = 1e-7
    Jfd = np.zeros((3, 6))
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        arm.set_q(q + dq)
        _, p1 = arm.tcp_pose()
        Jfd[:, i] = (p1 - p0) / eps
    assert np.max(np.abs(J[:3] - Jfd)) < 1e-5


# -- structure: the three Pieper stories ------------------------------------- #
def test_ur5e_has_parallel_arm_axes():
    rep = analyze_structure(load_robot("ur5e"))
    kinds = [t.kind for t in rep["triples"]]
    assert "parallel" in kinds
    par = [t.indices for t in rep["triples"] if t.kind == "parallel"]
    assert (1, 2, 3) in par                      # 0-based joints 2,3,4
    assert rep["off_wrist_center"] > 1e-3        # wrist is NOT spherical


def test_iiwa_is_true_srs():
    rep = analyze_structure(load_robot("iiwa14"))
    sph = [t.indices for t in rep["triples"] if t.kind == "spherical"]
    assert (0, 1, 2) in sph                      # spherical shoulder
    assert (4, 5, 6) in sph                      # spherical wrist
    assert rep["off_wrist_center"] < 1e-6        # wrist offset is exactly zero


def test_panda_shoulder_spherical_wrist_offset():
    rep = analyze_structure(load_robot("panda"))
    sph = [t.indices for t in rep["triples"] if t.kind == "spherical"]
    assert (0, 1, 2) in sph
    assert (4, 5, 6) not in sph
    assert 0.08 < rep["off_wrist_center"] < 0.10  # the famous 0.088 m


# -- S-R-S closed form ------------------------------------------------------- #
def test_reference_shoulder_axes_orthogonal(srs, iiwa):
    iiwa.set_q(srs.q_ref)
    _, ax = iiwa.world_lines()
    G = np.array([ax[0], ax[1], ax[2]])
    assert np.max(np.abs(G @ G.T - np.eye(3))) < 1e-6


def test_solve_at_psi_recovers_true_config(srs, iiwa):
    """Every sampled true config must come back exactly at its own arm angle."""
    rng = np.random.default_rng(5)
    recovered = 0
    worst = 0.0
    for _ in range(30):
        q_true = rng.uniform(iiwa.joint_low * 0.8, iiwa.joint_high * 0.8)
        iiwa.set_q(q_true)
        T = iiwa.tcp_transform()

        # arm angle of the true configuration
        a, ax = iiwa.world_lines()
        d4 = ax[3] / np.linalg.norm(ax[3])
        E = a[3] + ((srs.S - a[3]) @ d4) * d4
        n, h, v = srs._elbow_circle(T)[3:6]
        psi = np.arctan2((E - n) @ v, (E - n) @ h)

        sols = srs.solve_at_psi(T, psi)
        if any(np.linalg.norm(np.mod(s.q - q_true + np.pi, 2 * np.pi) - np.pi)
               < 1e-6 for s in sols):
            recovered += 1
        if sols:
            worst = max(worst, min(s.pos_err for s in sols))
    assert recovered == 30, f"only {recovered}/30 exact recoveries"
    assert worst < 1e-9, f"residual too large: {worst:.2e}"


def test_solve_returns_valid_multi_solutions(srs, iiwa):
    rng = np.random.default_rng(7)
    q_true = rng.uniform(iiwa.joint_low * 0.8, iiwa.joint_high * 0.8)
    iiwa.set_q(q_true)
    T = iiwa.tcp_transform()
    sols = srs.solve(T, max_solutions=16)
    assert len(sols) >= 4, f"only {len(sols)} solutions"
    for s in sols:
        iiwa.set_q(s.q)
        R, p = iiwa.tcp_pose()
        assert np.linalg.norm(p - T[:3, 3]) < 1e-9
        assert np.linalg.norm(R - T[:3, :3]) < 1e-9
        assert np.all(s.q >= iiwa.joint_low - 1e-12)
        assert np.all(s.q <= iiwa.joint_high + 1e-12)


def test_unreachable_target_returns_empty(srs):
    T = np.eye(4)
    T[:3, 3] = [5.0, 5.0, 5.0]
    assert srs.solve(T) == []
