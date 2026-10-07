"""Model geometry acceptance tests: model loading, calibration, FK cross-check."""

import numpy as np
import pytest

from ik import PandaArm, PANDA_MDH, fk_mdh, N_JOINTS

TOL = 1e-9


@pytest.fixture(scope="module")
def arm():
    return PandaArm()


def test_model_dimensions(arm):
    assert arm.nq == N_JOINTS
    assert arm.nv == N_JOINTS
    assert PANDA_MDH.shape == (N_JOINTS, 3)


def test_mdh_matches_mujoco_fk(arm):
    """Analytic MDH FK must match MuJoCo's link7 frame to machine precision."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        q = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q)
        T_mj = arm.tcp_transform()
        T_dh = fk_mdh(q)
        assert np.allclose(T_mj, T_dh, atol=TOL), "MDH FK != MuJoCo FK"


def test_shoulder_center(arm):
    geom = arm.geometry()
    assert np.allclose(geom.shoulder_center, [0.0, 0.0, 0.333], atol=1e-9)


def test_calibrated_lengths(arm):
    geom = arm.geometry()
    # Values verified against the MJCF geometry.
    assert geom.d_se == pytest.approx(0.3265962, abs=1e-5)
    assert geom.d_ew == pytest.approx(0.3927582, abs=1e-5)
    assert geom.wrist_offset == pytest.approx(0.088, abs=1e-9)


def test_calibration_is_config_independent(arm):
    """Calibration with different seeds must yield the same geometry."""
    g1 = arm.geometry(n_samples=4, seed=1)
    arm._geometry = None
    g2 = arm.geometry(n_samples=10, seed=99)
    assert np.allclose(g1.shoulder_center, g2.shoulder_center, atol=TOL)
    assert g1.d_se == pytest.approx(g2.d_se, abs=TOL)
    assert g1.d_ew == pytest.approx(g2.d_ew, abs=TOL)
    assert g1.wrist_offset == pytest.approx(g2.wrist_offset, abs=TOL)


def test_pose_error_roundtrip(arm):
    from ik import pose_error

    rng = np.random.default_rng(2)
    q = rng.uniform(arm.joint_low, arm.joint_high)
    arm.set_q(q)
    R, p = arm.tcp_pose()
    assert np.allclose(pose_error(R, p, R, p), 0.0, atol=1e-12)


def test_wrist_center_offset(arm):
    """W is a fixed distance (0.088) from the link7 origin."""
    rng = np.random.default_rng(3)
    geom = arm.geometry()
    for _ in range(10):
        q = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q)
        W = arm.wrist_center()
        _, p = arm.tcp_pose()
        assert np.linalg.norm(W - p) == pytest.approx(geom.wrist_offset, abs=1e-9)
