"""Independent FK/geometry tests for the displayed Cartesian TCP loops."""
import json

import numpy as np
import pytest

from ik import SrsSolver
from ik.kinematics import rot_log
from scripts.showcase.common import get_arm
from scripts.showcase.path_data import DEFAULT_FRAMES, collect_paths


@pytest.fixture(scope="module")
def paths():
    return collect_paths()


@pytest.mark.parametrize("key", ["ur5e", "lite6", "iiwa14", "panda"])
@pytest.mark.parametrize("kind", ["line", "circle"])
def test_every_displayed_frame_passes_fresh_fk_limits_and_loop_continuity(paths, key, kind):
    record = paths["robots"][key][kind]
    arm = get_arm(key)
    qs, Ts = np.asarray(record["qs"]), np.asarray(record["Ts"])
    assert len(qs) == len(Ts) == DEFAULT_FRAMES == 240
    np.testing.assert_allclose(Ts[:, :3, :3], np.repeat(Ts[:1, :3, :3], len(Ts), axis=0))
    arm.set_q(np.asarray(record["q_ref"]))
    np.testing.assert_allclose(arm.tcp_transform(), record["center_pose"], atol=1e-12)
    np.testing.assert_allclose(Ts[0, :3, :3], np.asarray(record["center_pose"])[:3, :3], atol=1e-12)
    np.testing.assert_allclose(np.asarray(record["center_pose"])[:3, 3], record["geometry"]["center"], atol=1e-12)
    assert np.all(qs >= arm.joint_low - 1e-10)
    assert np.all(qs <= arm.joint_high + 1e-10)
    positions, angles, points, margins = [], [], [], []
    for q, T in zip(qs, Ts):
        arm.set_q(q)
        R, p = arm.tcp_pose()
        points.append(p)
        positions.append(np.linalg.norm(p - T[:3, 3]))
        angles.append(np.linalg.norm(rot_log(T[:3, :3] @ R.T)))
        margins.append(np.min(np.minimum(q - arm.joint_low, arm.joint_high - q)))
    assert max(positions) < 1e-5 and max(angles) < 1e-5
    np.testing.assert_allclose(points, record["actual_points_m"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(positions, record["pos_error_m"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(angles, record["rot_error_rad"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(margins, record["joint_margin_rad"], atol=1e-12, rtol=0)
    steps = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    seam = np.linalg.norm(qs[-1] - qs[0])
    assert max(np.max(steps), seam) < .12
    assert record["summary"]["seam_dq_norm_rad"] == pytest.approx(seam)
    assert record["summary"]["max_dq_norm_rad"] == pytest.approx(np.max(steps))
    assert record["summary"]["closure_q_error_rad"] < 1e-4
    # Ground-plane clearance is geometric evidence only, not collision checking.
    assert np.min(np.asarray(points)[:, 2]) > (.03 if kind == "circle" else 0)
    assert record["summary"]["min_actual_tcp_z_m"] == pytest.approx(np.min(np.asarray(points)[:, 2]))


@pytest.mark.parametrize("key", ["ur5e", "lite6", "iiwa14", "panda"])
def test_tcp_circle_is_a_complete_cartesian_circle_not_an_elbow_locus(paths, key):
    record = paths["robots"][key]["circle"]
    geometry = record["geometry"]
    center, u, v = (np.asarray(geometry[x]) for x in ("center", "u", "v"))
    radius = geometry["radius"]
    target, actual = np.asarray(record["target_points_m"]), np.asarray(record["actual_points_m"])
    assert radius == pytest.approx(.26 if key == "lite6" else .40)
    assert np.linalg.norm(u) == pytest.approx(1)
    assert np.linalg.norm(v) == pytest.approx(1)
    assert np.dot(u, v) == pytest.approx(0)
    for points, atol in ((target, 1e-12), (actual, 1e-5)):
        delta = points - center
        np.testing.assert_allclose(np.linalg.norm(delta, axis=1), radius, atol=atol, rtol=0)
        np.testing.assert_allclose(delta @ np.cross(u, v), 0, atol=atol, rtol=0)
    # A steadily advancing full revolution, with no duplicate final frame.
    delta = target - center
    angle = np.unwrap(np.arctan2(delta @ v, delta @ u))
    np.testing.assert_allclose(np.diff(angle), 2 * np.pi / len(target), atol=1e-12)
    assert angle[-1] < 2 * np.pi
    assert angle[-1] + 2 * np.pi / len(target) == pytest.approx(2 * np.pi)


@pytest.mark.parametrize("key", ["ur5e", "lite6", "iiwa14", "panda"])
def test_line_reaches_both_endpoints_and_retraces_the_same_segment(paths, key):
    record = paths["robots"][key]["line"]
    start, end = (np.asarray(record["geometry"][k]) for k in ("line_start", "line_end"))
    points, actual = np.asarray(record["target_points_m"]), np.asarray(record["actual_points_m"])
    assert np.linalg.norm(end - start) == pytest.approx(.68 if key == "lite6" else 1.12)
    np.testing.assert_allclose(points[0], start, atol=1e-12)
    np.testing.assert_allclose(points[len(points) // 2], end, atol=1e-12)
    direction = (end - start) / np.linalg.norm(end - start)
    assert np.max(np.linalg.norm(np.cross(actual - start, direction), axis=1)) < 1e-5
    half = len(points) // 2
    np.testing.assert_allclose(points[1:half], points[:half:-1], atol=1e-12)


def test_redundancy_parameters_remain_fixed_around_the_circle(paths):
    panda = paths["robots"]["panda"]["circle"]
    constraint = panda["redundancy_constraint"]
    assert constraint["fixed_joint_index_zero_based"] == 2
    assert constraint["fixed_joint_name"] == "q3"
    q3 = np.asarray(panda["qs"])[:, 2]
    np.testing.assert_allclose(q3, constraint["fixed_joint_value_rad"], atol=1e-12)
    # q7 is now available to follow the large Cartesian circle.
    assert np.ptp(np.asarray(panda["qs"])[:, 6]) > .1
    record = paths["robots"]["iiwa14"]["circle"]
    arm = get_arm("iiwa14")
    solver = SrsSolver(arm)
    expected = record["redundancy_constraint"]["fixed_psi_rad"]
    for T, q in zip(record["Ts"], record["qs"]):
        _, _, _, center, h, v, _ = solver._elbow_circle(np.asarray(T))
        arm.set_q(q)
        anchors, axes = arm.world_lines()
        a = axes[3] / np.linalg.norm(axes[3])
        elbow = anchors[3] + np.dot(solver.S - anchors[3], a) * a
        psi = np.arctan2((elbow - center) @ v, (elbow - center) @ h)
        assert abs(np.arctan2(np.sin(psi - expected), np.cos(psi - expected))) < 1e-8


def test_path_payload_is_finite_json_and_sampling_is_explicit(paths):
    json.dumps(paths, allow_nan=False)
    assert "no collision" in paths["experiment"]["scope"]
    with pytest.raises(ValueError):
        collect_paths(n_frames=24)
    with pytest.raises(ValueError):
        collect_paths(n_frames=121)
