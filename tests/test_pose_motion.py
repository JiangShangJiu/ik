"""Fresh FK and physical geometry checks for the fixed-pose animations."""
import json

import numpy as np
import pytest

from ik import SrsSolver
from ik.kinematics import rot_log
from scripts.showcase.common import get_arm
from scripts.showcase.pose_data import collect_poses
from scripts.showcase.paths import metric_path


@pytest.fixture(scope="module")
def motion():
    return collect_poses(source=metric_path("solution_metrics.json"))


@pytest.mark.parametrize("key", ["ur5e", "lite6", "iiwa14", "panda"])
def test_every_frame_preserves_the_original_tcp_pose_with_fresh_fk(motion, key):
    source = json.loads(metric_path("solution_metrics.json").read_text())["robots"][key]
    record = motion["robots"][key]
    target, qs = np.asarray(record["target_pose"]), np.asarray(record["qs"])
    np.testing.assert_array_equal(target, source["target_pose"])
    assert len(qs) == 240
    arm = get_arm(key)
    errors, angles, points, margins, elbows = [], [], [], [], []
    srs = SrsSolver(arm) if key == "iiwa14" else None
    for q in qs:
        arm.set_q(q)
        R, p = arm.tcp_pose()
        anchors, axes = arm.world_lines()
        elbow = anchors[3].copy()
        if srs:
            axis = axes[3] / np.linalg.norm(axes[3])
            elbow += np.dot(srs.S - elbow, axis) * axis
        errors.append(np.linalg.norm(p - target[:3, 3]))
        angles.append(np.linalg.norm(rot_log(target[:3, :3] @ R.T)))
        margins.append(np.min(np.minimum(q - arm.joint_low, arm.joint_high - q)))
        points.append(p.copy())
        elbows.append(elbow)
    assert max(errors) < 1e-5 and max(angles) < 1e-5
    assert min(margins) >= 0
    for actual, field in ((errors, "pos_error_m"), (angles, "rot_error_rad"),
                          (margins, "joint_margin_rad"), (points, "actual_points_m"),
                          (elbows, "actual_elbows_m")):
        np.testing.assert_allclose(actual, record[field], atol=1e-12, rtol=0)
    summary = record["summary"]
    assert summary["max_pos_error_m"] == pytest.approx(max(errors))
    assert summary["max_rot_error_rad"] == pytest.approx(max(angles))
    assert summary["min_joint_margin_rad"] == pytest.approx(min(margins))
    assert summary["duration_s"] == 12 and summary["fps"] == 20


@pytest.mark.parametrize("key", ["ur5e", "lite6"])
def test_six_axis_solutions_are_held_and_switched_without_joint_interpolation(motion, key):
    source = json.loads(metric_path("solution_metrics.json").read_text())["robots"][key]
    record = motion["robots"][key]
    assert record["mode"] == "discrete_branches"
    assert record["parameter_name"] is None and record["parameter_values_rad"] is None
    assert record["summary"]["continuity_required"] is False
    assert record["summary"]["closure_q_error_rad"] is None
    np.testing.assert_array_equal(record["branch_indices"], np.repeat(np.arange(8), 30))
    expected = np.repeat(np.asarray([s["q"] for s in source["solutions"]]), 30, axis=0)
    np.testing.assert_array_equal(record["qs"], expected)
    assert np.count_nonzero(np.linalg.norm(np.diff(expected, axis=0), axis=1) > 0) == 7
    assert record["iterations_per_frame"] == [0] * 240


@pytest.mark.parametrize("key", ["iiwa14", "panda"])
def test_redundant_motion_is_a_continuous_cosine_round_trip(motion, key):
    record = motion["robots"][key]
    qs, parameters = np.asarray(record["qs"]), np.asarray(record["parameter_values_rad"])
    lo, hi = record["summary"]["parameter_interval_rad"]
    expected = lo + (hi - lo) * .5 * (1 - np.cos(np.linspace(0, 2*np.pi, 240, endpoint=False)))
    np.testing.assert_allclose(parameters, expected, atol=1e-12)
    assert record["mode"] == "continuous_self_motion"
    assert parameters[0] == pytest.approx(lo) and parameters[120] == pytest.approx(hi)
    assert hi - lo > (2.5 if key == "iiwa14" else .99)
    steps = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    seam = np.linalg.norm(qs[-1] - qs[0])
    assert max(np.max(steps), seam) < .12
    np.testing.assert_allclose(record["dq_norm_rad"], steps, atol=1e-12)
    assert record["summary"]["max_dq_norm_rad"] == pytest.approx(max(steps))
    assert record["summary"]["seam_dq_norm_rad"] == pytest.approx(seam)
    assert record["summary"]["closure_q_error_rad"] < 1e-4
    assert min(record["joint_margin_rad"]) > .05
    # Returning to each parameter follows the same branch instead of drifting
    # around a redundant family and jumping back to the first frame.
    assert np.max(np.linalg.norm(qs[1:120] - qs[:120:-1], axis=1)) < 1e-4


def test_iiwa_elbow_trace_matches_the_physical_srs_circle_and_arm_angle(motion):
    record = motion["robots"]["iiwa14"]
    g = record["geometry"]
    center, u, v, S, W = (np.asarray(g[k]) for k in ("center", "u", "v", "shoulder", "wrist"))
    elbows = np.asarray(record["actual_elbows_m"])
    delta = elbows - center
    assert g["kind"] == "elbow_circle" and g["elbow_joint_index_zero_based"] == 3
    np.testing.assert_allclose(elbows, g["elbows_m"], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(delta, axis=1), g["radius"], atol=1e-12)
    np.testing.assert_allclose(delta @ np.cross(u, v), 0, atol=1e-12)
    measured = np.arctan2(delta @ v, delta @ u)
    parameters = np.asarray(record["parameter_values_rad"])
    assert np.max(np.abs(np.arctan2(np.sin(measured - parameters), np.cos(measured - parameters)))) < 1e-10
    arm = get_arm("iiwa14")
    solver = SrsSolver(arm)
    np.testing.assert_allclose(np.linalg.norm(elbows - S, axis=1), solver.d_se, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(elbows - W, axis=1), solver.d_ew, atol=1e-12)
    assert record["iterations_per_frame"] == [0] * 240


def test_panda_prescribes_q7_and_records_actual_six_variable_dls_work(motion):
    record = motion["robots"]["panda"]
    assert record["parameter_name"] == "q7"
    np.testing.assert_array_equal(np.asarray(record["qs"])[:, 6], record["parameter_values_rad"])
    counts = record["iterations_per_frame"]
    assert len(counts) == 240 and all(1 <= n <= 120 for n in counts)
    assert record["summary"]["total_numerical_iterations"] == sum(counts)
    assert record["summary"]["max_numerical_iterations"] == max(counts)
    assert record["geometry"]["kind"] == "actual_joint_anchor_trace"
    assert record["geometry"]["elbow_joint_index_zero_based"] == 3
    assert np.max(record["summary"]["joint_excursion_rad"][:3]) > 1.5


def test_payload_is_finite_and_invalid_sampling_is_rejected(motion):
    json.dumps(motion, allow_nan=False)
    assert "No collision" in motion["experiment"]["scope"]
    for frames, fps in ((72, 20), (241, 20), (240, 0)):
        with pytest.raises(ValueError):
            collect_poses(n_frames=frames, fps=fps)
