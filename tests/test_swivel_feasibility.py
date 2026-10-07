"""Independent model checks for the target-specific swivel explanation."""
import json

import numpy as np
import pytest

from ik.kinematics import rot_log
from scripts.showcase.common import get_arm
from scripts.showcase.swivel_data import collect_swivel
from scripts.showcase.paths import metric_path


@pytest.fixture(scope="module")
def swivel():
    return collect_swivel(source=metric_path("pose_metrics.json"))


def test_dense_grid_keeps_the_same_target_and_every_angle_has_valid_ik(swivel):
    source = json.loads(metric_path("pose_metrics.json").read_text())["robots"]["iiwa14"]
    np.testing.assert_array_equal(swivel["target_pose"], source["target_pose"])
    assert len(swivel["samples"]) == 1441
    np.testing.assert_allclose([s["psi_deg"] for s in swivel["samples"]], np.arange(0, 360.001, .25), atol=1e-12)
    counts = [len(s["candidates"]) for s in swivel["samples"]]
    assert min(counts) == 2 and max(counts) == 8 and all(counts)
    assert counts == swivel["candidate_counts"]
    assert sum(counts) == swivel["summary"]["total_valid_candidates"]


def test_all_retained_candidates_pass_fresh_fk_limits_and_physical_elbow_geometry(swivel):
    arm = get_arm("iiwa14")
    target = np.asarray(swivel["target_pose"])
    geometry = swivel["geometry"]
    center, u, v, S = (np.asarray(geometry[k]) for k in ("center", "u", "v", "shoulder"))
    for sample in swivel["samples"]:
        expected_elbow = center + geometry["radius"] * (np.cos(sample["psi_rad"])*u + np.sin(sample["psi_rad"])*v)
        identities = []
        for c in sample["candidates"]:
            q = np.asarray(c["q"])
            arm.set_q(q)
            R, p = arm.tcp_pose()
            assert np.linalg.norm(p - target[:3, 3]) < 1e-8
            assert np.linalg.norm(rot_log(target[:3, :3] @ R.T)) < 1e-8
            margin = np.minimum(q - arm.joint_low, arm.joint_high - q)
            assert min(margin) >= 0 and c["valid"]
            assert c["joint_margin_rad"] == pytest.approx(min(margin))
            np.testing.assert_allclose(c["joint_margins_rad"], margin, atol=1e-12)
            anchors, axes = arm.world_lines()
            a = axes[3] / np.linalg.norm(axes[3])
            elbow = anchors[3] + np.dot(S - anchors[3], a)*a
            np.testing.assert_allclose(elbow, expected_elbow, atol=1e-12)
            signs = c["branch_signs"]
            assert signs == {"q4_sign": int(np.sign(q[3])), "q2_sign": int(np.sign(q[1])), "q6_sign": int(np.sign(q[5]))}
            identities.append(c["branch_id"])
        assert len(set(identities)) == len(identities)


def test_full_raw_branches_preserve_illegal_extensions_without_modulo_jumps(swivel):
    assert len(swivel["branches"]) == 8
    arm = get_arm("iiwa14")
    target = np.asarray(swivel["target_pose"])
    for branch in swivel["branches"]:
        principal = np.asarray(branch["raw_qs"])
        unwrapped = np.asarray(branch["unwrapped_qs"])
        assert principal.shape == unwrapped.shape == (1441, 7)
        np.testing.assert_allclose(unwrapped, np.unwrap(principal, axis=0), atol=1e-12)
        np.testing.assert_allclose(np.sin(unwrapped), np.sin(principal), atol=1e-12)
        assert unwrapped[-1, 0] - unwrapped[0, 0] == pytest.approx(2*np.pi)
        assert np.ptp(unwrapped[:, 0]) > arm.joint_high[0] - arm.joint_low[0]
        expected_mask = np.all((principal >= arm.joint_low) & (principal <= arm.joint_high), axis=1)
        np.testing.assert_array_equal(branch["valid_mask"], expected_mask)
        assert np.any(expected_mask) and not np.all(expected_mask)
        for i, q in enumerate(principal):
            arm.set_q(q)
            R, p = arm.tcp_pose()
            assert np.linalg.norm(p - target[:3, 3]) < 1e-8
            assert np.linalg.norm(rot_log(target[:3, :3] @ R.T)) < 1e-8
            expected = np.flatnonzero((q < arm.joint_low) | (q > arm.joint_high)).tolist()
            assert branch["violated_joint_indices_zero_based"][i] == expected


def test_sampled_candidate_graph_fails_for_both_initial_elbow_branches(swivel):
    for graph in swivel["graph_checks"]:
        assert graph["adjacent_dq_limit_rad"] in (.15, .30)
        assert not graph["any_full_turn_path_found"] and not graph["any_closed_path_found"]
        assert len(graph["starting_configurations"]) == 2
        for start in graph["starting_configurations"]:
            stop = start["termination"]
            assert not start["full_turn_path_found"] and not start["closed_path_found"]
            q = np.asarray(stop["last_valid_candidate"]["q"])
            other = np.asarray(stop["nearest_valid_alternative"]["q"])
            assert stop["nearest_alternative_dq_norm_rad"] == pytest.approx(np.linalg.norm(other-q))
            assert np.linalg.norm(other-q) > 4
            assert not stop["first_blocked_raw_candidate"]["valid"]
            if start["start_candidate"]["branch_signs"]["q4_sign"] > 0:
                assert stop["first_blocked_psi_deg"] == 1
                assert stop["violating_joint_indices_zero_based"] == [0]
            else:
                assert stop["first_blocked_psi_deg"] == 173
                assert stop["violating_joint_indices_zero_based"] == [6]


def test_current_target_geometry_encloses_joint1_axis_and_requires_excess_winding(swivel):
    geometry = swivel["geometry"]
    report = swivel["geometric_winding_report"]
    c, u, v, S = (np.asarray(geometry[k]) for k in ("center", "u", "v", "shoulder"))
    coeff = np.linalg.solve(geometry["radius"]*np.column_stack([u[:2], v[:2]]), (S-c)[:2])
    assert np.linalg.norm(coeff) == pytest.approx(report["xy_axis_enclosure_coefficient_norm"])
    assert np.linalg.norm(coeff) < 1
    parameters = np.asarray([s["psi_rad"] for s in swivel["samples"]])
    elbows = c + geometry["radius"]*(np.cos(parameters)[:, None]*u + np.sin(parameters)[:, None]*v)
    phi = np.unwrap(np.arctan2(elbows[:, 1]-S[1], elbows[:, 0]-S[0]))
    assert phi[-1]-phi[0] == pytest.approx(2*np.pi)
    assert np.min(np.linalg.norm(elbows[:, :2]-S[:2], axis=1)) > .008
    assert report["joint1_total_travel_rad"] < 2*np.pi
    assert report["joint1_axis_parallel_world_z_error"] < 1e-12
    assert report["shoulder_axes_to_S_max_gap_m"] < 1e-12
    assert report["wrist_axes_to_W_max_gap_m"] < 1e-12
    assert report["max_sampled_q1_elbow_azimuth_relation_error_rad"] < 1e-10
    assert report["target_specific_obstruction"] is True


def test_continue_to_limit_animation_uses_only_valid_same_branch_ik_then_holds(swivel):
    record = swivel["demonstration"]
    qs = np.asarray(record["qs"])
    ps = np.asarray(record["parameter_values_rad"])
    assert qs.shape == (240, 7)
    assert record["frame_phase"] == ["advance"]*180 + ["hold_limit"]*60
    assert np.all(np.diff(ps[:180]) > 0)
    np.testing.assert_array_equal(ps[180:], np.repeat(ps[179], 60))
    np.testing.assert_allclose(qs[180:], np.repeat(qs[179:180], 60, axis=0), atol=1e-12)
    assert np.degrees(ps[0]) == pytest.approx(6)
    assert np.degrees(ps[-1]) == pytest.approx(172.75)
    arm = get_arm("iiwa14")
    target = np.asarray(record["target_pose"])
    for q in qs:
        assert np.all(q >= arm.joint_low) and np.all(q <= arm.joint_high)
        arm.set_q(q)
        R, p = arm.tcp_pose()
        assert np.linalg.norm(p-target[:3, 3]) < 1e-8
        assert np.linalg.norm(rot_log(target[:3, :3] @ R.T)) < 1e-8
        assert q[1] < 0 and q[3] < 0 and q[5] < 0
    assert np.max(np.linalg.norm(np.diff(qs, axis=0), axis=1)) < .12
    assert np.linalg.norm(qs[-1]-qs[0]) > 4
    assert record["summary"]["seam_continuous"] is False
    assert record["summary"]["loop_reset_required"] is True
    blocked = record["first_blocked_candidate"]
    assert blocked["psi_deg"] == 173 and not blocked["valid"]
    assert record["violating_joint_indices_zero_based"] == [6]
    assert blocked["q"][6] < arm.joint_low[6]
    nearest = record["nearest_alternative"]
    assert nearest["valid"]
    assert record["nearest_alternative_dq_norm_rad"] == pytest.approx(np.linalg.norm(np.asarray(nearest["q"])-qs[-1]))


def test_original_interval_is_explained_as_selection_and_payload_is_finite(swivel):
    original = swivel["original_demonstration_interval"]
    np.testing.assert_allclose(np.degrees(original["parameter_interval_rad"]), [6, 156], atol=1e-12)
    assert "not a reachability" in original["meaning"]
    assert "One fixed target" in swivel["experiment"]["scope"]
    json.dumps(swivel, allow_nan=False)
