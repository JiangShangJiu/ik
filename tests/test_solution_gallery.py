"""Verify the new same-pose gallery against independent MuJoCo FK."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from ik.kinematics import rot_log
from ik.numerical import solve_numerical
from scripts.showcase.common import get_arm
from scripts.showcase.solution_data import collect_solutions, _FixedQ7
from scripts.showcase.paths import metric_path



@pytest.fixture(scope="module")
def gallery():
    source = metric_path("portfolio_metrics.json")
    before = source.read_bytes()
    result = collect_solutions(source=source)
    assert source.read_bytes() == before
    assert result["experiment"]["input_sha256"] == hashlib.sha256(before).hexdigest()
    json.dumps(result, allow_nan=False)
    return result


@pytest.mark.parametrize("key", ["ur5e", "lite6", "iiwa14", "panda"])
def test_every_candidate_preserves_the_original_target_and_model_limits(key, gallery):
    original = json.loads(Path(gallery["experiment"]["input_file"]).read_text())
    robot = gallery["robots"][key]
    np.testing.assert_array_equal(robot["target_pose"], original["robots"][key]["target_pose"])
    target = np.asarray(robot["target_pose"])
    arm = get_arm(key)
    assert robot["selected_count"] == len(robot["solutions"]) == 8
    assert robot["candidate_count"] == len(robot["candidates"])
    for record in robot["candidates"]:
        q = np.asarray(record["q"])
        assert np.all(q >= arm.joint_low - 1e-10)
        assert np.all(q <= arm.joint_high + 1e-10)
        arm.set_q(q)
        actual = arm.tcp_transform()
        pos = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        rot = np.linalg.norm(rot_log(target[:3, :3] @ actual[:3, :3].T))
        assert record["valid"] and pos < 1e-5 and rot < 1e-5
        assert record["pos_err_m"] == pytest.approx(pos, abs=1e-12)
        assert record["rot_err_rad"] == pytest.approx(rot, abs=1e-12)
        margin = np.min(np.minimum(q - arm.joint_low, arm.joint_high - q))
        assert record["joint_margin_rad"] == pytest.approx(margin, abs=1e-12)
    qs = np.asarray([r["q"] for r in robot["candidates"]])
    for i, q in enumerate(qs):
        if i:
            distances = np.linalg.norm((q - qs[:i] + np.pi) % (2*np.pi) - np.pi, axis=1)
            assert min(distances) >= 1e-3


def test_iiwa_representatives_span_the_full_sampled_angle_grid(gallery):
    robot = gallery["robots"]["iiwa14"]
    assert robot["candidate_count"] > 16
    assert len(robot["sampling"]["available_psi_rad"]) == 16
    np.testing.assert_allclose([r["psi_rad"] for r in robot["solutions"]],
                               np.arange(8) * np.pi / 4, atol=1e-12, rtol=0)
    assert robot["sampling"]["max_solutions"] == 0


def test_panda_representatives_are_interior_diverse_real_fixed_q7_solves(gallery):
    robot = gallery["robots"]["panda"]
    assert robot["candidate_count"] >= 8
    assert min(r["joint_margin_rad"] for r in robot["solutions"]) > 0.05
    # Actual joint-anchor displacement checks spatial diversity, rather than
    # counting seven nearly identical numerical refinements as a useful atlas.
    anchors = np.asarray([r["joint_anchors_m"] for r in robot["solutions"]])
    for i, a in enumerate(anchors):
        if i:
            distances = np.linalg.norm((a - anchors[:i]).reshape(i, -1), axis=1)
            assert min(distances) > 0.05
    arm = get_arm("panda")
    target = np.asarray(robot["target_pose"])
    settings = robot["sampling"]["solver_settings"]
    for record in robot["solutions"]:
        assert record["source"]["kind"] == "fixed-q7 DLS continuation"
        assert record["q"][-1] == pytest.approx(record["fixed_q7_rad"])
        view = _FixedQ7(arm, record["fixed_q7_rad"])
        replay = solve_numerical(view, target, np.asarray(record["source"]["previous_q"])[:-1],
                                 **settings)
        assert replay.success and replay.iters == record["iters"]
        np.testing.assert_allclose(replay.q, record["q"][:-1], atol=1e-12, rtol=0)
