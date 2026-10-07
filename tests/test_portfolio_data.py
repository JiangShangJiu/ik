"""Independent physical checks for the data published beside the portfolio."""

from types import SimpleNamespace
import json

import numpy as np
import pytest

from ik.kinematics import rot_log
from ik.numerical import solve_numerical
from scripts.showcase.common import get_arm
from scripts.showcase.portfolio_data import collect_data, _solution_record


@pytest.fixture(scope="module")
def measurements():
    return collect_data()


def test_measurements_are_json_safe_and_include_run_provenance(measurements):
    restored = json.loads(json.dumps(measurements, allow_nan=False))
    run = restored["experiment"]
    assert run["target_seed"] == 7
    assert run["path_seed"] == 11
    assert run["n_path_samples"] == 48
    assert run["dependencies"]["mujoco"]
    assert run["source_sha256"]["ik/srs.py"]
    assert restored["robots"]["iiwa14"]["branches"]["sampling"]["n_psi"] == 16


@pytest.mark.parametrize("key", ["ur5e", "lite6", "iiwa14", "panda"])
def test_published_path_survives_fresh_mujoco_fk_and_joint_limit_check(key, measurements):
    robot = measurements["robots"][key]
    arm = get_arm(key)
    path = robot["trajectory"]
    qs = np.asarray(path["qs"])
    Ts = np.asarray(path["Ts"])
    assert len(qs) == len(Ts) == 48
    assert np.all(qs >= arm.joint_low - 1e-10)
    assert np.all(qs <= arm.joint_high + 1e-10)
    actual = []
    pos_error, rot_error = [], []
    for q, T in zip(qs, Ts):
        arm.set_q(q)
        fresh = arm.tcp_transform()
        actual.append(fresh)
        pos_error.append(np.linalg.norm(fresh[:3, 3] - T[:3, 3]))
        rot_error.append(np.linalg.norm(rot_log(T[:3, :3] @ fresh[:3, :3].T)))
    np.testing.assert_allclose(actual, path["actual_Ts"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(pos_error, path["pos_error_m"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(rot_error, path["rot_error_rad"], atol=1e-12, rtol=0)
    assert max(pos_error) < 1e-5
    assert max(rot_error) < 1e-5
    # The reference comparison must describe an actual FK curve, not a drawn
    # quadratic added to make the Cartesian result look better.
    curved = []
    for q in path["q_linear"]:
        arm.set_q(q)
        curved.append(arm.tcp_pose()[1])
    np.testing.assert_allclose(curved, path["curved_points_m"], atol=1e-12, rtol=0)
    assert path["summary"]["max_curved_line_deviation_m"] > 1e-3
    assert path["summary"]["min_joint_margin_rad"] > 0


def test_exact_fk_alone_does_not_make_a_joint_limit_violation_valid():
    arm = get_arm("ur5e")
    q = 0.5 * (arm.joint_low + arm.joint_high)
    q[0] = arm.joint_high[0] + 0.01
    arm.set_q(q)
    record = _solution_record(arm, SimpleNamespace(q=q), arm.tcp_transform())
    assert record["pose_valid"]
    assert not record["within_limits"]
    assert not record["valid"]
    assert record["joint_margin_rad"] == pytest.approx(-0.01)


def test_panda_polishing_records_real_seed_errors_and_iteration_count(measurements):
    robot = measurements["robots"]["panda"]
    branch = robot["branches"]["solutions"][0]
    arm = get_arm("panda")
    target = np.asarray(robot["target_pose"])
    arm.set_q(branch["seed_q"])
    seed_pose = arm.tcp_transform()
    assert branch["seed_pos_err_m"] == pytest.approx(
        np.linalg.norm(seed_pose[:3, 3] - target[:3, 3]), abs=1e-12,
    )
    assert branch["seed_rot_err_rad"] == pytest.approx(
        np.linalg.norm(rot_log(target[:3, :3] @ seed_pose[:3, :3].T)), abs=1e-12,
    )
    replay = solve_numerical(arm, target, np.asarray(branch["seed_q"]), method="dls",
                             max_iters=60, tol_pos=1e-5, tol_rot=1e-5)
    assert replay.success
    assert branch["iters"] == replay.iters
    np.testing.assert_allclose(branch["q"], replay.q, atol=1e-12, rtol=0)
    for path in ("ik/model.py", "scripts/showcase/portfolio.py",
                 "scripts/showcase/portfolio_style.py"):
        assert path in measurements["experiment"]["source_sha256"]


def test_singularity_comparison_separates_units_and_uses_same_initial_pose(measurements):
    experiment = measurements["singularity"]
    arm = get_arm("iiwa14")
    T = np.asarray(experiment["target_pose"])
    q0 = np.asarray(experiment["q0"])
    arm.set_q(q0)
    fresh = arm.tcp_transform()
    initial_pos = np.linalg.norm(fresh[:3, 3] - T[:3, 3])
    initial_rot = np.linalg.norm(rot_log(T[:3, :3] @ fresh[:3, :3].T))
    for trace in experiment["methods"].values():
        np.testing.assert_allclose(trace["qs"][0], q0)
        assert trace["pos_error_m"][0] == pytest.approx(initial_pos)
        assert trace["rot_error_rad"][0] == pytest.approx(initial_rot)
        arm.set_q(trace["q_final"])
        final = arm.tcp_transform()
        assert trace["final_pos_err_m"] == pytest.approx(np.linalg.norm(final[:3, 3] - T[:3, 3]))
        assert trace["final_rot_err_rad"] == pytest.approx(
            np.linalg.norm(rot_log(T[:3, :3] @ final[:3, :3].T)), abs=1e-12,
        )
    assert "Independent teaching helper" in experiment["trace_scope"]
    assert "production_solver" in experiment
    assert experiment["methods"]["pinv"]["success"]
    assert experiment["methods"]["dls"]["success"]
