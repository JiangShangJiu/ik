"""Regression tests for the diagnostics shown to portfolio readers."""

import numpy as np

from ik import AnalyticSolution, Lite6Solver, PandaArm
from ik import analytic as analytic_module
from ik.kinematics import rot_log
from scripts.showcase import branch_diagnostics
from scripts.showcase.common import choose_target, get_arm
from scripts.showcase.trajectory_helpers import _bad_branch_samples, make_line


def test_polished_solution_preserves_its_actual_solver_diagnostics(monkeypatch):
    arm = PandaArm()
    arm.set_q(np.array([0.5, 0.8, 0.3, -1.6, 0.4, 1.7, 0.2]))
    target = arm.tcp_transform()
    attempts = []
    numerical = analytic_module.solve_numerical

    def record_attempt(robot, T, seed, **kwargs):
        result = numerical(robot, T, seed, **kwargs)
        attempts.append((seed.copy(), result))
        return result

    monkeypatch.setattr(analytic_module, "solve_numerical", record_attempt)
    solution = analytic_module.solve_analytic_best(arm, target)
    assert solution is not None
    seed, result = next((q, r) for q, r in attempts if r.success)
    assert solution.iters == result.iters
    assert np.array_equal(solution.seed_q, seed)
    arm.set_q(seed)
    R, p = arm.tcp_pose()
    assert np.isclose(solution.seed_pos_err, np.linalg.norm(p - target[:3, 3]))
    assert np.isclose(solution.seed_rot_err,
                      np.linalg.norm(rot_log(target[:3, :3] @ R.T)))


def test_near_limit_convergence_is_not_labelled_as_failure(monkeypatch):
    arm = PandaArm()
    q = 0.5 * (arm.joint_low + arm.joint_high)
    q[0] = arm.joint_low[0] + 1e-4
    solution = AnalyticSolution(q, True, 1e-8, 2e-8, iters=7)
    monkeypatch.setattr(branch_diagnostics, "solve_analytic", lambda *args, **kwargs: [solution])
    rows = branch_diagnostics.enumerate_branches("panda", arm, np.eye(4))
    assert len(rows[0]) == 5  # preserve callers that unpack the public API
    assert rows[0][3] == "converged"
    assert "近限位" in rows[0][4]
    display = branch_diagnostics._branch_displays("panda", arm, np.eye(4))[0]
    assert display.iters == 7 and display.near_limit


def test_bad_branch_frames_each_match_their_own_target():
    q_ref, _ = choose_target("lite6")
    arm = get_arm("lite6")
    targets, continuous = make_line("lite6", arm, q_ref,
                                    np.random.default_rng(11), n=48)
    indices, switched, jumps = _bad_branch_samples(targets, continuous,
                                                   Lite6Solver(arm))
    assert np.array_equal(np.diff(indices), [1, 1])
    for i, q in zip(indices, switched):
        arm.set_q(q)
        R, p = arm.tcp_pose()
        assert np.linalg.norm(p - targets[i][:3, 3]) < 1e-5
        assert np.linalg.norm(rot_log(targets[i][:3, :3] @ R.T)) < 1e-5
        assert np.all(q >= arm.joint_low) and np.all(q <= arm.joint_high)
    assert jumps[1] > 1.0  # the deliberately switched branch jumps visibly
    assert jumps[2] < 0.2  # subsequent motion follows that branch continuously
