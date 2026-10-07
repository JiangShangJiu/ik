"""Run the small, reproducible experiment behind the portfolio figures.

Run from the repository root::

    MPLCONFIGDIR=/tmp/ik-mpl MUJOCO_GL=egl \\
        python -m scripts.showcase.portfolio_data

The JSON is an observation of four seeded targets and four sampled paths, not
a benchmark, a workspace coverage claim, or a real-robot execution guarantee.
Every reported pose error is recomputed with MuJoCo FK.  The independent
singularity trace is an independent pedagogical helper; production
``solve_numerical`` results are recorded separately.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np

from ik import Lite6Solver, SrsSolver, Ur5eSolver, solve_analytic
from ik.kinematics import pose_error, rot_log
from ik.numerical import _dls_step, solve_numerical

from .common import all_targets, get_arm
from .trajectory_helpers import make_line
from .paths import ROOT, SHOW


ROBOT_KEYS = ("ur5e", "lite6", "iiwa14", "panda")
NAMES = {"ur5e": "UR5e", "lite6": "Lite 6", "iiwa14": "iiwa 14", "panda": "Panda"}
POSE_TOL_M = 1e-5
POSE_TOL_RAD = 1e-5
LIMIT_TOL_RAD = 1e-10


def _pose(arm, q):
    arm.set_q(np.asarray(q, dtype=float))
    return arm.tcp_transform().copy()


def _residual(arm, q, target):
    actual = _pose(arm, q)
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    # The rotation-log form avoids arccos roundoff at exact solutions.
    rotation = float(np.linalg.norm(rot_log(target[:3, :3] @ actual[:3, :3].T)))
    return actual, position, rotation


def _margin(arm, q):
    q = np.asarray(q, dtype=float)
    return float(np.min(np.minimum(q - arm.joint_low, arm.joint_high - q)))


def _solution_record(arm, solution, target):
    q = np.asarray(solution.q, dtype=float)
    _, pe, re = _residual(arm, q, target)
    margin = _margin(arm, q)
    finite = bool(np.all(np.isfinite(q)) and np.isfinite(pe) and np.isfinite(re))
    within_limits = finite and margin >= -LIMIT_TOL_RAD
    pose_valid = finite and pe < POSE_TOL_M and re < POSE_TOL_RAD
    record = {
        "q": q.tolist(), "pos_err_m": pe, "rot_err_rad": re,
        "joint_margin_rad": margin, "within_limits": within_limits,
        "pose_valid": pose_valid, "valid": within_limits and pose_valid,
    }
    if hasattr(solution, "psi"):
        record["psi_rad"] = float(solution.psi)
    if hasattr(solution, "iters"):
        # This is the solver's measured iteration count, not polish_iters.
        record["iters"] = int(solution.iters)
    if getattr(solution, "seed_q", None) is not None:
        seed_q = np.asarray(solution.seed_q, dtype=float)
        _, seed_pe, seed_re = _residual(arm, seed_q, target)
        record.update({"seed_q": seed_q.tolist(),
                       "seed_pos_err_m": seed_pe, "seed_rot_err_rad": seed_re})
    elif hasattr(solution, "seed_pos_err"):
        # Older results may have only scalar provenance, or no recorded seed.
        record["seed_q"] = None
        record["seed_pos_err_m"] = (None if solution.seed_pos_err is None
                                   else float(solution.seed_pos_err))
        record["seed_rot_err_rad"] = (None if solution.seed_rot_err is None
                                     else float(solution.seed_rot_err))
    return record


def _branches(key, arm, target):
    if key == "ur5e":
        method, sampling = "UR5e closed form", None
        sols = Ur5eSolver(arm).solve(target, tol_pos=POSE_TOL_M, tol_rot=POSE_TOL_RAD)
    elif key == "lite6":
        method, sampling = "Lite 6 spherical-wrist closed form", None
        sols = Lite6Solver(arm).solve(target, tol_pos=POSE_TOL_M, tol_rot=POSE_TOL_RAD)
    elif key == "iiwa14":
        method = "S-R-S closed form at sampled arm angles"
        sampling = {"n_psi": 16, "max_solutions": 16}
        sols = SrsSolver(arm, n_psi=16).solve(
            target, tol_pos=POSE_TOL_M, tol_rot=POSE_TOL_RAD, max_solutions=16,
        )
    else:
        method = "Panda geometric seeds with DLS polishing"
        sampling = {"n_theta": 8, "n_psi": 4, "polish_iters": 60, "max_solutions": 8}
        sols = solve_analytic(arm, target, tol_pos=POSE_TOL_M, tol_rot=POSE_TOL_RAD,
                              **sampling)
    records = [_solution_record(arm, sol, target) for sol in sols]
    valid = [r for r in records if r["valid"]]
    return {
        "method": method, "sampling": sampling,
        "count_scope": "Distinct returned solutions for this target; not all possible IK solutions.",
        "returned_count": len(records), "valid_count": len(valid),
        "valid_boundary_count": sum(r["joint_margin_rad"] <= 1e-3 for r in valid),
        "valid_interior_count": sum(r["joint_margin_rad"] > 1e-3 for r in valid),
        "boundary_threshold_rad": 1e-3,
        "limit_violation_count": sum(not r["within_limits"] for r in records),
        "pose_failure_count": sum(not r["pose_valid"] for r in records),
        "max_valid_pos_err_m": max((r["pos_err_m"] for r in valid), default=None),
        "max_valid_rot_err_rad": max((r["rot_err_rad"] for r in valid), default=None),
        "min_valid_joint_margin_rad": min((r["joint_margin_rad"] for r in valid), default=None),
        "solutions": records,
    }


def _line_deviations(points, start, end):
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-12:
        return np.linalg.norm(points - start, axis=1)
    return np.linalg.norm(np.cross(points - start, direction / length), axis=1)


def _trajectory(key, arm, q_ref, n, path_seed):
    Ts, qs = make_line(key, arm, q_ref, np.random.default_rng(path_seed), n=n)
    Ts, qs = np.asarray(Ts), np.asarray(qs)
    ts = np.linspace(0, 1, n)
    q_linear = (1 - ts[:, None]) * qs[0] + ts[:, None] * qs[-1]
    actual, position, rotation = [], [], []
    for T, q in zip(Ts, qs):
        fq, pe, re = _residual(arm, q, T)
        actual.append(fq)
        position.append(pe)
        rotation.append(re)
    actual = np.asarray(actual)
    curved_poses = np.asarray([_pose(arm, q) for q in q_linear])
    curved_position = np.linalg.norm(curved_poses[:, :3, 3] - Ts[:, :3, 3], axis=1)
    curved_rotation = [float(np.linalg.norm(rot_log(T[:3, :3] @ fq[:3, :3].T)))
                       for T, fq in zip(Ts, curved_poses)]
    p0, p1 = Ts[0, :3, 3], Ts[-1, :3, 3]
    line = _line_deviations(actual[:, :3, 3], p0, p1)
    curved_line = _line_deviations(curved_poses[:, :3, 3], p0, p1)
    dq = np.diff(qs, axis=0)
    step_norms = np.linalg.norm(dq, axis=1)
    margins = np.asarray([_margin(arm, q) for q in qs])
    finite = np.all(np.isfinite(qs)) and np.all(np.isfinite(actual))
    limits = margins >= -LIMIT_TOL_RAD
    valid_samples = limits & (np.asarray(position) < POSE_TOL_M) & (np.asarray(rotation) < POSE_TOL_RAD)
    return {
        "n_samples": n, "path_seed": path_seed,
        "parameter": ts.tolist(), "Ts": Ts.tolist(), "qs": qs.tolist(),
        "q_linear": q_linear.tolist(), "actual_Ts": actual.tolist(),
        "tcp_points_m": actual[:, :3, 3].tolist(),
        "curved_points_m": curved_poses[:, :3, 3].tolist(),
        "pos_error_m": position, "rot_error_rad": rotation,
        "line_deviation_m": line.tolist(),
        "curved_line_deviation_m": curved_line.tolist(),
        "curved_pos_error_m": curved_position.tolist(),
        "curved_rot_error_rad": curved_rotation,
        "dq_norm_rad": step_norms.tolist(), "joint_margin_rad": margins.tolist(),
        "summary": {
            "valid": bool(finite and np.all(valid_samples)),
            "valid_sample_count": int(np.sum(valid_samples)),
            "limit_violation_sample_count": int(np.sum(~limits)),
            "max_pos_error_m": max(position), "max_rot_error_rad": max(rotation),
            "max_line_deviation_m": float(np.max(line)),
            "max_curved_line_deviation_m": float(np.max(curved_line)),
            "max_curved_pos_error_m": float(np.max(curved_position)),
            "max_curved_rot_error_rad": max(curved_rotation),
            "max_dq_norm_rad": float(np.max(step_norms)),
            "max_joint_step_rad": float(np.max(np.abs(dq))),
            "min_joint_margin_rad": float(np.min(margins)),
            "line_length_m": float(np.linalg.norm(p1 - p0)),
        },
        "continuity_note": "Raw adjacent joint differences at uniform path parameters; no timing, velocity, acceleration, collision, or dynamics checks.",
        "limit_audit": "Every returned path configuration is checked independently; FK accuracy alone cannot establish joint-limit validity.",
    }


def _independent_trace(arm, target, q0, method, iterations=80, damping=0.05):
    """Fixed-damping teaching updates, deliberately separate from the solver.

    No max-step cap or LM acceptance test is used here.  Limit clipping is
    retained.  Attempted and applied steps differ when limits are reached.
    Position and rotation residuals stay separate because they have different
    units; the stacked six-vector norm must not be labelled metres or radians.
    """
    q = np.clip(np.asarray(q0, dtype=float), arm.joint_low, arm.joint_high)
    positions, rotations, attempted, applied, qs, margins = [], [], [], [], [], []
    for _ in range(iterations):
        actual, pe, re = _residual(arm, q, target)
        positions.append(pe)
        rotations.append(re)
        qs.append(q.tolist())
        margins.append(_margin(arm, q))
        e = pose_error(actual[:3, :3], actual[:3, 3], target[:3, :3], target[:3, 3])
        J = arm.tcp_jacobian()
        if method == "jt":
            largest = np.linalg.svd(J, compute_uv=False)[0]
            dq = J.T @ e / (largest ** 2 + 1e-12)
        elif method == "pinv":
            dq = np.linalg.pinv(J) @ e
        else:
            dq = _dls_step(J, e, damping)
        attempted.append(float(np.linalg.norm(dq)))
        q_next = np.clip(q + dq, arm.joint_low, arm.joint_high)
        applied.append(float(np.linalg.norm(q_next - q)))
        q = q_next
    _, pe, re = _residual(arm, q, target)
    return {
        "iterations": iterations, "qs": qs, "q_final": q.tolist(),
        "pos_error_m": positions, "rot_error_rad": rotations,
        "step_norm_rad": attempted, "applied_step_norm_rad": applied,
        "joint_margin_rad": margins, "final_pos_err_m": pe, "final_rot_err_rad": re,
        "success": bool(pe < 1e-6 and re < 1e-6),
    }


def _singularity(arm, target_ref, q_ref):
    sv = SrsSolver(arm, n_psi=32)
    R = target_ref[:3, :3]
    # Match the previous exhibit's target construction, including its direction
    # norm; record the actual reach fraction rather than assuming 0.9995.
    wrist = sv.S + 0.9995 * (sv.d_se + sv.d_ew) * np.array([0.28, -0.42, 0.86])
    target = np.eye(4)
    target[:3, :3], target[:3, 3] = R, wrist - R @ sv.c
    solutions = sv.solve_at_psi(target, 0.0) or sv.solve(target, max_solutions=8)
    solutions = [s for s in solutions if _solution_record(arm, s, target)["valid"]]
    if not solutions:
        raise RuntimeError("No independently verified, limit-valid near-singular iiwa solution")
    q_solution = min(solutions, key=lambda s: np.linalg.norm(s.q - q_ref)).q
    q0 = np.clip(q_solution + np.array([0.35, -0.25, 0.2, 0.0, 0.15, -0.1, 0.1]),
                 arm.joint_low, arm.joint_high)
    traces = {m: _independent_trace(arm, target, q0, m) for m in ("jt", "pinv", "dls")}
    production = {}
    for method in traces:
        result = solve_numerical(arm, target, q0, method=method, max_iters=80,
                                 tol_pos=1e-6, tol_rot=1e-6)
        _, pe, re = _residual(arm, result.q, target)
        production[method] = {
            "q": result.q.tolist(), "success": bool(result.success), "iters": result.iters,
            "pos_err_m": pe, "rot_err_rad": re, "joint_margin_rad": _margin(arm, result.q),
        }
    arm.set_q(q_solution)
    return {
        "robot": "iiwa14", "target_pose": target.tolist(), "q0": q0.tolist(),
        "q_solution": q_solution.tolist(), "elbow_radius_m": float(sv._elbow_circle(target)[-1]),
        "wrist_reach_fraction": float(np.linalg.norm(wrist - sv.S) / (sv.d_se + sv.d_ew)),
        "jacobian_sigma_min": float(np.linalg.svd(arm.tcp_jacobian(), compute_uv=False)[-1]),
        "jacobian_note": "Geometric Jacobian combines translation and rotation rows; singular values depend on the selected scaling.",
        "methods": traces,
        "trace_scope": "Independent teaching helper: 80 updates, fixed DLS damping 0.05, no step cap, no LM acceptance test, joint-limit clipping. These traces are not production solve_numerical histories.",
        "production_solver": {
            "settings": {"max_iters": 80, "max_step_rad": 0.5, "lambda0": 0.05,
                         "tol_pos_m": 1e-6, "tol_rot_rad": 1e-6, "clamp_limits": True},
            "results": production,
        },
    }


def _provenance():
    deps = {}
    for package in ("numpy", "mujoco", "matplotlib", "pillow", "pytest"):
        try:
            deps[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            deps[package] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    source_files = ("ik/numerical.py", "ik/closedform.py", "ik/srs.py", "ik/analytic.py",
                    "ik/model.py", "ik/robot.py", "ik/kinematics.py",
                    "scripts/showcase/common.py", "scripts/showcase/trajectory_helpers.py",
                    "scripts/showcase/paths.py",
                    "scripts/showcase/portfolio_data.py", "scripts/showcase/portfolio.py",
                    "scripts/showcase/portfolio_style.py")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0], "dependencies": deps,
        "machine": {"system": platform.system(), "release": platform.release(),
                    "architecture": platform.machine(), "logical_cpus": os.cpu_count()},
        "environment": {name: os.environ.get(name) for name in ("MUJOCO_GL", "MPLCONFIGDIR")},
        "git_commit": commit,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_files},
    }


def collect_data(seed=7, path_seed=11, n=48):
    """Return JSON-safe measurements of one deterministic experiment.

    ``n`` is the number of uniform Cartesian path samples.  Targets use the
    existing ``all_targets(seed=7)`` selection: highest sigma_min among 300
    configurations sampled inside the middle 50% of each model's joint ranges.
    This deliberately selects well-conditioned examples, not random success
    rates.  Defaults are used by the public portfolio.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    started = perf_counter()
    targets = all_targets(seed=seed)
    robots = {}
    for key in ROBOT_KEYS:
        arm = get_arm(key)
        q_ref, target = targets[key]
        xml = Path(arm.xml_path if hasattr(arm, "xml_path") else arm.spec.xml)
        robots[key] = {
            "name": NAMES[key], "dof": len(arm.joint_low),
            "q_ref": q_ref.tolist(), "target_pose": target.tolist(),
            "joint_limits": {"low_rad": arm.joint_low.tolist(), "high_rad": arm.joint_high.tolist()},
            "model": {"xml_name": f"{xml.parent.name}/{xml.name}",
                      "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest()},
            "branches": _branches(key, arm, target),
            "trajectory": _trajectory(key, arm, q_ref, n=n, path_seed=path_seed),
        }
    singularity = _singularity(get_arm("iiwa14"), targets["iiwa14"][1], targets["iiwa14"][0])
    experiment = _provenance()
    experiment.update({
        "target_seed": seed, "path_seed": path_seed, "n_path_samples": n,
        "elapsed_s": perf_counter() - started,
        "pose_tolerances": {"position_m": POSE_TOL_M, "rotation_rad": POSE_TOL_RAD,
                            "joint_limit_rad": LIMIT_TOL_RAD},
        "target_selection": "Per robot: highest geometric-Jacobian sigma_min of 300 seed-controlled samples in the middle 50% of model joint ranges.",
        "scope": "Four representative targets and four feasible Cartesian segments; not a workspace benchmark or hardware validation.",
        "command": "MPLCONFIGDIR=/tmp/ik-mpl MUJOCO_GL=egl python -m scripts.showcase.portfolio_data",
    })
    result = {"schema_version": 1, "experiment": experiment, "robots": robots,
              "singularity": singularity}
    # Reject NaNs or accidental NumPy values before an artifact is written.
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SHOW / "portfolio_metrics.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--path-seed", type=int, default=11)
    parser.add_argument("--samples", type=int, default=48)
    args = parser.parse_args()
    data = collect_data(seed=args.seed, path_seed=args.path_seed, n=args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(f"Wrote {args.output} ({data['experiment']['elapsed_s']:.2f} s)")
    for key, robot in data["robots"].items():
        branch, path = robot["branches"], robot["trajectory"]["summary"]
        print(f"{key}: {branch['valid_count']}/{branch['returned_count']} valid solutions; "
              f"path {path['valid_sample_count']}/{args.samples} valid samples; "
              f"max position error {path['max_pos_error_m']:.3e} m; "
              f"minimum joint margin {path['min_joint_margin_rad']:.3e} rad")


if __name__ == "__main__":
    main()
