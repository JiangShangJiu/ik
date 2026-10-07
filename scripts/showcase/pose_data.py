"""Verified fixed-TCP-pose branch switching and continuous redundant motions.

Run ``python -m scripts.showcase.pose_data``. Six-axis configurations are held
and switched without interpolating joints. Seven-axis frames are solved at the
same TCP pose while a redundancy parameter follows a cosine return trajectory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path

import numpy as np

from ik import SrsSolver
from ik.kinematics import rot_log
from ik.numerical import solve_numerical
from .common import get_arm
from .solution_data import _FixedQ7

from .paths import ROOT, SHOW, metric_path

DEFAULT_SOURCE = metric_path("solution_metrics.json")
DEFAULT_FRAMES, DEFAULT_FPS = 240, 20
POSITION_TOL = ROTATION_TOL = 1e-5
MAX_CONTINUOUS_DQ = .12
INTERIOR_MARGIN = .05
METHOD_LINES = {
    "ur5e": ["J2/J3/J4 三平行轴", "肩角 → 平面 2R → 腕角", "几何闭式 · 枚举分支"],
    "lite6": ["J4/J5/J6 球形腕", "腕心定位 → 2R → 球腕", "几何闭式 · 枚举分支"],
    "iiwa14": ["S-R-S 臂型角参数化", "逐帧 ψ → 闭式逆解", "选取相邻构型保持连续"],
    "panda": ["逐帧给定 q7", "其余六关节用 DLS 求解", "以前一帧作为初值"],
}


def _measure(arm, target, qs, *, srs=None):
    points, elbows, positions, rotations, margins = [], [], [], [], []
    for q in qs:
        arm.set_q(q)
        R, p = arm.tcp_pose()
        anchors, axes = arm.world_lines()
        elbow = anchors[3].copy()
        if srs is not None:
            axis = axes[3] / np.linalg.norm(axes[3])
            elbow += np.dot(srs.S - elbow, axis) * axis
        points.append(p.copy())
        elbows.append(elbow)
        positions.append(float(np.linalg.norm(p - target[:3, 3])))
        rotations.append(float(np.linalg.norm(rot_log(target[:3, :3] @ R.T))))
        margins.append(float(np.min(np.minimum(q - arm.joint_low, arm.joint_high - q))))
    if max(positions) >= POSITION_TOL or max(rotations) >= ROTATION_TOL or min(margins) < -1e-10:
        raise RuntimeError("Fixed-pose frame failed fresh FK or joint-limit validation")
    return {"actual_points_m": np.asarray(points).tolist(),
            "actual_elbows_m": np.asarray(elbows).tolist(),
            "pos_error_m": positions, "rot_error_rad": rotations,
            "joint_margin_rad": margins}


def _record(arm, target, qs, iterations, fps, *, parameter=None, parameter_name=None,
            close=None, srs=None):
    qs = np.asarray(qs)
    continuous = parameter is not None
    record = _measure(arm, target, qs, srs=srs)
    steps = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    seam = float(np.linalg.norm(qs[-1] - qs[0]))
    closure_q = closure_pos = closure_rot = None
    if continuous:
        closed_q = close(float(parameter[0]), qs[-1])
        closure_q = float(np.linalg.norm(closed_q - qs[0]))
        arm.set_q(closed_q)
        R, p = arm.tcp_pose()
        closure_pos = float(np.linalg.norm(p - target[:3, 3]))
        closure_rot = float(np.linalg.norm(rot_log(target[:3, :3] @ R.T)))
        if max(float(np.max(steps)), seam) >= MAX_CONTINUOUS_DQ or closure_q >= 1e-4:
            raise RuntimeError(f"Self-motion is discontinuous: dq={max(steps):.6f}, closure={closure_q:.3e}")
        if min(record["joint_margin_rad"]) < INTERIOR_MARGIN:
            raise RuntimeError("Self-motion has insufficient interior joint-limit margin")
    summary = {
        "valid": True, "n_frames": len(qs), "fps": fps, "duration_s": len(qs) / fps,
        "fixed_tcp_pose": True, "continuity_required": continuous,
        "max_pos_error_m": max(record["pos_error_m"]),
        "max_rot_error_rad": max(record["rot_error_rad"]),
        "min_joint_margin_rad": min(record["joint_margin_rad"]),
        "max_dq_norm_rad": float(np.max(steps)), "seam_dq_norm_rad": seam,
        "closure_q_error_rad": closure_q, "closure_pos_error_m": closure_pos,
        "closure_rot_error_rad": closure_rot,
        "frame_dq_limit_rad": MAX_CONTINUOUS_DQ if continuous else None,
        "parameter_interval_rad": [float(min(parameter)), float(max(parameter))] if continuous else None,
        "joint_excursion_rad": np.ptp(qs, axis=0).tolist(),
        "max_numerical_iterations": int(max(iterations)),
        "total_numerical_iterations": int(sum(iterations)),
    }
    return {"target_pose": target.tolist(), "qs": qs.tolist(), **record,
            "iterations_per_frame": iterations, "dq_norm_rad": steps.tolist(),
            "parameter_values_rad": np.asarray(parameter).tolist() if continuous else None,
            "parameter_name": parameter_name,
            "mode": "continuous_self_motion" if continuous else "discrete_branches",
            "branch_indices": None, "geometry": None,
            "elbow_point_definition": "world_lines joint anchor J4, zero-based index 3",
            "summary": summary}


def _discrete(arm, target, source, n_frames, fps):
    solutions = source["solutions"]
    if len(solutions) != 8:
        raise RuntimeError("Six-axis showcase requires eight independently valid source candidates")
    hold = n_frames // 8
    qs = np.repeat(np.asarray([s["q"] for s in solutions]), hold, axis=0)
    result = _record(arm, target, qs, [0] * n_frames, fps)
    result["branch_indices"] = np.repeat(np.arange(8), hold).tolist()
    result["sampling"] = {"kind": "eight discrete candidates, constant hold then instantaneous switch",
                          "frames_per_candidate": hold, "configuration_count": 8,
                          "branch_index_convention": "zero-based source solutions index",
                          "source_candidate_indices": [s["candidate_index"] for s in solutions],
                          "scope": "Each displayed state is valid; switching is not a continuous or executable trajectory. No joint interpolation."}
    return result


def _iiwa_interval(arm, target, source, solver):
    """Find a measured continuous branch on a finite one-degree parameter grid."""
    runs, current = [], []
    reference = np.asarray(source["solutions"][0]["q"])
    for psi in np.linspace(0, 2 * np.pi, 361):
        sols = solver.solve_at_psi(target, float(psi), tol_pos=1e-8, tol_rot=1e-8)
        sols = [s for s in sols if np.min(np.minimum(s.q - arm.joint_low, arm.joint_high - s.q)) > INTERIOR_MARGIN]
        if not sols:
            if current:
                runs.append(current)
                current = []
            continue
        prev = current[-1][1] if current else reference
        best = min(sols, key=lambda s: np.linalg.norm(s.q - prev))
        if current and np.linalg.norm(best.q - prev) > .15:
            runs.append(current)
            current = []
        current.append((float(psi), best.q.copy()))
    if current:
        runs.append(current)
    run = max(runs, key=len)
    if len(run) < 20:
        raise RuntimeError("No sufficiently long measured continuous iiwa branch")
    # Leave two measured parameter steps at either end of the selected run.
    return run[2][0], run[-3][0], run[2][1], {
        "scan_step_rad": float(np.pi / 180), "scan_interval_rad": [0., float(2 * np.pi)],
        "longest_measured_run_rad": [run[0][0], run[-1][0]],
        "endpoint_inset_rad": float(2 * np.pi / 180),
        "scope": "A finite measured legal branch interval, not a claim that the whole elbow circle is joint-limit feasible."}


def _iiwa(arm, target, source, n_frames, fps):
    solver = SrsSolver(arm)
    lo, hi, previous, scan = _iiwa_interval(arm, target, source, solver)
    phase = .5 * (1 - np.cos(np.linspace(0, 2 * np.pi, n_frames, endpoint=False)))
    parameters = lo + (hi - lo) * phase

    def solve(psi, prev):
        sols = solver.solve_at_psi(target, psi, tol_pos=1e-8, tol_rot=1e-8)
        if not sols:
            raise RuntimeError("No valid iiwa branch at the prescribed arm angle")
        return min(sols, key=lambda s: np.linalg.norm(s.q - prev)).q.copy()

    qs = []
    for psi in parameters:
        previous = solve(float(psi), previous)
        qs.append(previous.copy())
    result = _record(arm, target, qs, [0] * n_frames, fps,
                     parameter=parameters, parameter_name="psi", close=solve, srs=solver)
    _, _, wrist, center, u, v, radius = solver._elbow_circle(target)
    result["geometry"] = {"kind": "elbow_circle", "center": center.tolist(),
                          "radius": float(radius), "u": u.tolist(), "v": v.tolist(),
                          "shoulder": solver.S.tolist(), "wrist": wrist.tolist(),
                          "elbows_m": result["actual_elbows_m"],
                          "elbow_joint_index_zero_based": 3,
                          "elbow_point_definition": "orthogonal projection of shoulder S onto current J4 axis"}
    result["elbow_point_definition"] = result["geometry"]["elbow_point_definition"]
    result["sampling"] = {"kind": "cosine forward/backward, endpoint excluded", **scan}
    return result


def _panda(arm, target, source, n_frames, fps):
    candidates = sorted(source["candidates"], key=lambda s: s["q"][-1])
    interior = [s for s in candidates if s["joint_margin_rad"] > INTERIOR_MARGIN]
    # Exclude the first and last retained samples, then increase the low endpoint
    # until the actual cosine traversal satisfies the per-frame motion bound.
    high = float(interior[-2]["q"][-1])
    phase = .5 * (1 - np.cos(np.linspace(0, 2 * np.pi, n_frames, endpoint=False)))
    attempts = []
    settings = {"method": "dls", "max_iters": 120, "max_step": .2,
                "tol_pos": 2e-8, "tol_rot": 2e-8, "lambda0": .01}

    def solve(q7, prev, *, with_iters=False):
        solved = solve_numerical(_FixedQ7(arm, q7), target, prev[:-1], **settings)
        if not solved.success:
            raise RuntimeError(f"Fixed-q7 continuation failed: {solved.pos_err:.3e} m, {solved.rot_err:.3e} rad")
        q = np.r_[solved.q, q7]
        return (q, int(solved.iters)) if with_iters else q

    for seed in interior[1:-3]:
        low = float(seed["q"][-1])
        parameters = low + (high - low) * phase
        previous = np.asarray(seed["q"])
        qs, iterations = [], []
        attempt = {"interval_rad": [low, high]}
        try:
            for q7 in parameters:
                previous, count = solve(float(q7), previous, with_iters=True)
                qs.append(previous.copy())
                iterations.append(count)
            attempt["max_dq_norm_rad"] = float(np.max(np.linalg.norm(np.diff(qs, axis=0), axis=1)))
            result = _record(arm, target, qs, iterations, fps, parameter=parameters,
                             parameter_name="q7", close=solve)
        except RuntimeError as error:
            attempt.update(accepted=False, reason=str(error))
            attempts.append(attempt)
            continue
        attempt["accepted"] = True
        attempts.append(attempt)
        result["sampling"] = {
            "kind": "cosine forward/backward, endpoint excluded",
            "solver_settings": settings, "source_q7_range_rad": [candidates[0]["q"][-1], candidates[-1]["q"][-1]],
            "interval_trials": attempts,
            "scope": "One measured local redundant family; q7 varies across frames and is fixed only within each six-variable solve. This is not all Panda solutions or a reachability-boundary proof."}
        result["geometry"] = {"kind": "actual_joint_anchor_trace", "elbow_joint_index_zero_based": 3,
                              "elbow_point_definition": result["elbow_point_definition"],
                              "elbows_m": result["actual_elbows_m"]}
        return result
    raise RuntimeError("No safely continuous Panda interval found in the source family")


def collect_poses(source=None, n_frames=DEFAULT_FRAMES, fps=DEFAULT_FPS):
    """Return fixed-pose frames with independent FK and honest continuity labels."""
    if n_frames < 160 or n_frames % 8 or fps <= 0:
        raise ValueError("n_frames must be divisible by eight and >=160; fps must be positive")
    source = metric_path("solution_metrics.json") if source is None else Path(source)
    input_bytes = source.read_bytes()
    original = json.loads(input_bytes)
    robots = {}
    for key, src in original["robots"].items():
        arm, target = get_arm(key), np.asarray(src["target_pose"])
        if key in ("ur5e", "lite6"):
            result = _discrete(arm, target, src, n_frames, fps)
        elif key == "iiwa14":
            result = _iiwa(arm, target, src, n_frames, fps)
        elif key == "panda":
            result = _panda(arm, target, src, n_frames, fps)
        else:
            raise ValueError(f"Unexpected source robot: {key}")
        result.update(name=src["name"], dof=src["dof"], method_lines=METHOD_LINES[key],
                      target_pose_source="exactly preserved from solution_metrics input",
                      joint_limits={"low_rad": arm.joint_low.tolist(), "high_rad": arm.joint_high.tolist()})
        robots[key] = result
    sources = ("scripts/showcase/pose_data.py", "scripts/showcase/solution_data.py",
               "scripts/showcase/common.py", "scripts/showcase/paths.py", "ik/srs.py", "ik/numerical.py",
               "ik/kinematics.py", "ik/model.py", "ik/robot.py")
    result = {"schema_version": 1,
              "experiment": {"created_utc": datetime.now(timezone.utc).isoformat(),
                             "n_frames": n_frames, "fps": fps, "duration_s": n_frames / fps,
                             "input_file": str(source), "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                             "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                             "versions": {p: metadata.version(p) for p in ("numpy", "mujoco")},
                             "position_tol_m": POSITION_TOL, "rotation_tol_rad": ROTATION_TOL,
                             "scope": "Fresh FK and joint limits for every fixed-TCP-pose frame. Continuity checked only for the seven-axis self-motions; six-axis branch switches are discrete. No collision, dynamics or hardware validation."},
              "robots": robots}
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Input solutions JSON; defaults to build, then the published snapshot")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--output", type=Path, default=SHOW / "pose_metrics.json")
    args = parser.parse_args()
    data = collect_poses(args.source, args.frames, args.fps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    for key, robot in data["robots"].items():
        print(key, robot["mode"], robot["summary"], flush=True)


if __name__ == "__main__":
    main()
