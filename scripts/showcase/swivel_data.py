"""Target-specific evidence: every arm angle can have IK without a full loop.

The sampled candidate graph is finite evidence. A separate geometric winding
diagnostic explains the first-joint obstruction for this particular iiwa pose.
No old showcase data or assets are changed by this module.
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
from .common import get_arm

from .paths import ROOT, SHOW, metric_path

DEFAULT_SOURCE = metric_path("pose_metrics.json")
GRID_STEP_DEG = .25
GRAPH_DQ_LIMITS = (.15, .30)
POSITION_TOL = ROTATION_TOL = 1e-8


def _identity(q):
    signs = {"q4_sign": 1 if q[3] >= 0 else -1,
             "q2_sign": 1 if q[1] >= 0 else -1,
             "q6_sign": 1 if q[5] >= 0 else -1}
    name = "_".join(f"{joint}{'+' if sign > 0 else '-'}" for joint, sign in
                    (("q4", signs["q4_sign"]), ("q2", signs["q2_sign"]), ("q6", signs["q6_sign"])))
    return name, signs


def _candidate(arm, target, q):
    # Canonical representatives for algebraic-candidate legality. Continuous
    # extension curves are separately unwrapped and never folded into limits.
    q = (np.asarray(q).copy() + np.pi) % (2 * np.pi) - np.pi
    arm.set_q(q)
    R, p = arm.tcp_pose()
    margin = np.minimum(q - arm.joint_low, arm.joint_high - q)
    pe = float(np.linalg.norm(p - target[:3, 3]))
    re = float(np.linalg.norm(rot_log(target[:3, :3] @ R.T)))
    branch_id, signs = _identity(q)
    violations = []
    for i in np.flatnonzero(margin < -1e-10):
        lower = q[i] < arm.joint_low[i]
        violations.append({"joint_index_zero_based": int(i), "joint_name": f"q{i + 1}",
                           "side": "lower" if lower else "upper", "q_rad": float(q[i]),
                           "limit_rad": float(arm.joint_low[i] if lower else arm.joint_high[i]),
                           "violation_rad": float(-margin[i])})
    return {"branch_id": branch_id, "branch_signs": signs, "q": q.tolist(),
            "pos_error_m": pe, "rot_error_rad": re,
            "joint_margin_rad": float(min(margin)), "joint_margins_rad": margin.tolist(),
            "valid": bool(pe < POSITION_TOL and re < ROTATION_TOL and not violations),
            "violations": violations}


def _graph(samples, raw_samples, threshold):
    """Propagate all reachable candidate nodes, rather than choosing greedily."""
    checks = []
    for start_index, start in enumerate(samples[0]["candidates"]):
        reachable = {start_index}
        last_index, stop = 0, None
        for i in range(1, len(samples)):
            prev = [samples[i - 1]["candidates"][j] for j in sorted(reachable)]
            nxt = samples[i]["candidates"]
            distances = np.linalg.norm(np.asarray([c["q"] for c in prev])[:, None, :]
                                       - np.asarray([c["q"] for c in nxt])[None, :, :], axis=2)
            new = set(np.flatnonzero(np.any(distances < threshold, axis=0)).tolist())
            if not new:
                row, col = np.unravel_index(np.argmin(distances), distances.shape)
                last = prev[row]
                same_branch = next(c for c in raw_samples[i] if c["branch_id"] == last["branch_id"])
                stop = {"last_valid_psi_rad": samples[i - 1]["psi_rad"],
                        "last_valid_psi_deg": samples[i - 1]["psi_deg"], "last_valid_candidate": last,
                        "first_blocked_psi_rad": samples[i]["psi_rad"],
                        "first_blocked_psi_deg": samples[i]["psi_deg"],
                        "first_blocked_raw_candidate": same_branch,
                        "violating_joint_indices_zero_based": [v["joint_index_zero_based"] for v in same_branch["violations"]],
                        "nearest_valid_alternative": nxt[col],
                        "nearest_alternative_dq_norm_rad": float(distances[row, col]),
                        "reason": "same-branch joint limit and no sufficiently close valid alternative"
                                  if same_branch["violations"] else "no adjacent candidate satisfies this sampled step threshold"}
                break
            reachable, last_index = new, i
        closed = stop is None and any(np.linalg.norm(np.asarray(samples[-1]["candidates"][j]["q"])
                                                     - start["q"]) < 1e-6 for j in reachable)
        checks.append({"start_candidate_index": start_index, "start_candidate": start,
                       "full_turn_path_found": stop is None, "closed_path_found": bool(closed),
                       "last_reachable_psi_deg": samples[last_index]["psi_deg"],
                       "termination": stop})
    return {"adjacent_dq_limit_rad": threshold, "distance": "Euclidean norm of actual bounded joint coordinates; no modulo-angle jumps",
            "any_full_turn_path_found": any(c["full_turn_path_found"] for c in checks),
            "any_closed_path_found": any(c["closed_path_found"] for c in checks),
            "starting_configurations": checks,
            "scope": "All candidates at every sampled angle were considered. This finite grid and adjacency threshold alone are not a universal nonexistence proof."}


def _branches(samples, raw_samples):
    ids = sorted({c["branch_id"] for sample in samples for c in sample["candidates"]})
    branches = []
    for branch_id in ids:
        segments, current = [], []
        for i, sample in enumerate(samples):
            candidate = next((c for c in sample["candidates"] if c["branch_id"] == branch_id), None)
            continuous = (candidate is not None and
                          (not current or np.linalg.norm(np.asarray(candidate["q"]) - current[-1][1]["q"]) < .30))
            if not continuous and current:
                segments.append(current)
                current = []
            if candidate is not None:
                current.append((i, candidate))
        if current:
            segments.append(current)
        records = []
        for segment in segments:
            indices, candidates = zip(*segment)
            records.append({"sample_indices": list(indices),
                            "parameter_values_rad": [samples[i]["psi_rad"] for i in indices],
                            "parameter_interval_deg": [samples[indices[0]]["psi_deg"], samples[indices[-1]]["psi_deg"]],
                            "qs": [c["q"] for c in candidates],
                            "pos_error_m": [c["pos_error_m"] for c in candidates],
                            "rot_error_rad": [c["rot_error_rad"] for c in candidates],
                            "joint_margin_rad": [c["joint_margin_rad"] for c in candidates]})
        raw = [next(c for c in row if c["branch_id"] == branch_id) for row in raw_samples]
        raw_qs = np.asarray([c["q"] for c in raw])
        branches.append({"branch_id": branch_id, "branch_signs": raw[0]["branch_signs"], "segments": records,
                         "parameter_values_rad": [s["psi_rad"] for s in samples],
                         "raw_qs": raw_qs.tolist(), "unwrapped_qs": np.unwrap(raw_qs, axis=0).tolist(),
                         "valid_mask": [c["valid"] for c in raw],
                         "violated_joint_indices_zero_based": [[v["joint_index_zero_based"] for v in c["violations"]] for c in raw],
                         "raw_pos_error_m": [c["pos_error_m"] for c in raw],
                         "raw_rot_error_rad": [c["rot_error_rad"] for c in raw],
                         "raw_curve_scope": "Raw principal-angle candidates may violate limits. Unwrapped curves are geometric extensions, not valid motion frames; never modulo them back into a limit band."})
    return branches


def _winding(arm, solver, geometry, samples):
    center, u, v, S = (np.asarray(geometry[k]) for k in ("center", "u", "v", "shoulder"))
    radius = geometry["radius"]
    angles = np.asarray([s["psi_rad"] for s in samples])
    elbows = center + radius * (np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v)
    delta = elbows - S
    A = radius * np.column_stack([u[:2], v[:2]])
    coefficient = np.linalg.solve(A, (S - center)[:2])
    azimuth = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    arm.set_q(np.zeros(7))
    anchors, axes = arm.world_lines()
    axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    shoulder_gaps = [float(np.linalg.norm(np.cross(S - anchors[i], axes[i]))) for i in range(3)]
    wrist = np.asarray(geometry["wrist"])
    # Measure the spherical wrist at an actual pose for this target.
    arm.set_q(samples[0]["candidates"][0]["q"])
    target_anchors, target_axes = arm.world_lines()
    wrist_gaps = [float(np.linalg.norm(np.cross(wrist - target_anchors[i], target_axes[i]) /
                                     np.linalg.norm(target_axes[i]))) for i in (4, 5, 6)]
    errors = []
    for sample, E in zip(samples, elbows):
        phi = float(np.arctan2(E[1] - S[1], E[0] - S[0]))
        for candidate in sample["candidates"]:
            q = candidate["q"]
            expected = phi if q[1] > 0 else phi + np.pi
            errors.append(abs(float(np.arctan2(np.sin(q[0] - expected), np.cos(q[0] - expected)))))
    return {"xy_axis_enclosure_coefficient": coefficient.tolist(),
            "xy_axis_enclosure_coefficient_norm": float(np.linalg.norm(coefficient)),
            "shoulder_axis_enclosed": bool(np.linalg.norm(coefficient) < 1),
            "sampled_net_elbow_azimuth_rad": float(azimuth[-1] - azimuth[0]),
            "min_sampled_distance_to_joint1_axis_m": float(np.min(np.linalg.norm(delta[:, :2], axis=1))),
            "joint1_axis_world": axes[0].tolist(), "joint1_axis_parallel_world_z_error": float(np.linalg.norm(np.cross(axes[0], [0, 0, 1]))),
            "shoulder_axes_to_S_max_gap_m": max(shoulder_gaps),
            "wrist_axes_to_W_max_gap_m": max(wrist_gaps),
            "joint1_limits_rad": [float(arm.joint_low[0]), float(arm.joint_high[0])],
            "joint1_total_travel_rad": float(arm.joint_high[0] - arm.joint_low[0]),
            "max_sampled_q1_elbow_azimuth_relation_error_rad": max(errors),
            "target_specific_obstruction": bool(np.linalg.norm(coefficient) < 1 and arm.joint_high[0] - arm.joint_low[0] < 2*np.pi),
            "assumptions": ["This measured S-R-S model and this fixed TCP target only.",
                            "J1 is the world-Z shoulder axis; shoulder J1/J2/J3 axes meet at S.",
                            "The elbow circle XY ellipse encloses J1 without intersecting it, so the shoulder sign cannot change through q2=0.",
                            "On either continuous shoulder branch, a full elbow revolution requires a continuous J1 change of 2*pi, exceeding this model's 340-degree J1 range.",
                            "This does not assert that other TCP poses or other mechanisms lack full-circle self-motion."]}


def _demonstration(arm, solver, target, source, samples, raw_samples):
    branch_id = _identity(source["qs"][0])[0]
    start_psi = source["parameter_values_rad"][0]
    start_index = int(round(np.degrees(start_psi) / GRID_STEP_DEG))
    run = []
    for i in range(start_index, len(samples)):
        c = next((c for c in samples[i]["candidates"] if c["branch_id"] == branch_id), None)
        if c is None:
            blocked_index = i
            break
        run.append((i, c))
    else:
        raise RuntimeError("Expected this demonstration branch to encounter a sampled limit")
    indices, candidates = zip(*run)
    fine_qs = np.asarray([c["q"] for c in candidates])
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(fine_qs, axis=0), axis=1))]
    progress = .5 * (1 - np.cos(np.linspace(0, np.pi, 180)))
    # Interpolate the redundancy parameter from measured joint arc length, then
    # solve fresh IK at each prescribed angle. Joint vectors are never interpolated.
    parameters = np.interp(progress * lengths[-1], lengths, [samples[i]["psi_rad"] for i in indices])
    parameters = np.r_[parameters, np.repeat(parameters[-1], 60)]
    records, elbows = [], []
    for psi in parameters:
        solutions = solver.solve_at_psi(target, float(psi), tol_pos=POSITION_TOL, tol_rot=ROTATION_TOL)
        chosen = next((s for s in solutions if _identity(s.q)[0] == branch_id), None)
        if chosen is None:
            raise RuntimeError("Resampled demonstration frame has no valid same-branch IK")
        rec = _candidate(arm, target, chosen.q)
        if not rec["valid"]:
            raise RuntimeError("Demonstration failed independent FK/joint-limit validation")
        records.append(rec)
        anchors, axes = arm.world_lines()
        axis = axes[3] / np.linalg.norm(axes[3])
        elbows.append((anchors[3] + np.dot(solver.S - anchors[3], axis) * axis).tolist())
    qs = np.asarray([r["q"] for r in records])
    steps = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    if max(steps) >= .12:
        raise RuntimeError(f"Demonstration is too fast after true-IK arc-length resampling: {max(steps):.6f}")
    raw_blocked = next(c for c in raw_samples[blocked_index] if c["branch_id"] == branch_id)
    last = records[-1]
    nearest = min(samples[blocked_index]["candidates"], key=lambda c: np.linalg.norm(np.asarray(c["q"]) - last["q"]))
    return {"mode": "continue_to_limit_then_hold", "branch_id": branch_id,
            "name": "iiwa 14", "dof": 7, "target_pose": target.tolist(), "qs": qs.tolist(),
            "parameter_name": "psi", "parameter_values_rad": parameters.tolist(),
            "active_branch_id": branch_id,
            "frame_phase": ["advance"]*180 + ["hold_limit"]*60,
            "method_lines": ["S-R-S 臂型角参数化", "逐帧 ψ → 闭式逆解", "沿同一支路转至限位"],
            "actual_elbows_m": elbows, "iterations_per_frame": [0]*240,
            "pos_error_m": [r["pos_error_m"] for r in records],
            "rot_error_rad": [r["rot_error_rad"] for r in records],
            "joint_margin_rad": [r["joint_margin_rad"] for r in records],
            "dq_norm_rad": steps.tolist(),
            "sampling": {"moving_frames": 180, "hold_frames": 60,
                         "method": "cosine progress in measured joint arc length; parameter interpolation followed by fresh closed-form IK, never joint interpolation",
                         "last_valid_sample_deg": samples[indices[-1]]["psi_deg"],
                         "first_blocked_sample_deg": samples[blocked_index]["psi_deg"]},
            "last_valid_candidate": last,
            "last_valid_sample": {"psi_deg": samples[indices[-1]]["psi_deg"],
                                  "psi_rad": samples[indices[-1]]["psi_rad"], **last},
            "first_blocked_raw_candidate": raw_blocked,
            "first_blocked_candidate": {"psi_deg": samples[blocked_index]["psi_deg"],
                                        "psi_rad": samples[blocked_index]["psi_rad"], **raw_blocked},
            "violating_joint_indices_zero_based": [v["joint_index_zero_based"] for v in raw_blocked["violations"]],
            "nearest_valid_alternative": nearest, "nearest_alternative": nearest,
            "nearest_alternative_dq_norm_rad": float(np.linalg.norm(np.asarray(nearest["q"]) - last["q"])),
            "summary": {"valid": True, "n_frames": 240, "fps": 20, "duration_s": 12.,
                        "parameter_interval_rad": [float(parameters[0]), float(parameters[-1])],
                        "max_pos_error_m": max(r["pos_error_m"] for r in records),
                        "max_rot_error_rad": max(r["rot_error_rad"] for r in records),
                        "min_joint_margin_rad": min(r["joint_margin_rad"] for r in records),
                        "max_dq_norm_rad": float(max(steps)),
                        "seam_dq_norm_rad": float(np.linalg.norm(qs[-1] - qs[0])),
                        "seam_continuous": False, "loop_reset_required": True,
                        "scope": "All 240 shown frames are valid. The blocked raw candidate is explanatory data only. The end holds near the sampled joint limit; replay needs an explicit cut or fade reset."}}


def collect_swivel(source=None):
    source = metric_path("pose_metrics.json") if source is None else Path(source)
    input_bytes = source.read_bytes()
    original = json.loads(input_bytes)["robots"]["iiwa14"]
    target = np.asarray(original["target_pose"])
    arm = get_arm("iiwa14")
    solver = SrsSolver(arm)
    geo = solver._elbow_circle(target)
    _, _, W, center, u, v, radius = geo
    geometry = {"kind": "physical_elbow_circle", "center": center.tolist(), "radius": float(radius),
                "u": u.tolist(), "v": v.tolist(), "shoulder": solver.S.tolist(), "wrist": W.tolist(),
                "elbow_joint_index_zero_based": 3,
                "elbow_point_definition": "orthogonal projection of shoulder S onto current J4 axis"}
    samples, raw_samples = [], []
    for degrees in np.arange(0, 360 + GRID_STEP_DEG/2, GRID_STEP_DEG):
        psi = float(np.deg2rad(degrees))
        raw = [_candidate(arm, target, q) for q in solver._candidates(geo, psi)]
        valid = [c for c in raw if c["valid"]]
        public = solver.solve_at_psi(target, psi, tol_pos=POSITION_TOL, tol_rot=ROTATION_TOL)
        if len(public) != len(valid):
            raise RuntimeError("Public SRS candidate count disagrees with independent raw FK/limit validation")
        if not valid:
            raise RuntimeError("Unexpected empty arm-angle sample in this known target")
        samples.append({"psi_rad": psi, "psi_deg": float(degrees), "candidate_count": len(valid), "candidates": valid})
        raw_samples.append(raw)
    graph = [_graph(samples, raw_samples, limit) for limit in GRAPH_DQ_LIMITS]
    branches = _branches(samples, raw_samples)
    winding = _winding(arm, solver, geometry, samples)
    demo = _demonstration(arm, solver, target, original, samples, raw_samples)
    demo["geometry"] = {**geometry, "elbows_m": demo["actual_elbows_m"]}
    demo["joint_limits"] = {"low_rad": arm.joint_low.tolist(), "high_rad": arm.joint_high.tolist()}
    selected = [{"branch_id": b["branch_id"], "segment_index": max(range(len(b["segments"])),
                 key=lambda i: len(b["segments"][i]["sample_indices"]))} for b in branches]
    files = ("scripts/showcase/swivel_data.py", "scripts/showcase/common.py", "scripts/showcase/paths.py",
             "ik/srs.py", "ik/kinematics.py", "ik/robot.py", "ik/structure.py")
    model_path = arm.spec.xml
    counts = [s["candidate_count"] for s in samples]
    result = {"schema_version": 1, "robot_key": "iiwa14", "name": original["name"],
              "target_pose": target.tolist(), "geometry": geometry,
              "joint_limits": {"low_rad": arm.joint_low.tolist(), "high_rad": arm.joint_high.tolist()},
              "experiment": {"created_utc": datetime.now(timezone.utc).isoformat(),
                             "input_file": str(source), "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                             "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
                             "model_file": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                             "versions": {p: metadata.version(p) for p in ("numpy", "mujoco")},
                             "grid_step_deg": GRID_STEP_DEG, "grid_size": len(samples), "angle_interval_deg": [0., 360.],
                             "position_tol_m": POSITION_TOL, "rotation_tol_rad": ROTATION_TOL,
                             "branch_identity_definition": "sign(q4), sign(q2), sign(q6); stable away from their zero-angle branch mergers",
                             "scope": "One fixed target and this measured model/limits. Full sampled candidate graph plus target-specific geometric winding; no universal mechanism claim, collision or dynamics validation."},
              "samples": samples, "candidate_counts": counts, "branches": branches,
              "visualization_intervals": selected, "graph_checks": graph,
              "geometric_winding_report": winding, "demonstration": demo,
              "original_demonstration_interval": {"parameter_interval_rad": original["summary"]["parameter_interval_rad"],
                                                  "meaning": "The original 6-156 degree cosine round trip is a chosen smooth display interval, not a reachability or joint-limit boundary."},
              "summary": {"all_sampled_angles_have_valid_ik": all(counts), "sample_count": len(samples),
                          "candidate_count_min": min(counts), "candidate_count_max": max(counts),
                          "total_valid_candidates": sum(counts),
                          "total_raw_candidates": sum(len(row) for row in raw_samples),
                          "any_sampled_full_turn_path_found": any(g["any_full_turn_path_found"] for g in graph),
                          "any_sampled_closed_path_found": any(g["any_closed_path_found"] for g in graph),
                          "max_pos_error_m": max(c["pos_error_m"] for s in samples for c in s["candidates"]),
                          "max_rot_error_rad": max(c["rot_error_rad"] for s in samples for c in s["candidates"]),
                          "min_joint_margin_rad": min(c["joint_margin_rad"] for s in samples for c in s["candidates"]),
                          "max_raw_pos_error_m": max(c["pos_error_m"] for row in raw_samples for c in row),
                          "max_raw_rot_error_rad": max(c["rot_error_rad"] for row in raw_samples for c in row),
                          "target_specific_geometric_joint1_obstruction": winding["target_specific_obstruction"]}}
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Input pose JSON; defaults to build, then the published snapshot")
    parser.add_argument("--output", type=Path, default=SHOW / "swivel_metrics.json")
    args = parser.parse_args()
    data = collect_swivel(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(data["summary"])
    print(data["demonstration"]["summary"])


if __name__ == "__main__":
    main()
