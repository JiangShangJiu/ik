"""Verified same-pose IK representatives for the four portfolio robots.

    python -m scripts.showcase.solution_data

Six-axis examples retain the eight candidates in portfolio_metrics.json.
iiwa samples the entire 16-angle grid without truncating its returned pool.
Panda follows one locally reachable family by changing q7 and solving the six
remaining variables. None of these finite pools is a complete IK solution set.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from ik import SrsSolver
from ik.kinematics import rot_log
from ik.numerical import solve_numerical
from .common import get_arm
from .paths import ROOT, SHOW, metric_path

KEYS = ("ur5e", "lite6", "iiwa14", "panda")
POSITION_TOL = ROTATION_TOL = 1e-5
INTERIOR_MARGIN = 0.05
Q7_INCREMENT = 0.025
DEDUP_TOL = 1e-3


class _FixedQ7:
    """Six active variables with an explicitly prescribed Panda q7."""
    def __init__(self, arm, q7):
        self.arm, self.q7 = arm, float(q7)
        self.joint_low, self.joint_high = arm.joint_low[:-1], arm.joint_high[:-1]

    def set_q(self, q):
        self.arm.set_q(np.r_[q, self.q7])

    def tcp_pose(self):
        return self.arm.tcp_pose()

    def tcp_jacobian(self):
        return self.arm.tcp_jacobian()[:, :-1]


def _record(arm, target, q, source, **extra):
    q = np.asarray(q, dtype=float)
    arm.set_q(q)
    actual = arm.tcp_transform()
    pe = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    re = float(np.linalg.norm(rot_log(target[:3, :3] @ actual[:3, :3].T)))
    margin = float(np.min(np.minimum(q - arm.joint_low, arm.joint_high - q)))
    anchors, _ = arm.world_lines()
    valid = bool(np.all(np.isfinite(q)) and pe < POSITION_TOL and re < ROTATION_TOL
                 and margin >= -1e-10)
    return {"q": q.tolist(), "pos_err_m": pe, "rot_err_rad": re,
            "joint_margin_rad": margin, "valid": valid,
            "joint_anchors_m": anchors.tolist(), "source": source, **extra}


def _distance(q, other):
    return float(np.linalg.norm((np.asarray(q) - other + np.pi) % (2 * np.pi) - np.pi))


def _append_distinct(pool, record):
    if not record["valid"]:
        return False
    if any(_distance(record["q"], np.asarray(r["q"])) < DEDUP_TOL for r in pool):
        return False
    record["candidate_index"] = len(pool)
    pool.append(record)
    return True


def _diverse(pool, n, initial=None):
    """Max-min actual joint-anchor separation; joint angles break ties."""
    if not pool:
        return []
    n = min(n, len(pool))
    selected = [max(range(len(pool)), key=lambda i: pool[i]["joint_margin_rad"])
                if initial is None else initial]
    anchors = np.asarray([r["joint_anchors_m"] for r in pool])
    while len(selected) < n:
        def score(i):
            geo = min(float(np.linalg.norm(anchors[i] - anchors[j])) for j in selected)
            joints = min(_distance(pool[i]["q"], np.asarray(pool[j]["q"])) for j in selected)
            return geo, joints, pool[i]["joint_margin_rad"], -i
        selected.append(max((i for i in range(len(pool)) if i not in selected), key=score))
    return selected


def _iiwa_pool(arm, target):
    solver = SrsSolver(arm, n_psi=16)
    # max_solutions=0 is the existing API's uncapped mode. Keeping the whole
    # grid avoids mistaking the first 16 returned branches for 16 arm angles.
    sols = solver.solve(target, tol_pos=POSITION_TOL, tol_rot=ROTATION_TOL, max_solutions=0)
    pool = []
    for sol in sols:
        _append_distinct(pool, _record(arm, target, sol.q,
                         {"kind": "SrsSolver.solve", "n_psi": 16, "max_solutions": 0},
                         psi_rad=float(sol.psi), iters=0))
    return pool


def _iiwa_select(pool, n):
    angles = sorted(set(r["psi_rad"] for r in pool))
    # One representative at each selected arm angle. Uniformity refers only to
    # this available finite angle grid; legality of the whole circle is not claimed.
    grid_indices = np.floor(np.arange(min(n, len(angles))) * len(angles) /
                            min(n, len(angles))).astype(int)
    selected = []
    for gi in grid_indices:
        group = [i for i, r in enumerate(pool) if r["psi_rad"] == angles[gi]]
        if not selected:
            best = max(group, key=lambda i: pool[i]["joint_margin_rad"])
        else:
            def score(i):
                d = min(_distance(pool[i]["q"], np.asarray(pool[j]["q"])) for j in selected)
                return d, pool[i]["joint_margin_rad"], -i
            best = max(group, key=score)
        selected.append(best)
    return selected, angles


def _panda_pool(arm, target, original):
    starters = [(i, s) for i, s in enumerate(original)
                if s["valid"] and s["joint_margin_rad"] > INTERIOR_MARGIN]
    if not starters:
        raise RuntimeError("No interior Panda starting candidate in the source experiment")
    # Best measured task residual, rather than a hard-coded new posture.
    index, seed = min(starters, key=lambda item: item[1]["pos_err_m"] + item[1]["rot_err_rad"])
    start = np.asarray(seed["q"], dtype=float)
    pool, scans = [], []
    _append_distinct(pool, _record(arm, target, start,
                     {"kind": "portfolio_metrics candidate", "index": index,
                      "reported_iters": seed.get("iters")},
                     fixed_q7_rad=float(start[-1]), iters=None))
    settings = {"method": "dls", "max_iters": 120, "max_step": 0.2,
                "tol_pos": 1e-7, "tol_rot": 1e-7, "lambda0": 0.01}
    for direction in (-1, 1):
        previous = start.copy()
        solved, retained = 0, 0
        limit = arm.joint_low[-1] if direction < 0 else arm.joint_high[-1]
        scan = {"direction": direction, "initial_q": start.tolist()}
        for q7 in np.arange(start[-1] + direction * Q7_INCREMENT,
                            limit, direction * Q7_INCREMENT):
            view = _FixedQ7(arm, q7)
            result = solve_numerical(view, target, previous[:-1], **settings)
            if not result.success:
                scan["stopped_at_q7_rad"] = float(q7)
                scan["stop_pos_err_m"] = float(result.pos_err)
                scan["stop_rot_err_rad"] = float(result.rot_err)
                scan["stop_reason"] = "first numerical failure; not a reachability boundary proof"
                break
            q = np.r_[result.q, q7]
            rec = _record(arm, target, q,
                          {"kind": "fixed-q7 DLS continuation", "direction": direction,
                           "previous_q": previous.tolist()},
                          fixed_q7_rad=float(q7), iters=int(result.iters))
            solved += 1
            if rec["joint_margin_rad"] > INTERIOR_MARGIN:
                retained += int(_append_distinct(pool, rec))
            previous = q
        else:
            scan["stop_reason"] = "model q7 scan limit reached"
        scan.update({"converged_count": solved, "interior_retained_count": retained})
        scans.append(scan)
    sampling = {"starting_portfolio_candidate": index, "q7_increment_rad": Q7_INCREMENT,
                "minimum_retained_margin_rad": INTERIOR_MARGIN, "solver_settings": settings,
                "directional_scans": scans,
                "scope": "One locally followed redundant family; failed continuation does not rule out other branches."}
    return pool, sampling


def collect_solutions(source=None, samples=8):
    """Return verified finite representatives, preserving each input TCP pose."""
    if not 1 <= samples <= 8:
        raise ValueError("samples must be between 1 and 8")
    began = perf_counter()
    source = metric_path("portfolio_metrics.json") if source is None else Path(source)
    input_bytes = source.read_bytes()
    original = json.loads(input_bytes)
    robots = {}
    for key in KEYS:
        src = original["robots"][key]
        arm, target = get_arm(key), np.asarray(src["target_pose"])
        if key in ("ur5e", "lite6"):
            pool = []
            for index, s in enumerate(src["branches"]["solutions"]):
                _append_distinct(pool, _record(arm, target, s["q"],
                                 {"kind": "portfolio_metrics closed-form candidate", "index": index},
                                 iters=0))
            selected = list(range(min(samples, len(pool))))
            method = "Closed-form candidates, independently checked by MuJoCo FK"
            sampling = {"source_returned_count": len(src["branches"]["solutions"])}
            note = "Retained returned discrete candidates for this target; no completeness claim."
        elif key == "iiwa14":
            pool = _iiwa_pool(arm, target)
            selected, angles = _iiwa_select(pool, samples)
            method = "S-R-S closed form on the full sampled arm-angle grid"
            sampling = {"n_psi": 16, "max_solutions": 0,
                        "available_psi_rad": angles,
                        "selected_psi_rad": [pool[i]["psi_rad"] for i in selected]}
            note = "One representative per spaced available arm angle; branch choice maximizes joint-configuration separation. Finite grid, not all solutions."
        else:
            pool, sampling = _panda_pool(arm, target, src["branches"]["solutions"])
            selected = _diverse(pool, samples)
            method = "Panda fixed-q7 DLS continuation of one interior redundant family"
            note = "Interior candidates selected by max-min world joint-anchor separation; finite local scan, not all Panda solutions."
        if len(selected) < samples:
            raise RuntimeError(f"Only {len(selected)} distinct representatives available for {key}")
        solutions = [dict(pool[i], display_index=j + 1) for j, i in enumerate(selected)]
        qs = np.asarray([s["q"] for s in solutions])
        pair = [float(np.linalg.norm((qs[i] - qs[j] + np.pi) % (2*np.pi) - np.pi))
                for i in range(len(qs)) for j in range(i)]
        robots[key] = {
            "name": src["name"], "dof": src["dof"], "target_pose": target.tolist(),
            "joint_limits": {"low_rad": arm.joint_low.tolist(), "high_rad": arm.joint_high.tolist()},
            "solutions": solutions, "candidate_count": len(pool), "selected_count": len(solutions),
            "candidates": pool, "selected_candidate_indices": selected,
            "method": method, "selection_note": note, "sampling": sampling,
            "summary": {"max_pos_err_m": max(s["pos_err_m"] for s in solutions),
                        "max_rot_err_rad": max(s["rot_err_rad"] for s in solutions),
                        "min_joint_margin_rad": min(s["joint_margin_rad"] for s in solutions),
                        "min_pairwise_q_distance_rad": min(pair, default=None),
                        "boundary_selected_count": sum(s["joint_margin_rad"] <= 1e-3 for s in solutions)},
        }
    sources = ("scripts/showcase/solution_data.py", "scripts/showcase/common.py", "scripts/showcase/paths.py",
               "ik/kinematics.py", "ik/model.py", "ik/robot.py", "ik/srs.py", "ik/numerical.py")
    result = {
        "schema_version": 1, "experiment": {
            "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_s": perf_counter() - began,
            "input_file": str(source), "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
            "versions": {p: metadata.version(p) for p in ("numpy", "mujoco")},
            "position_tol_m": POSITION_TOL, "rotation_tol_rad": ROTATION_TOL,
            "dedup_joint_norm_tol_rad": DEDUP_TOL,
            "scope": "Independent valid configurations of the same TCP pose per robot. Candidate switching is not a continuous motion or a complete solution enumeration.",
        }, "robots": robots,
    }
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Input portfolio JSON; defaults to build, then the published snapshot")
    parser.add_argument("--output", type=Path, default=SHOW / "solution_metrics.json")
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    data = collect_solutions(source=args.source, samples=args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    for key, r in data["robots"].items():
        print(key, f"{r['selected_count']} selected / {r['candidate_count']} candidates", r["summary"], flush=True)


if __name__ == "__main__":
    main()
