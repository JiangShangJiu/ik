"""Verified Cartesian line and full-circle TCP motions for bright animations.

Every frame is solved at its prescribed Cartesian position and fixed rotation.
No joint interpolation is used. The circle is a TCP path, not an elbow circle.
Run ``python -m scripts.showcase.path_data`` to write path_metrics.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from ik import Lite6Solver, SrsSolver, Ur5eSolver
from ik.kinematics import rot_log
from ik.numerical import solve_numerical
from .common import get_arm
from .paths import ROOT, SHOW

KEYS = ("ur5e", "lite6", "iiwa14", "panda")
NAMES = {"ur5e": "UR5e", "lite6": "Lite 6", "iiwa14": "iiwa 14", "panda": "Panda"}
POSITION_TOL = ROTATION_TOL = 1e-5
MAX_FRAME_DQ = 0.12
DEFAULT_FRAMES = 240

# Presentation dimensions are explicit and never reduced after an IK failure.
# The line length is the distance between its two endpoints; one animation
# travels that segment forward and backward. Circle sizes are radii.
PATH_SIZES = {
    "ur5e": {"line_length_m": 1.12, "circle_radius_m": .40},
    "lite6": {"line_length_m": .68, "circle_radius_m": .26},
    "iiwa14": {"line_length_m": 1.12, "circle_radius_m": .40},
    "panda": {"line_length_m": 1.12, "circle_radius_m": .40},
}

# Tilted planes keep the larger TCP circles clear of the world z=0 plane.
# Every path still uses one fixed end-effector orientation.
PATH_PLANES = {key: {"u": [0., 1., 0.], "v": [0., 0., 1.]} for key in KEYS}
PATH_PLANES["ur5e"] = {"u": [0., 1., 0.],
                         "v": [np.sin(np.deg2rad(40)), 0., np.cos(np.deg2rad(40))]}
PATH_PLANES["panda"] = {"u": [0., 1., 0.],
                         "v": [-np.sin(np.pi / 12), 0., np.cos(np.pi / 12)]}
FIXED_ARM_ANGLES = {"iiwa14": -.5}

# Deliberately outward-reaching presentation postures, all subsequently checked
# against the model. These are joint-space seeds, not invented path solutions.
SEEDS = {
    "ur5e": [[0, -1.6, 2.1, -1.9, -1.5, 0]],
    "lite6": [[0, -.3, .6, 0, .9, 0]],
    "iiwa14": [[0, .2, .2, -2.0, -.1, .9, 0]],
    "panda": [[0, -.6, 0, -2.8, 0, 2., .785]],
}


class _FixedJoint:
    """Six-variable view with one explicitly indexed Panda joint held fixed."""
    def __init__(self, arm, q_ref, index):
        self.arm, self.q_ref, self.index = arm, np.asarray(q_ref).copy(), index
        self.active = np.arange(len(q_ref)) != index
        self.joint_low = arm.joint_low[self.active]
        self.joint_high = arm.joint_high[self.active]

    def expand(self, q):
        full_q = self.q_ref.copy()
        full_q[self.active] = q
        return full_q

    def set_q(self, q):
        self.arm.set_q(self.expand(q))

    def tcp_pose(self):
        return self.arm.tcp_pose()

    def tcp_jacobian(self):
        return self.arm.tcp_jacobian()[:, self.active]


class _Solver:
    def __init__(self, key, arm, q_ref, T_ref):
        self.key, self.arm = key, arm
        if key in ("ur5e", "lite6"):
            self.solver = Ur5eSolver(arm) if key == "ur5e" else Lite6Solver(arm)
            self.constraint = {"method": "closed-form candidates, nearest previous configuration"}
        elif key == "iiwa14":
            self.solver = SrsSolver(arm)
            arm.set_q(q_ref)
            anchors, axes = arm.world_lines()
            direction = axes[3] / np.linalg.norm(axes[3])
            E = anchors[3] + np.dot(self.solver.S - anchors[3], direction) * direction
            _, _, _, center, h, v, _ = self.solver._elbow_circle(T_ref)
            self.psi = float(np.arctan2((E - center) @ v, (E - center) @ h))
            self.psi = FIXED_ARM_ANGLES.get(key, self.psi)
            self.constraint = {"method": "S-R-S closed form, fixed arm angle, nearest branch",
                               "fixed_psi_rad": self.psi}
        else:
            # q3 permits a larger complete path than the former fixed-q7 slice.
            self.solver = _FixedJoint(arm, q_ref, index=2)
            self.constraint = {"method": "DLS with q3 fixed; six active joint variables",
                               "fixed_joint_index_zero_based": 2,
                               "fixed_joint_name": "q3", "fixed_joint_value_rad": float(q_ref[2])}

    def solve(self, T, previous):
        if self.key == "panda":
            result = solve_numerical(self.solver, T, np.asarray(previous)[self.solver.active],
                                     method="dls", max_iters=160, max_step=.3,
                                     tol_pos=2e-8, tol_rot=2e-8, lambda0=.005)
            if not result.success:
                raise RuntimeError(f"Panda fixed-q3 IK failed: {result.pos_err:.2e} m")
            return self.solver.expand(result.q)
        if self.key == "iiwa14":
            solutions = self.solver.solve_at_psi(T, self.psi,
                                                 tol_pos=POSITION_TOL, tol_rot=ROTATION_TOL)
        else:
            solutions = self.solver.solve(T, tol_pos=POSITION_TOL, tol_rot=ROTATION_TOL)
        if not solutions:
            raise RuntimeError(f"{self.key}: no valid IK candidate on Cartesian path")
        return min(solutions, key=lambda s: np.linalg.norm(s.q - previous)).q.copy()


def _transforms(rotation, points):
    Ts = np.repeat(np.eye(4)[None], len(points), axis=0)
    Ts[:, :3, :3], Ts[:, :3, 3] = rotation, points
    return Ts


def _record(arm, solver, Ts, q_ref, T_ref, geometry):
    # Move from the known reachable centre to the first Cartesian point by IK.
    # These approach samples are not substituted for the prescribed animation.
    previous = q_ref.copy()
    for t in np.linspace(0, 1, 13)[1:]:
        T = Ts[0].copy()
        T[:3, 3] = (1 - t) * T_ref[:3, 3] + t * Ts[0, :3, 3]
        previous = solver.solve(T, previous)
    qs, points, position, rotation, margins = [], [], [], [], []
    for T in Ts:
        q = solver.solve(T, previous)
        arm.set_q(q)
        R, p = arm.tcp_pose()
        qs.append(q.copy())
        points.append(p.copy())
        position.append(float(np.linalg.norm(p - T[:3, 3])))
        rotation.append(float(np.linalg.norm(rot_log(T[:3, :3] @ R.T))))
        margins.append(float(np.min(np.minimum(q - arm.joint_low, arm.joint_high - q))))
        previous = q
    qs, points = np.asarray(qs), np.asarray(points)
    steps = np.linalg.norm(np.diff(qs, axis=0), axis=1)
    seam = float(np.linalg.norm(qs[-1] - qs[0]))
    closed_q = solver.solve(Ts[0], qs[-1])
    arm.set_q(closed_q)
    closed_R, closed_p = arm.tcp_pose()
    closure_q = float(np.linalg.norm(closed_q - qs[0]))
    closure_pos = float(np.linalg.norm(closed_p - Ts[0, :3, 3]))
    closure_rot = float(np.linalg.norm(rot_log(Ts[0, :3, :3] @ closed_R.T)))
    if max(position) >= POSITION_TOL or max(rotation) >= ROTATION_TOL:
        raise RuntimeError("Cartesian frame failed position/orientation validation")
    if min(margins) < -1e-10:
        raise RuntimeError("Cartesian path exceeds a joint limit")
    if max(float(np.max(steps)), seam) >= MAX_FRAME_DQ or closure_q > 1e-4:
        raise RuntimeError(f"Cartesian path is discontinuous: step={max(steps):.3f}, "
                           f"seam={seam:.3f}, closure={closure_q:.3e} rad")
    radius = geometry.get("radius")
    if geometry["kind"] == "circle":
        path_length = 2 * np.pi * radius
        extent = 2 * radius
    else:
        extent = float(np.linalg.norm(np.asarray(geometry["line_end"]) - geometry["line_start"]))
        path_length = 2 * extent  # one complete forward/backward loop
    summary = {
        "valid": True, "n_frames": len(Ts), "fixed_orientation": True,
        "path_length_m": float(path_length), "path_extent_m": float(extent),
        "max_pos_error_m": max(position), "max_rot_error_rad": max(rotation),
        "min_joint_margin_rad": min(margins), "max_dq_norm_rad": float(np.max(steps)),
        "seam_dq_norm_rad": seam, "closure_q_error_rad": closure_q,
        "closure_pos_error_m": closure_pos, "closure_rot_error_rad": closure_rot,
        "frame_dq_limit_rad": MAX_FRAME_DQ,
        "min_actual_tcp_z_m": float(np.min(points[:, 2])),
    }
    return {"q_ref": q_ref.tolist(), "center_pose": T_ref.tolist(),
            "Ts": Ts.tolist(), "qs": qs.tolist(), "target_points_m": Ts[:, :3, 3].tolist(),
            "actual_points_m": points.tolist(), "pos_error_m": position,
            "rot_error_rad": rotation, "joint_margin_rad": margins,
            "dq_norm_rad": steps.tolist(), "geometry": geometry,
            "redundancy_constraint": solver.constraint, "summary": summary}


def _robot_paths(key, n_frames):
    arm = get_arm(key)
    errors = []
    # Keep the selected radius conspicuous rather than silently shrinking it.
    length = PATH_SIZES[key]["line_length_m"]
    radius = PATH_SIZES[key]["circle_radius_m"]
    u, v = (np.asarray(PATH_PLANES[key][axis], dtype=float) for axis in ("u", "v"))
    theta = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    for seed_index, seed in enumerate(SEEDS[key]):
        q_ref = np.asarray(seed, dtype=float)
        if np.any(q_ref < arm.joint_low) or np.any(q_ref > arm.joint_high):
            continue
        arm.set_q(q_ref)
        T_ref = arm.tcp_transform().copy()
        center = T_ref[:3, 3]
        solver = _Solver(key, arm, q_ref, T_ref)
        start, end = center - .5 * length * u, center + .5 * length * u
        progress = .5 * (1 - np.cos(theta))
        line_points = start + progress[:, None] * (end - start)
        circle_points = center + radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
        common = {"center": center.tolist(), "u": u.tolist(), "v": v.tolist(),
                  "line_start": start.tolist(), "line_end": end.tolist()}
        try:
            line = _record(arm, solver, _transforms(T_ref[:3, :3], line_points), q_ref, T_ref,
                           {**common, "kind": "line", "radius": None,
                            "sampling": "cosine forward/backward; endpoint excluded"})
            circle = _record(arm, solver, _transforms(T_ref[:3, :3], circle_points), q_ref, T_ref,
                             {**common, "kind": "circle", "radius": radius,
                              "line_start": None, "line_end": None,
                              "sampling": "one full TCP revolution; endpoint excluded"})
            return {"name": NAMES[key], "q_ref": q_ref.tolist(), "center_pose": T_ref.tolist(),
                    "seed_index": seed_index,
                    "joint_limits": {"low_rad": arm.joint_low.tolist(), "high_rad": arm.joint_high.tolist()},
                    "line": line, "circle": circle}
        except RuntimeError as error:
            errors.append(str(error))
    raise RuntimeError(f"No valid presentation paths for {key}: {'; '.join(errors)}")


def collect_paths(n_frames=DEFAULT_FRAMES):
    """Return freshly solved, JSON-safe line/circle paths for all four robots."""
    if n_frames < 72 or n_frames % 2:
        raise ValueError("n_frames must be an even integer >=72 for resolved closed loops")
    data = {"schema_version": 1,
            "experiment": {"created_utc": datetime.now(timezone.utc).isoformat(),
                           "n_frames": n_frames, "position_tol_m": POSITION_TOL,
                           "rotation_tol_rad": ROTATION_TOL,
                           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "scope": "Cartesian samples, fixed orientation, joint-limit and loop checks; no collision, timing, dynamics or hardware validation."},
            "robots": {key: _robot_paths(key, n_frames) for key in KEYS}}
    json.dumps(data, allow_nan=False)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--output", type=Path, default=SHOW / "path_metrics.json")
    args = parser.parse_args()
    data = collect_paths(args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)
    for key, robot in data["robots"].items():
        for kind in ("line", "circle"):
            print(key, kind, robot[kind]["summary"], flush=True)


if __name__ == "__main__":
    main()
