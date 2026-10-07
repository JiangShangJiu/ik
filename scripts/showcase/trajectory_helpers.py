"""Cartesian pose sampling, nearest-branch tracking and jump diagnostics.

The random short-segment experiment includes interpolated end orientations.
The fixed-orientation presentation line/circle paths live in paths.py.
"""
from __future__ import annotations

import numpy as np

from ik import Lite6Solver, SrsSolver, Ur5eSolver
from ik.numerical import solve_numerical


def _exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def slerp(R0, R1, t):
    from ik.kinematics import rot_log
    return R0 @ _exp_so3(t * rot_log(R0.T @ R1))


def path_poses(T0, T1, n):
    out = []
    p0, p1 = T0[:3, 3], T1[:3, 3]
    for t in np.linspace(0.0, 1.0, n):
        T = np.eye(4)
        T[:3, :3] = slerp(T0[:3, :3], T1[:3, :3], t)
        T[:3, 3] = (1 - t) * p0 + t * p1
        out.append(T)
    return out


def _solvers(key, arm):
    if key == "ur5e":
        return ("exact", Ur5eSolver(arm))
    if key == "lite6":
        return ("exact", Lite6Solver(arm))
    if key == "iiwa14":
        return ("exact", SrsSolver(arm, n_psi=16))
    return ("num", None)


def make_line(key, arm, q_ref, rng, n=48, delta=0.30):
    for _ in range(40):
        q_end = q_ref + rng.uniform(-delta, delta, len(arm.joint_low))
        q_end = np.clip(q_end, arm.joint_low, arm.joint_high)
        arm.set_q(q_ref)
        T0 = arm.tcp_transform()
        arm.set_q(q_end)
        T1 = arm.tcp_transform()
        Ts = path_poses(T0, T1, n)
        qs = solve_path(key, arm, Ts, q_ref)
        if qs is not None:
            return Ts, qs
        delta *= 0.8
    raise RuntimeError(f"no reachable line for {key}")


def solve_path(key, arm, Ts, q_start):
    kind, solver = _solvers(key, arm)
    qs = []
    prev = None
    for T in Ts:
        if kind == "exact":
            sols = solver.solve(T, tol_pos=1e-5, tol_rot=1e-5, max_solutions=16) \
                if key == "iiwa14" else solver.solve(T, tol_pos=1e-5, tol_rot=1e-5)
            if not sols:
                return None
            ref = prev if prev is not None else q_start
            best = min(sols, key=lambda s: float(np.linalg.norm(s.q - ref)))
            q = best.q
        else:
            res = solve_numerical(arm, T, prev if prev is not None else q_start,
                                  method="dls", max_iters=200,
                                  tol_pos=1e-7, tol_rot=1e-7)
            if res.pos_err > 1e-5 or res.rot_err > 1e-5:
                return None
            q = res.q
        qs.append(q)
        prev = q
    return np.array(qs)


def _bad_branch_samples(Ts, qs, solver):
    """Three adjacent, pose-correct frames with a deliberate middle jump.

    Re-solve the next target on the new branch, rather than displaying the
    middle configuration under the next target's marker.
    """
    mid = len(Ts) // 2
    idx = [mid - 1, mid, mid + 1]
    sols = solver.solve(Ts[mid], tol_pos=1e-5, tol_rot=1e-5)
    if not sols:
        raise RuntimeError("no IK solution at the branch-switch target")
    flipped = max(sols, key=lambda s: float(np.linalg.norm(s.q - qs[mid]))).q
    if np.linalg.norm(flipped - qs[mid]) < 0.1:
        raise RuntimeError("target has no distinct branch for the bad example")
    after = solver.solve(Ts[mid + 1], tol_pos=1e-5, tol_rot=1e-5)
    if not after:
        raise RuntimeError("new branch has no solution at the next target")
    continued = min(after, key=lambda s: float(np.linalg.norm(s.q - flipped))).q
    selected = [qs[mid - 1], flipped, continued]
    previous = [qs[mid - 2], selected[0], selected[1]]
    jumps = [float(np.linalg.norm(q - prev))
             for q, prev in zip(selected, previous)]
    return idx, selected, jumps
