"""Branch labels and truthful solver diagnostics used by showcase checks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ik import Lite6Solver, Ur5eSolver, solve_analytic
from .common import arm_frames, elbow_center, max_limit_hit


def _elbow_up(arm, q, idx):
    arm.set_q(np.asarray(q, float))
    anchors, axes, Sh, W = arm_frames(arm)
    if Sh is None or W is None:
        return True
    E = elbow_center(anchors, axes, Sh, idx)
    u = W - Sh
    u = u / np.linalg.norm(u)
    d = E - Sh
    perp = d - (d @ u) * u
    return bool(perp[2] >= 0.0)


def _label_6(arm, q, elbow_idx=2):
    elbow = "肘上" if _elbow_up(arm, q, elbow_idx) else "肘下"
    shoulder = "肩正" if q[0] >= 0 else "肩反"
    wrist = "腕正" if q[4] >= 0 else "腕反"
    return "·".join((elbow, shoulder, wrist))


@dataclass
class _BranchDisplay:
    q: np.ndarray
    pos_err: float
    rot_err: float
    kind: str
    name: str
    iters: int = 0
    near_limit: bool = False


def _branch_displays(key, arm, T, tol_pos=1e-4, tol_rot=1e-5):
    """Keep solver diagnostics together with each displayed configuration."""
    if key == "ur5e":
        sols = Ur5eSolver(arm).solve(T, tol_pos=tol_pos, tol_rot=tol_rot)
        return [_BranchDisplay(s.q, s.pos_err, s.rot_err, "exact", _label_6(arm, s.q))
                for s in sols]
    if key == "lite6":
        sols = Lite6Solver(arm).solve(T, tol_pos=tol_pos, tol_rot=tol_rot)
        return [_BranchDisplay(s.q, s.pos_err, s.rot_err, "exact", _label_6(arm, s.q))
                for s in sols]
    if key == "panda":
        out = []
        for s in solve_analytic(arm, T, max_solutions=8,
                                tol_pos=tol_pos, tol_rot=tol_rot):
            near_limit = max_limit_hit(arm, s.q)
            out.append(_BranchDisplay(
                s.q, s.pos_err, s.rot_err, "converged",
                "抛光收敛 · 近限位" if near_limit else "抛光收敛",
                s.iters, near_limit,
            ))
        return out
    raise KeyError(key)


def enumerate_branches(key, arm, T, tol_pos=1e-4, tol_rot=1e-5):
    """6R branches or finite Panda samples; retain the five-item tuple API."""
    return [(s.q, s.pos_err, s.rot_err, s.kind, s.name)
            for s in _branch_displays(key, arm, T, tol_pos, tol_rot)]
