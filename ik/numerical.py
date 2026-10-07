"""Numerical inverse kinematics for the Panda arm.

Implements three classic Jacobian-based iterative solvers:

* ``jt``   - Jacobian transpose (fixed gain)
* ``pinv`` - Moore-Penrose pseudo-inverse
* ``dls``  - damped least squares / Levenberg-Marquardt (adaptive damping)

All solvers share step limiting, joint-limit clamping and a 6-vector task
error ``[position ; rotation-vector]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import pose_error
from .model import PandaArm


@dataclass
class NumericalResult:
    q: np.ndarray
    success: bool
    iters: int
    pos_err: float
    rot_err: float
    method: str

    def cost(self) -> float:
        return self.pos_err + self.rot_err


def _dls_step(J: np.ndarray, e: np.ndarray, lam: float) -> np.ndarray:
    A = J @ J.T + (lam ** 2) * np.eye(J.shape[0])
    return J.T @ np.linalg.solve(A, e)


def solve_numerical(
    arm: PandaArm,
    T_target: np.ndarray,
    q0: np.ndarray,
    method: str = "dls",
    *,
    tol_pos: float = 1e-5,
    tol_rot: float = 1e-5,
    max_iters: int = 300,
    max_step: float = 0.5,
    jt_gain: float = 1.0,
    lambda0: float = 0.05,
    lam_min: float = 1e-8,
    lam_max: float = 1e3,
    rot_weight: float = 1.0,
    clamp_limits: bool = True,
) -> NumericalResult:
    """Solve IK for a single target pose.

    Parameters
    ----------
    T_target : 4x4 homogeneous target pose of the flange frame.
    q0 : initial joint configuration.
    method : ``"jt"``, ``"pinv"`` or ``"dls"``.
    tol_pos, tol_rot : convergence tolerances for the position (m) and
        orientation (rad) error magnitudes.
    max_step : maximum joint-space step norm per iteration (rad).
    rot_weight : relative weight of the orientation error in the task.
    clamp_limits : clamp joint values to the model limits.

    Returns
    -------
    NumericalResult with the best configuration found.
    """
    R_t = np.asarray(T_target, float)[:3, :3]
    p_t = np.asarray(T_target, float)[:3, 3]

    q = np.asarray(q0, float).copy()
    if clamp_limits:
        q = np.clip(q, arm.joint_low, arm.joint_high)

    def task_error(qq: np.ndarray) -> np.ndarray:
        arm.set_q(qq)
        R, p = arm.tcp_pose()
        e = pose_error(R, p, R_t, p_t)
        e[3:] *= rot_weight
        return e

    e = task_error(q)
    cost = float(e @ e)
    lam = lambda0
    iters = 0

    for iters in range(1, max_iters + 1):
        if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
            break

        J = arm.tcp_jacobian().copy()
        J[3:, :] *= rot_weight

        if method == "jt":
            # Adaptive gain: alpha < 2 / sigma_max^2 guarantees descent.
            sigma_max = np.linalg.svd(J, compute_uv=False)[0]
            dq = (jt_gain / (sigma_max ** 2 + 1e-12)) * (J.T @ e)
        elif method == "pinv":
            dq = np.linalg.pinv(J) @ e
        elif method == "dls":
            dq = _dls_step(J, e, lam)
        else:
            raise ValueError(f"unknown method: {method!r}")

        if not np.all(np.isfinite(dq)):
            break

        ndq = np.linalg.norm(dq)
        if ndq > max_step:
            dq *= max_step / ndq

        q_new = q + dq
        if clamp_limits:
            q_new = np.clip(q_new, arm.joint_low, arm.joint_high)

        e_new = task_error(q_new)
        cost_new = float(e_new @ e_new)

        if method == "dls":
            if cost_new < cost:
                q, e, cost = q_new, e_new, cost_new
                lam = max(lam / 1.5, lam_min)
            else:
                lam = min(lam * 2.0, lam_max)
                if lam >= lam_max:
                    break
                continue  # retry the same q with more damping
        else:
            # Guard against divergence for jt/pinv.
            if not np.all(np.isfinite(e_new)):
                break
            q, e, cost = q_new, e_new, cost_new

    e = task_error(q)
    pos_err = float(np.linalg.norm(e[:3]))
    rot_err = float(np.linalg.norm(e[3:]) / max(rot_weight, 1e-12))
    success = pos_err < tol_pos and rot_err < tol_rot
    return NumericalResult(q, success, iters, pos_err, rot_err, method)


def random_seeds(arm: PandaArm, n: int, seed: int | None = None) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        rng.uniform(arm.joint_low, arm.joint_high) for _ in range(max(1, n))
    ]


def solve_numerical_multiseed(
    arm: PandaArm,
    T_target: np.ndarray,
    seeds: list[np.ndarray],
    **kwargs,
) -> NumericalResult:
    """Run the solver from several seeds and return the best result."""
    best: NumericalResult | None = None
    for q0 in seeds:
        res = solve_numerical(arm, T_target, q0, **kwargs)
        if best is None or res.cost() < best.cost():
            best = res
        if best.success:
            break
    assert best is not None
    return best
