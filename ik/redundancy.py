"""Redundancy resolution: use the 1-DOF null space of the 7-DOF Panda.

For a 7-DOF arm solving a 6-DOF pose task the Jacobian ``J`` is 6x7, so the
task leaves a 1-dimensional self-motion manifold. The classic velocity-level
resolution is

    dq = J# e  +  N @ dq_secondary,        N = I - J# J

where ``J#`` is a (damped) pseudo-inverse and ``N`` projects the secondary
objective into the null space so that it does not disturb the primary task
to first order.

Provided secondary objectives:

* ``limit_avoidance``  - push joints away from their range limits
* ``posture``          - bias toward a desired (nominal) configuration
* ``manipulability``   - maximise sqrt(det(J J^T)) (distance from singularities)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .kinematics import pose_error
from .model import PandaArm
from .numerical import NumericalResult


@dataclass
class SecondaryTask:
    fn: Callable[[PandaArm, np.ndarray, np.ndarray], np.ndarray]
    weight: float
    name: str


# --------------------------------------------------------------------------- #
# Damped pseudo-inverse / null-space projector
# --------------------------------------------------------------------------- #
def damped_pinv(J: np.ndarray, lam: float) -> np.ndarray:
    """Damped (Levenberg-Marquardt) pseudo-inverse, shape (nv, m)."""
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(m))


def nullspace_projector(J: np.ndarray, lam: float) -> np.ndarray:
    """Null-space projector ``N = I - J# J`` (nv x nv)."""
    return np.eye(J.shape[1]) - damped_pinv(J, lam) @ J


# --------------------------------------------------------------------------- #
# Secondary objective factories
# --------------------------------------------------------------------------- #
def _limit_gradient(q, low, high, margin) -> np.ndarray:
    g = np.zeros_like(q)
    d_lo = q - low
    d_hi = high - q
    m = d_lo < margin
    g[m] += (margin - d_lo[m]) / margin
    m = d_hi < margin
    g[m] -= (margin - d_hi[m]) / margin
    return g


def limit_avoidance(margin: float = 0.35, weight: float = 0.5) -> SecondaryTask:
    def fn(arm: PandaArm, q: np.ndarray, J: np.ndarray) -> np.ndarray:
        return _limit_gradient(q, arm.joint_low, arm.joint_high, margin)

    return SecondaryTask(fn, weight, "limit_avoidance")


def posture(q_des: np.ndarray, weight: float = 0.5) -> SecondaryTask:
    q_des = np.asarray(q_des, dtype=float)

    def fn(arm: PandaArm, q: np.ndarray, J: np.ndarray) -> np.ndarray:
        return q_des - q

    return SecondaryTask(fn, weight, "posture")


def manipulability(weight: float = 0.3, eps: float = 1e-6) -> SecondaryTask:
    def fn(arm: PandaArm, q: np.ndarray, J: np.ndarray) -> np.ndarray:
        # d/dq sqrt(det(J J^T)) by finite differences (7 extra evaluations).
        def w(qq: np.ndarray) -> float:
            arm.set_q(qq)
            Jq = arm.tcp_jacobian()
            val = np.linalg.det(Jq @ Jq.T)
            return float(np.sqrt(max(val, 0.0)))

        g = np.zeros_like(q)
        for i in range(len(q)):
            qp = q.copy()
            qp[i] += eps
            qm = q.copy()
            qm[i] -= eps
            g[i] = (w(qp) - w(qm)) / (2 * eps)
        arm.set_q(q)  # restore
        return g

    return SecondaryTask(fn, weight, "manipulability")


# --------------------------------------------------------------------------- #
# Solver with redundancy resolution
# --------------------------------------------------------------------------- #
def solve_redundant(
    arm: PandaArm,
    T_target: np.ndarray,
    q0: np.ndarray,
    secondary: tuple[SecondaryTask, ...] = (),
    *,
    tol_pos: float = 1e-5,
    tol_rot: float = 1e-5,
    max_iters: int = 300,
    max_step: float = 0.5,
    lambda0: float = 0.05,
    lam_min: float = 1e-8,
    lam_max: float = 1e3,
    secondary_cap: float = 0.3,
    clamp_limits: bool = True,
) -> NumericalResult:
    """DLS primary task + null-space secondary objectives.

    The total secondary contribution is norm-bounded by ``secondary_cap`` so
    that it can never dominate the primary task regardless of its weight.
    """
    T_target = np.asarray(T_target, dtype=float)
    R_t, p_t = T_target[:3, :3], T_target[:3, 3]

    q = np.asarray(q0, dtype=float).copy()
    if clamp_limits:
        q = np.clip(q, arm.joint_low, arm.joint_high)

    def task_error(qq: np.ndarray) -> np.ndarray:
        arm.set_q(qq)
        R, p = arm.tcp_pose()
        return pose_error(R, p, R_t, p_t)

    e = task_error(q)
    cost = float(e @ e)
    lam = lambda0
    iters = 0

    for iters in range(1, max_iters + 1):
        if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
            break

        J = arm.tcp_jacobian()
        Jd = damped_pinv(J, lam)

        # --- primary step (bounded) ---
        dq_p = Jd @ e
        np_ = np.linalg.norm(dq_p)
        if np_ > max_step:
            dq_p *= max_step / np_
        q_p = q + dq_p
        if clamp_limits:
            q_p = np.clip(q_p, arm.joint_low, arm.joint_high)
        e_p = task_error(q_p)
        cost_p = float(e_p @ e_p)
        q_new, e_new, cost_new = q_p, e_p, cost_p

        # --- secondary step: only accepted if it does not worsen the task ---
        if secondary:
            N = np.eye(J.shape[1]) - Jd @ J
            dq_sec = np.zeros_like(dq_p)
            for task in secondary:
                dq_sec += N @ (task.weight * task.fn(arm, q, J))
            n_sec = np.linalg.norm(dq_sec)
            if n_sec > secondary_cap:
                dq_sec *= secondary_cap / n_sec
            if n_sec > 0:
                dq_tot = dq_p + dq_sec
                nt = np.linalg.norm(dq_tot)
                if nt > max_step:
                    dq_tot *= max_step / nt
                q_s = q + dq_tot
                if clamp_limits:
                    q_s = np.clip(q_s, arm.joint_low, arm.joint_high)
                e_s = task_error(q_s)
                cost_s = float(e_s @ e_s)
                if cost_s <= cost_p * (1 + 1e-9) + 1e-18:
                    q_new, e_new, cost_new = q_s, e_s, cost_s

        if cost_new < cost:
            q, e, cost = q_new, e_new, cost_new
            lam = max(lam / 1.5, lam_min)
        else:
            lam = min(lam * 2.0, lam_max)
            if lam >= lam_max:
                break

    e = task_error(q)
    pos_err = float(np.linalg.norm(e[:3]))
    rot_err = float(np.linalg.norm(e[3:]))
    success = pos_err < tol_pos and rot_err < tol_rot
    name = "dls+" + "+".join(t.name for t in secondary) if secondary else "dls"
    return NumericalResult(q, success, iters, pos_err, rot_err, name)


def nullspace_step(arm: PandaArm, q: np.ndarray, task: SecondaryTask, lam: float = 1e-3) -> np.ndarray:
    """A pure null-space step (for demonstrating self-motion)."""
    arm.set_q(q)
    J = arm.tcp_jacobian()
    N = nullspace_projector(J, lam)
    return N @ (task.weight * task.fn(arm, q, J))
