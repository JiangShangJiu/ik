"""Arm-angle (swivel) semi-analytic inverse kinematics for the Franka Panda.

The Panda is *almost* an S-R-S arm:

* the shoulder is spherical - joint axes 1, 2 and 3 all pass through
  ``S = (0, 0, 0.333)``;
* the elbow is a single revolute joint (axis normal to the arm plane);
* the wrist is **not** spherical - the flange (``link7`` origin) is offset by
  ``0.088 m`` from the wrist centre ``W = J5 ∩ J6``.

Because of the wrist offset, the ideal S-R-S position/orientation decoupling
does not apply directly. This implementation builds
**geometric seeds** from the arm-angle parameterisation and refines each seed
with damped-Newton steps. This is an implementation choice, not a claim that
Panda has no closed-form IK: analytical methods can fix a redundant parameter
such as q7. Without that extra constraint, regular target poses still have a
one-dimensional self-motion solution set.

Geometry (all verified numerically against MuJoCo):

* ``E - S = d3 * u + a3 * x3``  with ``u = [sin q2 cos q1, sin q2 sin q1, cos q2]``
* joint-4 axis ``A4 = x3 x u`` is the arm-plane normal
* ``q3`` from ``x3 = cos q3 * x2 + sin q3 * z2``
* ``q4`` from the forearm direction ``E -> W``
* ``q5, q6, q7`` from ``R4^7 = Rx(-pi/2) Rz(q5) Rx(pi/2) Rz(q6) Rx(pi/2) Rz(q7)``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import PANDA_MDH, mdh_transform, rot_log
from .model import PandaArm, PandaGeometry
from .numerical import solve_numerical

# --- constants derived from the DH table (no magic geometry numbers) -------- #
_D3 = float(PANDA_MDH[2, 2])                 # |S -> O3|
_A3 = float(PANDA_MDH[3, 1])                 # elbow x-offset (|O3 -> E|)
_K = np.array([PANDA_MDH[4, 1], PANDA_MDH[4, 2], 0.0])  # E->W dir in frame 4
_K = _K / np.linalg.norm(_K)


@dataclass
class AnalyticSolution:
    q: np.ndarray
    success: bool
    pos_err: float
    rot_err: float
    iters: int = 0
    seed_pos_err: float | None = None
    seed_rot_err: float | None = None
    seed_q: np.ndarray | None = None

    def cost(self) -> float:
        return self.pos_err + self.rot_err


def _Rx(a: float) -> np.ndarray:
    return mdh_transform(a, 0.0, 0.0, 0.0)[:3, :3]


def _Rz(a: float) -> np.ndarray:
    return mdh_transform(0.0, 0.0, 0.0, a)[:3, :3]


def _R03(q1: float, q2: float, q3: float) -> np.ndarray:
    R = np.eye(3)
    for i, qi in enumerate((q1, q2, q3)):
        R = R @ mdh_transform(
            PANDA_MDH[i, 0], PANDA_MDH[i, 1], PANDA_MDH[i, 2], qi
        )[:3, :3]
    return R


def _R04(q1: float, q2: float, q3: float, q4: float) -> np.ndarray:
    return _R03(q1, q2, q3) @ _Rx(np.pi / 2) @ _Rz(q4)


def wrist_branches(R: np.ndarray) -> list[tuple[float, float, float]]:
    """Both (q5, q6, q7) solutions of ``R = R4^7``."""
    c6 = float(np.clip(-R[1, 2], -1.0, 1.0))
    out: list[tuple[float, float, float]] = []
    for sign in (1.0, -1.0):
        s6 = sign * np.sqrt(max(1.0 - c6 ** 2, 0.0))
        q6 = float(np.arctan2(s6, c6))
        if abs(s6) < 1e-9:  # wrist singularity: q5, q7 not separable
            out.append((0.0, q6, float(np.arctan2(-R[1, 1], R[1, 0]))))
            continue
        q5 = float(np.arctan2(-R[2, 2] / s6, R[0, 2] / s6))
        q7 = float(np.arctan2(-R[1, 1] / s6, R[1, 0] / s6))
        out.append((q5, q6, q7))
    return out


def shoulder_solve(E: np.ndarray, A4: np.ndarray, S: np.ndarray) -> list[tuple[float, float, float]]:
    """Closed-form (q1, q2, q3) from elbow centre ``E`` and joint-4 axis ``A4``."""
    v = E - S
    A4 = A4 / np.linalg.norm(A4)
    e1 = v / np.linalg.norm(v)
    e2 = np.cross(A4, e1)
    beta0 = np.arctan2(_A3, _D3)
    out: list[tuple[float, float, float]] = []
    for beta in (beta0, -beta0):
        u = np.cos(beta) * e1 + np.sin(beta) * e2
        x3 = (v - _D3 * u) / _A3
        if abs(np.cross(x3, u) @ A4 - 1.0) > 1e-6:
            continue
        q2 = float(np.arccos(np.clip(u[2], -1.0, 1.0)))
        q1 = float(np.arctan2(u[1], u[0]))
        R2 = _Rz(q1) @ _Rx(-np.pi / 2) @ _Rz(q2)
        q3 = float(np.arctan2(x3 @ R2[:, 2], x3 @ R2[:, 0]))
        out.append((q1, q2, q3))
    return out


def elbow_circle(W: np.ndarray, S: np.ndarray, psi: float, geom: PandaGeometry):
    """Elbow centre for arm angle ``psi`` (or ``None`` if out of reach)."""
    d_se, d_ew = geom.d_se, geom.d_ew
    d = W - S
    L = float(np.linalg.norm(d))
    if not (abs(d_se - d_ew) <= L <= d_se + d_ew):
        return None
    ca = np.clip((d_se ** 2 + L ** 2 - d_ew ** 2) / (2 * d_se * L), -1.0, 1.0)
    alpha = np.arccos(ca)
    u = d / L
    n = S + d_se * np.cos(alpha) * u
    rho = d_se * np.sin(alpha)
    ref = np.array([0.0, 0.0, 1.0])
    h = ref - (ref @ u) * u
    if np.linalg.norm(h) < 1e-9:
        h = np.array([1.0, 0.0, 0.0])
    h = h / np.linalg.norm(h)
    v = np.cross(u, h)
    E = n + rho * (np.cos(psi) * h + np.sin(psi) * v)
    return E, n, h, v, rho


def _arm_seeds_for_W(W, target_R, S, geom, n_psi):
    R_t = target_R
    seeds: list[np.ndarray] = []
    for psi in np.linspace(0.0, 2 * np.pi, n_psi, endpoint=False):
        res = elbow_circle(W, S, psi, geom)
        if res is None:
            continue
        E = res[0]
        for A4 in (np.cross(E - S, W - E), -np.cross(E - S, W - E)):
            for (q1, q2, q3) in shoulder_solve(E, A4, S):
                dd = (W - E) / np.linalg.norm(W - E)
                w = _Rx(-np.pi / 2) @ (_R03(q1, q2, q3).T @ dd)
                q4 = np.arctan2(w[1], w[0]) - np.arctan2(_K[1], _K[0])
                q4 = (q4 + np.pi) % (2 * np.pi) - np.pi
                R_47 = _R04(q1, q2, q3, q4).T @ R_t
                for (q5, q6, q7) in wrist_branches(R_47):
                    seeds.append(np.array([q1, q2, q3, q4, q5, q6, q7]))
    return seeds


def analytic_seeds(
    arm: PandaArm,
    T_target: np.ndarray,
    *,
    n_theta: int = 8,
    n_psi: int = 4,
    theta_scan: int = 72,
) -> list[np.ndarray]:
    """Geometry-derived initial guesses covering the solution branches.

    ``W`` (the wrist centre) lies on a circle of radius ``wrist_offset`` around
    the flange, perpendicular to the approach axis. We scan ``theta`` around
    that circle, keep the feasible samples, and for each combine the arm-angle
    ``psi`` with the shoulder/wrist branches.
    """
    T_target = np.asarray(T_target, float)
    R_t, p_t = T_target[:3, :3], T_target[:3, 3]
    geom = arm.geometry()
    S = geom.shoulder_center
    off = geom.wrist_offset
    d_se, d_ew = geom.d_se, geom.d_ew

    feasible = []
    for theta in np.linspace(0.0, 2 * np.pi, theta_scan, endpoint=False):
        x6 = R_t @ np.array([np.cos(theta), -np.sin(theta), 0.0])
        W = p_t - off * x6
        L = np.linalg.norm(W - S)
        if abs(d_se - d_ew) <= L <= d_se + d_ew:
            feasible.append(W)
    if not feasible:
        return []
    if len(feasible) > n_theta:
        idx = np.linspace(0, len(feasible) - 1, n_theta).astype(int)
        feasible = [feasible[i] for i in idx]

    seeds: list[np.ndarray] = []
    for W in feasible:
        seeds.extend(_arm_seeds_for_W(W, R_t, S, geom, n_psi))
    return seeds


def _dedupe(sols: list[np.ndarray], tol: float = 1e-3) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for q in sols:
        if not any(
            np.linalg.norm((np.mod(q - o + np.pi, 2 * np.pi) - np.pi)) < tol
            for o in out
        ):
            out.append(q)
    return out


def solve_analytic(
    arm: PandaArm,
    T_target: np.ndarray,
    *,
    n_theta: int = 8,
    n_psi: int = 4,
    polish_iters: int = 60,
    tol_pos: float = 1e-4,
    tol_rot: float = 1e-4,
    max_solutions: int = 8,
) -> list[AnalyticSolution]:
    """Return distinct solution samples found by seeding + Newton polishing.

    The returned list is a finite sample of Panda's redundant solution set,
    not an enumeration of all IK solutions. ``max_solutions`` is a budget for
    successful polish attempts, before the final deduplication.
    """
    T_target = np.asarray(T_target, float)
    seeds = analytic_seeds(arm, T_target, n_theta=n_theta, n_psi=n_psi)

    polished: list[AnalyticSolution] = []
    seen_q: list[np.ndarray] = []
    for seed in seeds:
        seed = np.clip(seed, arm.joint_low, arm.joint_high)
        if any(
            np.linalg.norm((np.mod(seed - o + np.pi, 2 * np.pi) - np.pi)) < 1e-3
            for o in seen_q
        ):
            continue
        seen_q.append(seed)
        arm.set_q(seed)
        R_seed, p_seed = arm.tcp_pose()
        seed_pos_err = float(np.linalg.norm(p_seed - T_target[:3, 3]))
        seed_rot_err = float(np.linalg.norm(rot_log(T_target[:3, :3] @ R_seed.T)))
        res = solve_numerical(
            arm, T_target, seed, method="dls",
            max_iters=polish_iters, tol_pos=tol_pos, tol_rot=tol_rot,
        )
        if res.success:
            polished.append(
                AnalyticSolution(
                    res.q, True, res.pos_err, res.rot_err, res.iters,
                    seed_pos_err, seed_rot_err, seed.copy(),
                )
            )
        if len(polished) >= max_solutions:
            break

    polished.sort(key=lambda s: s.cost())
    unique: list[AnalyticSolution] = []
    for s in polished:
        if not any(
            np.linalg.norm(np.mod(s.q - u.q + np.pi, 2 * np.pi) - np.pi) < 1e-3
            for u in unique
        ):
            unique.append(s)
    return unique


def solve_analytic_best(arm: PandaArm, T_target: np.ndarray, **kwargs) -> AnalyticSolution | None:
    """Fast single-solution variant: stop at the first converged branch."""
    kwargs.setdefault("max_solutions", 1)
    sols = solve_analytic(arm, T_target, **kwargs)
    return sols[0] if sols else None
