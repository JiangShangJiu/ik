"""Closed-form IK for spherical-shoulder / spherical-wrist (S-R-S) arms.

The classic "arm angle / swivel angle" method needs, for a given elbow position
``E``, the shoulder angles ``q1..q3`` and the wrist angles ``q5..q7`` in closed
form.  Instead of re-deriving those per robot, we exploit the *product of
exponentials* about a reference configuration ``q_ref``:

    R(q) = Rw(w1, dq1) ... Rw(w7, dq7) R(q_ref),   dq = q - q_ref

where ``wi`` are the joint axes (in the base frame) *at* ``q_ref``.  Then

* the shoulder maps a fixed upper-arm vector ``u0`` and elbow-axis ``a0`` to the
  target ones, giving ``R_sh = [u,A4] [u0,a0]^T``; ``q1..q3`` follow from a
  generalized Euler extraction;
* the elbow ``q4`` is a signed angle in the arm plane;
* the wrist is a standard ZYZ extraction (its outer axes are parallel).

For a 7-DOF arm the solution set is a 1-D self-motion manifold parameterised by
the arm angle ``psi``; ``solve`` samples it, ``solve_at_psi`` evaluates one point.
Every returned solution is exact (no iteration, no polishing) for a true S-R-S
arm such as the KUKA iiwa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .structure import line_intersection


@dataclass
class SrsSolution:
    q: np.ndarray
    pos_err: float
    rot_err: float
    psi: float


def _Rw(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _rot_between(u0, a0, u1, a1):
    """Rotation taking the orthonormal pair (u0,a0) to (u1,a1)."""
    B0 = np.column_stack([u0, a0, np.cross(u0, a0)])
    B1 = np.column_stack([u1, a1, np.cross(u1, a1)])
    return B1 @ B0.T


def _euler_xyz(P):
    """Decompose P = Rx(a) Ry(b) Rz(c) -> up to two (a, b, c) branches.

    The ``/cb`` normalisation matters: without it the second branch uses
    formulas that only hold for the positive-cosine branch and silently returns
    wrong angles.
    """
    out = []
    sb = np.clip(P[0, 2], -1.0, 1.0)
    for b in (np.arcsin(sb), np.pi - np.arcsin(sb)):
        cb = np.cos(b)
        if abs(cb) < 1e-9:
            a, c = 0.0, np.arctan2(-P[0, 1], P[0, 0])
        else:
            a = np.arctan2(-P[1, 2] / cb, P[2, 2] / cb)
            c = np.arctan2(-P[0, 1] / cb, P[0, 0] / cb)
        out.append((a, b, c))
    return out


def _euler_zyz(M):
    """Decompose M = Rz(a) Ry(b) Rz(c) -> up to two (a, b, c) branches."""
    out = []
    cb = np.clip(M[2, 2], -1.0, 1.0)
    for b in (np.arccos(cb), -np.arccos(cb)):
        sb = np.sin(b)
        if abs(sb) < 1e-9:
            a, c = 0.0, np.arctan2(M[1, 0], M[0, 0])
        else:
            a = np.arctan2(M[1, 2] / sb, M[0, 2] / sb)
            c = np.arctan2(M[2, 1] / sb, -M[2, 0] / sb)
        out.append((a, b, c))
    return out


def _rot_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]]) / 2
    return th / (2 * np.sin(th)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class SrsSolver:
    """Arm-angle closed-form IK for an S-R-S arm (KUKA iiwa, Baxter, ...)."""

    def __init__(self, arm, q_ref: np.ndarray | None = None, n_psi: int = 12):
        self.arm = arm
        self.n_psi = n_psi
        if arm.n < 7:
            raise ValueError("SrsSolver expects a 7-DOF arm")
        self.q_ref = self._pick_reference() if q_ref is None else np.asarray(q_ref, float)
        self.arm.set_q(self.q_ref)
        self._measure()

    # -- setup ------------------------------------------------------------- #
    def _pick_reference(self):
        """Find a config whose shoulder axes are mutually orthogonal.

        Joints 1&2 and 2&3 are always perpendicular; we root-find ``w1·w3 = 0``.
        """
        arm = self.arm

        def f(q2):
            q = np.zeros(arm.n)
            q[1] = q2
            arm.set_q(q)
            _, ax = arm.world_lines()
            return float(ax[0] @ ax[2])

        grid = np.linspace(-np.pi, np.pi, 400)
        vals = [f(x) for x in grid]
        for i in range(len(grid) - 1):
            if vals[i] * vals[i + 1] < 0:
                lo, hi = grid[i], grid[i + 1]
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if f(lo) * f(mid) <= 0:
                        hi = mid
                    else:
                        lo = mid
                q = np.zeros(arm.n)
                q[1] = 0.5 * (lo + hi)
                arm.set_q(q)
                _, ax = arm.world_lines()
                G = np.array([ax[0], ax[1], ax[2]]) @ np.array([ax[0], ax[1], ax[2]]).T
                if np.max(np.abs(G - np.eye(3))) < 1e-6:
                    return q
        raise RuntimeError("no orthogonal shoulder reference found")

    def _measure(self):
        arm = self.arm
        a, ax = arm.world_lines()
        self.S = line_intersection(a[0], ax[0], a[1], ax[1])
        d4 = ax[3] / np.linalg.norm(ax[3])
        E0 = a[3] + ((self.S - a[3]) @ d4) * d4
        self.d_se = float(np.linalg.norm(E0 - self.S))
        self.u0 = (E0 - self.S) / self.d_se
        self.a0 = d4
        W0 = line_intersection(a[4], ax[4], a[5], ax[5])
        R0, p0 = arm.tcp_pose()
        self.c = R0.T @ (W0 - p0)          # W = p + R @ c
        self.d_ew = float(np.linalg.norm(W0 - E0))
        self.f0 = (W0 - E0) / self.d_ew
        self.w = ax.copy()
        self.R_ref = R0.copy()

    # -- geometry ---------------------------------------------------------- #
    def _elbow_circle(self, T):
        R_t, p_t = np.asarray(T, float)[:3, :3], np.asarray(T, float)[:3, 3]
        W = p_t + R_t @ self.c
        d = W - self.S
        L = np.linalg.norm(d)
        if not (abs(self.d_se - self.d_ew) - 1e-9 <= L <= self.d_se + self.d_ew + 1e-9):
            return None
        ca = np.clip((self.d_se ** 2 + L ** 2 - self.d_ew ** 2)
                     / (2 * self.d_se * L), -1.0, 1.0)
        alpha = np.arccos(ca)
        u_hat = d / L
        n = self.S + self.d_se * np.cos(alpha) * u_hat
        rho = self.d_se * np.sin(alpha)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(ref @ u_hat) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        h = ref - (ref @ u_hat) * u_hat
        h = h / np.linalg.norm(h)
        v = np.cross(u_hat, h)
        return R_t, p_t, W, n, h, v, rho

    def _candidates(self, geo, psi) -> Iterator[np.ndarray]:
        R_t, p_t, W, n, h, v, rho = geo
        E = n + rho * (np.cos(psi) * h + np.sin(psi) * v)
        u = (E - self.S) / self.d_se
        fe = (W - E) / np.linalg.norm(W - E)
        B_sh = np.column_stack([self.w[0], self.w[1], self.w[2]])
        z = self.w[4] / np.linalg.norm(self.w[4])
        y = self.w[5] / np.linalg.norm(self.w[5])
        B_wr = np.column_stack([np.cross(y, z), y, z])
        for sgn in (1.0, -1.0):
            A4 = sgn * np.cross(u, fe)
            nrm = np.linalg.norm(A4)
            if nrm < 1e-9:
                continue
            A4 = A4 / nrm
            R_sh = _rot_between(self.u0, self.a0, u, A4)
            for (a1, a2, a3) in _euler_xyz(B_sh.T @ R_sh @ B_sh):
                q = self.q_ref.copy()
                q[0] += a1
                q[1] += a2
                q[2] += a3
                g = R_sh.T @ fe
                q[3] = self.q_ref[3] + np.arctan2(np.cross(self.a0, self.f0) @ g,
                                                  self.f0 @ g)
                R04 = (_Rw(self.w[0], q[0] - self.q_ref[0])
                       @ _Rw(self.w[1], q[1] - self.q_ref[1])
                       @ _Rw(self.w[2], q[2] - self.q_ref[2])
                       @ _Rw(self.w[3], q[3] - self.q_ref[3]))
                Wm = R04.T @ R_t @ self.R_ref.T
                for (b1, b2, b3) in _euler_zyz(B_wr.T @ Wm @ B_wr):
                    yield q + np.array([0, 0, 0, 0, b1, b2, b3])

    def _verify(self, q, R_t, p_t, tol_pos, tol_rot):
        q = np.clip(q, self.arm.joint_low, self.arm.joint_high)
        self.arm.set_q(q)
        R, p = self.arm.tcp_pose()
        pe = float(np.linalg.norm(p - p_t))
        re = float(np.linalg.norm(_rot_log(R_t @ R.T)))
        if pe < tol_pos and re < tol_rot:
            return SrsSolution(q, pe, re, 0.0)
        return None

    # -- public API -------------------------------------------------------- #
    def solve_at_psi(self, T, psi, tol_pos=1e-6, tol_rot=1e-6):
        """Exact solutions for one arm angle ``psi`` (list of SrsSolution)."""
        geo = self._elbow_circle(T)
        if geo is None:
            return []
        out = []
        for q in self._candidates(geo, psi):
            sol = self._verify(q, geo[0], geo[1], tol_pos, tol_rot)
            if sol is not None:
                sol.psi = psi
                out.append(sol)
        return out

    def solve(self, T, tol_pos=1e-6, tol_rot=1e-6, max_solutions=16):
        """Sample the self-motion manifold and return exact solutions."""
        geo = self._elbow_circle(T)
        if geo is None:
            return []
        R_t, p_t = geo[0], geo[1]
        seen, good = set(), []
        for psi in np.linspace(0, 2 * np.pi, self.n_psi, endpoint=False):
            for q in self._candidates(geo, psi):
                sol = self._verify(q, R_t, p_t, tol_pos, tol_rot)
                if sol is None:
                    continue
                sol.psi = float(psi)
                key = tuple(np.round(np.mod(sol.q + np.pi, 2 * np.pi) / 1e-3).astype(int))
                if key in seen:
                    continue
                seen.add(key)
                good.append(sol)
        if max_solutions:
            good = good[:max_solutions]
        return good
