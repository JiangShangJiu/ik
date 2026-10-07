"""Exact closed-form inverse kinematics for two Pieper structures.

Both solvers read every geometric constant directly from a MuJoCo-backed
RobotArm at q = 0 -- no hand-typed link parameters. They return
machine-precision solutions with zero iterations.

* Lite6Solver -- UFACTORY Lite 6: 6R with a spherical wrist (J4-J6 meet in one
  point, zero tool offset). Position/orientation decouple.
* Ur5eSolver -- Universal Robots UR5e: 6R whose axes J2-J4 are mutually
  parallel. The shoulder angle is fixed by the constant lateral offset, which
  reduces the arm to a planar 2-link.

Geometry is built on the product of exponentials
R(q) = e^[w1]q1 ... e^[wn]qn R(0), wi = joint axes in the world frame at q=0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kinematics import rot_log
from .structure import line_intersection


def _skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def _R_axis(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = _skew(a)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def _euler_zyz(M):
    """Decompose M = Rz(a) Ry(b) Rz(c) -> up to two (a, b, c) branches."""
    out = []
    cb = float(np.clip(M[2, 2], -1.0, 1.0))
    for b in (np.arccos(cb), -np.arccos(cb)):
        sb = np.sin(b)
        if abs(sb) < 1e-9:
            a, c = 0.0, float(np.arctan2(M[1, 0], M[0, 0]))
        else:
            a = float(np.arctan2(M[1, 2] / sb, M[0, 2] / sb))
            c = float(np.arctan2(M[2, 1] / sb, -M[2, 0] / sb))
        out.append((a, b, c))
    return out


def _wrist_zyz(M, w4, w5, w6):
    """Exact wrist extraction for a ZYZ joint set given the *measured* axes.

    Returns the two ``(q4, q5, q6)`` branches of
    ``M = R_{w4}(q4) R_{w5}(q5) R_{w6}(q6)`` where ``w4`` and ``w6`` are
    collinear and ``w5`` is perpendicular to them.
    """
    o = np.asarray(w4, float)
    o = o / np.linalg.norm(o)
    m = np.asarray(w5, float)
    m = m - (m @ o) * o
    m = m / np.linalg.norm(m)
    B = np.column_stack([np.cross(m, o), m, o])
    Mt = B.T @ np.asarray(M, float) @ B
    return _euler_zyz(Mt)


@dataclass
class ClosedFormSolution:
    q: np.ndarray
    pos_err: float
    rot_err: float
    branch: str = ""

    @property
    def success(self):
        return True


class _PlanarPieperSolver:
    def __init__(self, arm):
        self.arm = arm
        self.n = arm.n
        arm.set_q(np.zeros(self.n))
        self.a, self.w_raw = arm.world_lines()
        self.w = self.w_raw / np.linalg.norm(self.w_raw, axis=1, keepdims=True)
        R0, p0 = arm.tcp_pose()
        self.R0 = R0.copy()
        self.p0 = p0.copy()
        W0 = line_intersection(self.a[-2], self.w[-2], self.a[-1], self.w[-1])
        self.o_w = self.R0.T @ (self.p0 - W0)
        self.wrist0 = W0
        self.normal = self.w[1] / np.linalg.norm(self.w[1])
        self.S = line_intersection(self.a[0], self.w[0], self.a[1], self.w[1])
        self.c1 = self.w[0] / np.linalg.norm(self.w[0])
        self.P0 = self.c1
        self.P1 = np.cross(self.normal, self.P0)
        self.plane_origin = self.S

    # -- helpers ------------------------------------------------------------ #
    def _R1(self, q1):
        return _R_axis(self.c1, q1)

    def _planar(self, v, q1):
        r = self._R1(q1)
        vv = v - self.plane_origin
        return float(vv @ (r @ self.P1)), float(vv @ (r @ self.P0))

    def _angle_of(self, v):
        return float(np.arctan2(v @ self.P1, v @ self.P0))

    @staticmethod
    def _u(theta):
        return np.array([np.sin(theta), np.cos(theta)])

    def _wrist_center(self, T):
        R, p = np.asarray(T, float)[:3, :3], np.asarray(T, float)[:3, 3]
        return p - R @ self.o_w

    def _shoulder_angles(self, W):
        cxn = np.cross(self.c1, self.normal)
        v = W - self.S
        A = float(v @ cxn)
        B = float(v @ self.normal)
        C = float((self.wrist0 - self.S) @ self.normal)
        R = np.hypot(A, B)
        if R < 1e-12:
            return [] if abs(C) > 1e-9 else [0.0]
        if abs(C) > R + 1e-9:
            return []
        s = float(np.arcsin(np.clip(C / R, -1.0, 1.0)))
        psi = float(np.arctan2(B, A))
        return [s - psi, np.pi - s - psi]

    def _two_link(self, d, l1, l2):
        D = float(np.linalg.norm(d))
        if D < 1e-12 or not (abs(l1 - l2) - 1e-9 <= D <= l1 + l2 + 1e-9):
            return []
        cden = (D ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
        if abs(cden) > 1 + 1e-9:
            return []
        delta0 = float(np.arccos(np.clip(cden, -1.0, 1.0)))
        theta_d = float(np.arctan2(d[0], d[1]))
        out = []
        for delta in (delta0, -delta0):
            phi1 = theta_d - np.arctan2(l2 * np.sin(delta), l1 + l2 * np.cos(delta))
            out.append((delta, phi1, phi1 + delta))
        return out

    def _wrap_to_limits(self, q):
        """Add multiples of 2*pi per joint so it lands inside the limits."""
        q = np.array(q, float)
        for i in range(len(q)):
            low, high = self.arm.joint_low[i], self.arm.joint_high[i]
            a = q[i]
            kmin = int(np.ceil((low - a) / (2 * np.pi) - 1e-9))
            kmax = int(np.floor((high - a) / (2 * np.pi) + 1e-9))
            if kmin > kmax:
                return None
            mid = 0.5 * (low + high)
            k = int(np.clip(round((mid - a) / (2 * np.pi)), kmin, kmax))
            q[i] = a + 2 * np.pi * k
        return q

    def _verify(self, q, T, tol_pos=1e-6, tol_rot=1e-6, branch=""):
        q = self._wrap_to_limits(np.asarray(q, float))
        if q is None:
            return None
        if np.any(q < self.arm.joint_low - 1e-9) or np.any(q > self.arm.joint_high + 1e-9):
            return None
        R_t, p_t = np.asarray(T, float)[:3, :3], np.asarray(T, float)[:3, 3]
        self.arm.set_q(q)
        R, p = self.arm.tcp_pose()
        pe = float(np.linalg.norm(p - p_t))
        re = float(np.linalg.norm(rot_log(R_t @ R.T)))
        if pe < tol_pos and re < tol_rot:
            return ClosedFormSolution(q.copy(), pe, re, branch)
        return None

    @staticmethod
    def _dedupe(sols):
        out = []
        for s in sols:
            if any(np.linalg.norm(np.mod(s.q - o.q + np.pi, 2 * np.pi) - np.pi) < 1e-5
                   for o in out):
                continue
            out.append(s)
        return out


class Lite6Solver(_PlanarPieperSolver):
    """Closed-form IK for a 6R with a spherical wrist (Pieper condition 1)."""

    def __init__(self, arm):
        super().__init__(arm)
        assert abs(abs(self.w[2] @ self.normal) - 1.0) < 1e-6, "J2,J3 not parallel"
        assert abs(self.c1 @ self.normal) < 1e-6, "J1 not perpendicular to J2"
        W = line_intersection(self.a[3], self.w[3], self.a[4], self.w[4])
        w6 = self.w[5]
        gap = float(np.linalg.norm(np.cross(W - self.a[5], w6)))
        assert gap < 1e-6, "wrist not spherical"
        self.s2 = 1.0 if self.w[1] @ self.normal > 0 else -1.0
        self.s3 = 1.0 if self.w[2] @ self.normal > 0 else -1.0
        self.theta1 = self._angle_of(self.a[2] - self.S)
        f0 = W - self.a[2]
        self.theta_f = self._angle_of(f0)
        self.l1 = float(np.linalg.norm(self.a[2] - self.S))
        self.l2 = float(np.linalg.norm(f0))

    def solve(self, T, tol_pos=1e-6, tol_rot=1e-6):
        T = np.asarray(T, float)
        R_d = T[:3, :3]
        W = self._wrist_center(T)
        sols = []
        for q1 in self._shoulder_angles(W):
            X, Z = self._planar(W, q1)
            for (delta, phi1, phi2) in self._two_link(np.array([X, Z]), self.l1, self.l2):
                q2 = (phi1 - self.theta1) / self.s2
                q3 = (phi2 - self.theta_f - self.s2 * q2) / self.s3
                A = (_R_axis(self.w[0], q1) @ _R_axis(self.w[1], q2)
                     @ _R_axis(self.w[2], q3))
                M = A.T @ R_d @ self.R0.T
                # M = R_{w4}(q4) R_{w5}(q5) R_{w6}(q6)
                for (q4, q5, q6) in _wrist_zyz(M, self.w[3], self.w[4], self.w[5]):
                    s = self._verify(np.array([q1, q2, q3, q4, q5, q6]),
                                     T, tol_pos, tol_rot, "spherical wrist")
                    if s is not None:
                        sols.append(s)
        return self._dedupe(sols)


class Ur5eSolver(_PlanarPieperSolver):
    """Closed-form IK for a 6R whose axes J2-J4 are parallel (condition 2)."""

    def __init__(self, arm):
        super().__init__(arm)
        for j in (1, 2, 3):
            assert abs(abs(self.w[j] @ self.normal) - 1.0) < 1e-6, "J2..J4 not parallel"
        assert abs(self.c1 @ self.normal) < 1e-6, "J1 not perpendicular to J2"
        self.s2 = 1.0 if self.w[1] @ self.normal > 0 else -1.0
        self.s3 = 1.0 if self.w[2] @ self.normal > 0 else -1.0
        self.s4 = 1.0 if self.w[3] @ self.normal > 0 else -1.0
        B = self.a[1]
        self.B2 = np.array(self._planar(B, 0.0))
        self.theta1 = self._angle_of(self.a[2] - B)
        self.l1 = self._planar_len(self.a[2] - B)
        self.theta2 = self._angle_of(self.a[3] - self.a[2])
        self.l2 = self._planar_len(self.a[3] - self.a[2])
        W0 = self.wrist0
        self.theta3 = self._angle_of(W0 - self.a[3])
        self.l3 = self._planar_len(W0 - self.a[3])
        self.w5 = self.w[4].copy()
        self.w6 = self.w[5].copy()
        self.axis6_tcp = self.R0.T @ self.w6

    def _theta_from_axis(self, q1, R_d):
        d6 = np.asarray(R_d, float) @ self.axis6_tcp
        h = self._R1(q1).T @ d6
        cxn = np.cross(self.normal, self.w5)
        A = float(self.w5 @ h)
        B = float(cxn @ h)
        if abs(A) < 1e-12 and abs(B) < 1e-12:
            return []
        base = float(np.arctan2(-A, B))
        return [base, base + np.pi]

    def _planar_len(self, v):
        v = v - (v @ self.normal) * self.normal
        return float(np.linalg.norm(v))

    def solve(self, T, tol_pos=1e-6, tol_rot=1e-6):
        T = np.asarray(T, float)
        R_d = T[:3, :3]
        W = self._wrist_center(T)
        sols = []
        for q1 in self._shoulder_angles(W):
            target = np.array(self._planar(W, q1))
            for Theta in self._theta_from_axis(q1, R_d):
                off = self.l3 * self._u(self.theta3 + self.s4 * Theta)
                d = target - self.B2 - off
                for (delta, phi1, phi2) in self._two_link(d, self.l1, self.l2):
                    q2 = (phi1 - self.theta1) / self.s2
                    q3 = (phi2 - self.theta2 - self.s2 * q2) / self.s3
                    q4 = (Theta - self.s2 * q2 - self.s3 * q3) / self.s4
                    A = (_R_axis(self.w[0], q1) @ _R_axis(self.w[1], q2)
                         @ _R_axis(self.w[2], q3))
                    M = A.T @ R_d @ self.R0.T
                    for (q4w, q5, q6) in _wrist_zyz(M, self.w[3], self.w[4], self.w[5]):
                        s = self._verify(np.array([q1, q2, q3, q4w, q5, q6]),
                                         T, tol_pos, tol_rot, "parallel axes")
                        if s is not None:
                            sols.append(s)
        return self._dedupe(sols)
