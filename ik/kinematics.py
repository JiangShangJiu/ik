"""Forward kinematics + geometric helpers for the Franka Panda arm.

The Panda is a 7R chain with all consecutive twist angles equal to 90 deg.
The modified-DH table below reproduces MuJoCo's ``link7`` pose exactly
(verified to machine precision), and ``link7`` origin is our IK target frame
(TCP = flange frame).
"""

from __future__ import annotations

import numpy as np

# Modified DH (Craig): (alpha_{i-1}, a_{i-1}, d_i), joint angle q_i is the
# revolute variable. Row i corresponds to joint i+1.
PANDA_MDH = np.array(
    [
        [0.0, 0.0, 0.333],        # joint 1
        [-np.pi / 2, 0.0, 0.0],   # joint 2
        [np.pi / 2, 0.0, 0.316],  # joint 3
        [np.pi / 2, 0.0825, 0.0], # joint 4
        [-np.pi / 2, -0.0825, 0.384],  # joint 5
        [np.pi / 2, 0.0, 0.0],    # joint 6
        [np.pi / 2, 0.088, 0.0],  # joint 7
    ],
    dtype=float,
)

N_JOINTS = 7


def mdh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Single modified-DH link transform (Craig convention)."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -sa * d],
            [st * sa, ct * sa, ca, ca * d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def fk_mdh(q: np.ndarray) -> np.ndarray:
    """Forward kinematics of the arm chain (7 joints) -> 4x4 flange pose."""
    q = np.asarray(q, dtype=float)
    T = np.eye(4)
    for i in range(N_JOINTS):
        alpha, a, d = PANDA_MDH[i]
        T = T @ mdh_transform(alpha, a, d, q[i])
    return T


# --------------------------------------------------------------------------- #
# SO(3) / pose utilities
# --------------------------------------------------------------------------- #
def rot_log(R: np.ndarray) -> np.ndarray:
    """Logarithm map of SO(3) -> rotation vector (robust near pi)."""
    R = np.asarray(R, dtype=float)
    cos_t = (np.trace(R) - 1.0) / 2.0
    cos_t = min(1.0, max(-1.0, cos_t))
    theta = np.arccos(cos_t)
    if theta < 1e-8:
        return np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
        ) / 2.0
    if np.pi - theta < 1e-6:
        # Near pi: use the symmetric part.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        # Recover signs from off-diagonal.
        k = int(np.argmax(axis))
        if axis[k] > 0:
            axis = axis * np.sign(A[k] + 1e-12)
        if np.linalg.norm(axis) < 1e-9:
            axis = np.array([1.0, 0.0, 0.0])
        return theta * axis / np.linalg.norm(axis)
    return theta / (2.0 * np.sin(theta)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )


def pose_error(
    R_cur: np.ndarray,
    p_cur: np.ndarray,
    R_tgt: np.ndarray,
    p_tgt: np.ndarray,
) -> np.ndarray:
    """6-vector task error: [position error ; orientation error (rotvec)].

    Orientation error uses ``log(R_tgt @ R_cur^T)`` so that a small change
    ``R_cur <- exp(dw) R_cur`` moves the error by ``dw``.
    """
    e_p = np.asarray(p_tgt) - np.asarray(p_cur)
    e_r = rot_log(np.asarray(R_tgt) @ np.asarray(R_cur).T)
    return np.concatenate([e_p, e_r])


# --------------------------------------------------------------------------- #
# Line geometry helpers (used for structural calibration)
# --------------------------------------------------------------------------- #
def point_line_distance(P: np.ndarray, A: np.ndarray, X: np.ndarray) -> float:
    """Distance from point X to the line through P with unit direction A."""
    A = np.asarray(A, dtype=float)
    A = A / np.linalg.norm(A)
    v = np.asarray(X, dtype=float) - np.asarray(P, dtype=float)
    return float(np.linalg.norm(v - np.dot(v, A) * A))


def closest_points_on_lines(Pi, Ai, Pk, Ak):
    """Closest points between two (possibly skew) infinite lines.

    Returns ``(point_i, point_k, gap)``. For intersecting lines ``gap ~ 0``
    and both points coincide with the intersection.
    """
    Pi, Pk = np.asarray(Pi, float), np.asarray(Pk, float)
    Ai = np.asarray(Ai, float)
    Ai = Ai / np.linalg.norm(Ai)
    Ak = np.asarray(Ak, float)
    Ak = Ak / np.linalg.norm(Ak)
    a = Ai @ Ai
    b = Ai @ Ak
    c = Ak @ Ak
    w0 = Pi - Pk
    d = Ai @ w0
    e = Ak @ w0
    den = a * c - b * b
    if abs(den) < 1e-12:  # parallel
        t = -d / a
        s = 0.0
    else:
        t = (b * e - c * d) / den
        s = (a * e - b * d) / den
    pi = Pi + t * Ai
    pk = Pk + s * Ak
    return pi, pk, float(np.linalg.norm(pi - pk))


def line_intersection(Pi, Ai, Pk, Ak) -> np.ndarray:
    """Midpoint of the closest points (best estimate of the intersection)."""
    pi, pk, _ = closest_points_on_lines(Pi, Ai, Pk, Ak)
    return 0.5 * (pi + pk)


def foot_of_perpendicular(P, a_point, a_dir) -> np.ndarray:
    """Foot of the perpendicular from point P onto the given line."""
    a_point = np.asarray(a_point, float)
    a_dir = np.asarray(a_dir, float)
    a_dir = a_dir / np.linalg.norm(a_dir)
    v = np.asarray(P, float) - a_point
    return a_point + np.dot(v, a_dir) * a_dir
