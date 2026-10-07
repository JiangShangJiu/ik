"""Structural analysis of a serial arm against Pieper's criterion.

Pieper (1968): a 6-DOF arm has a closed-form inverse kinematics if either

* three consecutive joint axes intersect at one point, or
* three consecutive joint axes are mutually parallel.

These helpers read the *actual* joint axes out of a (MuJoCo-backed) arm and
report, for every consecutive triple, whether it is a genuine spherical joint,
a parallel bundle, or nothing special.  A triple is "spherical" only if the
three axes are pairwise incident *and* the pairwise intersections coincide at
every sampled configuration -- checking a single axis is not enough, since
three coplanar lines can meet pairwise at the corners of a triangle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOL = 1e-6  # metres


@dataclass
class Triple:
    indices: tuple[int, int, int]   # 0-based joint indices
    kind: str                       # "spherical" | "parallel" | "none"
    world_fixed: bool | None = None  # spherical point fixed in base frame?


def line_line_dist(p1, d1, p2, d2) -> float:
    """Minimum distance between two infinite lines."""
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    n = np.cross(d1, d2)
    if np.linalg.norm(n) < 1e-9:  # parallel
        return float(np.linalg.norm(np.cross(p2 - p1, d1)))
    n = n / np.linalg.norm(n)
    return float(abs((p2 - p1) @ n))


def line_intersection(p1, d1, p2, d2):
    """Closest point on line 1 to line 2 (their meeting point when they meet)."""
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    w0 = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    d, e = d1 @ w0, d2 @ w0
    den = a * c - b * b
    if abs(den) < 1e-12:
        return None
    return p1 + ((b * e - c * d) / den) * d1


def parallel(d1, d2, tol: float = 1e-6) -> bool:
    return np.linalg.norm(np.cross(d1 / np.linalg.norm(d1),
                                   d2 / np.linalg.norm(d2))) < tol


def sample_joint_lines(arm, n: int = 8, seed: int = 0):
    """Return ``[(anchors (k,3), axes (k,3)), ...]`` over random configs."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        q = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q)
        anchors, axes = arm.world_lines()
        samples.append((np.asarray(anchors, float), np.asarray(axes, float)))
    return samples


def classify_triple(samples, i: int, j: int, k: int) -> Triple:
    """Classify the consecutive triple ``(i, j, k)`` of joint indices."""
    if all(parallel(ax[i], ax[j]) and parallel(ax[i], ax[k])
           for _, ax in samples):
        return Triple((i, j, k), "parallel")

    points = []
    for a, ax in samples:
        dij = line_line_dist(a[i], ax[i], a[j], ax[j])
        dik = line_line_dist(a[i], ax[i], a[k], ax[k])
        djk = line_line_dist(a[j], ax[j], a[k], ax[k])
        if max(dij, dik, djk) > TOL:
            return Triple((i, j, k), "none")
        p = line_intersection(a[i], ax[i], a[j], ax[j])
        q = line_intersection(a[i], ax[i], a[k], ax[k])
        if p is None or q is None or np.linalg.norm(p - q) > TOL:
            return Triple((i, j, k), "none")
        points.append(p)

    pts = np.array(points)
    fixed = bool(np.max(np.linalg.norm(pts - pts.mean(axis=0), axis=1)) < TOL)
    return Triple((i, j, k), "spherical", fixed)


def wrist_offset(anchors, axes, last: int = -1) -> float:
    """Distance from the last joint axis to the J(n-2) ∩ J(n-1) centre."""
    w = line_intersection(anchors[last - 2], axes[last - 2],
                          anchors[last - 1], axes[last - 1])
    if w is None:
        return float("nan")
    d = axes[last] / np.linalg.norm(axes[last])
    rel = w - anchors[last]
    return float(np.linalg.norm(rel - (rel @ d) * d))


def analyze_structure(arm, n: int = 8, seed: int = 0) -> dict:
    """Full Pieper report for ``arm`` (needs duck-typed ``set_q``/``world_lines``)."""
    samples = sample_joint_lines(arm, n=n, seed=seed)
    k = samples[0][0].shape[0]
    triples = [classify_triple(samples, t, t + 1, t + 2) for t in range(k - 2)]
    a, ax = samples[0]
    return {
        "n_joints": k,
        "triples": triples,
        "off_wrist_center": wrist_offset(a, ax),
    }
