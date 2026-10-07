"""Shared helpers for the showcase: geometry roles, target picking, metrics."""

from __future__ import annotations

import numpy as np
import mujoco

from ik import PandaArm, analyze_structure, load_robot
from ik.structure import line_intersection

# Light studio backdrop for every offscreen render.  The menagerie robot XMLs
# carry no skybox (only the sibling ``scene.xml`` does), so offscreen MuJoCo
# would otherwise clear to plain black; we inject a soft gradient skybox.
SKY_RGB1 = (0.97, 0.98, 1.00)   # zenith
SKY_RGB2 = (0.74, 0.79, 0.86)   # horizon

_SKYBOX_CACHE: dict = {}


def skybox_model(xml_path):
    """Compile ``xml_path`` with a light gradient skybox bolted on top.

    Everything else (bodies, joints, geoms, ids, ranges) is byte-identical, so
    the returned model is a drop-in replacement for the plain one.
    """
    key = str(xml_path)
    if key not in _SKYBOX_CACHE:
        spec = mujoco.MjSpec.from_file(key)
        tex = spec.add_texture()
        tex.name = "showcase_skybox"
        tex.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
        tex.rgb1[:] = SKY_RGB1
        tex.rgb2[:] = SKY_RGB2
        tex.width = 512
        tex.height = 3072
        _SKYBOX_CACHE[key] = spec.compile()
    return _SKYBOX_CACHE[key]


def _xml_path(arm):
    if hasattr(arm, "xml_path"):
        return arm.xml_path
    return arm.spec.xml


def _install_skybox(arm):
    arm.model = skybox_model(_xml_path(arm))
    arm.data = mujoco.MjData(arm.model)
    return arm


def arm_frames(arm):
    """Current joint anchors/axes plus shoulder ``S`` and wrist ``W`` centres."""
    anchors, axes = arm.world_lines()
    n = len(axes)
    S = line_intersection(anchors[0], axes[0], anchors[1], axes[1])
    W = line_intersection(anchors[n - 3], axes[n - 3], anchors[n - 2], axes[n - 2])
    return anchors, axes, S, W


def elbow_center(anchors, axes, S, idx=3):
    """Centre of joint ``idx`` (the elbow for a 7-DOF arm)."""
    d = axes[idx] / np.linalg.norm(axes[idx])
    return anchors[idx] + ((np.asarray(S) - anchors[idx]) @ d) * d


def get_arm(key):
    """Panda uses the dedicated ``PandaArm`` (needed by the analytic solver)."""
    arm = PandaArm() if key == "panda" else load_robot(key)
    return _install_skybox(arm)

# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #
def roles_for(arm, n: int = 4) -> dict:
    """Map 0-based joint index -> 'spherical' | 'parallel' (others stay unset)."""
    st = analyze_structure(arm, n=n)
    roles: dict = {}
    for t in st["triples"]:
        if t.kind == "spherical":
            for i in t.indices:
                roles[i] = "spherical"
        elif t.kind == "parallel":
            for i in t.indices:
                roles.setdefault(i, "parallel")
    return roles


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def sigma_min(arm, q):
    arm.set_q(np.asarray(q, float))
    J = arm.tcp_jacobian()
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def max_limit_hit(arm, q, margin=1e-3):
    return bool(np.any(q < arm.joint_low + margin) or np.any(q > arm.joint_high - margin))


# --------------------------------------------------------------------------- #
# targets -- one per robot, far from singular, loose limits
# --------------------------------------------------------------------------- #
def sample_cfg(arm, rng, margin=0.25):
    span = arm.joint_high - arm.joint_low
    lo = arm.joint_low + margin * span
    hi = arm.joint_high - margin * span
    return rng.uniform(lo, hi)


def choose_target(key, seed=7, tries=300):
    """Return (q_ref, T) with a well-conditioned, easily reachable target.

    Only manipulability is scored while sampling (cheap); the solver runs once
    on the winner to record how many branches actually exist for it.
    """
    arm = get_arm(key)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(tries):
        q = sample_cfg(arm, rng)
        s = sigma_min(arm, q)
        if best is None or s > best[1]:
            best = (q, s)
    q_ref = best[0]
    T = _fk(arm, q_ref)
    return q_ref, T


def _fk(arm, q):
    arm.set_q(np.asarray(q, float))
    return arm.tcp_transform()


def all_targets(seed=7):
    return {k: choose_target(k, seed=seed) for k in ("ur5e", "lite6", "iiwa14", "panda")}
