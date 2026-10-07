"""Panda arm model loading and geometric calibration.

The Panda is loaded from MuJoCo Menagerie (``panda_nohand.xml``: 7-DOF arm
without the gripper).  ``PandaArm`` wraps the raw ``MjModel``/``MjData`` and
exposes the quantities the IK solvers need.

``calibrate_geometry`` measures the S-R-S structure directly from the model
so that no geometric constant is hard-coded by hand:

* ``shoulder_center`` S : common intersection of axes 1, 2, 3
* ``d_se``              : |S -> elbow-foot on joint-4 axis|
* ``d_ew``              : |elbow-foot -> wrist center W|, W = J5 ∩ J6
* ``wrist_offset``      : |W -> link7 origin| (the non-spherical wrist offset)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mujoco

from .kinematics import (
    closest_points_on_lines,
    foot_of_perpendicular,
    line_intersection,
    point_line_distance,
)

DEFAULT_XML = Path.home() / "mujoco/mujoco_menagerie/franka_emika_panda/panda_nohand.xml"
ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
TCP_BODY = "link7"  # our IK target frame (flange / DH frame 7)
ELBOW_JOINT = "joint4"
SHOULDER_JOINTS = ("joint1", "joint2", "joint3")
WRIST_JOINTS = ("joint5", "joint6")


def resolve_xml(xml_path: str | os.PathLike | None = None) -> Path:
    if xml_path is None:
        env = os.environ.get("PANDA_XML")
        menagerie = os.environ.get("MUJOCO_MENAGERIE")
        if env:
            xml_path = Path(env)
        elif menagerie:
            xml_path = Path(menagerie) / "franka_emika_panda/panda_nohand.xml"
        else:
            xml_path = DEFAULT_XML
    xml_path = Path(xml_path).expanduser()
    if not xml_path.exists():
        raise FileNotFoundError(
            f"Panda MJCF not found: {xml_path}. Set MUJOCO_MENAGERIE or PANDA_XML, or pass xml_path."
        )
    return xml_path


@dataclass(frozen=True)
class PandaGeometry:
    """Calibrated S-R-S geometry of the arm (metres, base frame)."""

    shoulder_center: np.ndarray
    d_se: float
    d_ew: float
    wrist_offset: float
    joint_low: np.ndarray
    joint_high: np.ndarray

    def as_dict(self) -> dict:
        return {
            "shoulder_center": self.shoulder_center.tolist(),
            "d_se": self.d_se,
            "d_ew": self.d_ew,
            "wrist_offset": self.wrist_offset,
            "joint_low": self.joint_low.tolist(),
            "joint_high": self.joint_high.tolist(),
        }


class PandaArm:
    """Thin wrapper around the MuJoCo Panda arm model."""

    def __init__(self, xml_path: str | os.PathLike | None = None):
        self.xml_path = resolve_xml(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.nq = self.model.nq
        self.nv = self.model.nv

        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                for n in ARM_JOINT_NAMES
            ],
            dtype=int,
        )
        if np.any(self.joint_ids < 0):
            raise RuntimeError("Could not find all arm joints in the model.")

        self.tcp_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, TCP_BODY
        )
        self.elbow_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, ELBOW_JOINT
        )
        self.shoulder_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in SHOULDER_JOINTS
        ]
        self.wrist_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in WRIST_JOINTS
        ]

        self.joint_low = self.model.jnt_range[self.joint_ids, 0].copy()
        self.joint_high = self.model.jnt_range[self.joint_ids, 1].copy()
        self._geometry: PandaGeometry | None = None

    # -- forward kinematics ------------------------------------------------- #
    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[:] = np.asarray(q, dtype=float)
        mujoco.mj_forward(self.model, self.data)

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(R, p)`` of the flange (``link7``) frame in world frame."""
        R = self.data.xmat[self.tcp_body_id].reshape(3, 3).copy()
        p = self.data.xpos[self.tcp_body_id].copy()
        return R, p

    def tcp_transform(self) -> np.ndarray:
        R, p = self.tcp_pose()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def tcp_jacobian(self) -> np.ndarray:
        """6xN geometric Jacobian of the TCP (flange) frame.

        Rows 0-2 are the linear part, rows 3-5 the angular part. Requires
        ``set_q``/``mj_forward`` to have been called first.
        """
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        point = self.data.xpos[self.tcp_body_id]
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.tcp_body_id)
        return np.vstack([jacp, jacr])

    def world_lines(self) -> tuple[np.ndarray, np.ndarray]:
        """World-frame joint anchor points and unit axes, shape (7, 3)."""
        anchors = np.array(
            [self.data.xanchor[j] for j in self.joint_ids], dtype=float
        )
        axes = np.array([self.data.xaxis[j] for j in self.joint_ids], dtype=float)
        return anchors, axes

    # -- calibration -------------------------------------------------------- #
    def geometry(self, n_samples: int = 8, seed: int = 0) -> PandaGeometry:
        if self._geometry is None:
            self._geometry = calibrate_geometry(self, n_samples=n_samples, seed=seed)
        return self._geometry

    def wrist_center(self) -> np.ndarray:
        """Wrist center W = intersection of joint-5 and joint-6 axes."""
        anchors, axes = self.world_lines()
        return line_intersection(
            anchors[4], axes[4], anchors[5], axes[5]
        )


def calibrate_geometry(arm: PandaArm, n_samples: int = 8, seed: int = 0) -> PandaGeometry:
    """Measure the S-R-S geometry and assert it is configuration-independent."""
    rng = np.random.default_rng(seed)
    shoulder_points = []
    d_se_list = []
    d_ew_list = []
    wrist_off_list = []
    tol = 1e-9

    for _ in range(max(1, n_samples)):
        q = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q)
        a, ax = arm.world_lines()

        # Shoulder center: intersection of axes 1 and 2 (both contain S).
        S = line_intersection(a[0], ax[0], a[1], ax[1])
        # Joint 3 axis must also pass through S.
        assert point_line_distance(a[2], ax[2], S) < tol, "shoulder not spherical"
        shoulder_points.append(S)

        # Elbow foot: perpendicular foot from S onto the joint-4 axis.
        E = foot_of_perpendicular(S, a[3], ax[3])
        d_se = float(np.linalg.norm(E - S))

        # Wrist center W = J5 ∩ J6.
        W = line_intersection(a[4], ax[4], a[5], ax[5])
        # E must also be the perpendicular foot from W onto joint-4 axis.
        E_w = foot_of_perpendicular(W, a[3], ax[3])
        d_ew = float(np.linalg.norm(W - E_w))

        wrist_off = float(np.linalg.norm(W - arm.data.xpos[arm.tcp_body_id]))

        d_se_list.append(d_se)
        d_ew_list.append(d_ew)
        wrist_off_list.append(wrist_off)

    shoulder_center = np.mean(shoulder_points, axis=0)
    if np.max(np.linalg.norm(np.array(shoulder_points) - shoulder_center, axis=1)) > tol:
        raise AssertionError("shoulder center is not constant across configs")
    if np.ptp(d_se_list) > tol:
        raise AssertionError("d_se is not constant")
    if np.ptp(d_ew_list) > tol:
        raise AssertionError("d_ew is not constant")
    if np.ptp(wrist_off_list) > tol:
        raise AssertionError("wrist offset is not constant")

    return PandaGeometry(
        shoulder_center=shoulder_center,
        d_se=float(np.mean(d_se_list)),
        d_ew=float(np.mean(d_ew_list)),
        wrist_offset=float(np.mean(wrist_off_list)),
        joint_low=arm.joint_low.copy(),
        joint_high=arm.joint_high.copy(),
    )
