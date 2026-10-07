"""Generic MuJoCo-backed robot arm wrapper + registry.

``PandaArm`` in :mod:`ik.model` is a Panda-specific wrapper with S-R-S geometry
calibration.  The numerical solver, however, only needs a handful of methods
(``set_q``, ``tcp_pose``, ``tcp_jacobian``, ``joint_low``/``joint_high``), so it
already works for *any* arm that provides them.  :class:`RobotArm` is that
generic wrapper, letting the same solver drive the UR5e, the KUKA iiwa 14 and
the Panda without duplication.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

MENAGERIE = Path(
    os.environ.get("MUJOCO_MENAGERIE", Path.home() / "mujoco/mujoco_menagerie")
).expanduser()


@dataclass(frozen=True)
class RobotSpec:
    key: str
    label: str
    dof: int
    xml: Path
    joints: tuple[str, ...]
    tcp_body: str
    pieper: str  # short description of how it satisfies Pieper


REGISTRY: dict[str, RobotSpec] = {
    "panda": RobotSpec(
        "panda", "Franka Emika Panda (7R)", 7,
        MENAGERIE / "franka_emika_panda/panda_nohand.xml",
        tuple(f"joint{i}" for i in range(1, 8)), "link7",
        "shoulder spherical, wrist NOT spherical (0.088 m)",
    ),
    "ur5e": RobotSpec(
        "ur5e", "Universal Robots UR5e (6R)", 6,
        MENAGERIE / "universal_robots_ur5e/ur5e.xml",
        ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"), "wrist_3_link",
        "parallel arm axes (2,3,4), wrist NOT spherical (0.10 m)",
    ),
    "iiwa14": RobotSpec(
        "iiwa14", "KUKA LBR iiwa 14 (7R)", 7,
        MENAGERIE / "kuka_iiwa_14/iiwa14.xml",
        tuple(f"joint{i}" for i in range(1, 8)), "link7",
        "spherical shoulder AND wrist (true S-R-S)",
    ),
    "lite6": RobotSpec(
        "lite6", "UFACTORY Lite 6 (6R)", 6,
        MENAGERIE / "ufactory_lite6/lite6.xml",
        tuple(f"joint{i}" for i in range(1, 7)), "link6",
        "spherical WRIST (J4-J6 meet, zero offset), shoulder NOT spherical",
    ),
}


def load_robot(name: str) -> "RobotArm":
    if name not in REGISTRY:
        raise KeyError(f"unknown robot {name!r}; choose from {list(REGISTRY)}")
    return RobotArm(REGISTRY[name])


class RobotArm:
    """Minimal generic wrapper exposing what the numerical IK solver needs."""

    def __init__(self, spec: RobotSpec):
        self.spec = spec
        if not spec.xml.is_file():
            raise FileNotFoundError(f"Robot MJCF not found: {spec.xml}. Set MUJOCO_MENAGERIE.")
        self.model = mujoco.MjModel.from_xml_path(str(spec.xml))
        self.data = mujoco.MjData(self.model)

        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
             for n in spec.joints], dtype=int
        )
        if np.any(self.joint_ids < 0):
            raise RuntimeError(f"missing arm joints for {spec.key}")
        self.tcp_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, spec.tcp_body
        )
        if self.tcp_body_id < 0:
            raise RuntimeError(f"missing TCP body {spec.tcp_body!r}")

        self.dof_adr = self.model.jnt_dofadr[self.joint_ids]
        self.qpos_adr = self.model.jnt_qposadr[self.joint_ids]
        self.joint_low = self.model.jnt_range[self.joint_ids, 0].copy()
        self.joint_high = self.model.jnt_range[self.joint_ids, 1].copy()
        self.n = len(self.joint_ids)

    # -- state ------------------------------------------------------------- #
    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_adr] = np.asarray(q, float)
        mujoco.mj_forward(self.model, self.data)

    def qpos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].copy()

    # -- kinematics -------------------------------------------------------- #
    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        R = self.data.xmat[self.tcp_body_id].reshape(3, 3).copy()
        p = self.data.xpos[self.tcp_body_id].copy()
        return R, p

    def tcp_transform(self) -> np.ndarray:
        R, p = self.tcp_pose()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def fk(self, q: np.ndarray) -> np.ndarray:
        self.set_q(q)
        return self.tcp_transform()

    def tcp_jacobian(self) -> np.ndarray:
        """6xn geometric Jacobian of the TCP frame (rows: lin, ang)."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        point = self.data.xpos[self.tcp_body_id]
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.tcp_body_id)
        return np.vstack([jacp, jacr])[:, self.dof_adr]

    def world_lines(self) -> tuple[np.ndarray, np.ndarray]:
        anchors = np.array([self.data.xanchor[j] for j in self.joint_ids], float)
        axes = np.array([self.data.xaxis[j] for j in self.joint_ids], float)
        return anchors, axes

    def __repr__(self) -> str:
        return f"RobotArm({self.spec.key}, n={self.n})"
