"""Redundant-arm inverse kinematics for the Franka Panda on MuJoCo."""

from .kinematics import (
    PANDA_MDH,
    N_JOINTS,
    fk_mdh,
    mdh_transform,
    pose_error,
    rot_log,
)
from .model import (
    ARM_JOINT_NAMES,
    DEFAULT_XML,
    TCP_BODY,
    PandaArm,
    PandaGeometry,
    calibrate_geometry,
)
from .numerical import (
    NumericalResult,
    random_seeds,
    solve_numerical,
    solve_numerical_multiseed,
)
from .analytic import (
    AnalyticSolution,
    analytic_seeds,
    elbow_circle,
    shoulder_solve,
    solve_analytic,
    solve_analytic_best,
    wrist_branches,
)
from .redundancy import (
    SecondaryTask,
    damped_pinv,
    limit_avoidance,
    manipulability,
    nullspace_projector,
    nullspace_step,
    posture,
    solve_redundant,
)
from .robot import REGISTRY, RobotArm, RobotSpec, load_robot
from .structure import analyze_structure, classify_triple, wrist_offset
from .closedform import ClosedFormSolution, Lite6Solver, Ur5eSolver
from .srs import SrsSolution, SrsSolver

__all__ = [
    "PANDA_MDH",
    "N_JOINTS",
    "fk_mdh",
    "mdh_transform",
    "pose_error",
    "rot_log",
    "ARM_JOINT_NAMES",
    "DEFAULT_XML",
    "TCP_BODY",
    "PandaArm",
    "PandaGeometry",
    "calibrate_geometry",
    "NumericalResult",
    "random_seeds",
    "solve_numerical",
    "solve_numerical_multiseed",
    "SecondaryTask",
    "damped_pinv",
    "limit_avoidance",
    "manipulability",
    "nullspace_projector",
    "nullspace_step",
    "posture",
    "solve_redundant",
    "AnalyticSolution",
    "analytic_seeds",
    "elbow_circle",
    "shoulder_solve",
    "solve_analytic",
    "solve_analytic_best",
    "wrist_branches",
    "REGISTRY",
    "RobotArm",
    "RobotSpec",
    "load_robot",
    "analyze_structure",
    "classify_triple",
    "wrist_offset",
    "ClosedFormSolution",
    "Lite6Solver",
    "Ur5eSolver",
    "SrsSolution",
    "SrsSolver",
]
