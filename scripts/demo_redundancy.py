"""Phase 2 demo: quantify the benefit of null-space redundancy resolution.

For each sampled reachable target we solve
  * plain DLS (min-norm, redundancy left unused),
  * DLS + joint-limit avoidance,
  * DLS + manipulability maximisation,
and aggregate the joint-limit margin and manipulability of the solutions.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ik import (
    PandaArm,
    limit_avoidance,
    manipulability,
    solve_numerical,
    solve_redundant,
)


def min_limit_margin(arm, q):
    return float(np.min(np.minimum(q - arm.joint_low, arm.joint_high - q)))


def manip(arm, q):
    arm.set_q(q)
    J = arm.tcp_jacobian()
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def sigmin(arm, q):
    arm.set_q(q)
    return float(np.linalg.svd(arm.tcp_jacobian(), compute_uv=False)[-1])


def main() -> None:
    arm = PandaArm()
    rng = np.random.default_rng(10)

    margin_pairs = []
    manip_pairs = []

    for _ in range(300):
        q_true = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q_true)
        T = arm.tcp_transform()
        q0 = q_true + rng.normal(0, 0.3, arm.nq)

        r_plain = solve_numerical(arm, T, q0, method="dls")
        if not r_plain.success:
            continue

        r_limit = solve_redundant(
            arm, T, q0, (limit_avoidance(margin=0.4, weight=1.0),)
        )
        r_manip = solve_redundant(arm, T, q0, (manipulability(weight=1.0),))

        if r_limit.success:
            margin_pairs.append(
                (min_limit_margin(arm, r_plain.q), min_limit_margin(arm, r_limit.q))
            )
        if r_manip.success:
            manip_pairs.append(
                (
                    manip(arm, r_plain.q),
                    manip(arm, r_manip.q),
                    sigmin(arm, r_plain.q),
                )
            )

    m = np.array(margin_pairs)
    near = m[m[:, 0] < 0.5]
    print("Joint-limit avoidance (null space):")
    print(f"  all solutions      : margin {m[:, 0].mean():.3f} -> {m[:, 1].mean():.3f}")
    print(
        f"  near-limit subset  : margin {near[:, 0].mean():.3f} -> "
        f"{near[:, 1].mean():.3f}  (n={len(near)})"
    )

    mm = np.array(manip_pairs)
    sing = mm[mm[:, 2] < 0.15]
    print("\nManipulability maximisation (null space):")
    print(
        f"  near-singular subset: {sing[:, 0].mean():.4f} -> "
        f"{sing[:, 1].mean():.4f}  (n={len(sing)})"
    )
    print(
        "  note: with only a 1-DOF null space the achievable gain is small."
    )


if __name__ == "__main__":
    main()
