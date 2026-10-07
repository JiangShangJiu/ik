"""Phase 3 demo: arm-angle semi-analytic IK on the Panda.

Shows that one reachable pose admits several distinct solutions (the S-R-S
branches), all converged to the target pose.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ik import PandaArm, solve_analytic, solve_analytic_best


def main() -> None:
    arm = PandaArm()
    rng = np.random.default_rng(7)

    q_true = rng.uniform(arm.joint_low, arm.joint_high)
    arm.set_q(q_true)
    T = arm.tcp_transform()

    print("Target flange pose")
    print(f"  position: {np.round(T[:3, 3], 4)}")
    print(f"  (generated from q_true = {np.round(q_true, 3)})\n")

    sols = solve_analytic(arm, T)
    print(f"Arm-angle solver found {len(sols)} distinct solutions:")
    print(f"  {'#':>2}  {'pos_err':>10}  {'rot_err':>10}   q")
    for i, s in enumerate(sols):
        print(
            f"  {i:>2}  {s.pos_err:>10.2e}  {s.rot_err:>10.2e}   "
            f"{np.round(s.q, 3)}"
        )

    best = solve_analytic_best(arm, T)
    print(f"\nFast single-solution mode: success={best is not None} "
          f"pos_err={best.pos_err:.2e}" if best else "no solution")


if __name__ == "__main__":
    main()
