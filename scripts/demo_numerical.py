"""Phase 1 demo: compare numerical IK methods on a reachable target pose."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ik import PandaArm, solve_numerical, solve_numerical_multiseed, random_seeds


def main() -> None:
    arm = PandaArm()
    rng = np.random.default_rng(1)

    q_true = rng.uniform(arm.joint_low, arm.joint_high)
    arm.set_q(q_true)
    T = arm.tcp_transform()
    q0 = q_true + rng.normal(0, 0.2, arm.nq)

    print("Target pose (reachable, from a random configuration)")
    print(f"  position   : {np.round(T[:3, 3], 4)}")
    print(f"  seed error : {np.linalg.norm(q0 - q_true):.3f} rad from solution\n")
    print(f"{'method':<6} {'pos_err':>10} {'rot_err':>10} {'iters':>7} {'time_ms':>9}")
    print("-" * 46)

    for method in ("jt", "pinv", "dls"):
        t0 = time.perf_counter()
        res = solve_numerical(arm, T, q0, method=method, max_iters=5000)
        dt = (time.perf_counter() - t0) * 1e3
        print(
            f"{method:<6} {res.pos_err:>10.2e} {res.rot_err:>10.2e} "
            f"{res.iters:>7d} {dt:>9.1f}"
        )

    # Demonstrate multi-seed restart on a harder (near-singular) target.
    print("\nMulti-seed restart from random configurations (16 restarts):")
    q_hard = q_true.copy()
    q_hard[5] = 0.02  # push joint 6 close to its limit -> near-singular
    arm.set_q(q_hard)
    T_hard = arm.tcp_transform()
    seeds = random_seeds(arm, 16, seed=0)
    res1 = solve_numerical(arm, T_hard, seeds[0], method="dls")
    res2 = solve_numerical_multiseed(arm, T_hard, seeds, method="dls")
    print(f"  single seed : success={res1.success} cost={res1.cost():.2e}")
    print(f"  multi-seed  : success={res2.success} cost={res2.cost():.2e}")


if __name__ == "__main__":
    main()
