"""Numerical-IK benchmark across three real arms (UR5e / iiwa14 / Panda).

For each robot we generate reachable target poses and solve them with the two
robust Jacobian methods (pinv, DLS), comparing

* a *near* seed (q_true + noise)     - the "local tracking" regime,
* a single *random* seed             - the "cold start" regime,
* one near seed plus seven random seeds - a warm-start restart comparison.

We report success rate, iterations, wall time and residual error, plus each
arm's Pieper structure, to make the structure / solver relationship concrete.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ik.numerical import solve_numerical, solve_numerical_multiseed  # noqa: E402
from ik.robot import REGISTRY, load_robot  # noqa: E402
from ik.structure import analyze_structure  # noqa: E402


def make_targets(arm, n, rng):
    targets = []
    for _ in range(n):
        q = rng.uniform(arm.joint_low * 0.7, arm.joint_high * 0.7)
        arm.set_q(q)
        targets.append((q, arm.tcp_transform()))
    return targets


def run(arm, targets, rng, method, strategy, sigma=0.2, n_seeds=8):
    ok = 0
    iters, times, perr = [], [], []
    for q_true, T in targets:
        if strategy == "near":
            seeds = [q_true + rng.normal(0, sigma, size=arm.n)]
        elif strategy == "random":
            seeds = [rng.uniform(arm.joint_low, arm.joint_high)]
        else:  # multiseed restart
            seeds = [q_true + rng.normal(0, sigma, size=arm.n)]
            seeds += [rng.uniform(arm.joint_low, arm.joint_high)
                      for _ in range(n_seeds - 1)]

        t0 = time.perf_counter()
        res = solve_numerical_multiseed(arm, T, seeds, method=method, max_iters=300)
        times.append((time.perf_counter() - t0) * 1e3)
        iters.append(res.iters)
        perr.append(res.pos_err)
        ok += int(res.success)

    return {
        "success": ok / len(targets),
        "iters": float(np.median(iters)),
        "ms": float(np.median(times)),
        "perr": float(np.median(perr)),
    }


def sigma_min(arm, targets):
    vals = []
    # Sample the Jacobian at each target's own configuration.
    for q_true, _ in targets:
        arm.set_q(q_true)
        vals.append(np.linalg.svd(arm.tcp_jacobian(), compute_uv=False)[-1])
    return float(np.median(vals)), float(np.min(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    for key, spec in REGISTRY.items():
        arm = load_robot(key)
        targets = make_targets(arm, args.targets, rng)
        st = analyze_structure(arm, n=6)
        triples = "; ".join(
            f"{t.indices}:{t.kind}" for t in st["triples"] if t.kind != "none"
        ) or "none"
        smed, smin = sigma_min(arm, targets)

        print(f"\n{'=' * 74}")
        print(f"{spec.label}")
        print(f"  Pieper structure        : {triples}")
        print(f"  wrist offset            : {st['off_wrist_center']:.4f} m")
        print(f"  sigma_min (median / min): {smed:.4f} / {smin:.2e}")
        print(f"{'=' * 74}")
        print(f"  {'strategy':<22}{'method':<7}{'success':>9}{'iters':>8}{'ms':>8}{'pos_err':>11}")
        print(f"  {'-' * 66}")
        for strategy in ("near", "random", "multiseed(8)"):
            for method in ("pinv", "dls"):
                s = "multiseed" if strategy.startswith("multi") else strategy
                r = run(arm, targets, rng, method, s)
                print(f"  {strategy:<22}{method:<7}{r['success'] * 100:>8.0f}%"
                      f"{r['iters']:>8.0f}{r['ms']:>8.1f}{r['perr']:>11.1e}")


if __name__ == "__main__":
    main()
