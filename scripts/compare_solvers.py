"""Head-to-head comparison of the three Pieper stories, all in MuJoCo.

For each arm we solve the *same kind* of random reachable targets and report how
much numerical work the inverse kinematics costs:

* ``ur5e``   -- 6R, Pieper **parallel axes** (J2,J3,J4). This experiment uses
  DLS as its numerical baseline; the package also has a separate closed-form
  solver, which is not included in this historical comparison.
* ``iiwa14`` -- 7R, true **S-R-S**.  The arm-angle solution is exact: the
  ``SrsSolver`` closed form needs **zero** iterations, while the generic DLS
  solver still iterates.
* ``panda``  -- 7R, spherical shoulder but a 0.088 m wrist offset. This
  implementation refines geometric seeds numerically, giving a semi-analytic
  hybrid; that choice does not rule out other analytical Panda methods.

Run ``python scripts/compare_solvers.py`` to print a markdown table. ``--save``
writes ``build/solver_bench.md`` without overwriting the curated research notes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ik import (  # noqa: E402
    PandaArm,
    SrsSolver,
    load_robot,
    solve_analytic_best,
    solve_numerical,
)

N_TARGETS = 30
SEED_NOISE = 0.05  # rad; "near seed" used by the numerical baselines


def _targets(arm, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q = rng.uniform(arm.joint_low * 0.8, arm.joint_high * 0.8)
        arm.set_q(q)
        out.append((q, arm.tcp_transform()))
    return out


def bench_numerical(arm, targets, seed):
    rng = np.random.default_rng(seed + 100)
    iters, errs, times = [], [], []
    for q_true, T in targets:
        q0 = q_true + rng.normal(0.0, SEED_NOISE, len(q_true))
        t0 = time.perf_counter()
        res = solve_numerical(arm, T, q0, method="dls")
        times.append((time.perf_counter() - t0) * 1e3)
        iters.append(res.iters)
        errs.append(res.pos_err)
    return dict(iters=np.mean(iters), pos_err=max(errs), time=np.mean(times))


def bench_srs(arm, targets):
    solver = SrsSolver(arm, n_psi=12)
    iters, errs, times = [], [], []
    for q_true, T in targets:
        t0 = time.perf_counter()
        sols = solver.solve(T, max_solutions=16)
        times.append((time.perf_counter() - t0) * 1e3)
        iters.append(0)                       # closed form: no iterations
        errs.append(min(s.pos_err for s in sols))
    return dict(iters=np.mean(iters), pos_err=max(errs), time=np.mean(times))


def bench_analytic(arm, targets):
    errs, times = [], []
    for q_true, T in targets:
        t0 = time.perf_counter()
        best = solve_analytic_best(arm, T)
        times.append((time.perf_counter() - t0) * 1e3)
        errs.append(best.pos_err if best is not None else np.inf)
    return dict(iters=-1, pos_err=max(errs), time=np.mean(times))


def fmt_iters(v):
    return "0 (closed form)" if v == 0 else ("seeded polish" if v < 0 else f"{v:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="write build/solver_bench.md")
    args = ap.parse_args()

    ur5e = load_robot("ur5e")
    iiwa = load_robot("iiwa14")
    # The Panda analytic seed path needs the calibrated ``PandaArm`` geometry.
    panda = PandaArm()

    rows = [
        ("UR5e", 6, "parallel axes (J2,J3,J4)", "DLS (iterative)",
         bench_numerical(ur5e, _targets(ur5e, N_TARGETS, 1), 1)),
        ("iiwa 14", 7, "S-R-S (spherical shoulder+wrist)", "DLS (iterative)",
         bench_numerical(iiwa, _targets(iiwa, N_TARGETS, 2), 2)),
        ("iiwa 14", 7, "S-R-S (spherical shoulder+wrist)", "arm-angle closed form",
         bench_srs(iiwa, _targets(iiwa, N_TARGETS, 2))),
        ("Panda", 7, "shoulder spherical, wrist offset", "DLS (iterative)",
         bench_numerical(panda, _targets(panda, N_TARGETS, 3), 3)),
        ("Panda", 7, "shoulder spherical, wrist offset", "analytic seed + polish",
         bench_analytic(panda, _targets(panda, N_TARGETS, 3))),
    ]

    header = ("| Robot | DOF | Pieper structure | Method | Iterations | Max pos err | "
              "Mean time |")
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for name, dof, pieper, method, r in rows:
        lines.append(f"| {name} | {dof} | {pieper} | {method} | {fmt_iters(r['iters'])} | "
                     f"{r['pos_err']:.2e} m | {r['time']:.2f} ms |")
    table = "\n".join(lines)
    print(table)

    if args.save:
        output = Path(__file__).resolve().parents[1] / "build" / "solver_bench.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table + "\n", encoding="utf-8")
        print(f"\nsaved benchmark table to {output}")


if __name__ == "__main__":
    main()
