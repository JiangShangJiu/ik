"""Auto-classify real arms against Pieper's criterion (MuJoCo ground truth).

For each registered robot we sample random configurations, read every joint's
world-frame axis and test the geometric structure:

* three consecutive axes meeting at one (possibly moving) point -> spherical
* three consecutive axes mutually parallel                      -> parallel

A triple satisfying either is exactly Pieper's condition.  The test lives in
:mod:`ik.structure`; this script just presents it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ik.robot import REGISTRY, load_robot  # noqa: E402
from ik.structure import analyze_structure  # noqa: E402


def main() -> None:
    for key, spec in REGISTRY.items():
        arm = load_robot(key)
        st = analyze_structure(arm, n=8)
        names = list(spec.joints)

        print(f"\n{'=' * 70}")
        print(f"{spec.label}   -- {st['n_joints']} revolute joints")
        print(f"Pieper: {spec.pieper}")
        print(f"{'=' * 70}")
        found = False
        for t in st["triples"]:
            if t.kind == "none":
                continue
            found = True
            i, j, k = t.indices
            label = f"({i + 1},{j + 1},{k + 1})  {names[i][:6]}..{names[k][:6]}"
            if t.kind == "spherical":
                loc = "shoulder" if t.world_fixed else "wrist"
                note = f"spherical {loc} - axes meet at one point"
            else:
                note = "three axes mutually parallel"
            print(f"  {label:<26} {t.kind:<11} {note}")
        if not found:
            print("  (no Pieper triple)")
        off = st["off_wrist_center"]
        kind = "spherical" if off < 1e-6 else "NON-spherical"
        print(f"  last axis vs wrist-centre: {off:.5f} m  -> {kind} wrist")


if __name__ == "__main__":
    main()
