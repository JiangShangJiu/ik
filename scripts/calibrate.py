"""Phase 0 demo: load the Panda and print the calibrated S-R-S geometry."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ik import PandaArm  # noqa: E402


def main() -> None:
    arm = PandaArm()
    print(f"Model: {arm.xml_path}")
    print(f"nq={arm.nq}, nv={arm.nv}")
    geom = arm.geometry(n_samples=16, seed=0)
    print("\nCalibrated S-R-S geometry (base frame):")
    print(json.dumps(geom.as_dict(), indent=2))

    print("\nJoint limits (rad):")
    for name, lo, hi in zip(
        ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
        geom.joint_low,
        geom.joint_high,
    ):
        print(f"  {name}: [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
