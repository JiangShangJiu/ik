"""The animation must show actual IK solutions, including at its loop seam."""
import numpy as np

from ik import load_robot
from ik.kinematics import rot_log
from scripts.showcase.common import choose_target
from scripts.showcase.portfolio import self_motion


def test_self_motion_respects_limits_pose_and_loop_continuity():
    q_ref, T = choose_target("iiwa14", seed=7)
    data = {"robots": {"iiwa14": {"q_ref": q_ref.tolist(), "target_pose": T.tolist()}}}
    motion, _ = self_motion(data)
    qs = np.asarray(motion["qs"])
    arm = load_robot("iiwa14")
    assert motion["psi_interval_rad"][1] > motion["psi_interval_rad"][0]
    assert np.linalg.norm(qs[0] - qs[len(qs) // 2]) > 0.5
    assert np.all(qs >= arm.joint_low - 1e-9)
    assert np.all(qs <= arm.joint_high + 1e-9)
    jumps = np.linalg.norm(np.diff(np.vstack([qs, qs[:1]]), axis=0), axis=1)
    assert jumps.max() < 0.30
    for i, q in enumerate(qs):
        actual = arm.fk(q)
        pe = np.linalg.norm(actual[:3, 3] - T[:3, 3])
        re = np.linalg.norm(rot_log(T[:3, :3] @ actual[:3, :3].T))
        assert pe < 1e-8
        assert re < 1e-8
        assert abs(pe - motion["pos_error_m"][i]) < 1e-12
        assert abs(re - motion["rot_error_rad"][i]) < 1e-12
