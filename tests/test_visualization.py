"""Visualization smoke test: MuJoCo scene for visualizing IK solutions."""

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ik import PandaArm, solve_analytic  # noqa: E402


@pytest.fixture(scope="module")
def viz():
    try:
        from visualize_ik import Scene, render_frames, DEFAULT_Q
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"visualization import failed: {exc}")
    return Scene, render_frames, DEFAULT_Q


def test_scene_loads_with_target_marker(viz):
    Scene, _, _ = viz
    scene = Scene()
    assert scene.model.nq == 7          # arm only (nohand model)
    assert scene.model.nmocap >= 1      # target marker present


def test_render_solutions(viz):
    Scene, render_frames, DEFAULT_Q = viz
    arm = PandaArm()
    arm.set_q(DEFAULT_Q)
    T = arm.tcp_transform()
    sols = solve_analytic(arm, T, max_solutions=4)
    assert len(sols) >= 2

    scene = Scene()
    scene.set_target(T)
    cam = scene.camera(T)
    try:
        frames = render_frames(scene, sols, cam, width=160, height=120)
    except Exception as exc:  # pragma: no cover - depends on GL availability
        pytest.skip(f"offscreen rendering unavailable: {exc}")

    assert len(frames) == len(sols)
    assert frames[0].shape == (120, 160, 3)
    # The robot must be visible in the frame (not a blank background).
    assert float(np.asarray(frames[0]).std()) > 5.0
