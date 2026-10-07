"""Bring up the Franka MuJoCo model and visualize the inverse-kinematics solutions.

Loads the Panda arm (MuJoCo Menagerie) together with a translucent target
marker, solves IK for a chosen flange pose, and renders every distinct
solution.  Outputs
  * ``docs/img/ik_solutions.png``  - a labelled montage of all solutions
  * ``docs/img/ik_solutions.gif``  - an animation cycling through them
and can open an interactive viewer with ``--viewer``.

Examples
--------
    python scripts/visualize_ik.py                 # render montage + gif
    python scripts/visualize_ik.py --viewer        # interactive viewer
    python scripts/visualize_ik.py --q 0.7 1.4 1.6 -2.4 -1.16 3.28 -2.87
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from ik import PandaArm, solve_analytic  # noqa: E402
from ik.model import resolve_xml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).resolve().parent / "ik_scene_template.xml"
IMG_DIR = ROOT / "docs" / "img"

plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei", "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

# A reachable, visually clear pose (used when --q is not given).
DEFAULT_Q = np.array([0.7249, 1.4004, 1.5975, -2.3957, -1.1580, 3.2758, -2.8668])


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def build_scene_xml(panda_xml: Path) -> str:
    """Write the scene XML (template with absolute paths) to a temp file."""
    assets = panda_xml.parent / "assets"
    text = TEMPLATE.read_text()
    text = text.replace("__PANDA_XML__", str(panda_xml)).replace(
        "__ASSETS_DIR__", str(assets)
    )
    fh = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
    fh.write(text)
    fh.close()
    return fh.name


class Scene:
    """MuJoCo scene with the Panda arm and a movable target marker."""

    def __init__(self, panda_xml: Path | None = None):
        self.panda_xml = resolve_xml(panda_xml)
        self.xml_path = build_scene_xml(self.panda_xml)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.target_mocap_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ik_target"
        )
        if self.target_mocap_id < 0 or self.model.nmocap < 1:
            raise RuntimeError("target marker (mocap body) not found in scene")
        self.mocap_index = self.model.body_mocapid[self.target_mocap_id]

    def set_q(self, q: np.ndarray) -> None:
        self.data.qpos[:7] = np.asarray(q, float)
        mujoco.mj_forward(self.model, self.data)

    def set_target(self, T: np.ndarray) -> None:
        self.data.mocap_pos[self.mocap_index] = T[:3, 3]
        self.data.mocap_quat[self.mocap_index] = rotmat_to_quat(T[:3, :3])
        mujoco.mj_forward(self.model, self.data)

    def camera(self, T: np.ndarray) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        lookat = 0.5 * (np.array([0.0, 0.0, 0.30]) + T[:3, 3])
        cam.lookat[:] = lookat
        cam.distance = 1.45
        cam.azimuth = 128.0
        cam.elevation = -16.0
        return cam


def render_frames(scene: Scene, solutions, cam, width=520, height=400):
    """Render one RGB frame per solution."""
    renderer = mujoco.Renderer(scene.model, height, width)
    frames = []
    for sol in solutions:
        scene.set_q(sol.q)
        renderer.update_scene(scene.data, cam)
        frames.append(renderer.render().copy())
    renderer.close()
    return frames


def make_montage(scene, solutions, frames, out_path: Path) -> None:
    n = len(frames)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, (sol, img) in enumerate(zip(solutions, frames)):
        ax = axes[i]
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"解 #{i}   位置误差 {sol.pos_err:.1e} m   姿态误差 {sol.rot_err:.1e} rad",
            fontsize=11,
        )
        qs = " ".join(f"{v:+.2f}" for v in sol.q)
        ax.text(
            0.5, -0.02, f"q = [{qs}]", transform=ax.transAxes,
            ha="center", va="top", fontsize=8, family="monospace",
        )
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Franka Panda 逆运动学：同一目标位姿（橙色球 + RGB 坐标轴）的多组解",
        fontsize=14, y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def make_gif(scene, solutions, frames, out_path: Path, hold=6) -> None:
    """Animate through the solutions, labelling each frame."""
    gif_frames = []
    for i, (sol, img) in enumerate(zip(solutions, frames)):
        fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"解 #{i}  (共 {len(frames)} 组)", fontsize=13)
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        for _ in range(hold):
            gif_frames.append(Image.fromarray(buf))
        gif_frames.append(Image.fromarray(buf))  # brief pause frame
    gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:],
                       duration=150, loop=0)
    print(f"saved: {out_path}")


def run_viewer(scene: Scene, solutions, cam) -> None:
    """Interactive viewer: press SPACE to cycle through the solutions."""
    state = {"i": 0}

    def render():
        scene.set_q(solutions[state["i"]].q)

    def key_callback(keycode):
        if keycode == 32:  # space
            state["i"] = (state["i"] + 1) % len(solutions)
            print(f"solution {state['i']}/{len(solutions)}  q={np.round(solutions[state['i']].q, 3)}")

    render()
    print("Viewer: press SPACE to cycle through the IK solutions, ESC to quit.")
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        try:
            viewer.user_key_callback = key_callback
        except Exception:
            print("(key callback unavailable; solutions are not switchable)")
        while viewer.is_running():
            render()
            viewer.sync()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--q", type=float, nargs=7, default=None,
                    help="joint configuration (rad) whose flange pose becomes the target")
    ap.add_argument("--panda-xml", type=str, default=None, help="override the Panda MJCF path")
    ap.add_argument("--viewer", action="store_true", help="open the interactive viewer")
    ap.add_argument("--max-solutions", type=int, default=6)
    args = ap.parse_args()

    solver_arm = PandaArm(args.panda_xml)          # solver model (nohand, nq=7)
    q_true = np.array(args.q) if args.q is not None else DEFAULT_Q.copy()
    solver_arm.set_q(q_true)
    T = solver_arm.tcp_transform()

    solutions = solve_analytic(solver_arm, T, max_solutions=args.max_solutions)
    print(f"target position : {np.round(T[:3, 3], 4)}")
    print(f"IK solutions    : {len(solutions)}")

    scene = Scene(args.panda_xml)
    scene.set_target(T)
    cam = scene.camera(T)

    if args.viewer:
        run_viewer(scene, solutions, cam)
        return

    frames = render_frames(scene, solutions, cam)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    make_montage(scene, solutions, frames, IMG_DIR / "ik_solutions.png")
    make_gif(scene, solutions, frames, IMG_DIR / "ik_solutions.gif")


if __name__ == "__main__":
    main()
