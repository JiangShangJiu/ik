"""Draw verified configurations sharing one target pose for each robot.

    python -m scripts.showcase.solution_gallery
    python -m scripts.showcase.solution_gallery --reuse-data

Six-axis branches are discrete. Seven-axis panels are finite samples of a
redundant solution set. Each atlas panel is an independently verified IK
configuration; the atlas does not claim a continuous branch transition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw

from . import portfolio_style as V
from . import scene as S
from .portfolio import NAMES, Studio, target_extent
from .style import SHOW
from .paths import ROOT, metric_path

KEYS = tuple(NAMES)
KINDS = {"ur5e": "6R / 闭式离散候选", "lite6": "6R / 闭式离散候选",
         "iiwa14": "7R / 臂型角代表采样", "panda": "7R / 冗余构型代表采样"}


def _parameter(solution):
    if "psi_rad" in solution:
        return f"ψ {np.degrees(solution['psi_rad']):.1f}°"
    if "fixed_q7_rad" in solution:
        return f"q7 {solution['fixed_q7_rad']:.2f} rad"
    return "独立闭式候选"


def _frame(studio, q, target, camera):
    """Keep the fixed target marker visible even when a shell hides it."""
    frame = Image.fromarray(studio.render(q, target, camera, axis_len=.05, ball=.009))
    points = np.vstack([target[:3, 3], target[:3, 3] + .05 * target[:3, :3].T])
    u, v = S.project_points(studio.arm.model, studio.arm.data, camera, points,
                            width=studio.scene.width, height=studio.scene.height)
    pixels = np.column_stack([(u + 1) * studio.scene.width / 2,
                              (1 - v) * studio.scene.height / 2])
    draw = ImageDraw.Draw(frame)
    for tip, color in zip(pixels[1:], ("#D85850", "#47A064", "#547DC2")):
        draw.line([tuple(pixels[0]), tuple(tip)], fill=color, width=2)
    x, y = pixels[0]
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=V.ORANGE, outline="white", width=2)
    return frame


def atlas(key, record):
    solutions = record["solutions"]
    target = np.asarray(record["target_pose"])
    qs = np.asarray([s["q"] for s in solutions])
    studio = Studio(key, 318, 314)
    camera = studio.camera(qs, target_extent(target), margin=1.15)
    image = Image.new("RGB", (1440, 1120), V.BG)
    V.text(image, (48, 28), "IK / SAME TARGET, DIFFERENT CONFIGURATIONS", 18, V.MUTED, mono=True)
    V.text(image, (48, 78), f"{NAMES[key]} · 一个末端位姿，多种构型。", 42, bold=True)
    V.text(image, (50, 145), "固定末端位置与方向  /  统一相机与比例  /  每个候选单独回代验证", 23, V.MUTED)
    V.text(image, (50, 184), f"{KINDS[key]}    ·    本图 {len(solutions)} 个 / 候选池 {record['candidate_count']} 个", 18, V.TEAL)
    for i in range(8):
        x, y = 48 + i % 4 * 341, 224 + i // 4 * 425
        if i >= len(solutions):
            V.text(image, (x + 22, y + 130), "本次采样到此结束", 23, V.MUTED)
            continue
        solution = solutions[i]
        V.text(image, (x + 5, y), f"{i + 1:02d} / {_parameter(solution)}", 18, bold=True)
        V.inset(image, _frame(studio, qs[i], target, camera), (x, y + 34, 318, 314))
        V.text(image, (x + 5, y + 361), f"Δp {solution['pos_err_m']:.1e} m · ΔR {solution['rot_err_rad']:.1e} rad",
               14, V.MUTED, mono=True)
        margin = solution["joint_margin_rad"]
        V.text(image, (x + 5, y + 390), f"最小限位裕度 {np.degrees(margin):.1f}°",
               18, V.TEAL if margin > .05 else V.ORANGE)
    studio.close()
    V.rule(image, (48, 1080, 1392, 1080))
    V.text(image, (48, 1095), "彩色坐标系：固定目标位姿。七轴为有限代表采样，不是全部解；不包含碰撞检查。", 17, V.MUTED)
    V.save(image, SHOW / f"solutions_{key}.png")


def _representative_pair(studio, qs):
    """Use actual link locations rather than angular wraps to pick two views."""
    poses = []
    for q in qs:
        studio.arm.set_q(q)
        poses.append(studio.arm.data.xpos.copy().ravel())
    poses = np.asarray(poses)
    distance = np.linalg.norm(poses[:, None] - poses[None, :], axis=2)
    return np.unravel_index(np.argmax(distance), distance.shape)


def overview(robots):
    image = Image.new("RGB", (1440, 1320), V.BG)
    V.text(image, (48, 25), "IK / ONE POSE, MANY CONFIGURATIONS", 18, V.MUTED, mono=True)
    V.text(image, (48, 66), "末端相同，构型可以不同。", 44, bold=True)
    V.text(image, (50, 132), "每台选两种代表构型 · 同型号内目标与相机固定 · 每种型号另有完整八格图", 22, V.MUTED)
    for i, key in enumerate(KEYS):
        record = robots[key]
        solutions = record["solutions"]
        qs = np.asarray([s["q"] for s in solutions])
        target = np.asarray(record["target_pose"])
        studio = Studio(key, 304, 382)
        camera = studio.camera(qs, target_extent(target), margin=1.15)
        pair = _representative_pair(studio, qs)
        x, y = 48 + i % 2 * 696, 240 + i // 2 * 530
        V.text(image, (x + 6, y - 45), NAMES[key], 26, bold=True)
        V.text(image, (x + 172, y - 39), KINDS[key], 19, V.MUTED)
        for j, index in enumerate(pair):
            px = x + j * 332
            V.inset(image, _frame(studio, qs[index], target, camera), (px, y, 304, 382))
            V.text(image, (px + 10, y + 402), f"#{index + 1:02d} / {_parameter(solutions[index])}", 18, V.TEAL)
        studio.close()
    V.text(image, (52, 1272), "四台各自使用一个目标位姿；七轴是代表采样，每张图都通过完整位姿与关节限位检查。", 20, V.MUTED)
    V.save(image, SHOW / "solutions_quad.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-data", action="store_true")
    args = parser.parse_args(argv)
    path = SHOW / "solution_metrics.json"
    if args.reuse_data:
        data = json.loads(metric_path(path.name).read_text(encoding="utf-8"))
    else:
        from .solution_data import collect_solutions
        data = collect_solutions()
    data["render"] = {
        "format": "static candidate atlas; no continuous transition claim",
        "target_overlay": "camera-projected fixed target axes, visible through meshes for comparison",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / "scripts/showcase/portfolio.py",
                                    ROOT / "scripts/showcase/portfolio_style.py", ROOT / "scripts/showcase/scene.py")},
    }
    SHOW.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key in KEYS:
        atlas(key, data["robots"][key])
    overview(data["robots"])


if __name__ == "__main__":
    main()
