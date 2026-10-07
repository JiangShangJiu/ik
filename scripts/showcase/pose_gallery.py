"""Animate verified IK configurations while keeping the full TCP pose fixed.

    python -m scripts.showcase.pose_gallery
    python -m scripts.showcase.pose_gallery --reuse-data

UR/Lite hold and cut between discrete closed-form branches. iiwa/Panda are
freshly solved continuous self motions. Guides are camera-projected diagnostic
annotations; a complete elbow circle does not assert full-circle feasibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw

from . import portfolio_style as V
from . import scene as S
from .motion_gallery import _dashes, _encode, _read_bytes
from .portfolio import NAMES, Studio, target_extent
from .style import SHOW
from .paths import ROOT, metric_path, video_path

KEYS = tuple(NAMES)
VIEW = (36, 168, 804, 526)
GUIDE = "#378D91"
PALE = "#99B9B9"
ACTUAL = "#E4813E"
METHODS = {
    "ur5e": "三平行轴几何闭式 · 肩角 → 平面 2R → 腕角",
    "lite6": "球腕几何闭式 · 腕心定位 → 2R → 球腕",
    "iiwa14": "S-R-S · 臂型角 ψ 参数化 → 逐帧闭式解",
    "panda": "逐帧给定 q7 → 六关节 DLS · 前帧作初值",
}


def _projector(studio, camera):
    """Cache the camera basis instead of allocating a GL renderer each frame."""
    eye, up = S._camera_basis(studio.arm.model, studio.arm.data, camera)
    forward = -eye
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    width, height = studio.scene.width, studio.scene.height
    origin = np.asarray(camera.lookat) + camera.distance * eye
    tangent = np.tan(np.deg2rad(studio.arm.model.vis.global_.fovy) / 2)

    def project(points):
        delta = np.atleast_2d(points) - origin
        depth = delta @ forward
        if np.any(depth <= 0):
            raise RuntimeError("Fixed-pose annotation is behind the camera")
        u = (delta @ right) / (depth * tangent * width / height)
        v = (delta @ up) / (depth * tangent)
        return np.column_stack(((u + 1) * width / 2, (1 - v) * height / 2))

    return project


def _circle_points(geometry, angles):
    return (np.asarray(geometry["center"]) + geometry["radius"] *
            (np.cos(angles)[:, None] * np.asarray(geometry["u"]) +
             np.sin(angles)[:, None] * np.asarray(geometry["v"])))


def _camera(studio, record):
    target = np.asarray(record["target_pose"])
    extra = [target_extent(target)]
    azimuth, elevation = 145., -22.
    if "geometry" in record and record["parameter_name"] == "psi":
        geo = record["geometry"]
        extra.append(_circle_points(geo, np.linspace(0, 2 * np.pi, 161)))
        normal = np.cross(geo["u"], geo["v"])
        normal /= np.linalg.norm(normal)
        if normal[2] < 0:
            normal = -normal
        # A slight side view separates shoulder/elbow/wrist while the raised
        # camera still shows the elbow circle as a broad ellipse.
        azimuth = (np.degrees(np.arctan2(normal[1], normal[0])) + 244) % 360
        elevation = -float(np.clip(np.degrees(np.arcsin(normal[2])), 35, 48))
    return studio.scene.auto_camera(
        sweep_qs=np.asarray(record["qs"]), extra_points=np.vstack(extra),
        azimuth=azimuth, elevation=elevation, margin=1.17)


def _annotation(studio, camera, record):
    project = _projector(studio, camera)
    target = np.asarray(record["target_pose"])
    axes = np.vstack((target[:3, 3], target[:3, 3] + .055 * target[:3, :3].T))
    annotation = {"target": project(axes)}
    if record["parameter_name"] == "psi":
        geo = record["geometry"]
        interval = record["summary"]["parameter_interval_rad"]
        annotation.update(
            circle=project(_circle_points(geo, np.linspace(0, 2 * np.pi, 181))),
            legal=project(_circle_points(geo, np.linspace(*interval, 121))),
            elbows=project(geo["elbows_m"]),
            center=project(geo["center"])[0],
            sw=project([geo["shoulder"], geo["wrist"]]),
        )
    elif record["parameter_name"] == "q7":
        annotation["elbows"] = project(record["actual_elbows_m"])
        annotation["locus"] = annotation["elbows"][:len(record["qs"]) // 2 + 1]
    return annotation


def _marker(draw, point, color, radius=6):
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill=color, outline="white", width=2)


def _annotate(frame, annotation, k):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    if "circle" in annotation:
        _dashes(draw, annotation["circle"], color=PALE, width=2, dash=7, gap=6)
        draw.line([tuple(p) for p in annotation["legal"]], fill=GUIDE, width=4)
        _dashes(draw, annotation["sw"], color=PALE, width=2)
        draw.line([tuple(annotation["center"]), tuple(annotation["elbows"][k])],
                  fill=ACTUAL, width=2)
        for label, point in zip(("S", "W"), annotation["sw"]):
            _marker(draw, point, GUIDE, 4)
            V.text(image, (point[0] + 8, point[1] - 19), label, 17, GUIDE, bold=True)
        _marker(draw, annotation["elbows"][k], ACTUAL, 8)
        x, y = annotation["elbows"][k]
        V.text(image, (x + 12, y - 24), "E / 肘部", 17, V.ORANGE, bold=True)
    elif "locus" in annotation:
        _dashes(draw, annotation["locus"], color=GUIDE, width=3)
        _marker(draw, annotation["elbows"][k], ACTUAL, 7)
        x, y = annotation["elbows"][k]
        V.text(image, (x + 12, y - 24), "E / 肘部", 17, V.ORANGE, bold=True)
    # These are diagnostic projections, deliberately visible through shells.
    points = annotation["target"]
    x, y = points[0]
    draw.ellipse((x - 13, y - 13, x + 13, y + 13), outline=V.ORANGE, width=2)
    for tip, color in zip(points[1:], ("#D85850", "#47A064", "#547DC2")):
        draw.line([tuple(points[0]), tuple(tip)], fill=color, width=3)
    _marker(draw, points[0], V.ORANGE, 5)
    # Put the fixed-pose label in a constant corner, with a fine leader line.
    V.chip(image, (18, 16), "TCP 位姿固定", V.INK, size=17)
    draw.line([(165, 36), (x, y)], fill="#B6C3C5", width=1)
    return image


def _parameter(record, k):
    name = record["parameter_name"]
    if name == "psi":
        return f"ψ {np.degrees(record['parameter_values_rad'][k]):.1f}°"
    if name == "q7":
        return f"q7 {record['parameter_values_rad'][k]:.3f} rad"
    return f"分支 {record['branch_indices'][k] + 1:02d} / 08"


def _gauge(image, record, k):
    draw = ImageDraw.Draw(image)
    x0, x1, y = 884, 1228, 480
    if record["parameter_name"] is None:
        index = record["branch_indices"][k]
        for i in range(8):
            x = x0 + i * (x1 - x0) / 7
            _marker(draw, (x, y), V.TEAL if i == index else "#DCE7E8", 7)
        V.text(image, (880, 506), "离散候选逐个切换 · 无关节插值", 19, V.MUTED)
    else:
        lo, hi = record["summary"]["parameter_interval_rad"]
        value = record["parameter_values_rad"][k]
        draw.line((x0, y, x1, y), fill="#DAE6E6", width=7)
        x = x0 + (x1 - x0) * (value - lo) / (hi - lo)
        draw.line((x0, y, x, y), fill=GUIDE, width=7)
        _marker(draw, (x, y), ACTUAL, 8)
        interval = (f"合法 ψ 区间 {np.degrees(lo):.1f}° – {np.degrees(hi):.1f}°"
                    if record["parameter_name"] == "psi"
                    else f"q7 区间 {lo:.3f} – {hi:.3f} rad")
        V.text(image, (880, 506), interval, 18, V.MUTED)


def render_one(key, record, fps=20):
    studio = Studio(key, VIEW[2], VIEW[3])
    camera = _camera(studio, record)
    annotations = _annotation(studio, camera, record)
    continuous = record["mode"] == "continuous_self_motion"
    template = Image.new("RGB", (1280, 800), V.BG)
    V.text(template, (40, 24), "IK / FIXED TCP, CHANGING CONFIGURATION", 17, V.MUTED, mono=True)
    V.text(template, (40, 66), f"{NAMES[key]} · 末端不动，构型在变", 38, bold=True)
    V.text(template, (42, 124), "固定目标位置与方向  /  " +
           ("连续冗余自运动" if continuous else "独立闭式分支逐个展示") + "  /  逐帧 FK 验收", 21, V.MUTED)
    V.text(template, (880, 176), "本动画使用的求解方法", 22, V.TEAL, bold=True)
    for i, line in enumerate(record["method_lines"]):
        V.text(template, (880, 218 + 36 * i), line, 22)
    V.rule(template, (880, 338, 1234, 338))
    V.text(template, (880, 416), "当前冗余参数" if continuous else "当前闭式候选", 20, V.MUTED)
    V.rule(template, (880, 548, 1234, 548))
    if key == "iiwa14":
        legend = "浅虚线：完整肘部圆   青弧：合法区间   橙点：当前肘部"
    elif key == "panda":
        legend = "青色虚线：实际肘部轨迹   橙点：当前肘部"
    else:
        legend = "六轴一般对应离散分支；画面切换不表示连续换支运动"
    V.text(template, (40, 716), legend, 20, V.MUTED)
    position = np.asarray(record["target_pose"])[:3, 3]
    V.text(template, (40, 753), f"TCP fixed / x {position[0]:+.3f}  y {position[1]:+.3f}  z {position[2]:+.3f} m",
           16, V.MUTED, mono=True)
    V.text(template, (880, 716), "通过位姿与关节限位检查", 20, V.TEAL)
    V.text(template, (880, 754), "MuJoCo / fixed camera / kinematics", 14, V.MUTED, mono=True)
    frames = []
    target = np.asarray(record["target_pose"])
    for k, q in enumerate(record["qs"]):
        image = template.copy()
        frame = studio.render(q, target, camera, axis_len=.055, ball=.009)
        V.inset(image, _annotate(frame, annotations, k), VIEW)
        V.text(image, (880, 365), _parameter(record, k), 34, V.TEAL, mono=continuous)
        _gauge(image, record, k)
        V.text(image, (880, 573), f"Δp {record['pos_error_m'][k]:.1e} m", 23, V.INK, mono=True)
        V.text(image, (880, 612), f"ΔR {record['rot_error_rad'][k]:.1e} rad", 22, V.INK, mono=True)
        iterations = record["iterations_per_frame"][k]
        V.text(image, (880, 655), f"{'DLS' if key == 'panda' else '闭式'} · {iterations} 次迭代", 21, V.TEAL)
        frames.append(np.asarray(image))
    camera_report = dict(lookat=list(camera.lookat), distance=camera.distance,
                         azimuth=camera.azimuth, elevation=camera.elevation,
                         viewport=list(VIEW), target_pixel=annotations["target"][0].tolist())
    studio.close()
    _encode(frames, SHOW / f"pose_{key}", fps)
    return camera_report


def render_quad(robots, fps=20):
    template = Image.new("RGB", (1440, 1320), V.BG)
    V.text(template, (48, 25), "IK / ONE TCP POSE, DIFFERENT CONFIGURATIONS", 18, V.MUTED, mono=True)
    V.text(template, (48, 66), "末端不动，看看逆解怎么变。", 44, bold=True)
    V.text(template, (50, 132), "六轴：离散闭式分支  /  七轴：连续冗余自运动  /  求解方法直接标在画面中", 22, V.MUTED)
    readers = []
    width, height = 596, 390
    n = len(robots[KEYS[0]]["qs"])
    for i, key in enumerate(KEYS):
        x, y = 48 + i % 2 * 696, 190 + i // 2 * 540
        V.text(template, (x + 6, y), NAMES[key], 28, bold=True)
        mode = "连续自运动" if robots[key]["parameter_name"] else "离散分支切换"
        V.text(template, (x + 176, y + 5), mode, 20, V.TEAL)
        V.text(template, (x + 6, y + 43), METHODS[key], 19, V.MUTED)
        cmd = ["ffmpeg", "-v", "error", "-i", str(video_path(f"pose_{key}.mp4")),
               "-vf", f"crop={VIEW[2]}:{VIEW[3]}:{VIEW[0]}:{VIEW[1]},scale={width}:{height}",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        readers.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))
    V.text(template, (52, 1280), "橙色 TCP 坐标系固定 · iiwa 的圆是肘部轨迹 · 四台分别使用各自的目标位姿", 20, V.MUTED)
    frames = []
    try:
        for k in range(n):
            image = template.copy()
            for i, (key, reader) in enumerate(zip(KEYS, readers)):
                x, y = 48 + i % 2 * 696, 190 + i // 2 * 540
                raw = _read_bytes(reader.stdout, width * height * 3)
                V.inset(image, Image.frombytes("RGB", (width, height), raw), (x + 30, y + 76, width, height))
                record = robots[key]
                V.text(image, (x + 10, y + 480), _parameter(record, k), 23, V.TEAL)
                V.text(image, (x + 10, y + 515),
                       f"Δp {record['pos_error_m'][k]:.1e} m  /  ΔR {record['rot_error_rad'][k]:.1e} rad",
                       15, V.MUTED, mono=True)
            frames.append(np.asarray(image))
    finally:
        for reader in readers:
            reader.stdout.close()
            if reader.poll() is None:
                reader.terminate()
            reader.wait()
    _encode(frames, SHOW / "pose_quad", fps)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--quad-only", action="store_true")
    args = parser.parse_args(argv)
    SHOW.mkdir(parents=True, exist_ok=True)
    path = SHOW / "pose_metrics.json"
    if args.reuse_data or args.quad_only:
        data = json.loads(metric_path(path.name).read_text(encoding="utf-8"))
    else:
        from .pose_data import collect_poses
        data = collect_poses()
    fps = 20
    sources = (Path(__file__), ROOT / "scripts/showcase/portfolio.py",
               ROOT / "scripts/showcase/portfolio_style.py", ROOT / "scripts/showcase/scene.py",
               ROOT / "scripts/showcase/motion_gallery.py")
    data["render"] = {
        "fps": fps, "duration_s": len(data["robots"][KEYS[0]]["qs"]) / fps,
        "guides": "Camera-projected diagnostic markers, visible through meshes. iiwa: full geometric elbow circle and validated arc. Panda: actual elbow locus, not an assumed circle.",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "cameras": data.get("render", {}).get("cameras", {}),
    }
    if not args.quad_only:
        for key in KEYS:
            data["render"]["cameras"][key] = render_one(key, data["robots"][key], fps)
    render_quad(data["robots"], fps)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
