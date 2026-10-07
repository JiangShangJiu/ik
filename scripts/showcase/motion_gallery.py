"""Bright studio videos of four robots following Cartesian lines and circles.

    python -m scripts.showcase.motion_gallery
    python -m scripts.showcase.motion_gallery --reuse-data --kind circle

All visible guides are camera projections of the measured 3-D target paths.
There is no framewise camera motion and no interpolated joint-space shortcut.
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
from .portfolio import NAMES, Studio
from .style import SHOW, _write_mp4
from .paths import ROOT, metric_path, video_path

KEYS = tuple(NAMES)
VIEW = (36, 168, 820, 588)
PAGE_BG = "#F8FAFC"
GUIDE = "#378D91"
ACTUAL = "#E4813E"


def _world_guide(record):
    g = record["geometry"]
    if g["kind"] == "line":
        return np.linspace(g["line_start"], g["line_end"], 101)
    theta = np.linspace(0, 2 * np.pi, 161)
    return (np.asarray(g["center"]) + g["radius"] *
            (np.cos(theta)[:, None] * np.asarray(g["u"]) +
             np.sin(theta)[:, None] * np.asarray(g["v"])))


def _project(studio, cam, points, width=820, height=588):
    u, v = S.project_points(studio.arm.model, studio.arm.data, cam, points,
                            width=width, height=height)
    return np.column_stack([(u + 1) * width / 2, (1 - v) * height / 2])


def _dashes(draw, points, color=GUIDE, width=2, dash=8, gap=5):
    """Use pixel arc length, so a projected circle has regular dash spacing."""
    offset = 0.0
    for a, b in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(b - a))
        if length < 1e-8:
            continue
        pos = 0.0
        while pos < length:
            phase = offset % (dash + gap)
            active = phase < dash
            run = min((dash if active else dash + gap) - phase, length - pos)
            if run < 1e-8:
                run = min(1e-6, length - pos)
            if active:
                p0, p1 = a + pos / length * (b - a), a + (pos + run) / length * (b - a)
                draw.line([tuple(p0), tuple(p1)], fill=color, width=width)
            pos += run
            offset += run


def _camera(studio, qs, guide, record):
    """Look across the path plane so circles do not collapse to edge-on lines."""
    g = record["geometry"]
    normal = np.cross(np.asarray(g["u"]), np.asarray(g["v"]))
    normal = normal / np.linalg.norm(normal)
    # Prefer the upper face of tilted planes, then an outward view for vertical
    # paths. This keeps enlarged circles readable without viewing from below.
    center = np.asarray(g["center"])
    if abs(normal[2]) > .25:
        if normal[2] < 0:
            normal = -normal
    elif np.dot(normal[:2], center[:2]) < 0:
        normal = -normal
    # MuJoCo azimuth points from the camera toward the scene: its camera
    # position has the opposite horizontal bearing to the viewing normal.
    azimuth = (float(np.degrees(np.arctan2(normal[1], normal[0]))) + 180.0 + 35.0) % 360
    if np.linalg.norm(normal[:2]) < .2:
        azimuth = 135.0
    elevation = -float(np.clip(np.degrees(np.arcsin(max(0., normal[2]))), 18., 55.))
    Ts = np.asarray(record["Ts"])
    axis_tips = Ts[:, None, :3, 3] + .045 * Ts[:, :3, :3].transpose(0, 2, 1)
    framing_points = np.vstack([guide, axis_tips.reshape(-1, 3)])
    return studio.scene.auto_camera(sweep_qs=qs, extra_points=framing_points,
                                     azimuth=azimuth, elevation=elevation, margin=1.17)


def _guide_frame(frame, guide_px, actual_px, k, kind, end_px):
    image = Image.fromarray(frame)
    d = ImageDraw.Draw(image)
    _dashes(d, guide_px)
    if kind == "circle" and k:
        d.line([tuple(p) for p in actual_px[:k + 1]], fill=ACTUAL, width=3)
    if kind == "line":
        for label, (px, py) in zip(("A", "B"), end_px):
            d.ellipse((px - 4, py - 4, px + 4, py + 4), fill="white", outline=GUIDE, width=2)
            V.text(image, (px + 9, py - 20), label, 19, GUIDE, bold=True)
    x, y = actual_px[k]
    d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=ACTUAL, outline="white", width=3)
    return image


def _encode(frames, stem, fps):
    V.save(Image.fromarray(frames[0]), stem.with_name(stem.name + "_first.png"))
    _write_mp4(frames, stem.with_suffix(".mp4"), fps)
    print(f"saved: {stem.name}.mp4", flush=True)


def render_one(key, kind, record, fps=20):
    qs, Ts = np.asarray(record["qs"]), np.asarray(record["Ts"])
    guide = _world_guide(record)
    st = Studio(key, VIEW[2], VIEW[3])
    cam = _camera(st, qs, guide, record)
    guide_px = _project(st, cam, guide)
    actual_px = _project(st, cam, record["actual_points_m"])
    end_px = _project(st, cam, [record["geometry"]["line_start"], record["geometry"]["line_end"]]) \
        if kind == "line" else None
    spec = record["summary"]
    title = "末端沿直线往返" if kind == "line" else "末端沿圆周运动"
    template = Image.new("RGB", (1280, 800), PAGE_BG)
    V.text(template, (40, 24), "ROBOTICS / CARTESIAN PATH TRACKING", 17, V.MUTED, mono=True)
    V.text(template, (40, 66), f"{NAMES[key]} · {title}", 40, bold=True)
    V.text(template, (42, 124), "固定末端姿态  /  逐点 IK  /  关节限位与循环接缝检查", 21, V.MUTED)
    V.text(template, (900, 190), "LINE / BACK & FORTH" if kind == "line" else "CIRCLE / FULL REVOLUTION",
           16, V.MUTED, mono=True)
    g = record["geometry"]
    size = np.linalg.norm(np.asarray(g["line_end"]) - g["line_start"]) if kind == "line" else g["radius"]
    V.metric(template, (900, 240), f"{1000 * size:.0f} mm", "直线长度" if kind == "line" else "圆周半径", size=36)
    V.rule(template, (900, 352, 1235, 352))
    V.metric(template, (900, 386), f"{spec['max_pos_error_m']:.1e} m", "整段最大位置误差", size=27)
    V.metric(template, (900, 491), f"{spec['max_rot_error_rad']:.1e} rad", "整段最大姿态误差", size=25)
    V.text(template, (900, 612), "虚线  目标轨迹", 22, GUIDE)
    V.text(template, (900, 652), "橙点  当前实际末端", 22, ACTUAL)
    V.text(template, (900, 710), "已通过逐帧 FK 回代与限位检查", 18, V.MUTED)
    V.text(template, (40, 776), "MuJoCo / fixed camera / sampled kinematics", 15, V.MUTED, mono=True)
    frames = []
    for k, (q, T) in enumerate(zip(qs, Ts)):
        image = template.copy()
        frame = st.render(q, T, cam, axis_len=.045, ball=.009)
        frame = _guide_frame(frame, guide_px, actual_px, k, kind, end_px)
        V.inset(image, frame, VIEW)
        progress = k / len(qs)
        d = ImageDraw.Draw(image)
        d.rounded_rectangle((900, 752, 1230, 758), radius=3, fill="#E3EAF0")
        if k:
            d.rounded_rectangle((900, 752, 900 + 330 * progress, 758), radius=3, fill=GUIDE)
        frames.append(np.asarray(image))
    st.close()
    _encode(frames, SHOW / f"path_{kind}_{key}", fps)


def _read_bytes(stream, count):
    chunks, received = [], 0
    while received < count:
        chunk = stream.read(count - received)
        if not chunk:
            raise RuntimeError("Unexpected end of video while assembling the overview")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def render_quad(kind, robots, fps=20):
    """Decode the finished clips into an overview; no second IK or render run."""
    readers = []
    w, h = 640, 458
    n = len(robots[KEYS[0]][kind]["qs"])
    template = Image.new("RGB", (1440, 1320), PAGE_BG)
    V.text(template, (48, 25), "IK / FOUR ROBOTS, ONE PATH TYPE", 18, V.MUTED, mono=True)
    V.text(template, (48, 66), "让末端沿直线运动。" if kind == "line" else "让末端沿圆周运动。", 44, bold=True)
    V.text(template, (50, 132), "四台机械臂 · 固定末端姿态 · 实际 IK 构型 · 明亮背景", 23, V.MUTED)
    for i, key in enumerate(KEYS):
        x, y = 48 + i % 2 * 696, 205 + i // 2 * 530
        V.text(template, (x + 6, y - 10), NAMES[key], 26, bold=True)
        spec = robots[key][kind]["summary"]
        size = spec["path_extent_m"] if kind == "line" else robots[key][kind]["geometry"]["radius"]
        size_label = f"{'L' if kind == 'line' else 'r'} {1000 * size:.0f} mm"
        V.text(template, (x + 180, y - 5), f"{size_label} · max Δp {spec['max_pos_error_m']:.1e} m", 17, V.MUTED, mono=True)
        cmd = ["ffmpeg", "-v", "error", "-i", str(video_path(f"path_{kind}_{key}.mp4")),
               "-vf", f"crop={VIEW[2]}:{VIEW[3]}:{VIEW[0]}:{VIEW[1]},scale={w}:{h}",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        readers.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL))
    V.text(template, (52, 1272), "虚线：目标路径    橙点：实际末端    RGB 轴：目标姿态", 21, V.MUTED)
    frames = []
    try:
        for _ in range(n):
            im = template.copy()
            for i, reader in enumerate(readers):
                raw = _read_bytes(reader.stdout, w * h * 3)
                frame = Image.frombytes("RGB", (w, h), raw)
                x, y = 48 + i % 2 * 696, 240 + i // 2 * 530
                V.inset(im, frame, (x, y, w, h))
            frames.append(np.asarray(im))
    finally:
        for reader in readers:
            reader.stdout.close()
            if reader.poll() is None:
                reader.terminate()
            reader.wait()
    _encode(frames, SHOW / f"path_{kind}_quad", fps)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--kind", choices=("line", "circle", "all"), default="all")
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args(argv)
    SHOW.mkdir(parents=True, exist_ok=True)
    path = SHOW / "path_metrics.json"
    if args.reuse_data:
        data = json.loads(metric_path(path.name).read_text(encoding="utf-8"))
    else:
        from .path_data import collect_paths
        data = collect_paths(n_frames=args.frames)
    data["render"] = {
        "fps": args.fps, "background": "bright cool-white studio, no grey floor",
        "guide": "2-D camera projection of the actual 3-D target curve; shown over the mesh for visibility",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / "scripts/showcase/portfolio.py",
                                    ROOT / "scripts/showcase/portfolio_style.py", ROOT / "scripts/showcase/scene.py")},
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    kinds = ("line", "circle") if args.kind == "all" else (args.kind,)
    for kind in kinds:
        for key in KEYS:
            render_one(key, kind, data["robots"][key][kind], args.fps)
        render_quad(kind, data["robots"], args.fps)
    print(f"Path experiment: {path}", flush=True)


if __name__ == "__main__":
    main()
