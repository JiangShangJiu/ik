"""Show why pointwise IK existence does not imply a continuous full swivel.

    python -m scripts.showcase.swivel_gallery
    python -m scripts.showcase.swivel_gallery --reuse-data

The illustration is specific to one measured iiwa target and its joint limits.
The explanation clip advances on one branch, stops before its wrist limit, then
restarts with an explicitly labelled cut. It is not a continuous looping path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from . import portfolio_style as V
from .motion_gallery import _encode
from .portfolio import Studio
from .pose_gallery import _annotate, _annotation, _camera, _circle_points, _projector
from .style import SHOW
from .paths import ROOT, metric_path

VIEW = (36, 170, 770, 548)


def _branch_plot(data):
    samples = data["samples"]
    angles = np.asarray([s["psi_deg"] for s in samples])
    branches = data["branches"]
    selected = data["demonstration"]["branch_id"]
    fig, (counts, bands) = plt.subplots(
        2, 1, figsize=(12.4, 4.2), sharex=True, facecolor=V.BG,
        gridspec_kw={"height_ratios": [1, 3.2]})
    values = data["candidate_counts"]
    counts.fill_between(angles, values, 0, color=V.TEAL, alpha=.12)
    counts.plot(angles, values, color=V.TEAL, lw=1.8)
    counts.set_ylim(0, 9)
    counts.set_yticks([0, 4, 8])
    counts.set_ylabel("合法解数量", fontsize=13)
    counts.set_title("逐点存在性：整圈采样角度都有解", loc="left", fontsize=15, color=V.INK)
    labels = []
    for row, branch in enumerate(branches):
        identity = branch["branch_id"]
        labels.append(f"B{row + 1:02d}" + (" *" if identity == selected else ""))
        bands.plot([0, 360], [row, row], color="#E0E6E7", lw=6, solid_capstyle="butt")
        for segment in branch["segments"]:
            first, last = segment["parameter_interval_deg"]
            color = V.ORANGE if identity == selected else V.TEAL
            bands.plot([first, last], [row, row], color=color, lw=6,
                       solid_capstyle="butt")
            for angle in (first, last):
                if 0 < angle < 360:
                    bands.scatter([angle], [row], s=14, color=V.RED, zorder=4)
    lo, hi = np.degrees(data["original_demonstration_interval"]["parameter_interval_rad"])
    bands.axvspan(lo, hi, color=V.BLUE, alpha=.075)
    bands.set_yticks(range(len(branches)), labels)
    bands.set_ylim(len(branches) - .4, -.6)
    bands.set_ylabel("固定肩 / 肘 / 腕分支", fontsize=13)
    bands.set_xlabel("臂型角 ψ / °", fontsize=13)
    bands.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    bands.set_title("连续性：单条分支在限位处中断，不能把其他分支直接拼接", loc="left", fontsize=15, color=V.INK)
    for ax in (counts, bands):
        V.axes_style(ax)
        ax.set_xlim(0, 360)
        ax.grid(axis="x", color=V.LINE, alpha=.5, lw=.6)
    bands.tick_params(axis="y", labelsize=11)
    fig.subplots_adjust(left=.10, right=.98, top=.91, bottom=.13, hspace=.55)
    image = V.plot_image(fig)
    plt.close(fig)
    return image


def _projection_plot(data):
    geo = data["geometry"]
    points = _circle_points(geo, np.linspace(0, 2 * np.pi, 361))
    shoulder = np.asarray(geo["shoulder"])
    fig, ax = plt.subplots(figsize=(5.2, 3.3), facecolor=V.BG)
    x, y = points[:, 0], points[:, 1]
    ax.fill(x, y, color=V.TEAL, alpha=.10)
    ax.plot(x, y, color=V.TEAL, lw=2.3)
    ax.scatter([shoulder[0]], [shoulder[1]], color=V.RED, s=65, zorder=4)
    ax.annotate("S / 肩轴在投影内部", xy=shoulder[:2], xytext=(.10, -.23),
                fontsize=12, color=V.INK,
                arrowprops={"arrowstyle": "->", "color": V.RED})
    ax.set_aspect("equal")
    ax.set_xlim(-.08, .40)
    ax.set_ylim(-.30, .16)
    ax.set_xlabel("世界 X / m", fontsize=11)
    ax.set_ylabel("世界 Y / m", fontsize=11)
    V.axes_style(ax)
    fig.subplots_adjust(left=.13, right=.98, top=.98, bottom=.18)
    image = V.plot_image(fig)
    plt.close(fig)
    return image


def _joint_plot(data):
    angles = np.asarray([s["psi_rad"] for s in data["samples"]])
    demo = data["demonstration"]
    branch = next(b for b in data["branches"] if b["branch_id"] == demo["branch_id"])
    # Keep the freshly verified geometric branch unwrapped. Modulo 2*pi would
    # conceal the joint-limit crossing that this plot is meant to explain.
    q1 = np.asarray(branch["unwrapped_qs"])[:, 0]
    low, high = np.asarray(data["geometric_winding_report"]["joint1_limits_rad"])
    degrees = np.degrees(q1)
    inside = (q1 >= low) & (q1 <= high)
    fig, ax = plt.subplots(figsize=(6.7, 3.3), facecolor=V.BG)
    ax.axhspan(np.degrees(low), np.degrees(high), color=V.TEAL, alpha=.065)
    ax.plot(np.degrees(angles), degrees, color=V.RED, lw=2, ls="--", label="几何延拓 / 超限段仅示意")
    ax.plot(np.degrees(angles), np.where(inside, degrees, np.nan), color=V.TEAL, lw=2.5,
            label="J1 限位内的连续角度")
    for limit in (low, high):
        ax.axhline(np.degrees(limit), color=V.RED, ls=":", lw=1.3)
    ax.text(356, np.degrees(high) + 12, "+170°", color=V.RED, ha="right", fontsize=11)
    ax.text(356, np.degrees(low) + 12, "−170°", color=V.RED, ha="right", fontsize=11)
    ax.set_xlim(0, 360)
    ax.set_ylim(-210, max(360, degrees.max() + 20))
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("臂型角 ψ / °", fontsize=11)
    ax.set_ylabel("连续展开的 J1 / °", fontsize=11)
    V.axes_style(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.subplots_adjust(left=.12, right=.97, top=.97, bottom=.18)
    image = V.plot_image(fig)
    plt.close(fig)
    return image


def figure(data):
    image = Image.new("RGB", (1440, 1320), V.BG)
    V.text(image, (48, 25), "IK / EXISTENCE, CONTINUITY, JOINT LIMITS", 18, V.MUTED, mono=True)
    V.text(image, (48, 69), "逐点有解 ≠ 连续整圈。", 46, bold=True)
    V.text(image, (50, 140), "iiwa 14 · 同一个固定末端位姿 · 实际关节限位 · 0.25° 间隔扫描", 23, V.MUTED)
    summary = data["summary"]
    V.metric(image, (54, 211), f"{summary['sample_count']} / {summary['sample_count']}", "采样臂型角有合法逆解", size=38)
    V.metric(image, (545, 211), "0", "本次连接图中的连续整圈路径", size=38)
    total = np.degrees(data["geometric_winding_report"]["joint1_total_travel_rad"])
    V.metric(image, (1035, 211), f"{total:.0f}°", "J1 总行程 / 绕行需 360°", size=38)
    V.inset(image, _branch_plot(data), (48, 326, 1344, 449), rounded=0)
    V.text(image, (60, 785), "青色：该分支合法段   橙色：当前动画分支 *   红点：区间端点   淡蓝：原先 6°–156° 平稳展示段", 19, V.MUTED)
    V.rule(image, (48, 834, 1392, 834))
    V.text(image, (60, 855), "几何原因：肘圆的俯视投影围住肩轴", 24, bold=True)
    V.text(image, (682, 855), "连续展开 J1，不能用取模掩盖越限", 24, bold=True)
    V.inset(image, _projection_plot(data), (48, 905, 570, 315), rounded=0)
    V.inset(image, _joint_plot(data), (650, 905, 742, 315), rounded=0)
    V.text(image, (60, 1234), "同一肩部分支绕圆一圈，J1 也要累计转一圈。", 20, V.MUTED)
    V.text(image, (682, 1234), "当前目标需要 360°，模型允许约 340°。", 20, V.MUTED)
    V.rule(image, (48, 1276, 1392, 1276))
    V.text(image, (48, 1290), "结论限定当前目标与模型；6°–156° 不是运动极限。采样连接图与几何绕转证据分别记录在 JSON。", 17, V.MUTED)
    V.save(image, SHOW / "swivel_feasibility.png")


def _joint_meter(image, value, low, high):
    draw = ImageDraw.Draw(image)
    x0, x1, y = 861, 1230, 392
    draw.rounded_rectangle((x0, y - 4, x1, y + 4), radius=4, fill="#DDE8E7")
    draw.line((x0, y - 13, x0, y + 13), fill=V.RED, width=3)
    draw.line((x1, y - 13, x1, y + 13), fill=V.RED, width=3)
    x = x0 + (x1 - x0) * (value - low) / (high - low)
    draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=V.ORANGE, outline="white", width=2)
    V.text(image, (854, 417), f"{np.degrees(low):.0f}°", 18, V.RED)
    V.text(image, (1179, 417), f"+{np.degrees(high):.0f}°", 18, V.RED)


def animation(data):
    source = data["demonstration"]
    demo = dict(source)
    demo["geometry"] = {**data["geometry"], "elbows_m": demo["actual_elbows_m"]}
    studio = Studio("iiwa14", VIEW[2], VIEW[3])
    camera = _camera(studio, demo)
    annotation = _annotation(studio, camera, demo)
    project = _projector(studio, camera)
    joints = []
    for q in demo["qs"]:
        studio.arm.set_q(q)
        joints.append(studio.arm.world_lines()[0][6].copy())
    joint_pixels = project(joints)
    low, high = np.array(data["joint_limits"]["low_rad"])[6], np.array(data["joint_limits"]["high_rad"])[6]
    blocked = source["first_blocked_raw_candidate"]
    blocked_angle = source["sampling"]["first_blocked_sample_deg"]
    blocked_q7 = np.degrees(blocked["q"][6])
    jump = source["nearest_alternative_dq_norm_rad"]
    template = Image.new("RGB", (1280, 800), V.BG)
    V.text(template, (40, 24), "IK / POINTWISE SOLUTIONS DO NOT MAKE A CONTINUOUS PATH", 16, V.MUTED, mono=True)
    V.text(template, (40, 66), "每个角度有解，为什么还会停住？", 38, bold=True)
    V.text(template, (42, 124), "iiwa 14 · 同一目标位姿 · 沿同一分支继续转动 · 到限位前停止", 21, V.MUTED)
    V.text(template, (852, 177), "S-R-S · 臂型角闭式逆解", 22, V.TEAL, bold=True)
    V.text(template, (852, 279), "J7 / 腕部关节角", 20, V.MUTED)
    V.text(template, (852, 487), f"再到 ψ {blocked_angle:.2f}° 时", 21, V.MUTED)
    V.text(template, (852, 522), f"当前支路 J7 {blocked_q7:.2f}°", 23, V.RED)
    V.text(template, (852, 558), "超出 −175° 下限", 21, V.RED)
    V.text(template, (852, 610), "此时仍有其他合法解，但需换支", 20)
    V.text(template, (852, 647), f"最近替代解 Δq ≈ {jump:.2f} rad", 23, V.ORANGE)
    V.text(template, (852, 685), "不能将跳变画成连续自运动", 20, V.MUTED)
    V.text(template, (40, 744), "前 9 秒：同支运动 → 后 3 秒：限位前停住 → 重播时直接切回起点", 20, V.MUTED)
    V.text(template, (40, 778), "All shown configurations are valid / replay uses a cut, not a continuous return", 13, V.MUTED, mono=True)
    frames = []
    target = np.asarray(data["target_pose"])
    moving = demo["sampling"]["moving_frames"]
    for k, q in enumerate(demo["qs"]):
        image = template.copy()
        frame = _annotate(studio.render(q, target, camera, axis_len=.055, ball=.009), annotation, k)
        x, y = joint_pixels[k]
        draw = ImageDraw.Draw(frame)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline=V.RED, width=2)
        V.text(frame, (x + 15, y + 8), "J7", 19, V.RED, bold=True)
        V.inset(image, frame, VIEW)
        psi = np.degrees(demo["parameter_values_rad"][k])
        V.text(image, (852, 224), f"ψ {psi:.2f}°", 34, V.TEAL, mono=True)
        value = q[6]
        V.text(image, (852, 321), f"{np.degrees(value):+.2f}°", 37, V.RED if k >= moving else V.TEAL, mono=True)
        _joint_meter(image, value, low, high)
        if k >= moving:
            V.chip(image, (60, 658), "限位前停住 / 下一步将越限", V.RED, size=18)
        else:
            V.chip(image, (60, 658), "同一分支 · 固定 TCP · 0 次迭代", V.TEAL, size=18)
        V.text(image, (40, 723),
               f"FK / Δp {demo['pos_error_m'][k]:.1e} m   ΔR {demo['rot_error_rad'][k]:.1e} rad",
               14, V.TEAL, mono=True)
        frames.append(np.asarray(image))
    studio.close()
    _encode(frames, SHOW / "swivel_iiwa14", demo["summary"]["fps"])
    return dict(lookat=list(camera.lookat), distance=camera.distance,
                azimuth=camera.azimuth, elevation=camera.elevation, viewport=list(VIEW))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--stills", action="store_true")
    args = parser.parse_args(argv)
    SHOW.mkdir(parents=True, exist_ok=True)
    path = SHOW / "swivel_metrics.json"
    if args.reuse_data:
        data = json.loads(metric_path(path.name).read_text(encoding="utf-8"))
    else:
        from .swivel_data import collect_swivel
        data = collect_swivel()
    figure(data)
    sources = (Path(__file__), ROOT / "scripts/showcase/pose_gallery.py",
               ROOT / "scripts/showcase/portfolio.py", ROOT / "scripts/showcase/portfolio_style.py",
               ROOT / "scripts/showcase/scene.py", ROOT / "scripts/showcase/motion_gallery.py")
    data["render"] = {
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "scope": "A target-specific explanation. 9s advance + 3s hold; replay explicitly resets by a cut, not a continuous seam. Unwrapped J1 beyond limits is a plot only, never a valid animation configuration.",
    }
    if not args.stills:
        data["render"]["camera"] = animation(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
