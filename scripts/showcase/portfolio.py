"""Render the homepage portfolio from fresh, verifiable IK experiments.

    python -m scripts.showcase.portfolio
    python -m scripts.showcase.portfolio --reuse-data

The generated JSON is the source of every numerical label. A hardware-free
MuJoCo EGL context is sufficient; use MUJOCO_GL=glfw on a desktop if needed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from ik import SrsSolver
from ik.kinematics import rot_log
from . import scene as S
from . import portfolio_style as V
from .common import get_arm
from .style import SHOW
from .paths import metric_path

NAMES = {"ur5e": "UR5e", "lite6": "Lite 6", "iiwa14": "iiwa 14", "panda": "Panda"}


def studio_arm(key):
    """Compile an isolated studio model; leave shared cached models intact."""
    arm = get_arm(key)
    xml = arm.xml_path if hasattr(arm, "xml_path") else arm.spec.xml
    spec = mujoco.MjSpec.from_file(str(xml))
    for light in spec.lights:
        light.diffuse[:] = (0.25, 0.25, 0.24)
        light.ambient[:] = (0.04, 0.04, 0.04)
        light.specular[:] = (0.10, 0.10, 0.10)
        light.castshadow = False
    sky = spec.add_texture(name="portfolio_sky")
    sky.type = mujoco.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mujoco.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1[:] = (0.985, 0.990, 0.998)
    sky.rgb2[:] = (0.940, 0.958, 0.978)
    sky.width, sky.height = 256, 1536
    for name, pos, direction, diffuse in (
        ("portfolio_key", (2, -3, 3), (-2, 3, -3), (0.28, 0.28, 0.28)),
        ("portfolio_fill", (-2, -1, 2), (2, 1, -2), (0.24, 0.24, 0.25)),
        ("portfolio_rim", (0, 2, 3), (0, -2, -3), (0.12, 0.13, 0.12)),
    ):
        light = spec.worldbody.add_light(name=name)
        light.pos[:] = pos
        light.dir[:] = direction
        light.diffuse[:] = diffuse
        light.castshadow = False
    arm.model = spec.compile()
    arm.data = mujoco.MjData(arm.model)
    arm.model.vis.headlight.ambient[:] = (0.35, 0.35, 0.35)
    arm.model.vis.headlight.diffuse[:] = (0.44, 0.44, 0.44)
    arm.model.vis.headlight.specular[:] = (0.12, 0.12, 0.12)
    arm.model.vis.global_.offwidth = 1600
    arm.model.vis.global_.offheight = 1200
    # Use the same photographic finish while retaining manufacturers' colours.
    arm.model.mat_shininess[:] = 0.25
    arm.model.mat_specular[:] = 0.18
    if key in ("panda", "lite6"):
        # Keep pale shells readable against the bright studio without clipping
        # large white highlights. This affects display materials, not the model.
        pale_material = np.min(arm.model.mat_rgba[:, :3], axis=1) > 0.65
        pale_geom = np.min(arm.model.geom_rgba[:, :3], axis=1) > 0.65
        arm.model.mat_rgba[pale_material, :3] *= 0.85
        arm.model.geom_rgba[pale_geom, :3] *= 0.85
    return arm


class Studio:
    def __init__(self, key, width=640, height=540):
        self.arm = studio_arm(key)
        self.scene = S.RobotScene(key, width, height, arm=self.arm)

    def camera(self, qs, points=(), margin=1.20):
        return self.scene.auto_camera(sweep_qs=qs, extra_points=points,
                                      azimuth=145, elevation=-18, margin=margin)

    def render(self, q, T, cam, extra=(), *, axis_len=0.09, ball=0.017):
        def target(scn):
            S.add_target(scn, T, axis_len=axis_len, ball=ball)
        return self.scene.render(q=q, cam=cam, floor=False,
                                 overlays=[*extra, target])

    def close(self):
        self.scene.close()


def target_extent(T):
    return np.vstack([T[:3, 3], *[T[:3, 3] + 0.12 * v for v in T[:3, :3].T]])


def self_motion(data, n=72):
    """Pick a continuous, legal arm-angle interval and traverse it both ways.

    An elbow circle is a geometric locus. Joint limits may exclude parts of it,
    so no absent solution is clipped or interpolated to fabricate this loop.
    """
    arm = get_arm("iiwa14")
    sv = SrsSolver(arm)
    T = np.asarray(data["robots"]["iiwa14"]["target_pose"])
    runs, current = [], []
    for psi in np.linspace(0, 2 * np.pi, 361):
        sols = sv.solve_at_psi(T, float(psi), tol_pos=1e-8, tol_rot=1e-8)
        sols = [s for s in sols if np.all(s.q >= arm.joint_low - 1e-9)
                and np.all(s.q <= arm.joint_high + 1e-9)]
        if not sols:
            if current:
                runs.append(current)
                current = []
            continue
        ref = current[-1][1] if current else np.asarray(data["robots"]["iiwa14"]["q_ref"])
        best = min(sols, key=lambda s: np.linalg.norm(s.q - ref))
        if current and np.linalg.norm(best.q - ref) > 0.16:
            runs.append(current)
            current = []
        current.append((float(psi), best.q.copy()))
    if current:
        runs.append(current)
    run = max(runs, key=len)
    if len(run) < 20:
        raise RuntimeError("No sufficiently long continuous, limit-valid self-motion interval")
    # Cap the interval at 120 degrees to make an easily readable looping motion.
    span = min(len(run) - 1, 120)
    start = (len(run) - span - 1) // 2
    lo, hi = run[start][0], run[start + span][0]
    phases = (1 - np.cos(np.linspace(0, 2 * np.pi, n, endpoint=False))) / 2
    psis = lo + (hi - lo) * phases
    qs, pos, rot, margins = [], [], [], []
    prev = run[start][1]
    for psi in psis:
        sols = sv.solve_at_psi(T, float(psi), tol_pos=1e-8, tol_rot=1e-8)
        best = min(sols, key=lambda s: np.linalg.norm(s.q - prev))
        arm.set_q(best.q)
        R, p = arm.tcp_pose()
        pos.append(float(np.linalg.norm(p - T[:3, 3])))
        rot.append(float(np.linalg.norm(rot_log(T[:3, :3] @ R.T))))
        margins.append(float(np.min(np.minimum(best.q - arm.joint_low, arm.joint_high - best.q))))
        qs.append(best.q.copy())
        prev = best.q
    qs = np.asarray(qs)
    jump = float(np.max(np.linalg.norm(np.diff(np.vstack([qs, qs[:1]]), axis=0), axis=1)))
    if jump > 0.30 or min(margins) < -1e-9:
        raise RuntimeError("Self-motion failed the continuity/limit check")
    geo = sv._elbow_circle(T)
    _, _, W, center, h, v, rho = geo
    circle = center + rho * (np.cos(np.linspace(0, 2 * np.pi, 100))[:, None] * h
                              + np.sin(np.linspace(0, 2 * np.pi, 100))[:, None] * v)
    report = dict(target_pose=T.tolist(), qs=qs.tolist(), psi_rad=psis.tolist(),
                  pos_error_m=pos, rot_error_rad=rot, min_joint_margin_rad=min(margins),
                  max_dq_norm_rad=jump, max_pos_error_m=max(pos), max_rot_error_rad=max(rot),
                  psi_interval_rad=[lo, hi], sampling="continuous legal interval, cosine forward/backward")
    return report, (sv.S, W, center, h, v, rho, circle)


def circle_overlay(geo, psi):
    shoulder, wrist, center, h, v, rho, _ = geo
    elbow = center + rho * (np.cos(psi) * h + np.sin(psi) * v)
    normal = wrist - shoulder
    normal /= np.linalg.norm(normal)
    def draw(scn):
        S.add_circle(scn, center, normal, rho, [0.15, 0.58, 0.50, 0.48], width=0.003)
        S.add_capsule(scn, shoulder, wrist, 0.002, [0.47, 0.56, 0.51, 0.4])
        S.add_sphere(scn, elbow, 0.016, [0.15, 0.58, 0.50, 1])
        S.add_capsule(scn, center, elbow, 0.0025, [0.15, 0.58, 0.50, 0.7])
    return draw


def hero(data, motion, geo):
    qs = np.asarray(motion["qs"])
    T = np.asarray(motion["target_pose"])
    st = Studio("iiwa14", 750, 670)
    points = np.vstack([geo[-1], target_extent(T)])
    cam = st.camera(qs[::6], points, margin=1.18)
    im = Image.new("RGB", (1440, 850), V.BG)
    V.text(im, (56, 40), "ROBOTICS / INVERSE KINEMATICS", 19, V.MUTED, mono=True)
    V.rule(im, (56, 80, 1384, 80))
    V.text(im, (56, 117), "同一位姿，", 60, bold=True)
    V.text(im, (56, 199), "无数种可能。", 60, bold=True)
    V.text(im, (60, 294), "从机构几何到稳定运动", 28, V.MUTED)
    V.text(im, (60, 342), "闭式逆解 · 冗余参数化 · 阻尼迭代", 22, V.MUTED)
    V.chip(im, (60, 414), "4 台真实模型")
    V.chip(im, (233, 414), "6 / 7 自由度", V.INK)
    V.rule(im, (60, 494, 577, 494))
    V.metric(im, (60, 534), "0", "闭式迭代 / 次")
    V.metric(im, (270, 534), f"{motion['max_pos_error_m']:.1e}", "本段最大位置误差 / m", size=32)
    V.text(im, (60, 660), "末端位姿固定，肘部沿可行圆弧运动。", 23)
    V.text(im, (60, 702), "所有构型经过 FK、关节限位与连续性复核。", 20, V.MUTED)
    frame = st.render(qs[18], T, cam, [circle_overlay(geo, motion["psi_rad"][18])])
    V.inset(im, frame, (634, 109, 750, 670))
    V.chip(im, (665, 129), "KUKA iiwa 14 / 7R", V.INK)
    V.rule(im, (56, 799, 1384, 799))
    V.text(im, (56, 819), "PYTHON  /  NUMPY  /  MUJOCO                         GEOMETRY → SOLUTIONS → MOTION", 16, V.MUTED, mono=True)
    st.close()
    V.save(im, SHOW / "portfolio_hero.png")


def structure(data):
    im = V.plate("02", "解法，首先由结构决定。", "同一套模型接口，四种机构：闭式解用于枚举，阻尼迭代用于局部求解。", 880)
    facts = {
        "ur5e": ("6R / 平行轴", "J2–J4 平行", "平面几何 · 闭式分支", V.TEAL),
        "lite6": ("6R / 球形腕", "J4–J6 汇交", "位置 / 姿态解耦", V.TEAL),
        "iiwa14": ("7R / S–R–S", "球形肩 + 球形腕", "臂型角 ψ · 连续冗余", V.BLUE),
        "panda": ("7R / 偏置腕", "腕轴偏移 88 mm", "几何种子 + DLS 抛光", V.ORANGE),
    }
    for i, key in enumerate(NAMES):
        x = 48 + i * 341
        d = data["robots"][key]
        q, T = np.asarray(d["q_ref"]), np.asarray(d["target_pose"])
        st = Studio(key, 318, 355)
        cam = st.camera([q], target_extent(T), margin=1.18)
        st.arm.set_q(q)
        anchors, axes = st.arm.world_lines()
        indices = {"ur5e": (1, 2, 3), "lite6": (3, 4, 5),
                   "iiwa14": (0, 1, 2, 4, 5, 6), "panda": (0, 1, 2)}[key]
        def structural_axes(scn, anchors=anchors.copy(), axes=axes.copy(), indices=indices):
            for j in indices:
                S.add_capsule(scn, anchors[j] - .06 * axes[j], anchors[j] + .06 * axes[j],
                              .003, [.08, .58, .50, .85])
        V.inset(im, st.render(q, T, cam, [structural_axes]), (x, 236, 318, 355))
        st.close()
        V.text(im, (x + 12, 199), NAMES[key], 26, bold=True)
        tag, condition, method, color = facts[key]
        V.chip(im, (x + 12, 608), tag, color, 18)
        V.text(im, (x + 12, 666), condition, 24)
        V.text(im, (x + 12, 708), method, 21, V.MUTED)
    V.text(im, (60, 782), "图中琥珀色球与 RGB 轴表示目标位姿。Panda 的混合解法是本项目的实现选择。", 20, V.MUTED)
    V.save(im, SHOW / "portfolio_structure.png")


def branches(data):
    d = data["robots"]["ur5e"]
    sols = [s for s in d["branches"]["solutions"] if s["valid"]]
    T = np.asarray(d["target_pose"])
    qs = [np.asarray(s["q"]) for s in sols]
    im = V.plate("03", "同一个目标，逐支验证。", f"UR5e / 当前目标返回 {len(qs)} 个合法闭式解；数量来自求解，精度来自 FK 复核。", 1040)
    st = Studio("ur5e", 318, 295)
    cam = st.camera(qs, target_extent(T), margin=1.22)
    for i, (q, sol) in enumerate(zip(qs, sols)):
        x, y = 48 + i % 4 * 341, 226 + i // 4 * 356
        if i >= 8:
            break
        V.inset(im, st.render(q, T, cam), (x, y, 318, 295))
        V.chip(im, (x + 10, y + 10), f"{i + 1:02d} / 0 iterations", V.INK, 16)
        V.text(im, (x + 10, y + 309), f"Δp {sol['pos_err_m']:.1e} m", 17, V.TEAL, mono=True)
        V.text(im, (x + 10, y + 336), f"ΔR {sol['rot_err_rad']:.1e} rad", 17, V.MUTED, mono=True)
    st.close()
    V.text(im, (60, 956), "6 轴的离散构型用于分支选择；7 轴返回的有限样本用于探索连续解集。", 21, V.MUTED)
    V.save(im, SHOW / "portfolio_branches.png")


def polish(data):
    d = data["robots"]["panda"]
    sols = [s for s in d["branches"]["solutions"]
            if s["valid"] and s["joint_margin_rad"] > 0.05]
    sol = max(sols, key=lambda s: s["seed_pos_err_m"])
    q, seed, T = np.asarray(sol["q"]), np.asarray(sol["seed_q"]), np.asarray(d["target_pose"])
    im = V.plate("03B", "几何给出初值，迭代修正位姿。", "Panda / 同一目标、同一几何种子：直接回代种子与最终解，记录真实抛光迭代数。", 875)
    st = Studio("panda", 440, 450)
    cam = st.camera([seed, q], target_extent(T), margin=1.22)
    st.arm.set_q(seed)
    seed_p = st.arm.tcp_pose()[1].copy()
    def gap(scn):
        S.add_sphere(scn, seed_p, .014, [.81, .43, .21, 1])
        S.add_capsule(scn, seed_p, T[:3, 3], .004, [.81, .43, .21, .8])
    for x, cfg, label, color in ((48, seed, "01 / 几何种子", V.ORANGE),
                                  (510, q, "02 / DLS 抛光", V.TEAL)):
        frame = st.render(cfg, T, cam, [gap] if x == 48 else [])
        if x == 48:
            # A projected diagnostic line stays readable when the mesh hides
            # the world-space segment. Both endpoints are measured FK points.
            frame = Image.fromarray(frame)
            u, v = S.project_points(st.arm.model, st.arm.data, cam,
                                    [seed_p, T[:3, 3]], width=440, height=450)
            endpoints = np.column_stack([(u + 1) * 220, (1 - v) * 225])
            draw = ImageDraw.Draw(frame)
            a, b = endpoints
            for t0 in np.arange(0, 1, .12):
                p0 = a + t0 * (b - a)
                p1 = a + min(t0 + .07, 1) * (b - a)
                draw.line([tuple(p0), tuple(p1)], fill=V.ORANGE, width=3)
            for px, py in endpoints:
                draw.ellipse((px - 5, py - 5, px + 5, py + 5),
                             fill=V.WHITE, outline=V.ORANGE, width=3)
        V.inset(im, frame, (x, 226, 440, 450))
        V.chip(im, (x + 18, 245), label, color, 18)
    st.close()
    V.text(im, (66, 704), f"Δp {sol['seed_pos_err_m']:.2e} m", 21, V.ORANGE, mono=True)
    V.text(im, (66, 741), f"ΔR {sol['seed_rot_err_rad']:.2e} rad", 20, V.MUTED, mono=True)
    V.text(im, (528, 704), f"Δp {sol['pos_err_m']:.2e} m", 21, V.TEAL, mono=True)
    V.text(im, (528, 741), f"ΔR {sol['rot_err_rad']:.2e} rad", 20, V.MUTED, mono=True)
    V.metric(im, (998, 310), str(sol["iters"]), "记录的求解迭代次数", size=58)
    V.metric(im, (998, 458), f"{np.degrees(sol['joint_margin_rad']):.1f}°", "最终最小限位裕度", size=35)
    V.text(im, (998, 616), "橙色连线表示种子", 22, V.MUTED)
    V.text(im, (998, 653), "末端与目标间的误差。", 22, V.MUTED)
    V.save(im, SHOW / "portfolio_polish.png")


def tracking(data):
    import matplotlib.pyplot as plt
    im = V.plate("04", "位置到了，还要走得连续。", "逐点求解笛卡尔直线，并选取附近构型；同时检查位姿误差、相邻关节变化和限位裕度。", 1050)
    d = data["robots"]["panda"]["trajectory"]
    qs, Ts = np.asarray(d["qs"]), np.asarray(d["Ts"])
    points, curved = np.asarray(d["tcp_points_m"]), np.asarray(d["curved_points_m"])
    def lines(scn):
        S.add_capsule(scn, Ts[0, :3, 3], Ts[-1, :3, 3], .004, [0.07, 0.58, 0.45, 1])
        for a, b in zip(curved[:-1], curved[1:]):
            S.add_capsule(scn, a, b, .003, [0.82, 0.43, 0.20, 1])
    st = Studio("panda", 580, 500)
    cam = st.camera(qs[::4], np.vstack([curved, target_extent(Ts[len(qs)//2])]), margin=1.25)
    V.inset(im, st.render(qs[len(qs)//2], Ts[len(qs)//2], cam, [lines]), (48, 223, 580, 500))
    st.close()
    V.chip(im, (70, 243), "Panda / Cartesian tracking", V.INK, 18)
    V.text(im, (60, 747), "绿线：逐点 IK     橙线：关节线性插值", 22, V.MUTED)
    fig, axs = plt.subplots(2, 1, figsize=(5.15, 4.60), facecolor=V.BG)
    t = np.linspace(0, 1, len(qs))
    axs[0].plot(t, 1000 * np.asarray(d["curved_line_deviation_m"]), color=V.ORANGE, lw=2, label="Joint interpolation")
    axs[0].plot(t, 1000 * np.asarray(d["line_deviation_m"]), color=V.TEAL, lw=2, label="Cartesian IK")
    axs[0].set_ylabel("Line deviation / mm", fontsize=11)
    axs[0].set_title("Panda / same start and end configuration", fontsize=10, color=V.MUTED, loc="left")
    axs[0].legend(frameon=False, fontsize=10)
    for key, color in zip(NAMES, [V.BLUE, V.ORANGE, V.TEAL, V.INK]):
        tr = data["robots"][key]["trajectory"]
        axs[1].plot(t[1:], np.degrees(tr["dq_norm_rad"]), color=color, lw=1.8, label=NAMES[key])
    axs[1].set_ylabel("Adjacent step ||Δq|| / deg", fontsize=11)
    axs[1].set_xlabel(f"Path progress ({len(qs)} pose samples)", fontsize=11)
    axs[1].legend(frameon=False, fontsize=9, ncol=4)
    for ax in axs:
        V.axes_style(ax)
        ax.set_xlim(0, 1)
    fig.subplots_adjust(left=.16, right=.96, top=.95, bottom=.12, hspace=.42)
    V.inset(im, V.plot_image(fig), (660, 220, 730, 572))
    plt.close(fig)
    V.rule(im, (60, 818, 1374, 818))
    for i, key in enumerate(NAMES):
        tr = data["robots"][key]["trajectory"]
        pe = max(tr["pos_error_m"])
        re = max(tr["rot_error_rad"])
        margin = min(tr["joint_margin_rad"])
        x = 60 + i * 340
        V.text(im, (x, 845), NAMES[key], 23, bold=True)
        V.text(im, (x, 887), f"max Δp  {pe:.1e} m", 18, V.TEAL, mono=True)
        V.text(im, (x, 917), f"max ΔR  {re:.1e} rad", 18, V.MUTED, mono=True)
        V.text(im, (x, 947), f"limit margin  {np.degrees(margin):.1f}°", 18, V.MUTED, mono=True)
    V.save(im, SHOW / "portfolio_tracking.png")


def singularity(data):
    import matplotlib.pyplot as plt
    d = data["singularity"]
    im = V.plate("05", "接近奇异时，观察数值行为。", "同一目标、同一初值：比较原始更新量，并独立记录位置与姿态误差。", 980)
    q, T = np.asarray(d["q_solution"]), np.asarray(d["target_pose"])
    st = Studio("iiwa14", 422, 490)
    cam = st.camera([q, np.asarray(d["methods"]["dls"]["q_final"])], target_extent(T), 1.28)
    sv = SrsSolver(st.arm)
    geo = sv._elbow_circle(T)
    def target_circle(scn):
        _, _, W, center, _, _, rho = geo
        S.add_circle(scn, center, W - sv.S, rho, [.08, .58, .50, .9], width=.003)
    V.inset(im, st.render(q, T, cam, [target_circle]), (48, 223, 422, 490))
    st.close()
    V.chip(im, (68, 243), "iiwa / near extension", V.INK, 17)
    V.metric(im, (70, 740), f"{1000 * d['elbow_radius_m']:.1f} mm", "目标肘部圆半径", size=30)
    fig, axs = plt.subplots(3, 1, figsize=(7.0, 5.2), facecolor=V.BG, sharex=True)
    colors = {"jt": V.RED, "pinv": V.ORANGE, "dls": V.TEAL}
    names = {"jt": "Jacobian transpose", "pinv": "Pseudoinverse", "dls": "DLS (fixed λ=0.05)"}
    for method, tr in d["methods"].items():
        for ax, field in zip(axs, ["step_norm_rad", "pos_error_m", "rot_error_rad"]):
            values = np.maximum(np.asarray(tr[field]), 1e-16)
            ax.semilogy(np.arange(len(values)), values, color=colors[method], lw=1.8, label=names[method])
    for ax, label in zip(axs, ["Raw ||Δq|| / rad", "Position error / m", "Rotation error / rad"]):
        ax.set_ylabel(label, fontsize=10)
        V.axes_style(ax)
    axs[0].legend(frameon=False, fontsize=9, ncol=3, loc="upper right")
    axs[-1].set_xlabel("Iteration (diagnostic trace)", fontsize=11)
    fig.subplots_adjust(left=.14, right=.97, top=.96, bottom=.10, hspace=.20)
    V.inset(im, V.plot_image(fig), (484, 217, 908, 634))
    plt.close(fig)
    V.text(im, (60, 873), "诊断曲线使用固定阻尼与限位裁剪；生产求解器另有自适应阻尼、步长限制和误差验收。", 21, V.MUTED)
    V.save(im, SHOW / "portfolio_singularity.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-data", action="store_true", help="use the existing experiment JSON")
    args = parser.parse_args(argv)
    SHOW.mkdir(parents=True, exist_ok=True)
    path = SHOW / "portfolio_metrics.json"
    if args.reuse_data:
        data = json.loads(metric_path(path.name).read_text(encoding="utf-8"))
    else:
        from .portfolio_data import collect_data
        data = collect_data()
    motion, geo = self_motion(data)
    data["self_motion"] = motion
    data["experiment"]["render_command"] = "python -m scripts.showcase.portfolio"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    hero(data, motion, geo)
    structure(data)
    branches(data)
    polish(data)
    tracking(data)
    singularity(data)
    print(f"Experiment data: {path}", flush=True)


if __name__ == "__main__":
    main()
