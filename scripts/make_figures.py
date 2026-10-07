"""Generate figures for docs/ik_survey.md.

Produces
  * ``docs/img/srs_arm_angle.png`` : S-R-S arm, elbow circle, arm angle psi
  * ``docs/img/singularity.png``   : sigma_min distribution + elbow-circle
    degeneracy near the extended-arm singularity
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

# CJK fonts
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "WenQuanYi Zen Hei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"
OUT = IMG_DIR / "srs_arm_angle.png"
OUT_SING = IMG_DIR / "singularity.png"


def make_singularity_figure() -> None:
    """sigma_min distribution and elbow-circle degeneracy (measured)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from ik import PandaArm
    from ik.kinematics import line_intersection
    from ik.analytic import elbow_circle

    arm = PandaArm()
    S = arm.geometry().shoulder_center
    geom = arm.geometry()

    # --- sigma_min over random configurations ---
    rng = np.random.default_rng(0)
    sm = []
    for _ in range(600):
        q = rng.uniform(arm.joint_low, arm.joint_high)
        arm.set_q(q)
        sm.append(np.linalg.svd(arm.tcp_jacobian(), compute_uv=False)[-1])
    sm = np.array(sm)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.hist(sm, bins=40, color="#4C72B0", alpha=0.85)
    ax1.axvline(np.median(sm), color="#C44E52", ls="--",
                label=f"中位数 {np.median(sm):.3f}")
    ax1.axvline(np.percentile(sm, 1), color="#8172B3", ls=":",
                label=f"1% 分位 {np.percentile(sm, 1):.3f}")
    ax1.set_xlabel(r"$\sigma_{\min}(J)$")
    ax1.set_ylabel("配置数量")
    ax1.set_title(r"(a) 随机位形的 $\sigma_{\min}$ 分布（近奇异很常见）",
                  fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    # --- elbow-circle radius vs extension ---
    qe = np.array([-1.8625, -0.3658, -2.8635, -0.467, -0.4567, 0.3818, 0.7716])
    Lmax = geom.d_se + geom.d_ew
    ratios, rhos = [], []
    for dq4 in np.linspace(0.0, 3.0, 40):
        q = qe.copy()
        q[3] += dq4
        arm.set_q(q)
        a, ax = arm.world_lines()
        W = line_intersection(a[4], ax[4], a[5], ax[5])
        L = np.linalg.norm(W - S)
        res = elbow_circle(W, S, 0.0, geom)
        if res is None:
            continue
        ratios.append(L / Lmax)
        rhos.append(res[4])

    ax2.plot(ratios, rhos, "-o", ms=3, color="#55A868")
    ax2.axvline(1.0, color="#C44E52", ls="--", lw=1)
    ax2.text(0.985, max(rhos) * 0.55, "手臂完全伸直\n(肘部奇异)",
             color="#C44E52", fontsize=9, ha="right")
    ax2.set_xlabel(r"$|W-S| \,/\, (d_{se}+d_{ew})$")
    ax2.set_ylabel(r"肘部圆半径 $\rho$  [m]")
    ax2.set_title(r"(b) 接近伸直时 $\rho \to 0$，冗余（臂型角）塌缩", fontsize=12)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_SING, dpi=200, bbox_inches="tight")
    print(f"saved: {OUT_SING}")


def main() -> None:
    # --- S-R-S geometry (illustrative values) ------------------------------ #
    d_se, d_ew, L = 1.30, 1.20, 2.00
    cos_a = (d_se ** 2 + L ** 2 - d_ew ** 2) / (2 * d_se * L)
    alpha = np.arccos(cos_a)
    rho = d_se * np.sin(alpha)

    S = np.array([0.0, 0.0, 0.0])
    W = np.array([L, 0.0, 0.0])
    u = np.array([1.0, 0.0, 0.0])
    h = np.array([0.0, 1.0, 0.0])       # reference direction (psi = 0)
    v = np.cross(u, h)
    n = S + d_se * cos_a * u             # elbow-circle center

    def E_of(psi):
        return n + rho * (np.cos(psi) * h + np.sin(psi) * v)

    psi = np.deg2rad(58.0)
    E = E_of(psi)

    # sample the circle
    ts = np.linspace(0, 2 * np.pi, 200)
    circ = np.array([n + rho * (np.cos(t) * h + np.sin(t) * v) for t in ts])

    fig = plt.figure(figsize=(11, 4.6))

    # ---------------- (a) 3D view ---------------- #
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(circ[:, 0], circ[:, 1], circ[:, 2], color="#4C72B0", lw=2,
            label="肘部圆 (elbow circle)")
    ax.plot([S[0], W[0]], [S[1], W[1]], [S[2], W[2]], "--", color="0.5",
            lw=1.2, label=r"$\hat u$ (SW 轴)")
    for P, Q, c in [(S, E, "#C44E52"), (E, W, "#55A868")]:
        ax.plot([P[0], Q[0]], [P[1], Q[1]], [P[2], Q[2]], color=c, lw=3)
    ax.scatter(*S, color="k", s=60)
    ax.scatter(*W, color="k", s=60)
    ax.scatter(*E, color="#C44E52", s=70, zorder=5)
    ax.scatter(*n, color="0.4", s=30, marker="x")
    ax.text(*S, " $S$", fontsize=13)
    ax.text(*W, " $W$", fontsize=13)
    ax.text(*E, " $E$", color="#C44E52", fontsize=13)
    ax.text(*(n + 0.1 * h), " $n$", color="0.3", fontsize=12)

    ax.set_xlabel("x"), ax.set_ylabel("y"), ax.set_zlabel("z")
    ax.set_title("(a) S-R-S 结构与肘部圆", fontsize=12)
    ax.view_init(elev=22, azim=-58)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect((1.3, 1, 1))

    # ---------------- (b) view along u ---------------- #
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_aspect("equal")
    ax2.plot(circ[:, 1], circ[:, 2], color="#4C72B0", lw=2)
    # center & radii
    ax2.scatter([n[1]], [n[2]], color="0.4", marker="x", s=60)
    ax2.plot([n[1], E[1]], [n[2], E[2]], color="#C44E52", lw=1.5)
    ax2.text((n[1] + E[1]) / 2 + 0.03, (n[2] + E[2]) / 2, r"$\rho$",
             color="#C44E52", fontsize=14)
    # reference direction (psi=0)
    ax2.annotate("", xy=(n[1] + rho * 1.25, n[2]), xytext=(n[1], n[2]),
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=1.4))
    ax2.text(n[1] + rho * 1.28, n[2] + 0.03, r"$\hat h\;(\psi=0)$",
             color="0.3", fontsize=11)
    # elbow point
    ax2.scatter([E[1]], [E[2]], color="#C44E52", s=80, zorder=5)
    ax2.text(E[1] + 0.06, E[2] + 0.05, r"$E(\psi)$", color="#C44E52",
             fontsize=13)
    # psi arc
    arc = np.linspace(0, psi, 60)
    ax2.plot(n[1] + 0.55 * np.cos(arc), n[2] + 0.55 * np.sin(arc),
             color="#8172B3", lw=2)
    ax2.text(n[1] + 0.72 * np.cos(psi / 2), n[2] + 0.72 * np.sin(psi / 2),
             r"$\psi$", color="#8172B3", fontsize=16)

    lim = rho * 1.8
    ax2.set_xlim(-lim * 0.75, lim * 1.3)
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel(r"$\hat h$")
    ax2.set_ylabel(r"$\hat v$")
    ax2.set_title(r"(b) 沿 $\hat u$ 轴看：臂型角 $\psi$ 的定义", fontsize=12)
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"saved: {OUT}")

    make_singularity_figure()


if __name__ == "__main__":
    main()
