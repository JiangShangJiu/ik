"""Visualise the exact S-R-S closed form on the KUKA iiwa 14 (MuJoCo).

The iiwa is a *true* spherical-shoulder / spherical-wrist arm, so a single pose
admits a whole 1-D family of exact solutions parameterised by the arm angle
``psi``.  Every member of that family is computed in closed form -- no iteration.

Top row : MuJoCo renders of four **distinct** exact solutions that all place the
          flange at the *same* target pose (red marker).  The arm visibly swings
          through its redundancy while the TCP stays put.
Bottom  : the elbow circle ``E(psi)`` in 3-D, with the four elbow positions
          marked, plus the fixed shoulder centre and wrist centre.

Output: ``docs/img/iiwa_closed_form.png``
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

from ik import SrsSolver, load_robot  # noqa: E402

IMG_DIR = ROOT / "docs" / "img"
N_SOL = 4


def basis_from_z(z):
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def add_sphere(scene, p, radius, rgba):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius]), p, np.eye(3).flatten(), rgba,
    )
    scene.ngeom += 1


def add_line(scene, p0, p1, radius, rgba):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-9:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, L / 2, 0.0]), 0.5 * (p0 + p1),
        basis_from_z(d / L).flatten(), rgba,
    )
    scene.ngeom += 1


def render_solution(arm, q, tcp, circle_pts, S, W, width=430, height=400):
    arm.set_q(q)
    model, data = arm.model, arm.data
    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 1.7
    cam.lookat[:] = [0.0, 0.0, 0.35]
    cam.azimuth = 125.0
    cam.elevation = -18.0
    renderer.update_scene(data, cam)
    scn = renderer.scene

    # faint elbow circle
    for i in range(len(circle_pts) - 1):
        add_line(scn, circle_pts[i], circle_pts[i + 1], 0.003,
                 np.array([0.1, 0.5, 0.9, 0.55]))
    add_sphere(scn, S, 0.018, np.array([0.9, 0.1, 0.1, 0.9]))
    add_sphere(scn, W, 0.018, np.array([0.1, 0.2, 0.9, 0.9]))
    add_sphere(scn, tcp, 0.022, np.array([1.0, 0.0, 0.0, 1.0]))

    img = renderer.render().copy()
    renderer.close()
    return img


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    arm = load_robot("iiwa14")
    solver = SrsSolver(arm, n_psi=12)

    rng = np.random.default_rng(3)
    q_true = rng.uniform(arm.joint_low * 0.7, arm.joint_high * 0.7)
    arm.set_q(q_true)
    T = arm.tcp_transform()
    R_t, p_t = arm.tcp_pose()

    sols = solver.solve(T, max_solutions=32)
    # spread the picks over psi for visual variety
    picks = [sols[i] for i in np.linspace(0, len(sols) - 1, N_SOL).astype(int)]

    geo = solver._elbow_circle(T)
    n, h, v, rho = geo[3], geo[4], geo[5], geo[6]
    W = geo[2]
    psi = np.linspace(0, 2 * np.pi, 240)
    circle = n + rho * (np.cos(psi)[:, None] * h + np.sin(psi)[:, None] * v)

    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(2, N_SOL, height_ratios=[1.35, 1.0], hspace=0.28)

    elbows = []
    for k, sol in enumerate(picks):
        arm.set_q(sol.q)
        a, ax = arm.world_lines()
        d4 = ax[3] / np.linalg.norm(ax[3])
        E = a[3] + ((solver.S - a[3]) @ d4) * d4
        elbows.append(E)

        aximg = fig.add_subplot(gs[0, k])
        img = render_solution(arm, sol.q, p_t, circle, solver.S, W)
        aximg.imshow(img)
        aximg.axis("off")
        aximg.set_title(f"exact solution {k + 1}\nresidual {sol.pos_err:.1e} m, "
                        f"iterations 0", fontsize=10)

    ax3d = fig.add_subplot(gs[1, :], projection="3d")
    ax3d.plot(circle[:, 0], circle[:, 1], circle[:, 2], color="#1f77b4", lw=2,
              label="elbow circle $E(\\psi)$")
    for k, E in enumerate(elbows):
        ax3d.scatter(*E, color="#2ca02c", s=55, zorder=5,
                     label="closed-form elbows" if k == 0 else None)
        ax3d.plot([solver.S[0], E[0], W[0]], [solver.S[1], E[1], W[1]],
                  [solver.S[2], E[2], W[2]], color="#2ca02c", alpha=0.35, lw=1.2)
    ax3d.scatter(*solver.S, color="red", s=90, marker="*", zorder=6,
                 label="shoulder centre $S$")
    ax3d.scatter(*W, color="blue", s=70, zorder=6, label="wrist centre $W$")
    ax3d.scatter(*p_t, color="black", s=60, marker="x", zorder=6,
                 label="TCP (fixed target)")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.legend(fontsize=9, loc="upper left")
    ax3d.set_title("the 1-D self-motion manifold solved in closed form", fontsize=11)
    try:
        ax3d.set_box_aspect([1, 1, 1])
    except AttributeError:
        pass

    fig.suptitle("KUKA LBR iiwa 14: pure arm-angle closed-form IK in MuJoCo "
                 "($d_{se}=%.3f$ m, zero wrist offset)" % solver.d_se,
                 fontsize=13, y=0.99)
    out = IMG_DIR / "iiwa_closed_form.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)
    print(f"picked {len(picks)} exact solutions, residuals "
          f"{[f'{s.pos_err:.1e}' for s in picks]}")


if __name__ == "__main__":
    main()
