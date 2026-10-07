"""Render three real arms side by side, colouring each joint axis by the role it
plays in Pieper's criterion.

* green  - axes that are mutually parallel   (Pieper condition 2)
* red    - axes that meet at the shoulder centre (Pieper condition 1)
* blue   - axes that meet at the wrist centre    (Pieper condition 1)
* grey   - none of the above

Output: ``docs/img/pieper_robots.png``
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ik.robot import MENAGERIE

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "docs" / "img"

# name -> (xml, joints highlighted as shoulder / wrist, parallel)
ROBOTS = [
    ("Universal Robots UR5e  (6R)",
     MENAGERIE / "universal_robots_ur5e/ur5e.xml",
     {"shoulder": [], "wrist": [], "parallel": [1, 2, 3]}),
    ("KUKA LBR iiwa 14  (7R)",
     MENAGERIE / "kuka_iiwa_14/iiwa14.xml",
     {"shoulder": [0, 1, 2], "wrist": [4, 5, 6], "parallel": []}),
    ("Franka Emika Panda  (7R)",
     MENAGERIE / "franka_emika_panda/panda_nohand.xml",
     {"shoulder": [0, 1, 2], "wrist": [], "parallel": []}),
]

COLORS = {
    "other": np.array([0.75, 0.75, 0.75, 1.0]),
    "parallel": np.array([0.15, 0.65, 0.20, 1.0]),
    "shoulder": np.array([0.85, 0.15, 0.15, 1.0]),
    "wrist": np.array([0.15, 0.35, 0.85, 1.0]),
}
CAPTIONS = {
    "Universal Robots UR5e  (6R)":
        "Pieper via 3 PARALLEL axes (2,3,4)\nwrist NOT spherical (0.10 m offset)",
    "KUKA LBR iiwa 14  (7R)":
        "spherical SHOULDER (1,2,3) + WRIST (5,6,7)\n= true S-R-S  ->  pure arm-angle IK",
    "Franka Emika Panda  (7R)":
        "spherical SHOULDER (1,2,3), wrist NOT spherical (0.088 m)\n-> semi-analytic (seed + polish)",
}


def basis_from_z(z):
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def add_capsule(scene, p0, p1, radius, rgba):
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 1e-9:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, L / 2, 0.0]),
        0.5 * (p0 + p1), basis_from_z(d / L).flatten(), rgba,
    )
    scene.ngeom += 1


def add_sphere(scene, p, radius, rgba):
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius]), p, np.eye(3).flatten(), rgba,
    )
    scene.ngeom += 1


def arm_joint_ids(model):
    return [j for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def render_robot(xml, roles, width=480, height=420):
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)

    ids = arm_joint_ids(model)
    anchors = np.array([data.xanchor[j] for j in ids])
    axes = np.array([data.xaxis[j] for j in ids])

    renderer = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 1.7
    cam.lookat[:] = [0.0, 0.0, 0.35]
    cam.azimuth = 135.0
    cam.elevation = -18.0
    renderer.update_scene(data, cam)
    scn = renderer.scene

    # joint axes
    for k, j in enumerate(ids):
        if k in roles["parallel"]:
            col = COLORS["parallel"]
        elif k in roles["shoulder"]:
            col = COLORS["shoulder"]
        elif k in roles["wrist"]:
            col = COLORS["wrist"]
        else:
            col = COLORS["other"]
        p0 = anchors[k] - 0.07 * axes[k]
        p1 = anchors[k] + 0.07 * axes[k]
        add_capsule(scn, p0, p1, 0.008, col)

    # Mark the first anchor in each highlighted group.
    for group, rgba in (("shoulder", COLORS["shoulder"]), ("wrist", COLORS["wrist"])):
        idx = roles[group]
        if idx:
            add_sphere(scn, anchors[idx[:1]][0], 0.02, rgba)

    img = renderer.render().copy()
    renderer.close()
    return img


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (title, xml, roles) in zip(axes, ROBOTS):
        img = render_robot(xml, roles)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=12)
        ax.text(0.5, -0.02, CAPTIONS[title], transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
    fig.suptitle("Pieper structure of three real arms (MuJoCo joint axes)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    out = IMG_DIR / "pieper_robots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
