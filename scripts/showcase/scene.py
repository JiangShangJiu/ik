"""Offscreen MuJoCo rendering shared by every showcase exhibit.

All four robots are rendered with the same camera azimuth/elevation, the same
ground plane and the same lighting so the pictures read as one photo shoot.
The target pose marker (amber sphere + RGB frame) and the structural overlays
(joint axes, elbow circle, shoulder/wrist centres) are drawn as scene geoms on
top of the real robot meshes -- never as a separate stick figure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from ik import load_robot  # noqa: E402

# role colours of the joint axes -- the SAME three everywhere
AXIS_SPHERICAL = np.array([0.78, 0.20, 0.62, 1.0])   # axes that concur
AXIS_PARALLEL = np.array([0.10, 0.62, 0.60, 1.0])    # axes mutually parallel
AXIS_OTHER = np.array([0.72, 0.72, 0.74, 1.0])       # neither

TARGET_BALL = np.array([1.00, 0.62, 0.07, 0.42])
TARGET_X = np.array([0.95, 0.15, 0.15, 1.0])
TARGET_Y = np.array([0.15, 0.85, 0.20, 1.0])
TARGET_Z = np.array([0.25, 0.45, 1.00, 1.0])

FLOOR = np.array([0.62, 0.64, 0.66, 0.55])
SHOULDER = np.array([0.90, 0.10, 0.10, 0.9])
WRIST = np.array([0.10, 0.20, 0.90, 0.9])
ELBOW = np.array([0.10, 0.70, 0.20, 0.95])
CIRCLE = np.array([0.10, 0.45, 0.90, 0.6])
LINE = np.array([0.95, 0.55, 0.05, 0.9])


def _basis_from_z(z):
    z = np.asarray(z, float)
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def add_sphere(scn, p, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, radius, radius]), np.asarray(p, float),
                        np.eye(3).flatten(), rgba)
    scn.ngeom += 1


def add_capsule(scn, p0, p1, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.array([radius, L / 2, 0.0]), 0.5 * (p0 + p1),
                        _basis_from_z(d / L).flatten(), rgba)
    scn.ngeom += 1


def add_plane(scn, size, rgba, pos=(0, 0, 0)):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_PLANE,
                        np.array([size, size, 0.0]), np.asarray(pos, float),
                        np.eye(3).flatten(), rgba)
    scn.ngeom += 1


def add_circle(scn, center, normal, radius, rgba, n=72, width=0.004):
    center = np.asarray(center, float)
    B = _basis_from_z(normal)
    ts = np.linspace(0, 2 * np.pi, n + 1)
    pts = center + radius * (np.cos(ts)[:, None] * B[:, 0]
                             + np.sin(ts)[:, None] * B[:, 1])
    for i in range(n):
        add_capsule(scn, pts[i], pts[i + 1], width, rgba)


def add_target(scn, T, axis_len=0.13, ball=0.028):
    R, p = np.asarray(T, float)[:3, :3], np.asarray(T, float)[:3, 3]
    add_sphere(scn, p, ball, TARGET_BALL)
    for col, rgba in zip(R.T, (TARGET_X, TARGET_Y, TARGET_Z)):
        add_capsule(scn, p, p + axis_len * col, 0.007, rgba)


@dataclass
class Camera:
    lookat: tuple
    distance: float
    azimuth: float = 135.0
    elevation: float = -18.0


# per-robot framing so every base sits on the ground at a similar size
FRAMING = {
    "ur5e": Camera((0.0, -0.15, 0.35), 1.85),
    "lite6": Camera((0.05, 0.0, 0.32), 1.35),
    "iiwa14": Camera((0.0, 0.0, 0.42), 1.75),
    "panda": Camera((0.0, 0.0, 0.40), 1.75),
}


def _mjcam(cam):
    mjcam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(mjcam)
    mjcam.type = mujoco.mjtCamera.mjCAMERA_FREE
    mjcam.lookat[:] = cam.lookat
    mjcam.distance = cam.distance
    mjcam.azimuth = cam.azimuth
    mjcam.elevation = cam.elevation
    return mjcam


def _camera_basis(model, data, cam):
    """Unit look-at direction and camera up for the given free camera."""
    r = mujoco.Renderer(model, 64, 64)
    r.update_scene(data, _mjcam(cam))
    c = r.scene.camera[0]
    pos = np.array(c.pos, float)
    up = np.array(c.up, float)
    r.close()
    e = pos - np.asarray(cam.lookat, float)
    e /= np.linalg.norm(e)
    up /= np.linalg.norm(up)
    return e, up


def project_points(model, data, cam, points, width=560, height=480):
    """Normalised image coords (u right, v up in [-1, 1]) for world points."""
    e, up = _camera_basis(model, data, cam)
    fwd = -e
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_pos = np.asarray(cam.lookat, float) + cam.distance * e
    t = np.tan(np.deg2rad(model.vis.global_.fovy) / 2.0)
    asp = width / height
    d = np.atleast_2d(points) - cam_pos
    z = d @ fwd
    z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = (d @ right) / (z * t * asp)
    v = (d @ up) / (z * t)
    return u, v


def auto_camera(model, data, width, height, extra_points=(), margin=1.08,
                azimuth=135.0, elevation=-14.0, center=None, distance=None):
    """Frame the robot (plus ``extra_points``) tightly and consistently."""
    pts = [np.asarray(data.xpos, float)]
    for p in extra_points:
        a = np.atleast_2d(np.asarray(p, float))
        if a.size:
            pts.append(a)
    pts = np.vstack(pts)
    if center is None:
        center = 0.5 * (pts.min(0) + pts.max(0))
    center = np.asarray(center, float)
    if distance is not None:
        return Camera(tuple(center), float(distance), azimuth, elevation)

    cam0 = Camera(tuple(center), 1.0, azimuth, elevation)
    e, up = _camera_basis(model, data, cam0)
    fwd = -e
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    fovy = np.deg2rad(model.vis.global_.fovy)
    t = np.tan(fovy / 2.0)
    asp = width / height

    def extent(d):
        cam_pos = center + d * e
        d_ = pts - cam_pos
        z = d_ @ fwd
        if np.any(z <= 1e-6):
            return 1e9
        u = (d_ @ right) / (z * t * asp)
        v = (d_ @ up) / (z * t)
        return float(max(np.abs(u).max(), np.abs(v).max()))

    lo, hi = 0.2, 40.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if extent(mid) > 1.0 / margin:
            lo = mid
        else:
            hi = mid
    return Camera(tuple(center), hi, azimuth, elevation)


class RobotScene:
    def __init__(self, key: str, width: int = 560, height: int = 480,
                 arm=None):
        self.key = key
        self.arm = arm if arm is not None else load_robot(key)
        self.model = self.arm.model
        self.data = self.arm.data
        self.width, self.height = width, height
        self._renderer = None

    # -- state -------------------------------------------------------------- #
    def set_q(self, q):
        self.arm.set_q(np.asarray(q, float))

    def tcp(self):
        return self.arm.tcp_pose()

    # -- render ------------------------------------------------------------- #
    def render(self, q=None, cam: Camera | None = None, overlays=None,
               geom_rgba=None, floor=True):
        if q is not None:
            self.set_q(q)
        cam = cam or FRAMING[self.key]
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        mjcam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(mjcam)
        mjcam.type = mujoco.mjtCamera.mjCAMERA_FREE
        mjcam.lookat[:] = cam.lookat
        mjcam.distance = cam.distance
        mjcam.azimuth = cam.azimuth
        mjcam.elevation = cam.elevation

        if geom_rgba is not None:
            self.model.geom_rgba[:] = geom_rgba
        self._renderer.update_scene(self.data, mjcam)
        scn = self._renderer.scene
        if floor:
            add_plane(scn, 3.0, FLOOR, (0, 0, 0.0))
        if overlays:
            for ov in overlays:
                ov(scn)
        img = self._renderer.render().copy()
        return img

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def extent_points(self, max_pts=700):
        """World-space samples covering the whole silhouette.

        ``auto_camera`` used to frame only body origins (``data.xpos``), so long
        links -- and every branch in a montage rendered with one shared camera --
        used to poke past the image border.  We sample the actual mesh vertices
        (sub-sampled) and the bounding sphere of non-mesh geoms so the whole
        silhouette stays inside the frame.
        """
        m, d = self.model, self.data
        chunks, centres = [], []
        for gi in range(m.ngeom):
            c = d.geom_xpos[gi]
            if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[gi]
                if mid < 0:
                    centres.append(c)
                    continue
                va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                if vn <= 0:
                    centres.append(c)
                    continue
                verts = m.mesh_vert[va:va + vn]
                stride = max(1, vn // 200)
                R = d.geom_xmat[gi].reshape(3, 3)
                chunks.append(verts[::stride] @ R.T + c)
            else:
                r = float(m.geom_rbound[gi])
                centres.append(c + [r, 0.0, 0.0])
                centres.append(c - [r, 0.0, 0.0])
                centres.append(c + [0.0, r, 0.0])
                centres.append(c - [0.0, r, 0.0])
                centres.append(c + [0.0, 0.0, r])
                centres.append(c - [0.0, 0.0, r])
        pts = chunks + [np.asarray(centres, float)] if centres else chunks
        pts = np.vstack(pts) if pts else np.empty((0, 3))
        if len(pts) > max_pts:
            stride = int(np.ceil(len(pts) / max_pts))
            pts = pts[::stride]
        return pts

    def auto_camera(self, extra_points=(), margin=1.08, center=None,
                    distance=None, azimuth=135.0, elevation=-14.0,
                    sweep_qs=None):
        extra = list(extra_points)
        if sweep_qs is not None:
            for q in sweep_qs:
                self.set_q(np.asarray(q, float))
                extra.append(self.extent_points())
        else:
            extra.append(self.extent_points())
        return auto_camera(self.model, self.data, self.width, self.height,
                           extra_points=extra, margin=margin,
                           azimuth=azimuth, elevation=elevation,
                           center=center, distance=distance)

    # -- reusable overlays -------------------------------------------------- #
    def target_overlay(self, T):
        return lambda scn: add_target(scn, T)

    def joint_axes_overlay(self, roles, half=0.06, width=0.008, q=None):
        """``roles`` maps 0-based joint index -> 'spherical'|'parallel'|'other'."""
        if q is not None:
            self.set_q(q)
        anchors, axes = self.arm.world_lines()

        def _draw(scn):
            for k in range(len(axes)):
                col = {"spherical": AXIS_SPHERICAL,
                       "parallel": AXIS_PARALLEL}.get(roles.get(k), AXIS_OTHER)
                p = anchors[k]
                p0 = p - half * axes[k]
                p1 = p + half * axes[k]
                add_capsule(scn, p0, p1, width, col)
        return _draw
