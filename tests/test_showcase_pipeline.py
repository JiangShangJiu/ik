"""Showcase geometry checks and current documentation asset references."""

from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest

from scripts.showcase.common import all_targets, get_arm, roles_for
from scripts.showcase.branch_diagnostics import enumerate_branches
from scripts.showcase.trajectory_helpers import make_line
from ik import SrsSolver

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def targets():
    return all_targets()


def test_targets_have_finite_valid_poses(targets):
    for key, (q, T) in targets.items():
        assert T.shape == (4, 4)
        assert np.isfinite(q).all()
        assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-9)


def test_structure_roles():
    assert roles_for(get_arm("ur5e")) == {1: "parallel", 2: "parallel", 3: "parallel"}
    lite = roles_for(get_arm("lite6"))
    assert all(lite[i] == "spherical" for i in (3, 4, 5))
    iiwa = roles_for(get_arm("iiwa14"))
    assert all(iiwa[i] == "spherical" for i in (0, 1, 2, 4, 5, 6))
    panda = roles_for(get_arm("panda"))
    assert all(panda[i] == "spherical" for i in (0, 1, 2))


@pytest.mark.parametrize("key", ["ur5e", "lite6"])
def test_six_axis_branches_are_exact(key, targets):
    arm = get_arm(key)
    sols = enumerate_branches(key, arm, targets[key][1])
    assert len(sols) >= 4
    tol = 1e-3 if key == "lite6" else 1e-9
    for q, pe, re_, kind, name in sols:
        assert kind == "exact"
        assert pe < tol
        assert re_ < 1e-9
    assert len({s[4] for s in sols}) == len(sols)


def test_iiwa_family_is_exact(targets):
    arm = get_arm("iiwa14")
    sv = SrsSolver(arm, n_psi=16)
    sols = sv.solve(targets["iiwa14"][1], max_solutions=16)
    assert len(sols) >= 8
    assert max(s.pos_err for s in sols) < 1e-9


def test_straight_line_stays_on_the_line(targets):
    for key in ("ur5e", "lite6", "iiwa14", "panda"):
        arm = get_arm(key)
        q_ref, _ = targets[key]
        rng = np.random.default_rng(11)
        Ts, qs = make_line(key, arm, q_ref, rng, n=24)
        p0, p1 = Ts[0][:3, 3], Ts[-1][:3, 3]
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        devs = []
        for T, q in zip(Ts, qs):
            arm.set_q(q)
            p = arm.tcp_pose()[1]
            devs.append(np.linalg.norm(np.cross(p - p0, d)))
        assert max(devs) < 1e-4


@pytest.mark.parametrize("document", [
    "README.md", "docs/REPRODUCING.md", "docs/ik_survey.md", "docs/homepage/inverse-kinematics.md",
])
def test_current_documentation_references_existing_local_files(document):
    path = ROOT / document
    text = path.read_text(encoding="utf-8")
    refs = set(re.findall(r"\]\(([^)\s]+)(?:\s+[^)]*)?\)", text))
    refs.update(re.findall(r"\b(?:src|poster|href)=[\"']([^\"']+)[\"']", text))
    # Astro frontmatter also contains site-root asset paths outside HTML tags.
    refs.update(re.findall(r"[\"'](/assets/[^\"']+)[\"']", text))
    local, missing = [], []
    for ref in refs:
        url = urlsplit(ref.strip("<>"))
        if url.scheme or url.netloc or not url.path:
            continue
        asset = unquote(url.path)
        if asset.startswith("/assets/"):
            target = ROOT / "docs/homepage/public" / asset.lstrip("/")
        elif asset.startswith("/"):
            continue  # Routes such as /posts/... belong to the user's site.
        else:
            target = path.parent / asset
        local.append(asset)
        if not target.exists():
            missing.append(asset)
    assert local, f"no local file references found in {document}"
    assert not missing, f"{document} references missing local files: {sorted(missing)}"
