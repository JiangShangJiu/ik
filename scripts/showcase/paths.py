"""Locations for disposable renders and the published portfolio snapshots.

Generation writes only to ``build/showcase``. Reusing data prefers a fresh
generated JSON and falls back to the checked-in homepage snapshot, so a clean
checkout can run the numerical checks without rendering any media first.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHOW = ROOT / "build" / "showcase"
PUBLIC = ROOT / "docs" / "homepage" / "public"
DATA = PUBLIC / "assets" / "data" / "inverse-kinematics"
VIDEOS = PUBLIC / "assets" / "videos" / "inverse-kinematics"


def metric_path(name):
    """Resolve an input JSON at call time, without creating or changing files."""
    generated = SHOW / name
    return generated if generated.is_file() else DATA / name


def video_path(name):
    """Resolve a single-view MP4 when composing a gallery overview."""
    generated = SHOW / name
    return generated if generated.is_file() else VIDEOS / name
