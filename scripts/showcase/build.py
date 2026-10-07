"""Recompute and render the current portfolio, then export its used assets.

Run from the repository root: ``python -m scripts.showcase.build``.
Raw renders go to ignored ``build/showcase/``; the homepage manifest determines
which final WebP/MP4/JSON assets are copied into ``docs/homepage/public/``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from .paths import ROOT

STEPS = ("portfolio", "solution_gallery", "pose_gallery", "swivel_gallery", "motion_gallery")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    for name in STEPS:
        print(f"Generating {name}…", flush=True)
        subprocess.run([sys.executable, "-m", f"scripts.showcase.{name}"], cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "scripts.showcase.export_homepage", "--from-render"],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
