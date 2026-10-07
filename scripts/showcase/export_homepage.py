"""Package only the assets referenced by the homepage Markdown.

``python -m scripts.showcase.export_homepage`` packages checked-in assets.
``--from-render`` first imports fresh renders from ``build/showcase``.
Neither command modifies or deploys the actual website repository.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import zipfile

from .paths import ROOT, SHOW

HOMEPAGE = ROOT / "docs/homepage"
RENDERS = SHOW
ARCHIVE = ROOT / "build/ik-homepage.zip"
ASSET_URL = re.compile(r"/assets/(?:img|videos|data)/[^\s\)\"'<>]+")


def referenced_assets(markdown: str) -> list[Path]:
    """Return unique public-relative paths from YAML, Markdown and HTML."""
    paths = []
    for url in sorted(set(ASSET_URL.findall(markdown))):
        path = Path(url.removeprefix("/"))
        if ".." in path.parts or path.is_absolute():
            raise ValueError(f"Asset must stay inside public/: {url}")
        paths.append(path)
    if not paths:
        raise ValueError("Homepage Markdown does not reference any local assets")
    return paths


def import_renders(assets: list[Path], public: Path, renders: Path) -> None:
    """Convert/copy the manifest, validating all inputs before writing files."""
    sources = {}
    for asset in assets:
        if asset.suffix == ".webp":
            name = "portfolio_hero" if asset.stem == "portfolio_card" else asset.stem
            sources[asset] = renders / f"{name}.png"
        elif asset.suffix in (".mp4", ".json"):
            sources[asset] = renders / asset.name
        else:
            raise ValueError(f"Unsupported rendered asset: {asset}")
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing renders; run scripts.showcase.build first: " + ", ".join(missing))
    from PIL import Image

    for asset, source in sources.items():
        target = public / asset
        target.parent.mkdir(parents=True, exist_ok=True)
        if asset.suffix == ".webp":
            with Image.open(source) as image:
                if asset.stem == "portfolio_card":
                    image = image.crop((0, 90, 1440, 690))
                image.save(target, "WEBP", quality=92, method=6)
        else:
            shutil.copyfile(source, target)


def package_homepage(homepage: Path = HOMEPAGE, archive: Path = ARCHIVE,
                     *, from_render: bool = False, renders: Path = RENDERS) -> list[Path]:
    markdown = homepage / "inverse-kinematics.md"
    assets = referenced_assets(markdown.read_text(encoding="utf-8"))
    public = homepage / "public"
    if from_render:
        import_renders(assets, public, renders)
    missing = [str(asset) for asset in assets if not (public / asset).is_file()]
    if missing:
        raise FileNotFoundError("Missing published assets: " + ", ".join(missing))
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(markdown, "src/content/projects/inverse-kinematics.md")
        for asset in assets:
            bundle.write(public / asset, Path("public") / asset)
    return assets


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-render", action="store_true",
                        help="Update referenced public assets from build/showcase before packaging")
    args = parser.parse_args(argv)
    assets = package_homepage(from_render=args.from_render)
    print(f"Homepage bundle: {ARCHIVE} ({len(assets)} assets, {ARCHIVE.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
