"""The release bundle works without rendering and contains only used assets."""
from pathlib import Path
import zipfile

import pytest

from scripts.showcase.export_homepage import package_homepage, referenced_assets

ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_homepage_packages_without_render_intermediates(tmp_path):
    homepage = ROOT / "docs/homepage"
    archive = tmp_path / "homepage.zip"
    assets = package_homepage(homepage, archive, renders=tmp_path / "absent")
    expected = {"src/content/projects/inverse-kinematics.md"}
    expected.update("public/" + p.as_posix() for p in assets)
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == expected
        assert bundle.testzip() is None
        for asset in assets:
            assert bundle.read("public/" + asset.as_posix()) == (homepage / "public" / asset).read_bytes()


def test_manifest_handles_all_document_formats_and_rejects_parent_paths():
    markdown = '''heroImage: "/assets/img/cover.webp"
![cover](/assets/img/cover.webp)
<source src="/assets/videos/demo.mp4" />
[data](/assets/data/result.json)
'''
    assert set(referenced_assets(markdown)) == {
        Path("assets/img/cover.webp"), Path("assets/videos/demo.mp4"), Path("assets/data/result.json")}
    with pytest.raises(ValueError):
        referenced_assets('[bad](/assets/data/../../../private.json)')


def test_fresh_render_import_produces_only_the_documented_release_files(tmp_path):
    from PIL import Image

    homepage, renders = tmp_path / "homepage", tmp_path / "renders"
    homepage.mkdir()
    renders.mkdir()
    (homepage / "inverse-kinematics.md").write_text('''heroImage: "/assets/img/portfolio_card.webp"
![figure](/assets/img/figure.webp)
<source src="/assets/videos/demo.mp4" />
[data](/assets/data/result.json)
''')
    Image.new("RGB", (1440, 850), "white").save(renders / "portfolio_hero.png")
    Image.new("RGB", (40, 30), "teal").save(renders / "figure.png")
    (renders / "demo.mp4").write_bytes(b"video copied unchanged")
    (renders / "result.json").write_bytes(b'{"valid": true}\n')
    (renders / "unused.gif").write_bytes(b"not a release input")

    assets = package_homepage(homepage, tmp_path / "release.zip", from_render=True, renders=renders)
    public = homepage / "public"
    assert {p.relative_to(public) for p in public.rglob("*") if p.is_file()} == set(assets)
    with Image.open(public / "assets/img/portfolio_card.webp") as cover:
        assert cover.size == (1440, 600)
    with Image.open(public / "assets/img/figure.webp") as figure:
        assert figure.size == (40, 30)
    assert (public / "assets/videos/demo.mp4").read_bytes() == (renders / "demo.mp4").read_bytes()
    assert (public / "assets/data/result.json").read_bytes() == (renders / "result.json").read_bytes()
