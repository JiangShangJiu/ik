"""A clean checkout can reuse published data without changing its snapshots."""
import json

from scripts.showcase import paths


def test_input_resolution_prefers_new_build_files_and_preserves_public(tmp_path, monkeypatch):
    build, data, videos = (tmp_path / name for name in ("build", "data", "videos"))
    monkeypatch.setattr(paths, "SHOW", build)
    monkeypatch.setattr(paths, "DATA", data)
    monkeypatch.setattr(paths, "VIDEOS", videos)
    data.mkdir()
    videos.mkdir()
    snapshot = data / "pose_metrics.json"
    video = videos / "pose_iiwa14.mp4"
    snapshot.write_text("published snapshot")
    video.write_bytes(b"published video")

    assert paths.metric_path(snapshot.name) == snapshot
    assert paths.video_path(video.name) == video
    assert not build.exists()
    build.mkdir()
    fresh = build / snapshot.name
    fresh.write_text("fresh experiment")
    assert paths.metric_path(snapshot.name) == fresh
    # Resolution happens at call time, including after a generated file is
    # removed. An import-time default would keep pointing to a stale file.
    fresh.unlink()
    assert paths.metric_path(snapshot.name) == snapshot
    assert snapshot.read_text() == "published snapshot"
    assert video.read_bytes() == b"published video"


def test_reuse_data_writes_only_to_build_without_requiring_old_render_outputs(tmp_path, monkeypatch):
    from scripts.showcase import solution_gallery

    build, data = tmp_path / "build", tmp_path / "data"
    monkeypatch.setattr(paths, "SHOW", build)
    monkeypatch.setattr(paths, "DATA", data)
    monkeypatch.setattr(solution_gallery, "SHOW", build)
    data.mkdir()
    snapshot = data / "solution_metrics.json"
    snapshot.write_text(json.dumps({"robots": {key: {} for key in solution_gallery.KEYS}}))
    before = snapshot.read_bytes()
    rendered = []
    monkeypatch.setattr(solution_gallery, "atlas", lambda key, record: rendered.append(key))
    monkeypatch.setattr(solution_gallery, "overview", lambda robots: None)

    solution_gallery.main(["--reuse-data"])

    assert rendered == list(solution_gallery.KEYS)
    result = json.loads((build / snapshot.name).read_text())
    assert "render" in result
    assert snapshot.read_bytes() == before
