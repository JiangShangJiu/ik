"""The documented shared model directory must also work for Panda."""
from ik.model import resolve_xml


def test_panda_uses_shared_menagerie_and_explicit_override(tmp_path, monkeypatch):
    xml = tmp_path / "franka_emika_panda/panda_nohand.xml"
    xml.parent.mkdir()
    xml.touch()
    override = tmp_path / "override.xml"
    override.touch()
    monkeypatch.delenv("PANDA_XML", raising=False)
    monkeypatch.setenv("MUJOCO_MENAGERIE", str(tmp_path))
    assert resolve_xml() == xml
    monkeypatch.setenv("PANDA_XML", str(override))
    assert resolve_xml() == override
    assert resolve_xml(xml) == xml
