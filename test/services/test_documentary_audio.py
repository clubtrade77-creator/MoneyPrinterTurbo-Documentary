from pathlib import Path

import pytest

from app.models.documentary import DocumentaryScene, SceneType
from app.services.documentary.audio import (
    NarrationAudioError,
    attach_narration_audio,
    load_narration_audio,
)
from app.services.documentary.project import create_project, load_project, save_project


def _project_with_scene(tmp_path: Path):
    project = create_project(
        "Narration audio case",
        project_id="doc_narration_audio",
        root=tmp_path,
    )
    project.plan.scenes = [
        DocumentaryScene(
            id="scene_narration",
            scene_type=SceneType.title_card,
        )
    ]
    save_project(project, tmp_path)
    return project


def test_attach_narration_audio_copies_and_registers_project_asset(
    tmp_path: Path,
    monkeypatch,
):
    project = _project_with_scene(tmp_path)
    source = tmp_path / "voice.wav"
    source.write_bytes(b"fake-wave")

    monkeypatch.setattr(
        "app.services.documentary.audio._probe_audio",
        lambda path: (2.5, "pcm_s16le"),
    )

    asset = attach_narration_audio(
        project.id,
        "scene_narration",
        source,
        language="EN",
        root=tmp_path,
    )

    assert asset.language == "en"
    assert asset.duration_seconds == pytest.approx(2.5)
    assert asset.audio_codec == "pcm_s16le"
    assert Path(asset.local_path).is_file()
    assert Path(asset.local_path).read_bytes() == b"fake-wave"

    updated = load_project(project.id, tmp_path)
    assert updated.revision > project.revision
    assert len(updated.narration_audio) == 1
    assert updated.narration_audio[0] == asset


def test_load_narration_audio_rejects_modified_registered_file(
    tmp_path: Path,
    monkeypatch,
):
    project = _project_with_scene(tmp_path)
    source = tmp_path / "voice.wav"
    source.write_bytes(b"original-wave")
    monkeypatch.setattr(
        "app.services.documentary.audio._probe_audio",
        lambda path: (1.5, "pcm_s16le"),
    )

    asset = attach_narration_audio(
        project.id,
        "scene_narration",
        source,
        root=tmp_path,
    )
    Path(asset.local_path).write_bytes(b"changed-wave")

    with pytest.raises(NarrationAudioError, match="changed after registration"):
        load_narration_audio(
            project.id,
            "scene_narration",
            root=tmp_path,
        )


def test_load_narration_audio_rejects_stale_scene_text(
    tmp_path: Path,
    monkeypatch,
):
    project = _project_with_scene(tmp_path)
    project = load_project(project.id, tmp_path)
    project.plan.scenes[0].narration_text = "Original narration."
    save_project(project, tmp_path)

    source = tmp_path / "voice.wav"
    source.write_bytes(b"original-wave")
    monkeypatch.setattr(
        "app.services.documentary.audio._probe_audio",
        lambda path: (1.5, "pcm_s16le"),
    )

    attach_narration_audio(
        project.id,
        "scene_narration",
        source,
        root=tmp_path,
    )

    changed = load_project(project.id, tmp_path)
    changed.plan.scenes[0].narration_text = "Updated narration."
    save_project(changed, tmp_path)

    with pytest.raises(NarrationAudioError, match="narration text changed"):
        load_narration_audio(
            project.id,
            "scene_narration",
            root=tmp_path,
        )


def test_attach_narration_audio_replaces_previous_scene_language_asset(
    tmp_path: Path,
    monkeypatch,
):
    project = _project_with_scene(tmp_path)
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first.write_bytes(b"first-wave")
    second.write_bytes(b"second-wave")
    monkeypatch.setattr(
        "app.services.documentary.audio._probe_audio",
        lambda path: (1.0, "pcm_s16le"),
    )

    old_asset = attach_narration_audio(
        project.id,
        "scene_narration",
        first,
        language="en",
        root=tmp_path,
    )
    new_asset = attach_narration_audio(
        project.id,
        "scene_narration",
        second,
        language="en",
        root=tmp_path,
    )

    assert old_asset.local_path != new_asset.local_path
    assert not Path(old_asset.local_path).exists()
    assert Path(new_asset.local_path).read_bytes() == b"second-wave"

    updated = load_project(project.id, tmp_path)
    assert len(updated.narration_audio) == 1
    assert updated.narration_audio[0] == new_asset


def test_attach_narration_audio_rejects_unknown_scene(
    tmp_path: Path,
    monkeypatch,
):
    project = _project_with_scene(tmp_path)
    source = tmp_path / "voice.wav"
    source.write_bytes(b"fake-wave")
    monkeypatch.setattr(
        "app.services.documentary.audio._probe_audio",
        lambda path: (1.0, "pcm_s16le"),
    )

    with pytest.raises(ValueError, match="scene not found"):
        attach_narration_audio(
            project.id,
            "scene_missing",
            source,
            root=tmp_path,
        )


def test_attach_narration_audio_rejects_unsupported_extension(tmp_path: Path):
    project = _project_with_scene(tmp_path)
    source = tmp_path / "voice.txt"
    source.write_text("not audio", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported narration audio extension"):
        attach_narration_audio(
            project.id,
            "scene_narration",
            source,
            root=tmp_path,
        )
