from pathlib import Path

from app.models.documentary import (
    AudioMode,
    DocumentaryScene,
    ProvenanceType,
    RightsStatus,
    SourceAsset,
    SourceType,
)
from app.services.documentary import audio as audio_service
from app.services.documentary import narration_synthesis
from app.services.documentary.project import create_project, load_project, save_project


def _make_project(tmp_path: Path):
    project = create_project(
        "Narration synthesis",
        project_id="doc_narration_synthesis",
        root=tmp_path,
    )
    project.sources = [
        SourceAsset(
            id="source_video",
            source_type=SourceType.local_video,
            provenance=ProvenanceType.user_provided,
            rights_status=RightsStatus.user_owned,
        )
    ]
    project.plan.scenes = [
        DocumentaryScene(
            id="scene_narrated",
            scene_type="narration_over_source",
            source_id="source_video",
            source_start=0.0,
            source_end=5.0,
            audio_mode=AudioMode.narration,
            narration_text="Short narration.",
        )
    ]
    save_project(project, tmp_path)
    return load_project(project.id, tmp_path)


def test_synthesis_persists_voice_and_reuses_current_audio(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    calls = []

    def fake_tts(**kwargs):
        Path(kwargs["voice_file"]).write_bytes(b"audio")
        calls.append(kwargs["voice_name"])
        return object()

    monkeypatch.setattr(narration_synthesis.voice_service, "tts", fake_tts)
    monkeypatch.setattr(
        narration_synthesis.voice_service,
        "get_audio_duration",
        lambda path: 2.0,
    )
    monkeypatch.setattr(audio_service, "_probe_audio", lambda path: (2.0, "mp3"))

    first = narration_synthesis.synthesize_narration(
        project.id,
        "en-US-TestVoice",
        root=tmp_path,
    )
    revision_after_first = load_project(project.id, tmp_path).revision
    second = narration_synthesis.synthesize_narration(
        project.id,
        "en-US-TestVoice",
        root=tmp_path,
    )
    revision_after_second = load_project(project.id, tmp_path).revision

    assert revision_after_second == revision_after_first
    assert len(first.generated) == 1
    assert len(second.generated) == 0
    assert len(second.reused) == 1
    assert calls == ["en-US-TestVoice"]
    updated = load_project(project.id, tmp_path)
    assert updated.narrator_voices["en"] == "en-US-TestVoice"
    assert len(updated.narration_audio) == 1
    assert updated.narration_audio[0].voice_name == "en-US-TestVoice"


def test_synthesis_regenerates_after_voice_change(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    calls = []

    def fake_tts(**kwargs):
        Path(kwargs["voice_file"]).write_bytes(b"audio")
        calls.append(kwargs["voice_name"])
        return object()

    monkeypatch.setattr(narration_synthesis.voice_service, "tts", fake_tts)
    monkeypatch.setattr(
        narration_synthesis.voice_service,
        "get_audio_duration",
        lambda path: 2.0,
    )
    monkeypatch.setattr(audio_service, "_probe_audio", lambda path: (2.0, "mp3"))

    narration_synthesis.synthesize_narration(
        project.id,
        "en-US-FirstVoice",
        root=tmp_path,
    )
    narration_synthesis.synthesize_narration(
        project.id,
        "en-US-SecondVoice",
        root=tmp_path,
    )

    assert calls == ["en-US-FirstVoice", "en-US-SecondVoice"]
    updated = load_project(project.id, tmp_path)
    assert updated.narrator_voices["en"] == "en-US-SecondVoice"


def test_synthesis_resumes_without_repeating_completed_scene(tmp_path, monkeypatch):
    project = _make_project(tmp_path)
    project = load_project(project.id, tmp_path)
    project.plan.scenes.append(
        DocumentaryScene(
            id="scene_second",
            scene_type="narration_over_source",
            source_id="source_video",
            source_start=5.0,
            source_end=10.0,
            audio_mode=AudioMode.narration,
            narration_text="Second narration.",
        )
    )
    save_project(project, tmp_path)

    calls = []
    fail_second = {"value": True}

    def fake_tts(**kwargs):
        scene_name = Path(kwargs["voice_file"]).stem
        calls.append(scene_name)
        if scene_name == "scene_second" and fail_second["value"]:
            return None
        Path(kwargs["voice_file"]).write_bytes(b"audio")
        return object()

    monkeypatch.setattr(narration_synthesis.voice_service, "tts", fake_tts)
    monkeypatch.setattr(
        narration_synthesis.voice_service,
        "get_audio_duration",
        lambda path: 2.0,
    )
    monkeypatch.setattr(audio_service, "_probe_audio", lambda path: (2.0, "mp3"))

    try:
        narration_synthesis.synthesize_narration(
            project.id,
            "en-US-TestVoice",
            root=tmp_path,
        )
    except narration_synthesis.NarrationSynthesisError:
        pass
    else:
        raise AssertionError("the first run should fail on the second scene")

    fail_second["value"] = False
    calls.clear()
    result = narration_synthesis.synthesize_narration(
        project.id,
        "en-US-TestVoice",
        root=tmp_path,
    )

    assert calls == ["scene_second"]
    assert len(result.reused) == 1
    assert len(result.generated) == 1


def test_synthesis_wraps_tts_provider_errors(tmp_path, monkeypatch):
    project = _make_project(tmp_path)

    def failing_tts(**kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(narration_synthesis.voice_service, "tts", failing_tts)

    with pytest.raises(
        narration_synthesis.NarrationSynthesisError,
        match="TTS failed for scene scene_narrated",
    ):
        narration_synthesis.synthesize_narration(
            project.id,
            "en-US-TestVoice",
            root=tmp_path,
        )
