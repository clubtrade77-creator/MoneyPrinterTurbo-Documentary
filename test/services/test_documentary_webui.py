from types import SimpleNamespace
from io import BytesIO

import pytest

from app.services.documentary.localization import LocalizationError
import webui.documentary as documentary_ui
from webui.documentary import (
    _default_story_target_seconds,
    _documentary_voice_options,
    _documentary_voice_preview_text,
    _format_transcript_segment,
    _format_transcript_time,
    _load_clip_plan_if_available,
    _load_story_plan_if_available,
    _render_output_is_current,
    _load_transcript_if_available,
    _story_evidence_timecode,
    _write_uploaded_video_to_temp,
)


class _Upload(BytesIO):
    def __init__(self, payload: bytes, name: str):
        super().__init__(payload)
        self.name = name


def test_documentary_upload_helper_stages_video_bytes():
    upload = _Upload(b"video-bytes", "camera.MP4")

    temp_path = _write_uploaded_video_to_temp(upload)
    try:
        assert temp_path.suffix == ".mp4"
        assert temp_path.read_bytes() == b"video-bytes"
    finally:
        temp_path.unlink(missing_ok=True)


def test_documentary_upload_helper_rejects_non_video_extension():
    upload = _Upload(b"not-video", "notes.txt")

    with pytest.raises(ValueError, match="unsupported documentary video extension"):
        _write_uploaded_video_to_temp(upload)


def test_documentary_transcript_lookup_returns_none_when_missing(monkeypatch):
    def missing(project_id: str, source_id: str):
        raise FileNotFoundError(source_id)

    monkeypatch.setattr(documentary_ui, "load_source_transcript", missing)

    assert _load_transcript_if_available("doc_test", "source_test") is None


def test_documentary_transcript_lookup_returns_existing_transcript(monkeypatch):
    expected = object()

    monkeypatch.setattr(
        documentary_ui,
        "load_source_transcript",
        lambda project_id, source_id: expected,
    )

    assert _load_transcript_if_available("doc_test", "source_test") is expected


def test_documentary_transcript_time_formatting():
    assert _format_transcript_time(0.0) == "00:00.00"
    assert _format_transcript_time(3.36) == "00:03.36"
    assert _format_transcript_time(65.27) == "01:05.27"
    assert _format_transcript_time(3661.04) == "01:01:01.04"


def test_documentary_transcript_segment_includes_timecodes():
    segment = type(
        "Segment",
        (),
        {
            "start_seconds": 3.6,
            "end_seconds": 6.74,
            "text": " A few seconds later, the situation changed completely. ",
        },
    )()

    assert _format_transcript_segment(segment) == (
        "[00:03.60–00:06.74] "
        "A few seconds later, the situation changed completely."
    )


def test_documentary_story_evidence_timecode_uses_segment_bounds():
    transcript = type(
        "Transcript",
        (),
        {
            "segments": [
                type(
                    "Segment",
                    (),
                    {"id": 0, "start_seconds": 0.0, "end_seconds": 3.36},
                )(),
                type(
                    "Segment",
                    (),
                    {"id": 1, "start_seconds": 3.60, "end_seconds": 6.74},
                )(),
            ]
        },
    )()

    assert _story_evidence_timecode(transcript, [0, 1]) == (
        "[00:00.00–00:06.74]"
    )
    assert _story_evidence_timecode(transcript, [1]) == (
        "[00:03.60–00:06.74]"
    )
    assert _story_evidence_timecode(transcript, [99]) == ""


def test_documentary_story_plan_lookup_returns_none_when_missing(monkeypatch):
    def missing(project_id: str):
        raise FileNotFoundError(project_id)

    monkeypatch.setattr(documentary_ui, "load_story_plan", missing)

    assert _load_story_plan_if_available("doc_test") is None


def test_documentary_story_plan_lookup_returns_existing_plan(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        documentary_ui,
        "load_story_plan",
        lambda project_id: expected,
    )

    assert _load_story_plan_if_available("doc_test") is expected


def test_documentary_short_story_target_tracks_available_evidence():
    assert _default_story_target_seconds(7.04) == 7
    assert _default_story_target_seconds(2.0) == 5
    assert _default_story_target_seconds(59.6) == 60
    assert _default_story_target_seconds(120.0) == 600


def test_documentary_clip_plan_lookup_returns_none_when_missing(monkeypatch):
    def missing(project_id: str):
        raise FileNotFoundError(project_id)

    monkeypatch.setattr(documentary_ui, "load_clip_plan", missing)

    assert _load_clip_plan_if_available("doc_test") is None


def test_documentary_clip_plan_lookup_returns_existing_plan(monkeypatch):
    expected = object()
    monkeypatch.setattr(
        documentary_ui,
        "load_clip_plan",
        lambda project_id: expected,
    )

    assert _load_clip_plan_if_available("doc_test") is expected


def test_documentary_render_currentness_tracks_plan_files(
    tmp_path, monkeypatch
):
    output = tmp_path / "master.mp4"
    story_plan = tmp_path / "story-plan.json"
    clip_plan = tmp_path / "clip-plan.json"

    for path in (output, story_plan, clip_plan):
        path.write_bytes(b"x")

    monkeypatch.setattr(
        documentary_ui,
        "documentary_render_path",
        lambda project_id, language=None: output,
    )
    monkeypatch.setattr(
        documentary_ui,
        "story_plan_path",
        lambda project_id: story_plan,
    )
    monkeypatch.setattr(
        documentary_ui,
        "clip_plan_path",
        lambda project_id: clip_plan,
    )
    monkeypatch.setattr(
        documentary_ui,
        "load_project",
        lambda project_id: SimpleNamespace(
            master_language="en",
            plan=SimpleNamespace(scenes=[]),
            sources=[],
            narration_audio=[],
        ),
    )

    import os

    os.utime(story_plan, (11, 11))
    os.utime(clip_plan, (12, 12))
    os.utime(output, (13, 13))
    assert _render_output_is_current("doc_test") is True

    os.utime(clip_plan, (14, 14))
    assert _render_output_is_current("doc_test") is False


def test_documentary_render_currentness_rejects_missing_local_media(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "master.mp4"
    source_path = tmp_path / "source.mp4"
    output.write_bytes(b"render")
    source_path.write_bytes(b"source")

    monkeypatch.setattr(
        documentary_ui,
        "documentary_render_path",
        lambda project_id, language=None: output,
    )
    monkeypatch.setattr(
        documentary_ui,
        "story_plan_path",
        lambda project_id: tmp_path / "missing-story.json",
    )
    monkeypatch.setattr(
        documentary_ui,
        "clip_plan_path",
        lambda project_id: tmp_path / "missing-clip.json",
    )
    monkeypatch.setattr(
        documentary_ui,
        "load_project",
        lambda project_id: SimpleNamespace(
            master_language="en",
            plan=SimpleNamespace(
                scenes=[
                    SimpleNamespace(
                        id="scene_source",
                        source_id="source_media",
                        audio_mode=AudioMode.muted,
                    )
                ]
            ),
            sources=[
                SimpleNamespace(
                    id="source_media",
                    local_path=str(source_path),
                )
            ],
            narration_audio=[],
        ),
    )

    import os

    os.utime(source_path, (11, 11))
    os.utime(output, (12, 12))
    assert _render_output_is_current("doc_test") is True

    source_path.unlink()
    assert _render_output_is_current("doc_test") is False


def test_documentary_render_currentness_ignores_other_language_audio(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "master.mp4"
    source_path = tmp_path / "source.mp4"
    en_audio = tmp_path / "narration-en.mp3"
    ru_audio = tmp_path / "narration-ru.mp3"
    for path in (output, source_path, en_audio, ru_audio):
        path.write_bytes(b"x")

    monkeypatch.setattr(
        documentary_ui,
        "documentary_render_path",
        lambda project_id, language=None: output,
    )
    monkeypatch.setattr(
        documentary_ui,
        "story_plan_path",
        lambda project_id: tmp_path / "missing-story.json",
    )
    monkeypatch.setattr(
        documentary_ui,
        "clip_plan_path",
        lambda project_id: tmp_path / "missing-clip.json",
    )
    monkeypatch.setattr(
        documentary_ui,
        "load_project",
        lambda project_id: SimpleNamespace(
            master_language="en",
            plan=SimpleNamespace(
                scenes=[
                    SimpleNamespace(
                        id="scene_narrated",
                        source_id="source_media",
                        audio_mode=AudioMode.narration,
                    )
                ]
            ),
            sources=[
                SimpleNamespace(
                    id="source_media",
                    local_path=str(source_path),
                )
            ],
            narration_audio=[
                SimpleNamespace(
                    scene_id="scene_narrated",
                    language="en",
                    local_path=str(en_audio),
                ),
                SimpleNamespace(
                    scene_id="scene_narrated",
                    language="ru",
                    local_path=str(ru_audio),
                ),
            ],
        ),
    )

    import os

    os.utime(source_path, (10, 10))
    os.utime(en_audio, (11, 11))
    os.utime(output, (12, 12))
    os.utime(ru_audio, (20, 20))

    assert _render_output_is_current("doc_test") is True


def test_localized_render_currentness_requires_valid_localization(
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "master-ru.mp4"
    output.write_bytes(b"render")

    monkeypatch.setattr(
        documentary_ui,
        "documentary_render_path",
        lambda project_id, language=None: output,
    )
    monkeypatch.setattr(
        documentary_ui,
        "load_project",
        lambda project_id: SimpleNamespace(
            master_language="en",
            plan=SimpleNamespace(scenes=[]),
            sources=[],
            narration_audio=[],
        ),
    )

    def stale_localization(*args, **kwargs):
        raise LocalizationError("stale localization")

    monkeypatch.setattr(
        documentary_ui,
        "load_localization_plan",
        stale_localization,
    )

    assert _render_output_is_current(
        "doc_test",
        language="ru",
    ) is False


def test_documentary_render_currentness_requires_output(tmp_path, monkeypatch):
    output = tmp_path / "missing-master.mp4"
    monkeypatch.setattr(
        documentary_ui,
        "documentary_render_path",
        lambda project_id, language=None: output,
    )

    assert _render_output_is_current("doc_test") is False


def test_documentary_voice_preview_text_uses_project_language():
    assert "documentary" in _documentary_voice_preview_text("en").lower()
    assert "документального" in _documentary_voice_preview_text("ru").lower()
    assert "documental" in _documentary_voice_preview_text("es").lower()



def test_documentary_voice_options_prefers_configured_cartesia_voice(monkeypatch):
    monkeypatch.setattr(
        documentary_ui.voice_service,
        "get_cartesia_api_key",
        lambda: "cartesia-key",
    )
    monkeypatch.setattr(
        documentary_ui.config,
        "cartesia",
        {
            "voice_id": "voice-123",
            "model_id": "sonic-3.6",
        },
    )
    monkeypatch.setattr(
        documentary_ui.voice_service,
        "list_cartesia_voices",
        lambda language, limit: [
            {
                "id": "voice-999",
                "name": "Sergei",
                "gender": "masculine",
            },
            {
                "id": "voice-123",
                "name": "Alexei",
                "gender": "masculine",
            },
        ],
    )
    monkeypatch.setattr(
        documentary_ui.config,
        "app",
        {"gemini_api_key": ""},
    )
    monkeypatch.setattr(
        documentary_ui.voice_service,
        "get_minimax_tts_api_key",
        lambda: "",
    )
    monkeypatch.setattr(
        documentary_ui.voice_service,
        "get_fish_audio_api_key",
        lambda: "",
    )

    options = _documentary_voice_options("ru")

    assert options[:2] == [
        (
            "cartesia:voice-123:ru",
            "Cartesia · Alexei · masculine",
        ),
        (
            "cartesia:voice-999:ru",
            "Cartesia · Sergei · masculine",
        ),
    ]


def test_documentary_voice_options_falls_back_to_configured_cartesia_voice(monkeypatch):
    monkeypatch.setattr(
        documentary_ui.voice_service,
        "get_cartesia_api_key",
        lambda: "cartesia-key",
    )
    monkeypatch.setattr(
        documentary_ui.config,
        "cartesia",
        {
            "voice_id": "voice-123",
            "model_id": "sonic-3.6",
        },
    )
    monkeypatch.setattr(
        documentary_ui.voice_service,
        "list_cartesia_voices",
        lambda language, limit: [],
    )
    monkeypatch.setattr(
        documentary_ui.config,
        "app",
        {"gemini_api_key": ""},
    )
    monkeypatch.setattr(
        documentary_ui.voice_service,
        "get_minimax_tts_api_key",
        lambda: "",
    )
    monkeypatch.setattr(
        documentary_ui.voice_service,
        "get_fish_audio_api_key",
        lambda: "",
    )

    options = _documentary_voice_options("en")

    assert options[0] == (
        "cartesia:voice-123:en",
        "Cartesia · sonic-3.6",
    )
