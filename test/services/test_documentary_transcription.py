from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models.documentary import RightsStatus, SourceAsset, SourceType, VideoMetadata
from app.services.documentary import project as project_service
from app.services.documentary.project import (
    attach_local_file,
    attach_local_video,
    create_project,
    load_project,
    save_project,
)
from app.services.documentary.transcription import (
    TranscriptionError,
    load_source_transcript,
    transcript_path,
    transcribe_media,
    transcribe_source,
)


class FakeWhisperModel:
    def __init__(self, segments, info=None):
        self.segments = segments
        self.info = info or SimpleNamespace(
            language="en",
            language_probability=0.98,
        )
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return iter(self.segments), self.info


def _word(start, end, text, probability=0.9):
    return SimpleNamespace(
        start=start,
        end=end,
        word=text,
        probability=probability,
    )


def _segment(
    start,
    end,
    text,
    *,
    words=None,
    avg_logprob=-0.2,
    no_speech_prob=0.03,
):
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        words=words,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


def test_transcribe_media_builds_structured_segment_and_word_timecodes(tmp_path: Path):
    media_file = tmp_path / "interview.mp4"
    media_file.write_bytes(b"media")
    model = FakeWhisperModel(
        [
            _segment(
                1.0,
                3.5,
                " Hello world. ",
                words=[
                    _word(1.0, 1.6, " Hello", 0.94),
                    _word(1.7, 2.4, " world.", 0.91),
                ],
            )
        ]
    )

    transcript = transcribe_media(
        media_file,
        source_id="source_interview",
        source_checksum_sha256="abc123",
        media_duration_seconds=10.0,
        model_override=model,
        model_name="test-whisper",
    )

    assert transcript.source_id == "source_interview"
    assert transcript.source_checksum_sha256 == "abc123"
    assert transcript.language == "en"
    assert transcript.language_probability == pytest.approx(0.98)
    assert transcript.media_duration_seconds == 10.0
    assert transcript.model_size == "test-whisper"
    assert transcript.full_text == "Hello world."
    assert len(transcript.segments) == 1
    assert transcript.segments[0].start_seconds == 1.0
    assert transcript.segments[0].end_seconds == 3.5
    assert transcript.segments[0].words[0].text == "Hello"
    assert transcript.segments[0].words[0].probability == pytest.approx(0.94)
    assert model.calls[0][1]["word_timestamps"] is True
    assert model.calls[0][1]["vad_filter"] is True


def test_transcribe_media_keeps_text_segment_when_word_alignment_is_missing(
    tmp_path: Path,
):
    media_file = tmp_path / "bodycam.mp4"
    media_file.write_bytes(b"media")
    model = FakeWhisperModel([_segment(0.2, 1.8, "Stop right there.", words=None)])

    transcript = transcribe_media(
        media_file,
        source_id="source_bodycam",
        model_override=model,
        model_name="test-whisper",
    )

    assert transcript.full_text == "Stop right there."
    assert len(transcript.segments) == 1
    assert transcript.segments[0].words == []


def test_transcribe_source_persists_and_loads_json_transcript(tmp_path: Path):
    project = create_project(
        "Transcript case", project_id="doc_transcript_case", root=tmp_path
    )
    media_file = tmp_path / "interview.wav"
    media_file.write_bytes(b"audio-placeholder")
    source = attach_local_file(
        project.id,
        media_file,
        source_type=SourceType.audio,
        rights_status=RightsStatus.user_owned,
        root=tmp_path,
    )
    model = FakeWhisperModel(
        [_segment(0.0, 1.2, "Recorded statement.", words=[_word(0.0, 1.0, "Recorded")])]
    )

    transcript = transcribe_source(
        project.id,
        source.id,
        root=tmp_path,
        model_override=model,
        model_name="test-whisper",
    )

    path = transcript_path(project.id, source.id, tmp_path)
    assert path.is_file()
    loaded = load_source_transcript(project.id, source.id, root=tmp_path)
    assert loaded == transcript
    assert loaded.source_checksum_sha256 == source.checksum_sha256


def test_transcribe_source_rejects_video_without_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project = create_project(
        "Silent video", project_id="doc_silent_video", root=tmp_path
    )
    media_file = tmp_path / "silent.mp4"
    media_file.write_bytes(b"silent-video")
    monkeypatch.setattr(
        project_service,
        "probe_video_metadata",
        lambda path: VideoMetadata(
            duration_seconds=4,
            width=1280,
            height=720,
            fps=30,
            has_audio=False,
            video_codec="h264",
            container="mov,mp4",
            file_size_bytes=Path(path).stat().st_size,
        ),
    )
    source = attach_local_video(project.id, media_file, root=tmp_path)

    with pytest.raises(TranscriptionError, match="no audio stream"):
        transcribe_source(
            project.id,
            source.id,
            root=tmp_path,
            model_override=FakeWhisperModel([]),
        )

    assert not transcript_path(project.id, source.id, tmp_path).exists()


def test_load_source_transcript_rejects_stale_source_checksum(tmp_path: Path):
    project = create_project(
        "Stale transcript", project_id="doc_stale_transcript", root=tmp_path
    )
    media_file = tmp_path / "statement.wav"
    media_file.write_bytes(b"audio-one")
    source = attach_local_file(
        project.id,
        media_file,
        source_type=SourceType.audio,
        root=tmp_path,
    )
    transcribe_source(
        project.id,
        source.id,
        root=tmp_path,
        model_override=FakeWhisperModel([_segment(0.0, 1.0, "First version.")]),
        model_name="test-whisper",
    )

    changed = load_project(project.id, tmp_path)
    changed.sources[0].checksum_sha256 = "different-checksum"
    save_project(changed, tmp_path)

    with pytest.raises(TranscriptionError, match="stale"):
        load_source_transcript(project.id, source.id, root=tmp_path)


def test_transcription_failure_does_not_leave_partial_json(tmp_path: Path):
    project = create_project(
        "Failed transcript", project_id="doc_failed_transcript", root=tmp_path
    )
    media_file = tmp_path / "audio.wav"
    media_file.write_bytes(b"audio")
    source = attach_local_file(
        project.id,
        media_file,
        source_type=SourceType.audio,
        root=tmp_path,
    )

    class BrokenModel:
        def transcribe(self, path, **kwargs):
            def broken_segments():
                raise RuntimeError("decoder failed")
                yield

            return broken_segments(), SimpleNamespace(
                language="en", language_probability=1.0
            )

    with pytest.raises(TranscriptionError, match="decoder failed"):
        transcribe_source(
            project.id,
            source.id,
            root=tmp_path,
            model_override=BrokenModel(),
            model_name="broken",
        )

    assert not transcript_path(project.id, source.id, tmp_path).exists()


def test_source_asset_rejects_unsafe_id_used_for_transcript_paths():
    with pytest.raises(ValueError, match="invalid documentary source id"):
        SourceAsset(id="../../escape", source_type=SourceType.audio)


def test_transcript_path_rejects_unsafe_source_id(tmp_path: Path):
    project = create_project(
        "Path safety", project_id="doc_transcript_path", root=tmp_path
    )

    with pytest.raises(ValueError, match="invalid documentary source id"):
        transcript_path(project.id, "../escape", tmp_path)
