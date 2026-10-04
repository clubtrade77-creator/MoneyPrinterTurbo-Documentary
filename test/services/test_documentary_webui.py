from io import BytesIO

import pytest

import webui.documentary as documentary_ui
from webui.documentary import (
    _format_transcript_segment,
    _format_transcript_time,
    _load_transcript_if_available,
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
