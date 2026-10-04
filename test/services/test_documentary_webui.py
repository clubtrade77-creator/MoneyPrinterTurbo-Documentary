from io import BytesIO

import pytest

import webui.documentary as documentary_ui
from webui.documentary import (
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
