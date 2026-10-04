from io import BytesIO

import pytest

from webui.documentary import _write_uploaded_video_to_temp


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
