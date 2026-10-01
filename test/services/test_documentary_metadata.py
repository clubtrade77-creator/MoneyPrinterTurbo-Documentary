import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from app.models.documentary import RightsStatus, VideoMetadata
from app.services.documentary import metadata as metadata_service
from app.services.documentary import project as project_service
from app.services.documentary.metadata import MediaProbeError, probe_video_metadata
from app.services.documentary.project import (
    attach_local_video,
    create_project,
    load_project,
    project_dir,
)


def _probe_result(payload: Any, *, returncode: int = 0, stderr: str = ""):
    return SimpleNamespace(
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr=stderr,
    )


def test_probe_video_metadata_parses_video_and_audio(tmp_path: Path, monkeypatch):
    media_file = tmp_path / "clip.mp4"
    media_file.write_bytes(b"video-bytes")
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "sample_rate": "48000",
            },
        ],
        "format": {
            "duration": "12.345",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        },
    }
    monkeypatch.setattr(
        metadata_service.subprocess,
        "run",
        lambda *args, **kwargs: _probe_result(payload),
    )

    metadata = probe_video_metadata(media_file, ffprobe_binary="ffprobe-test")

    assert metadata.duration_seconds == pytest.approx(12.345)
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.fps == pytest.approx(29.97002997)
    assert metadata.has_audio is True
    assert metadata.video_codec == "h264"
    assert metadata.audio_codec == "aac"
    assert metadata.audio_channels == 2
    assert metadata.audio_sample_rate == 48000
    assert metadata.file_size_bytes == len(b"video-bytes")


def test_probe_video_metadata_applies_display_rotation(tmp_path: Path, monkeypatch):
    media_file = tmp_path / "phone.mov"
    media_file.write_bytes(b"phone-video")
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "hevc",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "side_data_list": [{"rotation": -90}],
            }
        ],
        "format": {"duration": "4.0", "format_name": "mov,mp4"},
    }
    monkeypatch.setattr(
        metadata_service.subprocess,
        "run",
        lambda *args, **kwargs: _probe_result(payload),
    )

    metadata = probe_video_metadata(media_file, ffprobe_binary="ffprobe-test")

    assert metadata.width == 1080
    assert metadata.height == 1920
    assert metadata.rotation_degrees == 270
    assert metadata.has_audio is False


def test_probe_video_metadata_skips_attached_picture_stream(tmp_path: Path, monkeypatch):
    media_file = tmp_path / "with-cover.mp4"
    media_file.write_bytes(b"video-with-cover")
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "mjpeg",
                "width": 600,
                "height": 600,
                "avg_frame_rate": "1/1",
                "disposition": {"attached_pic": 1},
            },
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "24/1",
                "disposition": {"attached_pic": 0},
            },
        ],
        "format": {"duration": "5.0", "format_name": "mov,mp4"},
    }
    monkeypatch.setattr(
        metadata_service.subprocess,
        "run",
        lambda *args, **kwargs: _probe_result(payload),
    )

    metadata = probe_video_metadata(media_file, ffprobe_binary="ffprobe-test")

    assert metadata.video_codec == "h264"
    assert metadata.width == 1280
    assert metadata.height == 720


def test_probe_video_metadata_uses_stream_fallbacks(tmp_path: Path, monkeypatch):
    media_file = tmp_path / "fallback.mp4"
    media_file.write_bytes(b"fallback")
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 640,
                "height": 360,
                "duration": "7.5",
                "avg_frame_rate": "0/0",
                "r_frame_rate": "25/1",
            }
        ],
        "format": {"format_name": "mov,mp4"},
    }
    monkeypatch.setattr(
        metadata_service.subprocess,
        "run",
        lambda *args, **kwargs: _probe_result(payload),
    )

    metadata = probe_video_metadata(media_file, ffprobe_binary="ffprobe-test")

    assert metadata.duration_seconds == pytest.approx(7.5)
    assert metadata.fps == pytest.approx(25.0)


def test_probe_video_metadata_rejects_missing_video_stream(tmp_path: Path, monkeypatch):
    media_file = tmp_path / "audio.mp4"
    media_file.write_bytes(b"audio-only")
    payload = {
        "streams": [{"codec_type": "audio", "codec_name": "aac"}],
        "format": {"duration": "2.0", "format_name": "mov,mp4"},
    }
    monkeypatch.setattr(
        metadata_service.subprocess,
        "run",
        lambda *args, **kwargs: _probe_result(payload),
    )

    with pytest.raises(MediaProbeError, match="no video stream"):
        probe_video_metadata(media_file, ffprobe_binary="ffprobe-test")


def test_probe_video_metadata_rejects_non_object_payload(tmp_path: Path, monkeypatch):
    media_file = tmp_path / "bad-payload.mp4"
    media_file.write_bytes(b"bad-payload")
    monkeypatch.setattr(
        metadata_service.subprocess,
        "run",
        lambda *args, **kwargs: _probe_result([]),
    )

    with pytest.raises(MediaProbeError, match="invalid payload"):
        probe_video_metadata(media_file, ffprobe_binary="ffprobe-test")


def test_probe_video_metadata_reports_ffprobe_failure(tmp_path: Path, monkeypatch):
    media_file = tmp_path / "broken.mp4"
    media_file.write_bytes(b"broken")
    monkeypatch.setattr(
        metadata_service.subprocess,
        "run",
        lambda *args, **kwargs: _probe_result(
            {}, returncode=1, stderr="Invalid data found when processing input"
        ),
    )

    with pytest.raises(MediaProbeError, match="Invalid data found"):
        probe_video_metadata(media_file, ffprobe_binary="ffprobe-test")


def test_probe_video_metadata_reports_timeout(tmp_path: Path, monkeypatch):
    media_file = tmp_path / "slow.mp4"
    media_file.write_bytes(b"slow")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="ffprobe", timeout=1)

    monkeypatch.setattr(metadata_service.subprocess, "run", timeout)

    with pytest.raises(MediaProbeError, match="timed out"):
        probe_video_metadata(
            media_file, ffprobe_binary="ffprobe-test", timeout_seconds=1
        )


def test_probe_video_metadata_rejects_non_positive_timeout(tmp_path: Path, monkeypatch):
    media_file = tmp_path / "clip.mp4"
    media_file.write_bytes(b"video")

    def should_not_run(*args, **kwargs):
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(metadata_service.subprocess, "run", should_not_run)

    with pytest.raises(ValueError, match="greater than zero"):
        probe_video_metadata(media_file, timeout_seconds=0)


def test_attach_local_video_persists_metadata(tmp_path: Path, monkeypatch):
    project = create_project(
        "Video ingest", project_id="doc_video_ingest", root=tmp_path
    )
    media_file = tmp_path / "bodycam.MP4"
    media_file.write_bytes(b"real-video-placeholder")
    expected = VideoMetadata(
        duration_seconds=42.5,
        width=1920,
        height=1080,
        fps=30,
        has_audio=True,
        video_codec="h264",
        audio_codec="aac",
        container="mov,mp4",
        file_size_bytes=len(b"real-video-placeholder"),
        audio_channels=2,
        audio_sample_rate=48000,
    )

    def fake_probe(path):
        assert Path(path).is_file()
        assert project_dir(project.id, tmp_path) / "sources" in Path(path).parents
        return expected

    monkeypatch.setattr(project_service, "probe_video_metadata", fake_probe)

    source = attach_local_video(
        project.id,
        media_file,
        rights_status=RightsStatus.user_owned,
        root=tmp_path,
    )

    assert source.video_metadata == expected
    assert source.local_path.endswith(".mp4")
    loaded = load_project(project.id, tmp_path)
    assert loaded.sources[0].video_metadata == expected


def test_attach_local_video_rejects_unsupported_extension(tmp_path: Path):
    project = create_project(
        "Bad extension", project_id="doc_bad_extension", root=tmp_path
    )
    media_file = tmp_path / "clip.mkv"
    media_file.write_bytes(b"mkv")

    with pytest.raises(ValueError, match="unsupported documentary video extension"):
        attach_local_video(project.id, media_file, root=tmp_path)

    assert list((project_dir(project.id, tmp_path) / "sources").iterdir()) == []
    assert load_project(project.id, tmp_path).sources == []


def test_attach_local_video_probe_failure_rolls_back_copy(tmp_path: Path, monkeypatch):
    project = create_project(
        "Bad video", project_id="doc_bad_video", root=tmp_path
    )
    media_file = tmp_path / "broken.mp4"
    media_file.write_bytes(b"not-a-video")

    def fail_probe(path):
        raise MediaProbeError("simulated invalid video")

    monkeypatch.setattr(project_service, "probe_video_metadata", fail_probe)

    with pytest.raises(MediaProbeError, match="simulated invalid video"):
        attach_local_video(project.id, media_file, root=tmp_path)

    assert list((project_dir(project.id, tmp_path) / "sources").iterdir()) == []
    assert load_project(project.id, tmp_path).sources == []
