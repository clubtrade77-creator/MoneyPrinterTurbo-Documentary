from pathlib import Path

import pytest

from app.models.documentary import RightsStatus, SourceAsset, SourceType, VideoMetadata
from app.services.documentary import project as project_service
from app.services.documentary.metadata import MediaProbeError
from app.services.documentary.project import (
    add_source,
    attach_local_copy_to_source,
    create_project,
    load_project,
    project_dir,
)
from app.services.documentary.youtube_source import build_youtube_source_asset


def test_youtube_local_copy_is_probed_and_metadata_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project = create_project(
        "YouTube ingest", project_id="doc_youtube_ingest", root=tmp_path
    )
    source = build_youtube_source_asset(
        "https://youtu.be/dQw4w9WgXcQ",
        rights_status=RightsStatus.unknown_review_required,
    )
    add_source(project.id, source, root=tmp_path)

    media_file = tmp_path / "authorized.mp4"
    media_file.write_bytes(b"youtube-video-copy")
    expected = VideoMetadata(
        duration_seconds=33.25,
        width=1920,
        height=1080,
        fps=29.97,
        has_audio=True,
        video_codec="h264",
        audio_codec="aac",
        container="mov,mp4",
        file_size_bytes=len(b"youtube-video-copy"),
        audio_channels=2,
        audio_sample_rate=48000,
    )

    def fake_probe(path):
        assert Path(path).is_file()
        return expected

    monkeypatch.setattr(project_service, "probe_video_metadata", fake_probe)

    attached = attach_local_copy_to_source(
        project.id, source.id, media_file, root=tmp_path
    )

    assert attached.video_metadata == expected
    assert attached.rights_status == RightsStatus.unknown_review_required
    assert attached.is_publishable is False
    loaded = load_project(project.id, tmp_path)
    assert loaded.sources[0].video_metadata == expected


def test_youtube_local_copy_rejects_non_mp4_mov_without_leaving_file(tmp_path: Path):
    project = create_project(
        "YouTube bad copy", project_id="doc_youtube_bad_copy", root=tmp_path
    )
    source = build_youtube_source_asset("https://youtu.be/dQw4w9WgXcQ")
    add_source(project.id, source, root=tmp_path)

    media_file = tmp_path / "copy.mkv"
    media_file.write_bytes(b"mkv")

    with pytest.raises(ValueError, match="unsupported local copy for video source"):
        attach_local_copy_to_source(
            project.id, source.id, media_file, root=tmp_path
        )

    assert list((project_dir(project.id, tmp_path) / "sources").iterdir()) == []
    loaded = load_project(project.id, tmp_path)
    assert loaded.sources[0].local_path == ""
    assert loaded.sources[0].video_metadata is None


def test_youtube_local_copy_probe_failure_rolls_back_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project = create_project(
        "YouTube broken copy", project_id="doc_youtube_broken_copy", root=tmp_path
    )
    source = build_youtube_source_asset("https://youtu.be/dQw4w9WgXcQ")
    add_source(project.id, source, root=tmp_path)

    media_file = tmp_path / "broken.mp4"
    media_file.write_bytes(b"broken")

    def fail_probe(path):
        raise MediaProbeError("simulated broken external video")

    monkeypatch.setattr(project_service, "probe_video_metadata", fail_probe)

    with pytest.raises(MediaProbeError, match="simulated broken external video"):
        attach_local_copy_to_source(
            project.id, source.id, media_file, root=tmp_path
        )

    assert list((project_dir(project.id, tmp_path) / "sources").iterdir()) == []
    loaded = load_project(project.id, tmp_path)
    assert loaded.sources[0].local_path == ""
    assert loaded.sources[0].checksum_sha256 == ""
    assert loaded.sources[0].video_metadata is None


def test_news_video_local_copy_is_probed_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project = create_project(
        "News video ingest", project_id="doc_news_video_ingest", root=tmp_path
    )
    source = SourceAsset(
        id="source_news_video",
        source_type=SourceType.news,
        title="News footage",
    )
    add_source(project.id, source, root=tmp_path)

    media_file = tmp_path / "news.mov"
    media_file.write_bytes(b"news-video-copy")
    expected = VideoMetadata(
        duration_seconds=18.0,
        width=1280,
        height=720,
        fps=25.0,
        has_audio=True,
        video_codec="h264",
        audio_codec="aac",
        container="mov,mp4",
        file_size_bytes=len(b"news-video-copy"),
    )

    monkeypatch.setattr(project_service, "probe_video_metadata", lambda path: expected)

    attached = attach_local_copy_to_source(
        project.id, source.id, media_file, root=tmp_path
    )

    assert attached.video_metadata == expected
    loaded = load_project(project.id, tmp_path)
    assert loaded.sources[0].video_metadata == expected
