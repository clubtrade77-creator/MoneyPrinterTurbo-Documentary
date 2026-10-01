from pathlib import Path

import pytest

from app.models.documentary import RightsStatus, SourceType
from app.services.documentary.project import (
    PROJECT_SUBDIRS,
    add_source,
    attach_local_copy_to_source,
    attach_local_file,
    create_project,
    load_project,
    project_dir,
)
from app.services.documentary.youtube_source import (
    build_youtube_source_asset,
    extract_youtube_video_id,
)


def test_create_documentary_project_layout(tmp_path: Path):
    project = create_project(
        "Test Case", project_id="doc_test_case", root=tmp_path
    )

    assert project.id == "doc_test_case"
    base = project_dir(project.id, tmp_path)
    assert (base / "project.json").is_file()
    for subdir in PROJECT_SUBDIRS:
        assert (base / subdir).is_dir()

    loaded = load_project(project.id, tmp_path)
    assert loaded.title == "Test Case"
    assert loaded.sources == []


@pytest.mark.parametrize(
    ("url", "video_id"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=10", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extract_youtube_video_id(url: str, video_id: str):
    assert extract_youtube_video_id(url) == video_id


def test_youtube_asset_is_registered_but_not_renderable_without_local_copy(tmp_path: Path):
    project = create_project(
        "YouTube Case", project_id="doc_youtube_case", root=tmp_path
    )
    source = build_youtube_source_asset(
        "https://youtu.be/dQw4w9WgXcQ",
        title="Reference footage",
        channel="Example Channel",
    )
    assert source.rights_status == RightsStatus.unknown_review_required
    assert source.is_renderable is False

    add_source(project.id, source, root=tmp_path)
    loaded = load_project(project.id, tmp_path)
    assert loaded.sources[0].youtube_video_id == "dQw4w9WgXcQ"
    assert loaded.sources[0].source_type == SourceType.youtube

    media_file = tmp_path / "authorized.mp4"
    media_file.write_bytes(b"authorized-local-copy")
    attached = attach_local_copy_to_source(
        project.id, source.id, media_file, root=tmp_path
    )
    assert attached.is_renderable is True
    assert Path(attached.local_path).is_file()
    assert attached.checksum_sha256


def test_attach_local_file_copies_and_registers_source(tmp_path: Path):
    project = create_project(
        "Local Case", project_id="doc_local_case", root=tmp_path
    )
    media_file = tmp_path / "bodycam.mp4"
    media_file.write_bytes(b"test-video-bytes")

    source = attach_local_file(
        project.id,
        media_file,
        source_type=SourceType.bodycam,
        root=tmp_path,
    )

    assert source.source_type == SourceType.bodycam
    assert source.original_filename == "bodycam.mp4"
    assert Path(source.local_path).is_file()
    loaded = load_project(project.id, tmp_path)
    assert loaded.sources[0].id == source.id
