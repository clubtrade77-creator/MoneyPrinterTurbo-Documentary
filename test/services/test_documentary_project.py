from pathlib import Path

import pytest

from app.models.documentary import (
    ProvenanceType,
    RightsStatus,
    SourceAsset,
    SourceType,
)
from app.services.documentary import project as project_service
from app.services.documentary.project import (
    PROJECT_SUBDIRS,
    ProjectConflictError,
    add_source,
    attach_local_copy_to_source,
    attach_local_file,
    create_project,
    load_project,
    project_dir,
    save_project,
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
    assert project.revision == 1
    base = project_dir(project.id, tmp_path)
    assert (base / "project.json").is_file()
    for subdir in PROJECT_SUBDIRS:
        assert (base / subdir).is_dir()

    loaded = load_project(project.id, tmp_path)
    assert loaded.title == "Test Case"
    assert loaded.sources == []
    assert loaded.revision == 1


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


def test_youtube_asset_keeps_provenance_separate_from_rights(tmp_path: Path):
    project = create_project(
        "YouTube Case", project_id="doc_youtube_case", root=tmp_path
    )
    source = build_youtube_source_asset(
        "https://youtu.be/dQw4w9WgXcQ",
        title="Reference footage",
        channel="Example Channel",
    )
    assert source.provenance == ProvenanceType.third_party_platform
    assert source.rights_status == RightsStatus.unknown_review_required
    assert source.is_renderable is False
    assert source.is_publishable is False

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
    assert attached.is_publishable is False
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
        rights_status=RightsStatus.user_owned,
        root=tmp_path,
    )

    assert source.source_type == SourceType.bodycam
    assert source.provenance == ProvenanceType.user_provided
    assert source.original_filename == "bodycam.mp4"
    assert Path(source.local_path).is_file()
    assert source.is_renderable is True
    assert source.is_publishable is True
    loaded = load_project(project.id, tmp_path)
    assert loaded.sources[0].id == source.id


def test_renderability_requires_existing_file_and_publishability_requires_rights(
    tmp_path: Path,
):
    missing = SourceAsset(
        source_type=SourceType.local_video,
        local_path=str(tmp_path / "missing.mp4"),
        rights_status=RightsStatus.user_owned,
    )
    assert missing.is_renderable is False
    assert missing.is_publishable is False

    media_file = tmp_path / "exists.mp4"
    media_file.write_bytes(b"placeholder")
    review_required = SourceAsset(
        source_type=SourceType.local_video,
        local_path=str(media_file),
        rights_status=RightsStatus.unknown_review_required,
    )
    assert review_required.is_renderable is True
    assert review_required.is_publishable is False


def test_legacy_official_public_source_is_migrated_without_granting_rights():
    source = SourceAsset.model_validate(
        {
            "source_type": "news",
            "rights_status": "official_public_source",
        }
    )
    assert source.provenance == ProvenanceType.official_public_source
    assert source.rights_status == RightsStatus.unknown_review_required
    assert source.is_publishable is False


def test_stale_project_snapshot_cannot_overwrite_newer_revision(tmp_path: Path):
    project = create_project(
        "Revision Case", project_id="doc_revision_case", root=tmp_path
    )
    first = load_project(project.id, tmp_path)
    stale = load_project(project.id, tmp_path)

    first.title = "First writer"
    save_project(first, tmp_path)
    assert first.revision == 2

    stale.title = "Stale writer"
    with pytest.raises(ProjectConflictError):
        save_project(stale, tmp_path)

    loaded = load_project(project.id, tmp_path)
    assert loaded.title == "First writer"
    assert loaded.revision == 2


def test_failed_local_copy_manifest_save_rolls_back_new_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project = create_project(
        "Rollback Case", project_id="doc_rollback_case", root=tmp_path
    )
    source = build_youtube_source_asset("https://youtu.be/dQw4w9WgXcQ")
    add_source(project.id, source, root=tmp_path)

    media_file = tmp_path / "authorized.mp4"
    media_file.write_bytes(b"authorized-local-copy")

    def fail_write(*args, **kwargs):
        raise RuntimeError("simulated manifest write failure")

    monkeypatch.setattr(project_service, "_atomic_write_json", fail_write)
    with pytest.raises(RuntimeError, match="simulated manifest write failure"):
        attach_local_copy_to_source(
            project.id, source.id, media_file, root=tmp_path
        )

    source_files = list((project_dir(project.id, tmp_path) / "sources").iterdir())
    assert source_files == []
    loaded = load_project(project.id, tmp_path)
    assert loaded.sources[0].local_path == ""
    assert loaded.sources[0].checksum_sha256 == ""
