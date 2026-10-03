from pathlib import Path

import pytest

from app.models.documentary import (
    DocumentaryPlan,
    DocumentaryProject,
    DocumentaryScene,
    ProvenanceType,
    RightsStatus,
    SceneType,
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
    default_documentary_root,
    load_project,
    project_dir,
    save_project,
)
from app.services.documentary.youtube_source import (
    build_youtube_source_asset,
    extract_youtube_video_id,
)


def test_default_documentary_root_is_independent_of_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    root = default_documentary_root()

    assert root.is_absolute()
    assert root.name == "documentary"
    assert root.parent.name == "storage"


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


def test_duplicate_project_creation_does_not_destroy_existing_project(tmp_path: Path):
    project = create_project(
        "Original", project_id="doc_duplicate_case", root=tmp_path
    )

    with pytest.raises(FileExistsError):
        create_project("Replacement", project_id=project.id, root=tmp_path)

    loaded = load_project(project.id, tmp_path)
    assert loaded.title == "Original"
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
        published_at="2026-09-30",
    )
    assert source.provenance == ProvenanceType.third_party_platform
    assert source.rights_status == RightsStatus.unknown_review_required
    assert source.publication_date == "2026-09-30"
    assert source.youtube_published_at == "2026-09-30"
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
    assert Path(attached.local_path).is_absolute()
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
    assert Path(source.local_path).is_absolute()
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


def test_project_rejects_duplicate_source_ids():
    first = SourceAsset(id="source_duplicate", source_type=SourceType.news)
    second = SourceAsset(id="source_duplicate", source_type=SourceType.photo)

    with pytest.raises(ValueError, match="duplicate source ids"):
        DocumentaryProject(id="doc_duplicate_sources", title="Duplicate", sources=[first, second])


def test_project_rejects_duplicate_scene_ids():
    source = SourceAsset(id="source_one", source_type=SourceType.bodycam)
    first = DocumentaryScene(
        id="scene_duplicate",
        scene_type=SceneType.original_clip,
        source_id=source.id,
        source_start=0,
        source_end=1,
    )
    second = first.model_copy()

    with pytest.raises(ValueError, match="duplicate scene ids"):
        DocumentaryProject(
            id="doc_duplicate_scenes",
            title="Duplicate scenes",
            sources=[source],
            plan=DocumentaryPlan(scenes=[first, second]),
        )


def test_project_rejects_scene_reference_to_unknown_source():
    scene = DocumentaryScene(
        scene_type=SceneType.original_clip,
        source_id="source_missing",
        source_start=0,
        source_end=2,
    )

    with pytest.raises(ValueError, match="unknown source_id"):
        DocumentaryProject(
            id="doc_bad_reference",
            title="Bad reference",
            plan=DocumentaryPlan(scenes=[scene]),
        )


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


def test_stale_process_lock_is_recovered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    project = create_project(
        "Lock Case", project_id="doc_lock_case", root=tmp_path
    )
    lock_path = project_dir(project.id, tmp_path) / ".project.lock"
    lock_path.write_text("pid=12345\n", encoding="utf-8")
    monkeypatch.setattr(project_service, "_process_is_alive", lambda pid: False)

    loaded = load_project(project.id, tmp_path)
    loaded.title = "Recovered"
    save_project(loaded, tmp_path)

    assert load_project(project.id, tmp_path).title == "Recovered"
    assert not lock_path.exists()


def test_source_local_path_cannot_escape_project_sources(tmp_path: Path):
    project = create_project(
        "Path Case", project_id="doc_path_case", root=tmp_path
    )
    outside_file = tmp_path / "outside.mp4"
    outside_file.write_bytes(b"outside")
    source = SourceAsset(
        source_type=SourceType.local_video,
        local_path=str(outside_file),
    )

    with pytest.raises(ValueError, match="escapes project sources directory"):
        add_source(project.id, source, root=tmp_path)

    assert load_project(project.id, tmp_path).sources == []


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
