from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from app.models.documentary import (
    DocumentaryProject,
    ProvenanceType,
    RightsStatus,
    SourceAsset,
    SourceType,
    VideoMetadata,
    utc_now,
)
from app.services.documentary.metadata import probe_video_metadata
from app.utils.utils import storage_dir

PROJECT_SUBDIRS = (
    "sources",
    "transcripts",
    "research",
    "plans",
    "audio/en",
    "audio/ru",
    "audio/es",
    "subtitles",
    "renders",
    "manifests",
)
_PROJECT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,120}$")
_PROJECT_LOCK_TIMEOUT_SECONDS = 10.0
_PROJECT_LOCK_POLL_SECONDS = 0.05
_MALFORMED_LOCK_STALE_SECONDS = 5.0
_LOCAL_VIDEO_EXTENSIONS = {".mp4", ".mov"}
_VIDEO_SOURCE_TYPES = {
    SourceType.local_video,
    SourceType.youtube,
    SourceType.bodycam,
    SourceType.cctv,
    SourceType.court,
    SourceType.interview,
    SourceType.news,
    SourceType.broll,
}


class ProjectConflictError(RuntimeError):
    """Raised when a stale project snapshot would overwrite newer project data."""


def default_documentary_root() -> Path:
    """Return a repository-anchored storage root, independent of process cwd."""
    return Path(storage_dir("documentary")).resolve()


def _validate_project_id(project_id: str) -> str:
    if not _PROJECT_ID_RE.fullmatch(project_id or ""):
        raise ValueError("invalid documentary project id")
    return project_id


def project_dir(project_id: str, root: str | os.PathLike | None = None) -> Path:
    base = Path(root) if root is not None else default_documentary_root()
    return base / _validate_project_id(project_id)


def project_manifest_path(
    project_id: str, root: str | os.PathLike | None = None
) -> Path:
    return project_dir(project_id, root) / "project.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _remove_stale_lock(lock_path: Path) -> bool:
    """Best-effort recovery for a lock left behind by a crashed local process."""
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return True
    except OSError:
        return False

    match = re.fullmatch(r"pid=(\d+)", raw)
    if match:
        if _process_is_alive(int(match.group(1))):
            return False
    else:
        try:
            age = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            return True
        except OSError:
            return False
        if age < _MALFORMED_LOCK_STALE_SECONDS:
            return False

    try:
        lock_path.unlink()
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


@contextmanager
def _project_lock(
    project_id: str,
    root: str | os.PathLike | None = None,
    *,
    timeout: float = _PROJECT_LOCK_TIMEOUT_SECONDS,
):
    """Cross-process lock for short project manifest mutations.

    Atomic ``O_EXCL`` creation works on the supported local filesystems without adding
    a new dependency. A lock left by a crashed local process is recovered by checking
    the recorded PID. The lock is deliberately held only around project mutations.
    """
    target = project_dir(project_id, root)
    if not target.is_dir():
        raise FileNotFoundError(f"documentary project not found: {project_id}")

    lock_path = target / ".project.lock"
    deadline = time.monotonic() + timeout
    fd = None
    while fd is None:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
            except Exception:
                os.close(fd)
                fd = None
                lock_path.unlink(missing_ok=True)
                raise
        except FileExistsError:
            if _remove_stale_lock(lock_path):
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for documentary project lock: {project_id}"
                )
            time.sleep(_PROJECT_LOCK_POLL_SECONDS)

    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock_path.unlink(missing_ok=True)


def _load_project_unlocked(
    project_id: str, root: str | os.PathLike | None = None
) -> DocumentaryProject:
    manifest_path = project_manifest_path(project_id, root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"documentary project not found: {project_id}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DocumentaryProject.model_validate(payload)


def _validate_local_source_paths(
    project: DocumentaryProject, root: str | os.PathLike | None = None
) -> None:
    """Keep persisted local media references inside this documentary project."""
    base_dir = project_dir(project.id, root).resolve()
    sources_dir = (base_dir / "sources").resolve()
    audio_dir = (base_dir / "audio").resolve()

    for source in project.sources:
        if not source.local_path:
            continue
        resolved = Path(source.local_path).expanduser().resolve()
        if sources_dir != resolved.parent and sources_dir not in resolved.parents:
            raise ValueError(
                f"documentary source local_path escapes project sources directory: {source.id}"
            )

    for asset in project.narration_audio:
        resolved = Path(asset.local_path).expanduser().resolve()
        if audio_dir != resolved.parent and audio_dir not in resolved.parents:
            raise ValueError(
                "documentary narration audio path escapes project audio directory: "
                f"{asset.scene_id}/{asset.language}"
            )


def _save_project_unlocked(
    project: DocumentaryProject, root: str | os.PathLike | None = None
) -> Path:
    """Persist one already-locked project snapshot with optimistic revision checking."""
    manifest_path = project_manifest_path(project.id, root)
    current_revision = 0
    if manifest_path.is_file():
        current_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_revision = int(current_payload.get("revision", 0))

    if project.revision != current_revision:
        raise ProjectConflictError(
            f"stale documentary project revision: expected {current_revision}, "
            f"got {project.revision}"
        )

    _validate_local_source_paths(project, root)

    updated_at = utc_now()
    next_revision = current_revision + 1
    payload = project.model_dump(mode="json")
    payload["updated_at"] = updated_at.isoformat()
    payload["revision"] = next_revision
    _atomic_write_json(manifest_path, payload)

    project.updated_at = updated_at
    project.revision = next_revision
    return manifest_path


def save_project(
    project: DocumentaryProject, root: str | os.PathLike | None = None
) -> Path:
    with _project_lock(project.id, root):
        return _save_project_unlocked(project, root)


def create_project(
    title: str,
    *,
    master_language: str = "en",
    project_id: str | None = None,
    root: str | os.PathLike | None = None,
) -> DocumentaryProject:
    clean_title = (title or "").strip()
    if not clean_title:
        raise ValueError("documentary project title is required")

    project_id = project_id or f"doc_{uuid4().hex[:12]}"
    target = project_dir(project_id, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.mkdir(exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"documentary project already exists: {project_id}") from None

    try:
        for subdir in PROJECT_SUBDIRS:
            (target / subdir).mkdir(parents=True, exist_ok=False)

        project = DocumentaryProject(
            id=project_id,
            title=clean_title,
            master_language=(master_language or "en").strip() or "en",
        )
        save_project(project, root)
        return project
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise


def load_project(
    project_id: str, root: str | os.PathLike | None = None
) -> DocumentaryProject:
    # project.json is replaced atomically, so readers either see the previous complete
    # manifest or the next complete manifest and do not need to hold the write lock.
    return _load_project_unlocked(project_id, root)


def list_projects(
    root: str | os.PathLike | None = None,
) -> list[DocumentaryProject]:
    """Return valid documentary projects sorted by most recently updated first.

    Unknown directories and malformed manifests are ignored so one abandoned or
    partially copied folder cannot make the Documentary UI unusable.
    """
    base = Path(root) if root is not None else default_documentary_root()
    if not base.is_dir():
        return []

    projects: list[DocumentaryProject] = []
    for candidate in base.iterdir():
        if not candidate.is_dir() or not _PROJECT_ID_RE.fullmatch(candidate.name):
            continue
        if not (candidate / "project.json").is_file():
            continue
        try:
            projects.append(_load_project_unlocked(candidate.name, root))
        except (OSError, ValueError, json.JSONDecodeError):
            continue

    return sorted(
        projects,
        key=lambda project: (project.updated_at, project.id),
        reverse=True,
    )


def add_source(
    project_id: str,
    source: SourceAsset,
    *,
    root: str | os.PathLike | None = None,
) -> DocumentaryProject:
    with _project_lock(project_id, root):
        project = _load_project_unlocked(project_id, root)
        existing_ids = {item.id for item in project.sources}
        if source.id in existing_ids:
            raise ValueError(f"source already exists in project: {source.id}")
        project.sources.append(source)
        _save_project_unlocked(project, root)
        return project


def set_narrator_voice(
    project_id: str,
    language: str,
    voice_name: str,
    *,
    root: str | os.PathLike | None = None,
) -> DocumentaryProject:
    language_key = str(language or "").strip().lower()
    voice_value = str(voice_name or "").strip()
    if not re.fullmatch(r"[a-z]{2,8}(?:-[a-z0-9]{1,8}){0,3}", language_key):
        raise ValueError("invalid documentary narrator voice language")
    if not voice_value or len(voice_value) > 300:
        raise ValueError("invalid documentary narrator voice")

    with _project_lock(project_id, root):
        project = _load_project_unlocked(project_id, root)
        project.narrator_voices[language_key] = voice_value
        _save_project_unlocked(project, root)
        return project


def sha256_file(path: str | os.PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_local_source(
    project_id: str,
    source_path: Path,
    *,
    title: str,
    source_type: SourceType,
    rights_status: RightsStatus,
    rights_note: str,
    original_filename: str,
    root: str | os.PathLike | None,
    probe_video: bool,
) -> SourceAsset:
    target_dir = project_dir(project_id, root) / "sources"
    if not target_dir.is_dir():
        raise FileNotFoundError(f"documentary project not found: {project_id}")

    source_id = f"source_{uuid4().hex[:12]}"
    target_path = target_dir / f"{source_id}{source_path.suffix.lower()}"
    shutil.copy2(source_path, target_path)
    target_path = target_path.resolve()

    try:
        video_metadata: VideoMetadata | None = None
        if probe_video:
            video_metadata = probe_video_metadata(target_path)

        source = SourceAsset(
            id=source_id,
            source_type=source_type,
            provenance=ProvenanceType.user_provided,
            title=(title or source_path.stem).strip(),
            original_filename=(
                Path(original_filename).name
                if original_filename
                else source_path.name
            ),
            local_path=str(target_path),
            checksum_sha256=sha256_file(target_path),
            video_metadata=video_metadata,
            rights_status=rights_status,
            rights_note=rights_note,
        )
        add_source(project_id, source, root=root)
        return source
    except Exception:
        target_path.unlink(missing_ok=True)
        raise


def attach_local_file(
    project_id: str,
    source_path: str | os.PathLike,
    *,
    title: str = "",
    source_type: SourceType = SourceType.local_video,
    rights_status: RightsStatus = RightsStatus.unknown_review_required,
    rights_note: str = "",
    original_filename: str = "",
    root: str | os.PathLike | None = None,
) -> SourceAsset:
    """Copy a generic user-provided file into the project and register it."""
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source file not found: {source_path}")

    return _copy_local_source(
        project_id,
        source_path,
        title=title,
        source_type=source_type,
        rights_status=rights_status,
        rights_note=rights_note,
        original_filename=original_filename,
        root=root,
        probe_video=False,
    )


def attach_local_video(
    project_id: str,
    source_path: str | os.PathLike,
    *,
    title: str = "",
    source_type: SourceType = SourceType.local_video,
    rights_status: RightsStatus = RightsStatus.unknown_review_required,
    rights_note: str = "",
    original_filename: str = "",
    root: str | os.PathLike | None = None,
) -> SourceAsset:
    """Copy an MP4/MOV into the project and persist verified ffprobe metadata.

    The copied project file, not the external original, is probed. This avoids recording
    metadata for a different file if the original changes while ingestion is running.
    """
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source file not found: {source_path}")
    if source_path.suffix.lower() not in _LOCAL_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(_LOCAL_VIDEO_EXTENSIONS))
        raise ValueError(f"unsupported documentary video extension; expected one of: {allowed}")

    return _copy_local_source(
        project_id,
        source_path,
        title=title,
        source_type=source_type,
        rights_status=rights_status,
        rights_note=rights_note,
        original_filename=original_filename,
        root=root,
        probe_video=True,
    )


def attach_local_copy_to_source(
    project_id: str,
    source_id: str,
    source_path: str | os.PathLike,
    *,
    root: str | os.PathLike | None = None,
) -> SourceAsset:
    """Attach a local media copy to an existing external source asset.

    Rights metadata is intentionally unchanged: possessing a local copy does not prove
    permission to publish it. Video-like sources require MP4/MOV and are probed before
    the manifest is updated so downstream transcription/rendering sees verified media
    metadata. The copied file is removed if probing or manifest update fails.
    """
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source file not found: {source_path}")

    with _project_lock(project_id, root):
        project = _load_project_unlocked(project_id, root)
        source = next((item for item in project.sources if item.id == source_id), None)
        if source is None:
            raise ValueError(f"source not found in project: {source_id}")

        suffix = source_path.suffix.lower()
        should_probe_video = source.source_type in _VIDEO_SOURCE_TYPES
        if should_probe_video and suffix not in _LOCAL_VIDEO_EXTENSIONS:
            allowed = ", ".join(sorted(_LOCAL_VIDEO_EXTENSIONS))
            raise ValueError(
                f"unsupported local copy for video source; expected one of: {allowed}"
            )

        target_dir = project_dir(project_id, root) / "sources"
        target_path = target_dir / f"{source.id}-{uuid4().hex[:8]}{suffix}"
        old_local_path = source.local_path
        old_video_metadata = source.video_metadata
        shutil.copy2(source_path, target_path)
        target_path = target_path.resolve()

        try:
            video_metadata = (
                probe_video_metadata(target_path) if should_probe_video else old_video_metadata
            )
            source.original_filename = source_path.name
            source.local_path = str(target_path)
            source.checksum_sha256 = sha256_file(target_path)
            source.video_metadata = video_metadata
            _save_project_unlocked(project, root)
        except Exception:
            target_path.unlink(missing_ok=True)
            raise

        if old_local_path:
            old_path = Path(old_local_path)
            try:
                old_resolved = old_path.expanduser().resolve()
                target_dir_resolved = target_dir.resolve()
                if old_resolved != target_path and target_dir_resolved in old_resolved.parents:
                    old_path.unlink(missing_ok=True)
            except OSError:
                # The manifest already points at the new valid copy; stale-file cleanup
                # is best-effort and must not roll back the successful attachment.
                pass
        return source
