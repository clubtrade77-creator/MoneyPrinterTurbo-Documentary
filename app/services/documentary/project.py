from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from uuid import uuid4

from app.models.documentary import DocumentaryProject, SourceAsset, SourceType, utc_now

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


def default_documentary_root() -> Path:
    return Path("storage") / "documentary"


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
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temp_path, path)


def save_project(
    project: DocumentaryProject, root: str | os.PathLike | None = None
) -> Path:
    project.updated_at = utc_now()
    manifest_path = project_manifest_path(project.id, root)
    _atomic_write_json(manifest_path, project.model_dump(mode="json"))
    return manifest_path


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
    if target.exists():
        raise FileExistsError(f"documentary project already exists: {project_id}")

    for subdir in PROJECT_SUBDIRS:
        (target / subdir).mkdir(parents=True, exist_ok=True)

    project = DocumentaryProject(
        id=project_id,
        title=clean_title,
        master_language=(master_language or "en").strip() or "en",
    )
    save_project(project, root)
    return project


def load_project(
    project_id: str, root: str | os.PathLike | None = None
) -> DocumentaryProject:
    manifest_path = project_manifest_path(project_id, root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"documentary project not found: {project_id}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DocumentaryProject.model_validate(payload)


def add_source(
    project_id: str,
    source: SourceAsset,
    *,
    root: str | os.PathLike | None = None,
) -> DocumentaryProject:
    project = load_project(project_id, root)
    existing_ids = {item.id for item in project.sources}
    if source.id in existing_ids:
        raise ValueError(f"source already exists in project: {source.id}")
    project.sources.append(source)
    save_project(project, root)
    return project


def sha256_file(path: str | os.PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def attach_local_file(
    project_id: str,
    source_path: str | os.PathLike,
    *,
    title: str = "",
    source_type: SourceType = SourceType.local_video,
    root: str | os.PathLike | None = None,
) -> SourceAsset:
    """Copy a user-provided file into the project's sources directory and register it."""
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source file not found: {source_path}")

    target_dir = project_dir(project_id, root) / "sources"
    if not target_dir.is_dir():
        raise FileNotFoundError(f"documentary project not found: {project_id}")

    source_id = f"source_{uuid4().hex[:12]}"
    safe_suffix = source_path.suffix.lower()
    target_path = target_dir / f"{source_id}{safe_suffix}"
    shutil.copy2(source_path, target_path)

    source = SourceAsset(
        id=source_id,
        source_type=source_type,
        title=(title or source_path.stem).strip(),
        original_filename=source_path.name,
        local_path=str(target_path),
        checksum_sha256=sha256_file(target_path),
    )
    try:
        add_source(project_id, source, root=root)
    except Exception:
        target_path.unlink(missing_ok=True)
        raise
    return source


def attach_local_copy_to_source(
    project_id: str,
    source_id: str,
    source_path: str | os.PathLike,
    *,
    root: str | os.PathLike | None = None,
) -> SourceAsset:
    """Attach an authorized/local media copy to an existing external source asset."""
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source file not found: {source_path}")

    project = load_project(project_id, root)
    source = next((item for item in project.sources if item.id == source_id), None)
    if source is None:
        raise ValueError(f"source not found in project: {source_id}")

    target_dir = project_dir(project_id, root) / "sources"
    target_path = target_dir / f"{source.id}{source_path.suffix.lower()}"
    shutil.copy2(source_path, target_path)
    source.original_filename = source_path.name
    source.local_path = str(target_path)
    source.checksum_sha256 = sha256_file(target_path)
    save_project(project, root)
    return source
