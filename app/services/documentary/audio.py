from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from app.models.documentary import NarrationAudioAsset
from app.services.documentary.metadata import resolve_ffprobe_binary
from app.services.documentary.project import (
    load_project,
    project_dir,
    save_project,
    sha256_file,
)

_SUPPORTED_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}
_DEFAULT_PROBE_TIMEOUT_SECONDS = 30.0


class NarrationAudioError(RuntimeError):
    """Raised when documentary narration audio is missing, stale, or invalid."""


def _normalize_language(language: str) -> str:
    value = (language or "").strip().lower()
    if not value:
        raise ValueError("documentary narration language is required")
    parts = value.split("-")
    if not 1 <= len(parts) <= 4:
        raise ValueError("invalid documentary narration language")
    if not (2 <= len(parts[0]) <= 8 and parts[0].isalpha()):
        raise ValueError("invalid documentary narration language")
    for part in parts[1:]:
        if not 1 <= len(part) <= 8 or not part.isalnum():
            raise ValueError("invalid documentary narration language")
    return value


def _positive_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _probe_audio(
    path: Path,
    *,
    timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> tuple[float, str]:
    command = [
        resolve_ffprobe_binary(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NarrationAudioError(
            f"could not inspect narration audio: {path.name}"
        ) from exc

    if completed.returncode != 0:
        raise NarrationAudioError(
            f"ffprobe could not read narration audio: {path.name}"
        )

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise NarrationAudioError(
            f"ffprobe returned invalid narration audio metadata: {path.name}"
        ) from exc

    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise NarrationAudioError(
            f"narration audio has no stream list: {path.name}"
        )
    audio_stream = next(
        (
            stream
            for stream in streams
            if isinstance(stream, dict) and stream.get("codec_type") == "audio"
        ),
        None,
    )
    if not isinstance(audio_stream, dict):
        raise NarrationAudioError(
            f"file contains no narration audio stream: {path.name}"
        )

    format_info = payload.get("format")
    if not isinstance(format_info, dict):
        format_info = {}
    duration = _positive_float(format_info.get("duration")) or _positive_float(
        audio_stream.get("duration")
    )
    if duration is None:
        raise NarrationAudioError(
            f"narration audio duration is unavailable: {path.name}"
        )
    return duration, str(audio_stream.get("codec_name") or "")


def _audio_root(
    project_id: str,
    root: str | os.PathLike | None,
) -> Path:
    return (project_dir(project_id, root) / "audio").resolve()


def _validate_asset_path(
    project_id: str,
    asset: NarrationAudioAsset,
    *,
    root: str | os.PathLike | None,
) -> Path:
    path = Path(asset.local_path).expanduser().resolve()
    audio_root = _audio_root(project_id, root)
    if audio_root != path.parent and audio_root not in path.parents:
        raise NarrationAudioError(
            f"narration audio path escapes project storage: "
            f"{asset.scene_id}/{asset.language}"
        )
    return path


def attach_narration_audio(
    project_id: str,
    scene_id: str,
    source_path: str | os.PathLike,
    *,
    language: str | None = None,
    root: str | os.PathLike | None = None,
) -> NarrationAudioAsset:
    """Copy validated narration audio into a project and track it in project.json."""
    project = load_project(project_id, root)
    if not any(scene.id == scene_id for scene in project.plan.scenes):
        raise ValueError(f"documentary scene not found: {scene_id}")

    resolved_language = _normalize_language(language or project.master_language)
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"narration audio file not found: {source}")
    suffix = source.suffix.lower()
    if suffix not in _SUPPORTED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(_SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(
            f"unsupported narration audio extension; expected one of: {allowed}"
        )

    language_dir = _audio_root(project_id, root) / resolved_language
    language_dir.mkdir(parents=True, exist_ok=True)
    target = language_dir / f"narration-{uuid4().hex[:16]}{suffix}"
    shutil.copy2(source, target)
    target = target.resolve()

    previous = next(
        (
            item
            for item in project.narration_audio
            if item.scene_id == scene_id and item.language == resolved_language
        ),
        None,
    )

    try:
        duration, codec = _probe_audio(target)
        asset = NarrationAudioAsset(
            scene_id=scene_id,
            language=resolved_language,
            local_path=str(target),
            checksum_sha256=sha256_file(target),
            duration_seconds=duration,
            audio_codec=codec,
            file_size_bytes=target.stat().st_size,
        )
        project.narration_audio = [
            item
            for item in project.narration_audio
            if not (
                item.scene_id == scene_id
                and item.language == resolved_language
            )
        ]
        project.narration_audio.append(asset)
        save_project(project, root)
    except Exception:
        target.unlink(missing_ok=True)
        raise

    if previous is not None:
        old_path = _validate_asset_path(project_id, previous, root=root)
        if old_path != target:
            try:
                old_path.unlink(missing_ok=True)
            except OSError:
                pass
    return asset


def load_narration_audio(
    project_id: str,
    scene_id: str,
    *,
    language: str | None = None,
    root: str | os.PathLike | None = None,
) -> NarrationAudioAsset:
    """Load one narration asset and reject missing or modified project audio."""
    project = load_project(project_id, root)
    resolved_language = _normalize_language(language or project.master_language)
    asset = next(
        (
            item
            for item in project.narration_audio
            if item.scene_id == scene_id and item.language == resolved_language
        ),
        None,
    )
    if asset is None:
        raise NarrationAudioError(
            f"narration audio not found for scene {scene_id} "
            f"in language {resolved_language}"
        )

    path = _validate_asset_path(project_id, asset, root=root)
    if not path.is_file():
        raise NarrationAudioError(
            f"narration audio file is missing: {scene_id}/{resolved_language}"
        )
    if sha256_file(path) != asset.checksum_sha256:
        raise NarrationAudioError(
            f"narration audio changed after registration: "
            f"{scene_id}/{resolved_language}"
        )
    return asset
