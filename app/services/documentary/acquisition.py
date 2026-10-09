from __future__ import annotations

import ipaddress
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

from app.models.documentary import ProvenanceType, SourceAsset, SourceType
from app.services.documentary.project import (
    attach_local_copy_to_source,
    load_project,
)
from app.utils import utils

_MAX_ACQUISITION_BYTES = 1024 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_MAX_REDIRECTS = 3
_DIRECT_VIDEO_EXTENSIONS = {".mp4", ".mov"}
_YOUTUBE_FORMAT = (
    "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/"
    "b[ext=mp4][height<=1080]/"
    "bv*[height<=1080]+ba/b[height<=1080]"
)
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    )
}


class SourceAcquisitionError(RuntimeError):
    """Raised when an external documentary source cannot be acquired safely."""


def _source_by_id(project, source_id: str) -> SourceAsset:
    source = next((item for item in project.sources if item.id == source_id), None)
    if source is None:
        raise ValueError(f"source not found in project: {source_id}")
    return source


def _direct_extension(url: str) -> str:
    try:
        return Path(urlparse(url).path).suffix.lower()
    except ValueError:
        return ""


def _is_public_http_url(url: str) -> bool:
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return False
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return False

    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return False

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return bool(address.is_global)


def can_auto_acquire(source: SourceAsset) -> bool:
    """Whether the current acquisition service can fetch this source automatically."""
    if source.has_local_copy or not source.source_url:
        return False
    if source.source_type == SourceType.youtube:
        return True
    return (
        source.provenance == ProvenanceType.official_public_source
        and _direct_extension(source.source_url) in _DIRECT_VIDEO_EXTENSIONS
        and _is_public_http_url(source.source_url)
    )


def _download_direct_video(
    url: str,
    target_dir: Path,
    *,
    timeout_seconds: float,
    session=None,
) -> Path:
    if not _is_public_http_url(url):
        raise SourceAcquisitionError("direct source URL is not a safe public HTTP(S) URL")

    extension = _direct_extension(url)
    if extension not in _DIRECT_VIDEO_EXTENSIONS:
        raise SourceAcquisitionError("direct source must be an MP4 or MOV URL")

    request = session or requests
    current_url = url
    response = None
    target_path = target_dir / f"source{extension}"

    try:
        for redirect_index in range(_MAX_REDIRECTS + 1):
            response = request.get(
                current_url,
                headers=_REQUEST_HEADERS,
                timeout=(20, max(20.0, float(timeout_seconds))),
                stream=True,
                allow_redirects=False,
            )
            status = int(getattr(response, "status_code", 0) or 0)
            if 300 <= status < 400:
                if redirect_index >= _MAX_REDIRECTS:
                    raise SourceAcquisitionError("direct source redirected too many times")
                location = str(
                    (getattr(response, "headers", {}) or {}).get("Location") or ""
                ).strip()
                next_url = urljoin(current_url, location)
                if not location or not _is_public_http_url(next_url):
                    raise SourceAcquisitionError(
                        "direct source redirected to an unsafe URL"
                    )
                close = getattr(response, "close", None)
                if callable(close):
                    close()
                response = None
                current_url = next_url
                continue

            if status != 200:
                raise SourceAcquisitionError(
                    f"direct source returned HTTP {status or 'unknown'}"
                )
            break

        if response is None:
            raise SourceAcquisitionError("direct source returned no response")

        headers = getattr(response, "headers", {}) or {}
        content_type = str(headers.get("Content-Type") or "").lower()
        if content_type.startswith("text/") or "html" in content_type:
            raise SourceAcquisitionError("direct source returned non-video content")

        try:
            declared_size = int(headers.get("Content-Length") or 0)
        except (TypeError, ValueError):
            declared_size = 0
        if declared_size > _MAX_ACQUISITION_BYTES:
            raise SourceAcquisitionError("source video exceeds the 1 GB safety limit")

        downloaded = 0
        with open(target_path, "wb") as output:
            for chunk in response.iter_content(chunk_size=_COPY_CHUNK_BYTES):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > _MAX_ACQUISITION_BYTES:
                    raise SourceAcquisitionError(
                        "source video exceeds the 1 GB safety limit"
                    )
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())

        if downloaded == 0:
            raise SourceAcquisitionError("direct source download was empty")
        return target_path
    except requests.RequestException as exc:
        raise SourceAcquisitionError("direct source download failed") from exc
    finally:
        if response is not None:
            close = getattr(response, "close", None)
            if callable(close):
                close()


def _download_youtube_video(
    source: SourceAsset,
    target_dir: Path,
    *,
    timeout_seconds: float,
    ydl_factory=None,
) -> Path:
    if source.source_type != SourceType.youtube:
        raise SourceAcquisitionError("source is not a YouTube asset")

    if ydl_factory is None:
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:
            raise SourceAcquisitionError(
                "yt-dlp is not installed in the project environment"
            ) from exc
        ydl_factory = YoutubeDL

    output_template = str(target_dir / "youtube-%(id)s.%(ext)s")
    options = {
        "format": _YOUTUBE_FORMAT,
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreconfig": True,
        "merge_output_format": "mp4",
        "socket_timeout": min(max(float(timeout_seconds), 10.0), 60.0),
        "max_filesize": _MAX_ACQUISITION_BYTES,
        "ffmpeg_location": utils.get_ffmpeg_binary(),
    }

    try:
        with ydl_factory(options) as downloader:
            downloader.extract_info(source.source_url, download=True)
    except Exception as exc:
        raise SourceAcquisitionError("YouTube source download failed") from exc

    candidates = sorted(
        (
            path
            for path in target_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _DIRECT_VIDEO_EXTENSIONS
        ),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if not candidates:
        raise SourceAcquisitionError(
            "YouTube download did not produce an MP4/MOV file"
        )

    result = candidates[0]
    if result.stat().st_size <= 0:
        raise SourceAcquisitionError("YouTube source download was empty")
    if result.stat().st_size > _MAX_ACQUISITION_BYTES:
        raise SourceAcquisitionError("source video exceeds the 1 GB safety limit")
    return result


def acquire_source(
    project_id: str,
    source_id: str,
    *,
    root: str | os.PathLike | None = None,
    timeout_seconds: float = 600.0,
    session=None,
    ydl_factory=None,
) -> SourceAsset:
    """Acquire a local analysis copy for a registered external video source.

    Reuse/publication rights are deliberately left unchanged. Acquisition only makes
    the bytes locally available for transcription, clip selection and review.
    """
    project = load_project(project_id, root)
    source = _source_by_id(project, source_id)
    if source.has_local_copy:
        return source
    if not can_auto_acquire(source):
        raise SourceAcquisitionError(
            "source is not supported for automatic local acquisition"
        )

    with tempfile.TemporaryDirectory(prefix="documentary-source-") as temp_dir_value:
        temp_dir = Path(temp_dir_value)
        if source.source_type == SourceType.youtube:
            downloaded_path = _download_youtube_video(
                source,
                temp_dir,
                timeout_seconds=timeout_seconds,
                ydl_factory=ydl_factory,
            )
        else:
            downloaded_path = _download_direct_video(
                source.source_url,
                temp_dir,
                timeout_seconds=timeout_seconds,
                session=session,
            )

        return attach_local_copy_to_source(
            project_id,
            source.id,
            downloaded_path,
            root=root,
        )
