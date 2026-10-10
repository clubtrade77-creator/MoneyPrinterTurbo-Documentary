from __future__ import annotations

import ipaddress
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.models.documentary import SourceAsset, SourceType
from app.services.documentary.project import (
    attach_local_copy_to_source,
    load_project,
)


DEFAULT_MEDIA_FETCH_TIMEOUT_SECONDS = 900
DEFAULT_DIRECT_DOWNLOAD_TIMEOUT_SECONDS = 120
DEFAULT_MAX_DIRECT_DOWNLOAD_BYTES = 750 * 1024 * 1024


class DocumentaryMediaFetchError(RuntimeError):
    """Raised when Autopilot cannot obtain a local review copy of source media."""


def _is_public_http_url(value: str) -> bool:
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.strip().lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _yt_dlp_command() -> list[str]:
    executable = shutil.which("yt-dlp")
    if executable:
        return [executable]

    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        return [sys.executable, "-m", "yt_dlp"]

    raise DocumentaryMediaFetchError(
        "yt-dlp is required once on this machine for automatic YouTube source "
        "acquisition"
    )


def _download_direct_video(
    url: str,
    destination: Path,
    *,
    timeout_seconds: int = DEFAULT_DIRECT_DOWNLOAD_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_DIRECT_DOWNLOAD_BYTES,
) -> Path:
    if not _is_public_http_url(url):
        raise DocumentaryMediaFetchError("source media URL is not a public HTTP(S) URL")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(10, timeout_seconds),
            allow_redirects=True,
            headers={
                "User-Agent": "MoneyPrinterTurbo-Documentary/1.3 (+autopilot)",
                "Accept": "video/*,*/*;q=0.8",
            },
        ) as response:
            if response.status_code != 200:
                raise DocumentaryMediaFetchError(
                    f"direct source download returned HTTP {response.status_code}"
                )
            if not _is_public_http_url(response.url):
                raise DocumentaryMediaFetchError(
                    "direct source download redirected to a non-public URL"
                )

            content_type = str(response.headers.get("Content-Type") or "").lower()
            if content_type and "video/" not in content_type:
                raise DocumentaryMediaFetchError(
                    f"direct source URL did not return video content: {content_type}"
                )

            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except (TypeError, ValueError):
                    declared_size = 0
                if declared_size > max_bytes:
                    raise DocumentaryMediaFetchError(
                        "direct source video exceeds the automatic download size limit"
                    )

            total = 0
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise DocumentaryMediaFetchError(
                            "direct source video exceeds the automatic download size limit"
                        )
                    handle.write(chunk)
    except DocumentaryMediaFetchError:
        destination.unlink(missing_ok=True)
        raise
    except requests.RequestException as exc:
        destination.unlink(missing_ok=True)
        raise DocumentaryMediaFetchError(
            f"direct source video download failed: {exc}"
        ) from exc

    if not destination.is_file() or destination.stat().st_size <= 0:
        destination.unlink(missing_ok=True)
        raise DocumentaryMediaFetchError("direct source video download was empty")
    return destination


def _yt_dlp_js_runtime_args() -> list[str]:
    """Use any already-installed JS runtime that modern YouTube extraction supports."""
    for runtime, executable_names in (
        ("deno", ("deno",)),
        ("node", ("node", "nodejs")),
        ("bun", ("bun",)),
        ("quickjs", ("qjs", "quickjs")),
    ):
        for executable_name in executable_names:
            executable = shutil.which(executable_name)
            if executable:
                return ["--js-runtimes", f"{runtime}:{executable}"]
    return []


def _download_candidates(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
            and not path.name.endswith(".part")
        ),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )


def _ensure_mp4(source: Path, directory: Path) -> Path:
    if source.suffix.lower() in {".mp4", ".mov"}:
        return source

    target = directory / "source-converted.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise DocumentaryMediaFetchError(
            "automatic source download produced a non-MP4 video and ffmpeg "
            "is unavailable for conversion"
        )

    command = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "21",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(target),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=DEFAULT_MEDIA_FETCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise DocumentaryMediaFetchError(
            "automatic source conversion timed out"
        ) from exc
    except OSError as exc:
        raise DocumentaryMediaFetchError(
            f"could not start automatic source conversion: {exc}"
        ) from exc

    if result.returncode != 0 or not target.is_file() or target.stat().st_size <= 0:
        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        raise DocumentaryMediaFetchError(
            "automatic source conversion failed"
            + (f": {detail}" if detail else "")
        )
    return target


def _download_with_yt_dlp(
    url: str,
    directory: Path,
    *,
    timeout_seconds: int = DEFAULT_MEDIA_FETCH_TIMEOUT_SECONDS,
) -> Path:
    template = directory / "source.%(ext)s"
    runtime_args = _yt_dlp_js_runtime_args()
    selectors = (
        "bv*[height<=1080]+ba/b[height<=1080]/best",
        "bestvideo*+bestaudio/best",
        None,
    )
    attempts: list[str] = []

    for selector in selectors:
        for existing in directory.iterdir():
            if existing.is_file():
                existing.unlink(missing_ok=True)

        command = _yt_dlp_command()
        command.extend(runtime_args)
        command.extend(
            [
                "--no-playlist",
                "--no-progress",
                "--newline",
                "--max-filesize",
                "750M",
                "--merge-output-format",
                "mp4",
                "--remux-video",
                "mp4",
            ]
        )
        if selector:
            command.extend(["-f", selector])
        command.extend(["-o", str(template), url])

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise DocumentaryMediaFetchError(
                f"automatic source download timed out after {timeout_seconds} seconds"
            ) from exc
        except OSError as exc:
            raise DocumentaryMediaFetchError(
                f"could not start automatic source downloader: {exc}"
            ) from exc

        candidates = _download_candidates(directory)
        if result.returncode == 0 and candidates:
            return _ensure_mp4(candidates[0], directory)

        detail = (result.stderr or result.stdout or "").strip()
        if len(detail) > 1200:
            detail = detail[-1200:]
        attempts.append(detail or "requested format unavailable")

    summary = " | ".join(item for item in attempts if item)
    if len(summary) > 3000:
        summary = summary[-3000:]
    raise DocumentaryMediaFetchError(
        "automatic source download failed after format fallbacks"
        + (f": {summary}" if summary else "")
    )



def fetch_source_local_copy(
    project_id: str,
    source_id: str,
    *,
    root: str | Path | None = None,
) -> SourceAsset:
    """Obtain a local technical-review copy and attach it to an existing source.

    Publication rights remain unchanged. This function only supplies media needed
    for analysis and preview rendering; final masters remain rights-gated elsewhere.
    """
    project = load_project(project_id, root)
    source = next((item for item in project.sources if item.id == source_id), None)
    if source is None:
        raise ValueError(f"source not found in project: {source_id}")
    if source.has_local_copy:
        return source

    url = str(source.source_url or "").strip()
    if not url:
        raise DocumentaryMediaFetchError(
            f"source has no downloadable URL: {source_id}"
        )
    if not _is_public_http_url(url):
        raise DocumentaryMediaFetchError(
            f"source URL is not a public HTTP(S) URL: {source_id}"
        )

    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="documentary-source-fetch-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        if suffix in {".mp4", ".mov"} and source.source_type != SourceType.youtube:
            local_path = _download_direct_video(
                url,
                temp_dir_path / f"source{suffix}",
            )
        else:
            local_path = _download_with_yt_dlp(url, temp_dir_path)

        return attach_local_copy_to_source(
            project_id,
            source_id,
            local_path,
            original_filename=local_path.name,
            root=root,
        )
