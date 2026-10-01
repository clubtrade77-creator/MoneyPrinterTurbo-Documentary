from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from app.models.documentary import VideoMetadata
from app.utils.utils import get_ffmpeg_binary


class MediaProbeError(RuntimeError):
    """Raised when a local video cannot be inspected reliably."""


def resolve_ffprobe_binary(explicit: str | None = None) -> str:
    """Resolve ffprobe without assuming a particular operating system layout."""
    if explicit:
        return explicit

    configured = os.environ.get("FFPROBE_EXE")
    if configured:
        return configured

    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe:
        return system_ffprobe

    ffmpeg_binary = get_ffmpeg_binary()
    ffmpeg_path = Path(ffmpeg_binary)
    if ffmpeg_path.name.lower().startswith("ffmpeg") and ffmpeg_path.exists():
        sibling_name = "ffprobe.exe" if ffmpeg_path.suffix.lower() == ".exe" else "ffprobe"
        sibling = ffmpeg_path.with_name(sibling_name)
        if sibling.is_file():
            return str(sibling)

    return "ffprobe"


def _positive_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _positive_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_fps(value: Any) -> float | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    try:
        if isinstance(value, (int, float)):
            fps = float(value)
        else:
            fps = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(fps) or fps <= 0:
        return None
    return fps


def _rotation_degrees(video_stream: dict[str, Any]) -> int:
    raw_rotation: Any = None
    for side_data in video_stream.get("side_data_list", []) or []:
        if "rotation" in side_data:
            raw_rotation = side_data.get("rotation")
            break
    if raw_rotation is None:
        raw_rotation = (video_stream.get("tags") or {}).get("rotate")

    try:
        rotation = int(round(float(raw_rotation))) if raw_rotation is not None else 0
    except (TypeError, ValueError):
        return 0
    return rotation % 360


def _trim_probe_error(stderr: str, limit: int = 1200) -> str:
    clean = (stderr or "").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def probe_video_metadata(
    file_path: str | os.PathLike,
    *,
    ffprobe_binary: str | None = None,
    timeout_seconds: float = 30.0,
) -> VideoMetadata:
    """Inspect one local video with ffprobe and return normalized technical metadata."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"video file not found: {path}")

    command = [
        resolve_ffprobe_binary(ffprobe_binary),
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
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise MediaProbeError(
            "ffprobe was not found; install FFmpeg/ffprobe or set FFPROBE_EXE"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaProbeError(
            f"ffprobe timed out after {timeout_seconds:g}s while inspecting {path.name}"
        ) from exc

    if completed.returncode != 0:
        detail = _trim_probe_error(completed.stderr)
        suffix = f": {detail}" if detail else ""
        raise MediaProbeError(f"ffprobe could not read video {path.name}{suffix}")

    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaProbeError(f"ffprobe returned invalid JSON for {path.name}") from exc

    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise MediaProbeError(f"ffprobe returned no stream list for {path.name}")

    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"), None
    )
    if not isinstance(video_stream, dict):
        raise MediaProbeError(f"file contains no video stream: {path.name}")

    audio_streams = [
        stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    ]
    audio_stream = audio_streams[0] if audio_streams else None
    format_info = payload.get("format") if isinstance(payload.get("format"), dict) else {}

    duration = _positive_float(format_info.get("duration")) or _positive_float(
        video_stream.get("duration")
    )
    width = _positive_int(video_stream.get("width"))
    height = _positive_int(video_stream.get("height"))
    fps = _parse_fps(video_stream.get("avg_frame_rate")) or _parse_fps(
        video_stream.get("r_frame_rate")
    )

    missing = []
    if duration is None:
        missing.append("duration")
    if width is None:
        missing.append("width")
    if height is None:
        missing.append("height")
    if fps is None:
        missing.append("fps")
    if missing:
        raise MediaProbeError(
            f"ffprobe metadata is incomplete for {path.name}: missing {', '.join(missing)}"
        )

    rotation = _rotation_degrees(video_stream)
    display_width = width
    display_height = height
    if rotation in {90, 270}:
        display_width, display_height = height, width

    audio_channels = _positive_int(audio_stream.get("channels")) if audio_stream else None
    audio_sample_rate = (
        _positive_int(audio_stream.get("sample_rate")) if audio_stream else None
    )

    return VideoMetadata(
        duration_seconds=duration,
        width=display_width,
        height=display_height,
        fps=fps,
        has_audio=bool(audio_streams),
        video_codec=str(video_stream.get("codec_name") or ""),
        audio_codec=str(audio_stream.get("codec_name") or "") if audio_stream else "",
        container=str(format_info.get("format_name") or ""),
        file_size_bytes=path.stat().st_size,
        rotation_degrees=rotation,
        audio_channels=audio_channels,
        audio_sample_rate=audio_sample_rate,
    )
