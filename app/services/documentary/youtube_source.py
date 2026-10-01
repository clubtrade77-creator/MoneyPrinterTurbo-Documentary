from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from app.models.documentary import RightsStatus, SourceAsset, SourceType

_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
}


def extract_youtube_video_id(url: str) -> str:
    """Return the canonical 11-character YouTube video ID from a supported URL."""
    raw_url = (url or "").strip()
    if not raw_url:
        raise ValueError("YouTube URL is required")

    parsed = urlparse(raw_url if "://" in raw_url else f"https://{raw_url}")
    host = parsed.netloc.lower().split(":", 1)[0]
    candidate = ""

    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/", 1)[0]
    elif host in _YOUTUBE_HOSTS:
        path_parts = [part for part in parsed.path.split("/") if part]
        if parsed.path.rstrip("/") == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        elif len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            candidate = path_parts[1]

    if not _YOUTUBE_ID_RE.fullmatch(candidate):
        raise ValueError(f"unsupported or invalid YouTube URL: {url}")
    return candidate


def canonical_youtube_url(video_id: str) -> str:
    if not _YOUTUBE_ID_RE.fullmatch(video_id or ""):
        raise ValueError("invalid YouTube video ID")
    return f"https://www.youtube.com/watch?v={video_id}"


def build_youtube_source_asset(
    url: str,
    *,
    title: str = "",
    channel: str = "",
    published_at: str | None = None,
    rights_status: RightsStatus = RightsStatus.unknown_review_required,
    rights_note: str = "",
) -> SourceAsset:
    """Create a traceable SourceAsset from a YouTube URL without downloading media.

    Metadata enrichment and authorized/local media acquisition are deliberately kept
    separate. A public YouTube URL is discoverable, but that alone does not establish
    permission to reuse the footage.
    """
    video_id = extract_youtube_video_id(url)
    return SourceAsset(
        id=f"youtube_{video_id}",
        source_type=SourceType.youtube,
        title=title,
        source_url=canonical_youtube_url(video_id),
        publisher=channel,
        rights_status=rights_status,
        rights_note=rights_note,
        youtube_video_id=video_id,
        youtube_channel=channel,
        youtube_published_at=published_at,
    )
