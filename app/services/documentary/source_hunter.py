from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import urlencode

import requests
from loguru import logger

from app.models.documentary import RightsStatus, SourceAsset
from app.services.documentary.youtube_source import build_youtube_source_asset

_YOUTUBE_SEARCH_URL = "https://www.youtube.com/results"
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/150.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_OFFICIAL_CHANNEL_TERMS = (
    "police",
    "sheriff",
    "department",
    "city of",
    "county",
    "state patrol",
    "highway patrol",
    "district attorney",
    "prosecutor",
    "court",
    "fire department",
    "public safety",
    "official",
)
_STRONG_VIDEO_TERMS = (
    "bodycam",
    "body camera",
    "dashcam",
    "cctv",
    "surveillance",
    "footage",
    "full video",
    "raw video",
    "video released",
    "released footage",
)
_QUERY_NOISE_TERMS = {
    "video", "watch", "footage", "shows", "show", "the", "that", "with", "from",
    "after", "into", "over", "under", "and", "for", "police", "bodycam", "camera",
}


class SourceHunterError(RuntimeError):
    """Raised when original-video candidate discovery cannot be completed."""


@dataclass(frozen=True)
class SourceVideoCandidate:
    video_id: str
    title: str
    channel: str
    url: str
    published_text: str
    duration_text: str
    view_count_text: str
    score: int
    title_overlap_score: int
    source_quality_score: int
    video_signal_score: int
    reasons: tuple[str, ...]


def _text(value) -> str:
    if not isinstance(value, dict):
        return ""
    if isinstance(value.get("simpleText"), str):
        return value["simpleText"].strip()
    runs = value.get("runs")
    if isinstance(runs, list):
        return "".join(
            str(run.get("text") or "")
            for run in runs
            if isinstance(run, dict)
        ).strip()
    return ""


def _extract_initial_data(html: str) -> dict:
    markers = (
        "var ytInitialData = ",
        "window[\"ytInitialData\"] = ",
        "ytInitialData = ",
    )
    decoder = json.JSONDecoder()
    for marker in markers:
        start = html.find(marker)
        if start < 0:
            continue
        start += len(marker)
        while start < len(html) and html[start].isspace():
            start += 1
        try:
            payload, _ = decoder.raw_decode(html[start:])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            return payload
    raise SourceHunterError("YouTube search returned an unsupported page")


def _walk_video_renderers(value):
    if isinstance(value, dict):
        renderer = value.get("videoRenderer")
        if isinstance(renderer, dict):
            yield renderer
        for child in value.values():
            yield from _walk_video_renderers(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_video_renderers(child)


def _story_tokens(title: str) -> set[str]:
    value = re.sub(r"\bk[\s_-]*9\b", "k9", (title or "").lower())
    tokens = re.findall(r"[a-z0-9]{3,}", value)
    return {
        token
        for token in tokens
        if token not in _QUERY_NOISE_TERMS
    }


def _score_video(story_title: str, video_title: str, channel: str):
    story_tokens = _story_tokens(story_title)
    video_tokens = _story_tokens(video_title)
    shared = story_tokens & video_tokens

    overlap_score = min(45, len(shared) * 9)
    normalized_title = video_title.lower()
    normalized_channel = channel.lower()

    video_matches = [term for term in _STRONG_VIDEO_TERMS if term in normalized_title]
    video_signal_score = min(30, len(video_matches) * 10)

    official_matches = [
        term for term in _OFFICIAL_CHANNEL_TERMS if term in normalized_channel
    ]
    source_quality_score = 25 if official_matches else 0

    reasons = []
    if shared:
        reasons.append("story match: " + ", ".join(sorted(shared)[:5]))
    if video_matches:
        reasons.append("video signal: " + ", ".join(video_matches[:3]))
    if official_matches:
        reasons.append("official-channel signal: " + ", ".join(official_matches[:2]))

    score = min(100, overlap_score + video_signal_score + source_quality_score)
    return score, overlap_score, source_quality_score, video_signal_score, tuple(reasons)


def _build_search_query(story_title: str) -> str:
    clean = re.sub(r"\s+", " ", (story_title or "").strip())
    if not clean:
        raise ValueError("story title is required for source discovery")
    if len(clean) > 300:
        clean = clean[:300].rsplit(" ", 1)[0]
    return f'{clean} bodycam OR footage OR "full video"'


def find_source_videos(
    story_title: str,
    *,
    limit: int = 8,
    timeout_seconds: float = 20.0,
    session=None,
) -> list[SourceVideoCandidate]:
    """Find likely source-video candidates on YouTube without downloading them."""
    try:
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("source hunter limit must be an integer") from exc
    if not 1 <= limit <= 20:
        raise ValueError("source hunter limit must be between 1 and 20")

    query = _build_search_query(story_title)
    request = session or requests

    try:
        response = request.get(
            _YOUTUBE_SEARCH_URL,
            params={"search_query": query},
            headers=_REQUEST_HEADERS,
            timeout=timeout_seconds,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        raise SourceHunterError("YouTube source search request failed") from exc

    if response.status_code != 200:
        raise SourceHunterError(
            f"YouTube source search returned HTTP {response.status_code}"
        )

    initial_data = _extract_initial_data(response.text)
    candidates: list[SourceVideoCandidate] = []
    seen_ids: set[str] = set()

    for renderer in _walk_video_renderers(initial_data):
        video_id = str(renderer.get("videoId") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            continue
        if video_id in seen_ids:
            continue

        title = _text(renderer.get("title"))
        channel = _text(renderer.get("ownerText")) or _text(
            renderer.get("longBylineText")
        )
        if not title:
            continue

        score, overlap, source_quality, video_signal, reasons = _score_video(
            story_title, title, channel
        )
        candidates.append(
            SourceVideoCandidate(
                video_id=video_id,
                title=title,
                channel=channel,
                url=f"https://www.youtube.com/watch?v={video_id}",
                published_text=_text(renderer.get("publishedTimeText")),
                duration_text=_text(renderer.get("lengthText")),
                view_count_text=_text(renderer.get("viewCountText")),
                score=score,
                title_overlap_score=overlap,
                source_quality_score=source_quality,
                video_signal_score=video_signal,
                reasons=reasons,
            )
        )
        seen_ids.add(video_id)

    candidates.sort(
        key=lambda item: (
            item.score,
            item.source_quality_score,
            item.video_signal_score,
            item.title_overlap_score,
            item.title.lower(),
        ),
        reverse=True,
    )
    logger.info(
        f"Documentary Source Hunter found {len(candidates)} YouTube candidates"
    )
    return candidates[:limit]


def candidate_to_youtube_source(
    candidate: SourceVideoCandidate,
) -> SourceAsset:
    """Create a traceable project source while keeping reuse rights unverified."""
    return build_youtube_source_asset(
        candidate.url,
        title=candidate.title,
        channel=candidate.channel,
        rights_status=RightsStatus.unknown_review_required,
        rights_note=(
            "Discovered automatically by Source Hunter. Verify that this is the "
            "original/authoritative upload and confirm reuse/publication rights before use."
        ),
    )
