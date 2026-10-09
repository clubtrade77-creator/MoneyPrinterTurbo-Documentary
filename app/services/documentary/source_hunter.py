from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
    "left", "officer",
}
_GENERIC_MATCH_TERMS = {
    "officer", "suspect", "police", "incident", "man", "woman", "people", "person",
}
_STRONG_EVENT_TERMS = {
    "k9", "shooting", "homicide", "murder", "dead", "fatal", "chase", "rescue",
    "crash", "arrest", "standoff", "kidnapping", "hostage", "explosion",
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
    freshness_score: int
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


def _freshness_score(published_text: str) -> tuple[int, str]:
    value = (published_text or "").strip().lower()
    if not value:
        return 4, "video age unavailable"

    match = re.search(
        r"(\d+)\s*(minute|hour|day|week|month|year)s?\s+ago",
        value,
    )
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        age_days = {
            "minute": amount / 1440,
            "hour": amount / 24,
            "day": amount,
            "week": amount * 7,
            "month": amount * 30,
            "year": amount * 365,
        }[unit]
        if age_days <= 7:
            return 20, "published within 7 days"
        if age_days <= 30:
            return 14, "published within 30 days"
        if age_days <= 365:
            return 6, "published within 1 year"
        return 0, "older than 1 year"

    if any(token in value for token in ("today", "just now", "streamed")):
        return 20, "published recently"
    return 4, "video age unavailable"


def _score_video(
    story_title: str,
    video_title: str,
    channel: str,
    published_text: str,
):
    story_tokens = _story_tokens(story_title)
    video_tokens = _story_tokens(video_title)
    shared = story_tokens & video_tokens

    weighted_overlap = 0
    for token in shared:
        if token in _STRONG_EVENT_TERMS:
            weighted_overlap += 12
        elif token in _GENERIC_MATCH_TERMS:
            weighted_overlap += 2
        else:
            weighted_overlap += 6
    overlap_score = min(45, weighted_overlap)

    normalized_title = video_title.lower()
    normalized_channel = channel.lower()

    video_matches = [term for term in _STRONG_VIDEO_TERMS if term in normalized_title]
    video_signal_score = min(25, len(video_matches) * 10)

    official_matches = [
        term for term in _OFFICIAL_CHANNEL_TERMS if term in normalized_channel
    ]
    source_quality_score = 20 if official_matches else 0
    freshness_score, freshness_reason = _freshness_score(published_text)

    reasons = []
    if shared:
        reasons.append("story match: " + ", ".join(sorted(shared)[:6]))
    if video_matches:
        reasons.append("video signal: " + ", ".join(video_matches[:3]))
    if official_matches:
        reasons.append("official-channel signal: " + ", ".join(official_matches[:2]))
    reasons.append(freshness_reason)

    score = min(
        100,
        overlap_score
        + video_signal_score
        + source_quality_score
        + freshness_score,
    )
    return (
        score,
        overlap_score,
        source_quality_score,
        video_signal_score,
        freshness_score,
        tuple(reasons),
    )


def _build_search_query(story_title: str) -> str:
    clean = re.sub(r"\s+", " ", (story_title or "").strip())
    if not clean:
        raise ValueError("story title is required for source discovery")

    tokens = list(_story_tokens(clean))
    strong = [token for token in tokens if token in _STRONG_EVENT_TERMS]
    specific = [
        token
        for token in tokens
        if token not in _STRONG_EVENT_TERMS
        and token not in _GENERIC_MATCH_TERMS
    ]
    generic = [
        token
        for token in tokens
        if token in _GENERIC_MATCH_TERMS
    ]
    selected = (strong + specific + generic)[:7]
    if not selected:
        selected = re.findall(r"[A-Za-z0-9]+", clean)[:7]
    return " ".join(selected) + ' bodycam footage'


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

        published_text = _text(renderer.get("publishedTimeText"))
        (
            score,
            overlap,
            source_quality,
            video_signal,
            freshness,
            reasons,
        ) = _score_video(
            story_title,
            title,
            channel,
            published_text,
        )
        candidates.append(
            SourceVideoCandidate(
                video_id=video_id,
                title=title,
                channel=channel,
                url=f"https://www.youtube.com/watch?v={video_id}",
                published_text=published_text,
                duration_text=_text(renderer.get("lengthText")),
                view_count_text=_text(renderer.get("viewCountText")),
                score=score,
                title_overlap_score=overlap,
                source_quality_score=source_quality,
                video_signal_score=video_signal,
                freshness_score=freshness,
                reasons=reasons,
            )
        )
        seen_ids.add(video_id)

    candidates.sort(
        key=lambda item: (
            item.score,
            item.source_quality_score,
            item.freshness_score,
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
