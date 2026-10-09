from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests
from loguru import logger

from app.models.documentary import (
    ProvenanceType,
    RightsStatus,
    SourceAsset,
    SourceType,
)

_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
_DEFAULT_DISCOVERY_QUERY = (
    '("bodycam" OR "body camera" OR CCTV OR "surveillance video" OR dashcam '
    'OR "caught on camera" OR "police released video" OR "court footage" '
    'OR "security camera" OR "video shows")'
)
_FOOTAGE_TERMS = (
    "bodycam",
    "body camera",
    "cctv",
    "surveillance",
    "dashcam",
    "caught on camera",
    "security camera",
    "footage",
    "video",
    "camera",
)
_STORY_TERMS = (
    "rescue",
    "escape",
    "missing",
    "crash",
    "chase",
    "caught",
    "dramatic",
    "survived",
    "survivor",
    "mystery",
    "investigation",
    "arrest",
    "trial",
    "court",
    "disaster",
    "collapse",
    "fire",
    "flood",
)
_MAX_QUERY_LENGTH = 240
_MAX_DISCOVERY_RESULTS = 50


class StoryDiscoveryError(RuntimeError):
    """Raised when the external story-discovery feed cannot be queried safely."""


@dataclass(frozen=True)
class StoryCandidate:
    id: str
    title: str
    url: str
    publisher: str
    published_at: str
    language: str
    source_country: str
    image_url: str
    discovery_query: str
    score: int
    footage_score: int
    freshness_score: int
    story_score: int
    reasons: tuple[str, ...]


def _clean_query(value: str) -> str:
    value = re.sub(r"\s+", " ", (value or "").strip())
    if len(value) > _MAX_QUERY_LENGTH:
        raise ValueError(
            f"documentary discovery query must be at most {_MAX_QUERY_LENGTH} characters"
        )
    return value


def _build_query(user_query: str) -> str:
    clean = _clean_query(user_query)
    if not clean:
        return _DEFAULT_DISCOVERY_QUERY

    # Treat the UI query as a literal phrase so punctuation or boolean syntax typed by
    # a user cannot unexpectedly broaden the fixed discovery query.
    phrase = clean.replace('"', " ").strip()
    phrase = re.sub(r"\s+", " ", phrase)
    if not phrase:
        return _DEFAULT_DISCOVERY_QUERY
    return f'"{phrase}" {_DEFAULT_DISCOVERY_QUERY}'


def _parse_gdelt_datetime(value: str) -> datetime | None:
    raw = (value or "").strip()
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y%m%dT%H%M%S"):
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone.utc)
    return None


def _freshness_score(published_at: str, *, now: datetime) -> tuple[int, str]:
    published = _parse_gdelt_datetime(published_at)
    if published is None:
        return 8, "publication time unavailable"

    age_hours = max(0.0, (now - published).total_seconds() / 3600)
    if age_hours <= 24:
        return 30, "published within 24 hours"
    if age_hours <= 72:
        return 24, "published within 3 days"
    if age_hours <= 24 * 7:
        return 18, "published within 7 days"
    return 10, "older than 7 days"


def _score_text(title: str, *, published_at: str, now: datetime):
    normalized = re.sub(r"\s+", " ", (title or "").lower())

    footage_matches = [term for term in _FOOTAGE_TERMS if term in normalized]
    story_matches = [term for term in _STORY_TERMS if term in normalized]

    footage_score = min(40, 12 + len(footage_matches) * 8) if footage_matches else 0
    story_score = min(30, len(story_matches) * 6)
    freshness_score, freshness_reason = _freshness_score(published_at, now=now)

    reasons = []
    if footage_matches:
        reasons.append("visual-source signal: " + ", ".join(footage_matches[:3]))
    if story_matches:
        reasons.append("story signal: " + ", ".join(story_matches[:3]))
    reasons.append(freshness_reason)

    score = min(100, footage_score + story_score + freshness_score)
    return score, footage_score, freshness_score, story_score, tuple(reasons)


def _candidate_id(url: str) -> str:
    return "story_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _safe_http_url(value: str) -> str:
    raw = (value or "").strip()
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return raw


def discover_stories(
    query: str = "",
    *,
    lookback_hours: int = 72,
    limit: int = 20,
    timeout_seconds: float = 20.0,
    session=None,
    now: datetime | None = None,
) -> list[StoryCandidate]:
    """Find recent documentary story candidates from GDELT.

    This stage discovers leads only. It deliberately does not claim publication rights
    and does not download third-party media.
    """
    try:
        lookback_hours = int(lookback_hours)
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid documentary discovery limits") from exc

    if not 1 <= lookback_hours <= 24 * 30:
        raise ValueError("documentary discovery lookback must be between 1 hour and 30 days")
    if not 1 <= limit <= _MAX_DISCOVERY_RESULTS:
        raise ValueError(
            f"documentary discovery limit must be between 1 and {_MAX_DISCOVERY_RESULTS}"
        )

    discovery_query = _build_query(query)
    request = session or requests
    params = {
        "query": discovery_query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(250, max(limit * 3, 25)),
        "timespan": f"{lookback_hours}h",
        "sort": "datedesc",
    }

    try:
        response = request.get(
            _GDELT_DOC_API,
            params=params,
            timeout=timeout_seconds,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise StoryDiscoveryError("story discovery request failed") from exc

    if 300 <= response.status_code < 400:
        raise StoryDiscoveryError("story discovery endpoint returned a redirect")
    if response.status_code != 200:
        raise StoryDiscoveryError(
            f"story discovery endpoint returned HTTP {response.status_code}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise StoryDiscoveryError("story discovery returned invalid JSON") from exc

    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    if not isinstance(articles, list):
        raise StoryDiscoveryError("story discovery returned an invalid article list")

    current_time = now or datetime.now(timezone.utc)
    candidates: list[StoryCandidate] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for article in articles:
        if not isinstance(article, dict):
            continue
        url = _safe_http_url(str(article.get("url") or ""))
        title = re.sub(r"\s+", " ", str(article.get("title") or "").strip())
        if not url or not title:
            continue

        title_key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
        if url in seen_urls or (title_key and title_key in seen_titles):
            continue

        published_at = str(article.get("seendate") or "").strip()
        (
            score,
            footage_score,
            freshness_score,
            story_score,
            reasons,
        ) = _score_text(title, published_at=published_at, now=current_time)

        candidates.append(
            StoryCandidate(
                id=_candidate_id(url),
                title=title,
                url=url,
                publisher=str(article.get("domain") or "").strip(),
                published_at=published_at,
                language=str(article.get("language") or "").strip(),
                source_country=str(article.get("sourcecountry") or "").strip(),
                image_url=_safe_http_url(str(article.get("socialimage") or "")),
                discovery_query=discovery_query,
                score=score,
                footage_score=footage_score,
                freshness_score=freshness_score,
                story_score=story_score,
                reasons=reasons,
            )
        )
        seen_urls.add(url)
        if title_key:
            seen_titles.add(title_key)

    candidates.sort(
        key=lambda item: (
            item.score,
            item.freshness_score,
            item.footage_score,
            item.title.lower(),
        ),
        reverse=True,
    )
    logger.info(
        f"Documentary Story Discovery found {len(candidates)} unique candidates"
    )
    return candidates[:limit]


def candidate_to_source(candidate: StoryCandidate) -> SourceAsset:
    """Convert a discovery lead into a traceable project research source."""
    return SourceAsset(
        id=candidate.id,
        source_type=SourceType.news,
        provenance=ProvenanceType.third_party_platform,
        title=candidate.title,
        source_url=candidate.url,
        publisher=candidate.publisher,
        publication_date=candidate.published_at or None,
        rights_status=RightsStatus.unknown_review_required,
        rights_note=(
            "Discovered as a research lead. Publication/reuse rights for any linked "
            "media must be verified separately before use."
        ),
    )
