from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
from xml.etree import ElementTree

import requests
from loguru import logger

from app.models.documentary import (
    ProvenanceType,
    RightsStatus,
    SourceAsset,
    SourceType,
)

_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
_GOOGLE_NEWS_RSS_API = "https://news.google.com/rss/search"
_DEFAULT_DISCOVERY_QUERY = (
    '("bodycam" OR "body camera" OR CCTV OR "surveillance video" OR dashcam '
    'OR "caught on camera" OR "police released video" OR "court footage" '
    'OR "security camera" OR "video shows")'
)
_GOOGLE_VISUAL_QUERY = (
    '"bodycam" OR CCTV OR dashcam OR "caught on camera" OR '
    '"surveillance video" OR "video shows" OR footage'
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
_FOOTAGE_PUBLISHED_PATTERNS = (
    "video shows",
    "video captures",
    "video captured",
    "video released",
    "released video",
    "footage shows",
    "footage captures",
    "footage captured",
    "footage released",
    "released footage",
    "bodycam shows",
    "bodycam captures",
    "bodycam footage",
    "dashcam shows",
    "dashcam captures",
    "caught on camera",
    "caught on video",
    "surveillance video shows",
    "cctv shows",
    "cctv footage shows",
)
_FOOTAGE_NOT_AVAILABLE_PATTERNS = (
    "calls for footage",
    "calls on",
    "challenges",
    "demands",
    "asks for",
    "urges",
    "to release cctv",
    "to release footage",
    "release cctv footage",
    "harbor cameras",
    "install cameras",
    "installation of cameras",
)
_STORY_TERM_WEIGHTS = {
    "shooting": 10,
    "homicide": 10,
    "murder": 10,
    "kidnapping": 10,
    "hostage": 10,
    "standoff": 9,
    "chase": 9,
    "rescue": 9,
    "escape": 8,
    "missing": 8,
    "crash": 8,
    "survived": 8,
    "survivor": 8,
    "explosion": 8,
    "collapse": 8,
    "attack": 8,
    "robbery": 7,
    "arrest": 7,
    "suspect": 6,
    "officer": 5,
    "k9": 5,
    "dramatic": 5,
    "mystery": 5,
    "investigation": 5,
    "trial": 5,
    "court": 5,
    "disaster": 7,
    "fire": 6,
    "flood": 6,
}
_TITLE_NOISE_PREFIX_RE = re.compile(
    r"^(?:(?:breaking|watch|video|update|new|exclusive|developing)\s*[:\-–—]\s*)+",
    re.IGNORECASE,
)
_EVENT_STOPWORDS = {
    "after", "and", "body", "bodycam", "camera", "captures", "caught",
    "deadly", "footage", "from", "in", "left", "names", "new", "news",
    "of", "on", "police", "released", "releases", "shows", "struck", "that",
    "the", "video", "watch", "with",
}
_EVENT_TOKEN_ALIASES = {
    "k-9": "k9",
    "k_9": "k9",
    "body-camera": "bodycam",
    "bodycamera": "bodycam",
    "shoot": "shooting",
}
_MAX_QUERY_LENGTH = 240
_MAX_DISCOVERY_RESULTS = 50
_REQUEST_HEADERS = {
    "User-Agent": "MoneyPrinterTurbo-Documentary/1.3 (+story-discovery)",
    "Accept": "application/json, application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
}


class StoryDiscoveryError(RuntimeError):
    """Raised when all configured story-discovery feeds fail."""


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


def _google_when_token(lookback_hours: int) -> str:
    if lookback_hours <= 24:
        return "when:1d"
    days = max(1, min(30, (lookback_hours + 23) // 24))
    return f"when:{days}d"


def _build_google_query(user_query: str, lookback_hours: int) -> str:
    clean = _clean_query(user_query)
    visual_query = _GOOGLE_VISUAL_QUERY
    if clean:
        phrase = re.sub(r"\s+", " ", clean.replace('"', " ")).strip()
        if phrase:
            return f'"{phrase}" ({visual_query}) {_google_when_token(lookback_hours)}'
    return f"({visual_query}) {_google_when_token(lookback_hours)}"


def _parse_published_datetime(value: str) -> datetime | None:
    raw = (value or "").strip()
    if not raw:
        return None

    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y%m%dT%H%M%S"):
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone.utc)

    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_published_at(value: str) -> str:
    parsed = _parse_published_datetime(value)
    if parsed is None:
        return (value or "").strip()
    return parsed.strftime("%Y%m%dT%H%M%SZ")


def _freshness_score(published_at: str, *, now: datetime) -> tuple[int, str]:
    published = _parse_published_datetime(published_at)
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
    published_matches = [
        term for term in _FOOTAGE_PUBLISHED_PATTERNS if term in normalized
    ]
    unavailable_matches = [
        term for term in _FOOTAGE_NOT_AVAILABLE_PATTERNS if term in normalized
    ]
    story_matches = [
        term for term in _STORY_TERM_WEIGHTS if term in normalized
    ]

    footage_score = 0
    if footage_matches:
        footage_score = min(24, 8 + len(footage_matches) * 4)
    if published_matches:
        footage_score = min(40, footage_score + 16)
    if unavailable_matches:
        footage_score = max(0, footage_score - 18)

    story_score = min(
        30,
        sum(_STORY_TERM_WEIGHTS[term] for term in story_matches),
    )
    freshness_score, freshness_reason = _freshness_score(published_at, now=now)

    reasons = []
    if published_matches:
        reasons.append(
            "published-footage signal: " + ", ".join(published_matches[:2])
        )
    elif footage_matches:
        reasons.append("visual-source mention: " + ", ".join(footage_matches[:3]))
    if unavailable_matches:
        reasons.append(
            "footage availability uncertain: " + ", ".join(unavailable_matches[:2])
        )
    if story_matches:
        reasons.append("story signal: " + ", ".join(story_matches[:4]))
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


def _story_title_key(title: str) -> str:
    """Normalize editorial prefixes/punctuation so syndicated headlines deduplicate."""
    value = _TITLE_NOISE_PREFIX_RE.sub("", (title or "").strip())
    value = value.lower()
    # Preserve short but meaningful entities before punctuation is stripped.
    value = re.sub(r"\bk[\s_-]*9\b", "k9", value)
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _near_duplicate_title(left: str, right: str) -> bool:
    left_key = _story_title_key(left)
    right_key = _story_title_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True

    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    if min(len(left_tokens), len(right_tokens)) < 5:
        return False

    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return union > 0 and overlap / union >= 0.82


def _event_tokens(title: str) -> set[str]:
    value = _story_title_key(title)
    value = value.replace("k-9", "k9")
    tokens = []
    for token in value.split():
        token = _EVENT_TOKEN_ALIASES.get(token, token)
        if token in _EVENT_STOPWORDS or (len(token) < 3 and token != "k9"):
            continue
        tokens.append(token)
    return set(tokens)


def _same_event_title(left: str, right: str) -> bool:
    """Detect different headlines about the same underlying event."""
    left_tokens = _event_tokens(left)
    right_tokens = _event_tokens(right)
    if min(len(left_tokens), len(right_tokens)) < 4:
        return False

    shared = left_tokens & right_tokens
    if len(shared) < 4:
        return False

    overlap_coeff = len(shared) / min(len(left_tokens), len(right_tokens))
    story_terms = set(_STORY_TERM_WEIGHTS)
    strong_shared = shared & story_terms

    return overlap_coeff >= 0.5 and len(strong_shared) >= 2


def _dedupe_and_rank(
    raw_items: list[dict],
    *,
    discovery_query: str,
    limit: int,
    now: datetime,
) -> list[StoryCandidate]:
    candidates: list[StoryCandidate] = []
    seen_urls: set[str] = set()
    seen_titles: list[str] = []

    for article in raw_items:
        if not isinstance(article, dict):
            continue
        url = _safe_http_url(str(article.get("url") or ""))
        title = re.sub(r"\s+", " ", str(article.get("title") or "").strip())
        if not url or not title:
            continue

        title_key = _story_title_key(title)
        if url in seen_urls or any(
            _near_duplicate_title(title, seen_title)
            or _same_event_title(title, seen_title)
            for seen_title in seen_titles
        ):
            continue

        published_at = _canonical_published_at(
            str(article.get("published_at") or article.get("seendate") or "")
        )
        (
            score,
            footage_score,
            freshness_score,
            story_score,
            reasons,
        ) = _score_text(title, published_at=published_at, now=now)

        candidates.append(
            StoryCandidate(
                id=_candidate_id(url),
                title=title,
                url=url,
                publisher=str(
                    article.get("publisher") or article.get("domain") or ""
                ).strip(),
                published_at=published_at,
                language=str(article.get("language") or "").strip(),
                source_country=str(article.get("source_country") or article.get("sourcecountry") or "").strip(),
                image_url=_safe_http_url(
                    str(article.get("image_url") or article.get("socialimage") or "")
                ),
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
            seen_titles.append(title)

    candidates.sort(
        key=lambda item: (
            item.score,
            item.freshness_score,
            item.footage_score,
            item.title.lower(),
        ),
        reverse=True,
    )
    return candidates[:limit]


def _discover_gdelt(
    query: str,
    *,
    lookback_hours: int,
    limit: int,
    timeout_seconds: float,
    request,
) -> list[dict]:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": min(250, max(limit * 3, 25)),
        "timespan": f"{lookback_hours}h",
        "sort": "datedesc",
    }
    response = request.get(
        _GDELT_DOC_API,
        params=params,
        headers=_REQUEST_HEADERS,
        timeout=timeout_seconds,
        allow_redirects=False,
    )
    if 300 <= response.status_code < 400:
        raise StoryDiscoveryError("GDELT returned a redirect")
    if response.status_code != 200:
        raise StoryDiscoveryError(f"GDELT returned HTTP {response.status_code}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise StoryDiscoveryError("GDELT returned invalid JSON") from exc

    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    if not isinstance(articles, list):
        raise StoryDiscoveryError("GDELT returned an invalid article list")
    return articles


def _discover_google_news(
    user_query: str,
    *,
    lookback_hours: int,
    limit: int,
    timeout_seconds: float,
    request,
) -> tuple[str, list[dict]]:
    google_query = _build_google_query(user_query, lookback_hours)
    response = request.get(
        _GOOGLE_NEWS_RSS_API,
        params={
            "q": google_query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        },
        headers=_REQUEST_HEADERS,
        timeout=timeout_seconds,
        allow_redirects=True,
    )
    if response.status_code != 200:
        raise StoryDiscoveryError(
            f"Google News fallback returned HTTP {response.status_code}"
        )

    try:
        root = ElementTree.fromstring(response.text)
    except (ElementTree.ParseError, AttributeError) as exc:
        raise StoryDiscoveryError("Google News fallback returned invalid RSS") from exc

    articles: list[dict] = []
    for item in root.findall(".//item")[: max(limit * 3, 25)]:
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        published_at = (item.findtext("pubDate") or "").strip()
        source_node = item.find("source")
        publisher = ""
        if source_node is not None and source_node.text:
            publisher = source_node.text.strip()
        if title and publisher:
            suffix = f" - {publisher}"
            if title.endswith(suffix):
                title = title[: -len(suffix)].strip()

        articles.append(
            {
                "url": url,
                "title": title,
                "published_at": published_at,
                "publisher": publisher,
                "language": "English",
                "source_country": "",
                "image_url": "",
            }
        )
    return google_query, articles


def discover_stories(
    query: str = "",
    *,
    lookback_hours: int = 72,
    limit: int = 20,
    timeout_seconds: float = 20.0,
    session=None,
    now: datetime | None = None,
) -> list[StoryCandidate]:
    """Find recent documentary story candidates with a resilient provider fallback.

    GDELT is attempted first. If it rate-limits, fails, or returns no usable leads,
    Google News RSS is used automatically. Discovery only finds research leads; it does
    not grant publication rights and does not download third-party media.
    """
    try:
        lookback_hours = int(lookback_hours)
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid documentary discovery limits") from exc

    if not 1 <= lookback_hours <= 24 * 30:
        raise ValueError(
            "documentary discovery lookback must be between 1 hour and 30 days"
        )
    if not 1 <= limit <= _MAX_DISCOVERY_RESULTS:
        raise ValueError(
            f"documentary discovery limit must be between 1 and {_MAX_DISCOVERY_RESULTS}"
        )

    discovery_query = _build_query(query)
    request = session or requests
    current_time = now or datetime.now(timezone.utc)
    provider_errors: list[str] = []

    try:
        gdelt_articles = _discover_gdelt(
            discovery_query,
            lookback_hours=lookback_hours,
            limit=limit,
            timeout_seconds=timeout_seconds,
            request=request,
        )
        candidates = _dedupe_and_rank(
            gdelt_articles,
            discovery_query=discovery_query,
            limit=limit,
            now=current_time,
        )
        if candidates:
            logger.info(
                f"Documentary Story Discovery found {len(candidates)} candidates via GDELT"
            )
            return candidates
        provider_errors.append("GDELT returned no usable candidates")
    except (requests.RequestException, StoryDiscoveryError) as exc:
        provider_errors.append(str(exc))
        logger.warning(f"Documentary Story Discovery GDELT fallback trigger: {exc}")

    try:
        google_query, google_articles = _discover_google_news(
            query,
            lookback_hours=lookback_hours,
            limit=limit,
            timeout_seconds=timeout_seconds,
            request=request,
        )
        candidates = _dedupe_and_rank(
            google_articles,
            discovery_query=google_query,
            limit=limit,
            now=current_time,
        )
        if candidates:
            logger.info(
                "Documentary Story Discovery found "
                f"{len(candidates)} candidates via Google News fallback"
            )
            return candidates
        provider_errors.append("Google News fallback returned no usable candidates")
    except (requests.RequestException, StoryDiscoveryError) as exc:
        provider_errors.append(str(exc))
        logger.warning(f"Documentary Story Discovery Google News fallback failed: {exc}")

    detail = "; ".join(provider_errors) or "no discovery providers were available"
    raise StoryDiscoveryError(f"story discovery failed: {detail}")


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
