from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urlparse
from xml.etree import ElementTree

import requests
from loguru import logger

from app.models.documentary import (
    ProvenanceType,
    RightsStatus,
    SourceAsset,
    SourceType,
)
from app.services.documentary.youtube_source import build_youtube_source_asset

_YOUTUBE_SEARCH_URL = "https://www.youtube.com/results"
_DUCKDUCKGO_SEARCH_URL = "https://html.duckduckgo.com/html/"
_BING_SEARCH_URL = "https://www.bing.com/search"
_SEARCH_ENGINE_DOMAINS = {
    "bing.com",
    "www.bing.com",
    "duckduckgo.com",
    "html.duckduckgo.com",
    "google.com",
    "www.google.com",
}
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
class SourceWebCandidate:
    title: str
    url: str
    domain: str
    snippet: str
    score: int
    title_overlap_score: int
    official_score: int
    video_signal_score: int
    reasons: tuple[str, ...]


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


class _DuckDuckGoParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._current = None
        self._capture = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self._current = {
                "title": "",
                "url": attrs.get("href") or "",
                "snippet": "",
            }
            self._capture = "title"
        elif self._current is not None and "result__snippet" in classes:
            self._capture = "snippet"

    def handle_data(self, data):
        if self._current is not None and self._capture in {"title", "snippet"}:
            self._current[self._capture] += data

    def handle_endtag(self, tag):
        if self._current is None:
            return
        if self._capture == "title" and tag == "a":
            self._capture = None
        elif self._capture == "snippet" and tag in {"a", "div", "span"}:
            self._capture = None
            if self._current.get("title") and self._current.get("url"):
                self.results.append(self._current)
            self._current = None


class _BingParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._inside_result = False
        self._inside_h2 = False
        self._capture = None
        self._current = None
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set((attrs.get("class") or "").split())

        if tag == "li" and "b_algo" in classes:
            self._inside_result = True
            self._depth = 1
            self._current = {"title": "", "url": "", "snippet": ""}
            return

        if not self._inside_result:
            return

        if tag == "li":
            self._depth += 1
        elif tag == "h2":
            self._inside_h2 = True
        elif tag == "a" and self._inside_h2 and self._current is not None:
            self._current["url"] = attrs.get("href") or ""
            self._capture = "title"
        elif tag == "p" and self._current is not None:
            self._capture = "snippet"

    def handle_data(self, data):
        if self._inside_result and self._current is not None and self._capture:
            self._current[self._capture] += data

    def handle_endtag(self, tag):
        if not self._inside_result:
            return

        if tag == "a" and self._capture == "title":
            self._capture = None
        elif tag == "p" and self._capture == "snippet":
            self._capture = None
        elif tag == "h2":
            self._inside_h2 = False
        elif tag == "li":
            self._depth -= 1
            if self._depth <= 0:
                if (
                    self._current is not None
                    and self._current.get("title")
                    and self._current.get("url")
                ):
                    self.results.append(self._current)
                self._inside_result = False
                self._current = None
                self._capture = None
                self._inside_h2 = False
                self._depth = 0


def _parse_bing_rss(text: str) -> list[dict]:
    try:
        root = ElementTree.fromstring(text)
    except (ElementTree.ParseError, TypeError) as exc:
        raise SourceHunterError("Bing returned invalid RSS") from exc

    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        url = (item.findtext("link") or "").strip()
        snippet = (item.findtext("description") or "").strip()
        if title and url:
            items.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                }
            )
    return items


def _normalize_search_result_url(value: str) -> str:
    raw = unescape((value or "").strip())
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw
    parsed = urlparse(raw)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            raw = unquote(target)
            parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return raw


def _is_official_web_source(domain: str, title: str, snippet: str) -> tuple[int, list[str]]:
    host = (domain or "").lower()
    haystack = f"{host} {title} {snippet}".lower()
    reasons = []

    if host.endswith(".gov") or ".gov." in host:
        reasons.append("government domain")
        return 35, reasons

    agency_terms = [
        term
        for term in (
            "police department",
            "police bureau",
            "sheriff",
            "district attorney",
            "public safety",
            "state patrol",
            "highway patrol",
            "city of",
            "county",
        )
        if term in haystack
    ]
    if agency_terms:
        reasons.append("agency signal: " + ", ".join(agency_terms[:2]))
        return 18, reasons

    return 0, reasons


def _score_web_source(story_title: str, title: str, snippet: str, domain: str):
    story_tokens = _story_tokens(story_title)
    candidate_tokens = _story_tokens(f"{title} {snippet}")
    shared = story_tokens & candidate_tokens

    overlap_score = 0
    for token in shared:
        if token in _STRONG_EVENT_TERMS:
            overlap_score += 12
        elif token in _GENERIC_MATCH_TERMS:
            overlap_score += 2
        else:
            overlap_score += 6
    overlap_score = min(45, overlap_score)

    normalized = f"{title} {snippet}".lower()
    video_matches = [term for term in _STRONG_VIDEO_TERMS if term in normalized]
    video_signal_score = min(25, len(video_matches) * 10)

    official_score, official_reasons = _is_official_web_source(
        domain, title, snippet
    )

    reasons = []
    if shared:
        reasons.append("story match: " + ", ".join(sorted(shared)[:6]))
    if video_matches:
        reasons.append("video signal: " + ", ".join(video_matches[:3]))
    reasons.extend(official_reasons)

    score = min(100, overlap_score + video_signal_score + official_score)
    return score, overlap_score, official_score, video_signal_score, tuple(reasons)


def _build_web_search_query(story_title: str) -> str:
    base = _build_search_query(story_title)
    return base + ' official police sheriff city "video released"'


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
    tokens = re.findall(r"k9|[a-z0-9]{3,}", value)
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

    normalized = re.sub(r"\bk[\s_-]*9\b", "k9", clean.lower())
    ordered_tokens = []
    for token in re.findall(r"k9|[a-z0-9]{3,}", normalized):
        if token in _QUERY_NOISE_TERMS or token in ordered_tokens:
            continue
        ordered_tokens.append(token)

    strong = [token for token in ordered_tokens if token in _STRONG_EVENT_TERMS]
    specific = [
        token
        for token in ordered_tokens
        if token not in _STRONG_EVENT_TERMS
        and token not in _GENERIC_MATCH_TERMS
    ]
    generic = [
        token
        for token in ordered_tokens
        if token in _GENERIC_MATCH_TERMS
    ]
    selected = (strong + specific + generic)[:7]
    if not selected:
        selected = re.findall(r"[A-Za-z0-9]+", clean)[:7]
    return " ".join(selected) + ' bodycam footage'


def _web_candidates_from_items(
    story_title: str,
    items: list[dict],
    *,
    limit: int,
) -> list[SourceWebCandidate]:
    candidates = []
    seen_urls = set()

    for item in items:
        url = _normalize_search_result_url(item.get("url") or "")
        if not url or url in seen_urls:
            continue

        title = re.sub(r"\s+", " ", unescape(item.get("title") or "")).strip()
        snippet = re.sub(
            r"\s+", " ", unescape(item.get("snippet") or "")
        ).strip()
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        if not title or domain in _SEARCH_ENGINE_DOMAINS:
            continue

        (
            score,
            overlap,
            official_score,
            video_signal,
            reasons,
        ) = _score_web_source(
            story_title,
            title,
            snippet,
            domain,
        )

        # Source Hunter is intentionally strict: a web result must match the event
        # and either contain a real video/footage signal or look like an official source.
        if overlap < 12 or (video_signal <= 0 and official_score <= 0):
            continue

        candidates.append(
            SourceWebCandidate(
                title=title,
                url=url,
                domain=domain,
                snippet=snippet,
                score=score,
                title_overlap_score=overlap,
                official_score=official_score,
                video_signal_score=video_signal,
                reasons=reasons,
            )
        )
        seen_urls.add(url)

    candidates.sort(
        key=lambda item: (
            item.score,
            item.official_score,
            item.video_signal_score,
            item.title_overlap_score,
            item.title.lower(),
        ),
        reverse=True,
    )
    return candidates[:limit]


def find_web_sources(
    story_title: str,
    *,
    limit: int = 8,
    timeout_seconds: float = 20.0,
    session=None,
) -> list[SourceWebCandidate]:
    """Find likely original/official web sources with a search-provider fallback."""
    try:
        limit = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("source hunter web limit must be an integer") from exc
    if not 1 <= limit <= 20:
        raise ValueError("source hunter web limit must be between 1 and 20")

    query = _build_web_search_query(story_title)
    request = session or requests
    errors = []

    try:
        response = request.get(
            _DUCKDUCKGO_SEARCH_URL,
            params={"q": query},
            headers=_REQUEST_HEADERS,
            timeout=timeout_seconds,
            allow_redirects=True,
        )
        if response.status_code == 200:
            parser = _DuckDuckGoParser()
            parser.feed(response.text)
            candidates = _web_candidates_from_items(
                story_title,
                parser.results,
                limit=limit,
            )
            if candidates:
                logger.info(
                    f"Documentary Source Hunter found {len(candidates)} "
                    "web candidates via DuckDuckGo"
                )
                return candidates
            errors.append("DuckDuckGo returned no usable results")
        else:
            errors.append(f"DuckDuckGo returned HTTP {response.status_code}")
    except requests.RequestException as exc:
        errors.append(f"DuckDuckGo request failed: {exc}")

    try:
        response = request.get(
            _BING_SEARCH_URL,
            params={"q": query, "setlang": "en-US", "format": "rss"},
            headers=_REQUEST_HEADERS,
            timeout=timeout_seconds,
            allow_redirects=True,
        )
        if response.status_code == 200:
            try:
                bing_items = _parse_bing_rss(response.text)
            except SourceHunterError:
                parser = _BingParser()
                parser.feed(response.text)
                bing_items = parser.results

            candidates = _web_candidates_from_items(
                story_title,
                bing_items,
                limit=limit,
            )
            if candidates:
                logger.info(
                    f"Documentary Source Hunter found {len(candidates)} "
                    "web candidates via Bing fallback"
                )
                return candidates
            errors.append("Bing returned no relevant source results")
        else:
            errors.append(f"Bing returned HTTP {response.status_code}")
    except requests.RequestException as exc:
        errors.append(f"Bing request failed: {exc}")

    raise SourceHunterError(
        "web source search failed: " + "; ".join(errors)
    )


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


def candidate_to_web_source(candidate: SourceWebCandidate) -> SourceAsset:
    """Create a traceable web source without assuming publication rights."""
    source_id = (
        "web_" + hashlib.sha256(candidate.url.encode("utf-8")).hexdigest()[:16]
    )
    normalized = f"{candidate.title} {candidate.snippet}".lower()
    source_type = SourceType.news
    if "bodycam" in normalized or "body camera" in normalized:
        source_type = SourceType.bodycam
    elif "cctv" in normalized or "surveillance" in normalized:
        source_type = SourceType.cctv
    elif "court" in normalized:
        source_type = SourceType.court

    provenance = (
        ProvenanceType.official_public_source
        if candidate.official_score >= 35
        else ProvenanceType.third_party_platform
    )

    return SourceAsset(
        id=source_id,
        source_type=source_type,
        provenance=provenance,
        title=candidate.title,
        source_url=candidate.url,
        publisher=candidate.domain,
        rights_status=RightsStatus.unknown_review_required,
        rights_note=(
            "Discovered automatically by Source Hunter. Verify this page is the "
            "authoritative/original source and confirm reuse/publication rights before use."
        ),
    )


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
