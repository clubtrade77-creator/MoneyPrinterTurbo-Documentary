from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urljoin, urlparse
from xml.etree import ElementTree

import requests
from loguru import logger

from app.models.documentary import (
    ProvenanceType,
    RightsStatus,
    SourceAsset,
    SourceType,
)
from app.services.documentary.youtube_source import (
    build_youtube_source_asset,
    canonical_youtube_url,
    extract_youtube_video_id,
)

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
    "youtube.com",
    "www.youtube.com",
    "youtu.be",
}
_GOVERNMENT_DOMAIN_SUFFIXES = (
    ".gov",
    ".gov.uk",
    ".gov.au",
    ".gov.in",
    ".gov.br",
    ".gov.sg",
    ".gov.hk",
    ".gov.ie",
    ".gov.za",
    ".govt.nz",
    ".gob.mx",
    ".go.jp",
    ".go.kr",
    ".gouv.fr",
    ".gc.ca",
)
_GOVERNMENT_DOMAIN_EXACT = {
    "canada.ca",
    "service-public.fr",
}
_COUNTRY_GOV_SEARCH_SUFFIX = {
    "united states": ".gov",
    "usa": ".gov",
    "us": ".gov",
    "united kingdom": ".gov.uk",
    "uk": ".gov.uk",
    "india": ".gov.in",
    "australia": ".gov.au",
    "brazil": ".gov.br",
    "singapore": ".gov.sg",
    "hong kong": ".gov.hk",
    "ireland": ".gov.ie",
    "south africa": ".gov.za",
    "new zealand": ".govt.nz",
    "mexico": ".gob.mx",
    "japan": ".go.jp",
    "south korea": ".go.kr",
    "korea": ".go.kr",
    "france": ".gouv.fr",
    "canada": "canada.ca",
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
    "police department",
    "police bureau",
    "sheriff",
    "county sheriff",
    "city of",
    "state patrol",
    "highway patrol",
    "district attorney",
    "prosecutor",
    "court",
    "fire department",
    "public safety",
    "u.s. marshals",
    "us marshals",
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
    "security camera",
    "interview",
)
_QUERY_NOISE_TERMS = {
    "video", "watch", "footage", "shows", "show", "the", "that", "with", "from",
    "after", "into", "over", "under", "and", "for", "police", "bodycam", "camera",
    "left", "officer", "new", "update", "updates", "release", "released",
    "releases", "involved", "official",
}
_GENERIC_MATCH_TERMS = {
    "officer", "suspect", "police", "incident", "man", "woman", "people", "person",
}
_STRONG_EVENT_TERMS = {
    "k9", "shooting", "homicide", "murder", "dead", "fatal", "chase", "rescue",
    "crash", "arrest", "standoff", "kidnapping", "hostage", "explosion", "attack",
    "trial", "court", "fire", "flood", "escape", "missing", "robbery", "collapse",
}
_AGENCY_ACRONYMS = {
    "PPB", "NYPD", "LAPD", "LASD", "ICE", "FBI", "DEA", "ATF", "CBP",
    "DHS", "DOJ", "USMS",
}


class SourceHunterError(RuntimeError):
    """Raised when original-video candidate discovery cannot be completed."""


@dataclass(frozen=True)
class EmbeddedMediaCandidate:
    platform: str
    url: str
    label: str
    source_page_url: str


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


class _OfficialMediaParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str]] = []
        self._anchor_url = ""
        self._anchor_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a":
            self._anchor_url = str(attrs.get("href") or "").strip()
            self._anchor_text = []
            return

        if tag in {"iframe", "video", "source"}:
            src = str(attrs.get("src") or "").strip()
            if src:
                label = str(attrs.get("title") or attrs.get("aria-label") or tag).strip()
                self.links.append((src, label))

    def handle_data(self, data):
        if self._anchor_url:
            self._anchor_text.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._anchor_url:
            label = re.sub(r"\s+", " ", "".join(self._anchor_text)).strip()
            self.links.append((self._anchor_url, label or "linked media"))
            self._anchor_url = ""
            self._anchor_text = []


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


def _is_government_domain(domain: str) -> bool:
    host = (domain or "").lower().strip(".")
    if not host:
        return False
    if host in _GOVERNMENT_DOMAIN_EXACT:
        return True
    if any(
        host == suffix.lstrip(".") or host.endswith(suffix)
        for suffix in _GOVERNMENT_DOMAIN_SUFFIXES
    ):
        return True
    return any(host.endswith("." + exact) for exact in _GOVERNMENT_DOMAIN_EXACT)


def _is_official_web_source(domain: str, title: str, snippet: str) -> tuple[int, list[str]]:
    host = (domain or "").lower()
    haystack = f"{host} {title} {snippet}".lower()
    reasons = []

    if _is_government_domain(host):
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


def _entity_anchor_tokens(story_title: str) -> list[str]:
    anchors = []
    normalized = re.sub(r"\bk[\s_-]*9\b", "K9", story_title or "", flags=re.IGNORECASE)
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9'-]{2,}\b", normalized):
        raw_token = match.group(0)
        # Short all-caps tokens are usually publisher/agency acronyms (PPB, NYPD,
        # ICE). They are useful search terms but too brittle to be mandatory event
        # anchors because official sources may spell the organization out.
        if raw_token.upper() in _AGENCY_ACRONYMS:
            continue
        token = raw_token.lower()
        if (
            token in _QUERY_NOISE_TERMS
            or token in _STRONG_EVENT_TERMS
            or token in _GENERIC_MATCH_TERMS
            or token in anchors
        ):
            continue
        anchors.append(token)
    return anchors


def _event_match_ok(
    story_title: str,
    candidate_text: str,
    *,
    official: bool,
) -> bool:
    story_tokens = _story_tokens(story_title)
    candidate_tokens = _story_tokens(candidate_text)

    story_strong = story_tokens & _STRONG_EVENT_TERMS
    candidate_strong = candidate_tokens & _STRONG_EVENT_TERMS
    if story_strong:
        required = min(1 if official else 2, len(story_strong))
        if len(story_strong & candidate_strong) < required:
            return False

    entity_anchors = set(_entity_anchor_tokens(story_title))
    if entity_anchors and not (entity_anchors & candidate_tokens):
        return False

    return True


def _source_query_terms(story_title: str) -> str:
    normalized = (story_title or "").lower()
    if "bodycam" in normalized or "body camera" in normalized:
        return "bodycam footage"
    if (
        "cctv" in normalized
        or "surveillance" in normalized
        or "security camera" in normalized
    ):
        return "cctv surveillance footage"
    if "dashcam" in normalized or "dash cam" in normalized:
        return "dashcam footage"
    if "court" in normalized or "trial" in normalized:
        return "court footage video"
    if "interview" in normalized:
        return "original interview video"
    return "original video footage"


def _agency_query_terms(story_title: str) -> str:
    normalized = (story_title or "").lower()
    if any(
        term in normalized
        for term in (
            "police", "officer", "sheriff", "bodycam", "body camera",
            "arrest", "shooting", "chase", "suspect",
        )
    ):
        return "police sheriff official"
    if "court" in normalized or "trial" in normalized:
        return "court official"
    return "official"


def _build_web_search_query(story_title: str) -> str:
    clean = re.sub(r"\s+", " ", (story_title or "").strip())
    if not clean:
        raise ValueError("story title is required for web source discovery")

    normalized = re.sub(r"\bk[\s_-]*9\b", "k9", clean.lower())
    ordered_tokens = []
    for token in re.findall(r"k9|[a-z0-9]{3,}", normalized):
        if token in _QUERY_NOISE_TERMS or token in ordered_tokens:
            continue
        ordered_tokens.append(token)

    entity_anchors = _entity_anchor_tokens(clean)
    strong_priority = [
        token for token in ordered_tokens if token in _STRONG_EVENT_TERMS
    ]
    specific = [
        token
        for token in ordered_tokens
        if token not in _STRONG_EVENT_TERMS
        and token not in _GENERIC_MATCH_TERMS
        and token not in entity_anchors
    ]
    selected = (entity_anchors[:3] + strong_priority[:3] + specific[:1])[:6]
    if not selected:
        selected = ordered_tokens[:5]

    return (
        " ".join(selected)
        + " "
        + _source_query_terms(story_title)
        + ' "video released" '
        + _agency_query_terms(story_title)
    ).strip()


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
    else:
        short_match = re.search(
            r"(\d+)\s*(m|h|d|w|mo|y)\s+ago",
            value,
        )
        if short_match:
            amount = int(short_match.group(1))
            unit = short_match.group(2)
            age_days = {
                "m": amount / 1440,
                "h": amount / 24,
                "d": amount,
                "w": amount * 7,
                "mo": amount * 30,
                "y": amount * 365,
            }[unit]
        else:
            age_days = None

    if age_days is not None:
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
    video_tokens = _story_tokens(f"{video_title} {channel}")
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


def _video_title_key(title: str) -> str:
    value = re.sub(r"\bk[\s_-]*9\b", "k9", (title or "").lower())
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _near_duplicate_video_title(left: str, right: str) -> bool:
    left_key = _video_title_key(left)
    right_key = _video_title_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True

    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    if min(len(left_tokens), len(right_tokens)) < 5:
        return False
    shared = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return bool(union) and len(shared) / len(union) >= 0.9


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
    return (" ".join(selected) + " " + _source_query_terms(story_title)).strip()


def _official_page_allowed_for_inspection(url: str) -> bool:
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower()
    return (
        parsed.scheme in {"http", "https"}
        and bool(host)
        and _is_government_domain(host)
        and parsed.username is None
        and parsed.password is None
    )


def _normalize_embedded_media_url(
    source_page_url: str,
    raw_url: str,
) -> tuple[str, str] | None:
    absolute = urljoin(source_page_url, unescape((raw_url or "").strip()))
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    host = (parsed.hostname or "").lower()
    if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
        try:
            video_id = extract_youtube_video_id(absolute)
        except ValueError:
            return None
        return "youtube", canonical_youtube_url(video_id)

    if host in {"youtube-nocookie.com", "www.youtube-nocookie.com"}:
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2 and parts[0] == "embed":
            video_id = parts[1]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                return "youtube", canonical_youtube_url(video_id)
        return None

    if host == "vimeo.com" or host.endswith(".vimeo.com"):
        return "vimeo", absolute

    path = parsed.path.lower()
    if path.endswith((".mp4", ".mov", ".webm", ".m3u8")):
        return "direct_video", absolute

    return None


def find_embedded_media(
    source_page_url: str,
    *,
    limit: int = 6,
    timeout_seconds: float = 8.0,
    session=None,
) -> list[EmbeddedMediaCandidate]:
    """Inspect an official US .gov source page for embedded/linked video media.

    The fetch is deliberately restricted to recognized government domains discovered
    by Source Hunter; arbitrary third-party URLs are not fetched by this helper.
    """
    if not _official_page_allowed_for_inspection(source_page_url):
        raise SourceHunterError(
            "embedded-media inspection is restricted to recognized government domains"
        )

    try:
        limit = int(limit)
        timeout_seconds = float(timeout_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid embedded-media inspection settings") from exc
    if not 1 <= limit <= 20:
        raise ValueError("embedded-media limit must be between 1 and 20")
    if timeout_seconds <= 0:
        raise ValueError("embedded-media timeout must be positive")

    request = session or requests
    try:
        response = request.get(
            source_page_url,
            headers=_REQUEST_HEADERS,
            timeout=min(timeout_seconds, 10.0),
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise SourceHunterError("official source page request failed") from exc

    if 300 <= response.status_code < 400:
        raise SourceHunterError(
            "official source page redirected; media inspection skipped"
        )
    if response.status_code != 200:
        raise SourceHunterError(
            f"official source page returned HTTP {response.status_code}"
        )

    parser = _OfficialMediaParser()
    parser.feed((response.text or "")[:2_000_000])

    results: list[EmbeddedMediaCandidate] = []
    seen_urls: set[str] = set()
    for raw_url, label in parser.links:
        normalized = _normalize_embedded_media_url(source_page_url, raw_url)
        if normalized is None:
            continue
        platform, media_url = normalized
        if media_url in seen_urls:
            continue
        results.append(
            EmbeddedMediaCandidate(
                platform=platform,
                url=media_url,
                label=re.sub(r"\s+", " ", label or "").strip() or platform,
                source_page_url=source_page_url,
            )
        )
        seen_urls.add(media_url)
        if len(results) >= limit:
            break

    return results


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

        # Source Hunter is intentionally strict: a web result must match the
        # concrete event and either contain footage/video evidence or be a government source.
        if overlap < 12:
            continue
        if video_signal <= 0 and official_score < 35:
            continue
        if not _event_match_ok(
            story_title,
            f"{title} {snippet} {domain}",
            official=official_score >= 35,
        ):
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


def _official_search_query(query: str, source_country: str) -> str:
    country = re.sub(r"\s+", " ", (source_country or "").strip().lower())
    suffix = _COUNTRY_GOV_SEARCH_SUFFIX.get(country, "")
    if suffix:
        return f"{query} site:{suffix}"
    return f"{query} official"


def find_web_sources(
    story_title: str,
    *,
    source_country: str = "",
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
    official_query = _official_search_query(query, source_country)
    request = session or requests
    provider_timeout = min(float(timeout_seconds), 10.0)
    errors = []
    collected_items: list[dict] = []

    try:
        response = request.get(
            _DUCKDUCKGO_SEARCH_URL,
            params={"q": query},
            headers=_REQUEST_HEADERS,
            timeout=provider_timeout,
            allow_redirects=True,
        )
        if response.status_code == 200:
            parser = _DuckDuckGoParser()
            parser.feed(response.text)
            if parser.results:
                collected_items.extend(parser.results)
            else:
                errors.append("DuckDuckGo returned no usable results")
        else:
            errors.append(f"DuckDuckGo returned HTTP {response.status_code}")
    except requests.RequestException as exc:
        errors.append(f"DuckDuckGo request failed: {exc}")

    try:
        response = request.get(
            _BING_SEARCH_URL,
            params={"q": official_query, "setlang": "en-US", "format": "rss"},
            headers=_REQUEST_HEADERS,
            timeout=provider_timeout,
            allow_redirects=True,
        )
        if response.status_code == 200:
            try:
                bing_items = _parse_bing_rss(response.text)
            except SourceHunterError:
                bing_items = []

            if not bing_items:
                parser = _BingParser()
                parser.feed(response.text)
                bing_items = parser.results

            if bing_items:
                collected_items.extend(bing_items)
            else:
                errors.append("Bing returned no relevant source results")
        else:
            errors.append(f"Bing returned HTTP {response.status_code}")
    except requests.RequestException as exc:
        errors.append(f"Bing request failed: {exc}")

    candidates = _web_candidates_from_items(
        story_title,
        collected_items,
        limit=limit,
    )
    if candidates:
        logger.info(
            f"Documentary Source Hunter found {len(candidates)} aggregated web candidates"
        )
        return candidates

    detail = "; ".join(errors) or "providers returned no relevant source results"
    raise SourceHunterError("web source search failed: " + detail)


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
    seen_titles: list[str] = []

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

        # Source Hunter is for likely original/source footage, not ordinary news
        # recaps. Keep a candidate only if the title strongly signals source video
        # or the upload comes from an official/agency channel.
        if video_signal <= 0 and source_quality <= 0:
            continue

        if not _event_match_ok(
            story_title,
            f"{title} {channel}",
            official=source_quality > 0,
        ):
            continue

        # Very old non-official uploads are almost never the source for a fresh
        # story lead and should not survive merely because generic event terms match.
        if freshness == 0 and source_quality <= 0:
            continue

        if any(
            _near_duplicate_video_title(title, seen_title)
            for seen_title in seen_titles
        ):
            continue

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
        seen_titles.append(title)

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


def embedded_media_to_source(
    candidate: EmbeddedMediaCandidate,
    *,
    title: str = "",
) -> SourceAsset:
    """Convert media embedded on an official page into a traceable source asset."""
    rights_note = (
        "Media discovered on an official source page. Verify reuse/publication "
        "rights separately before using the media in a published documentary."
    )

    if candidate.platform == "youtube":
        return build_youtube_source_asset(
            candidate.url,
            title=(title or candidate.label).strip(),
            rights_status=RightsStatus.unknown_review_required,
            rights_note=rights_note,
        )

    source_id = (
        "media_"
        + hashlib.sha256(candidate.url.encode("utf-8")).hexdigest()[:16]
    )
    return SourceAsset(
        id=source_id,
        source_type=SourceType.news,
        provenance=ProvenanceType.official_public_source,
        title=(title or candidate.label).strip(),
        source_url=candidate.url,
        publisher=(urlparse(candidate.source_page_url).hostname or ""),
        rights_status=RightsStatus.unknown_review_required,
        rights_note=rights_note,
    )


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
