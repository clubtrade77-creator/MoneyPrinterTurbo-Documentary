from datetime import datetime, timezone

import pytest

from app.models.documentary import ProvenanceType, RightsStatus, SourceType
from app.services.documentary.story_discovery import (
    StoryDiscoveryError,
    _build_query,
    candidate_to_source,
    discover_stories,
)


class _Response:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response

class _SequenceSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def test_discover_stories_scores_visual_recent_candidates_first():
    session = _Session(
        _Response(
            {
                "articles": [
                    {
                        "url": "https://example.com/bodycam-story",
                        "title": "New bodycam video shows dramatic rescue after crash",
                        "seendate": "20261009T060000Z",
                        "domain": "example.com",
                        "language": "English",
                        "sourcecountry": "United States",
                        "socialimage": "https://example.com/image.jpg",
                    },
                    {
                        "url": "https://example.org/policy-story",
                        "title": "City council discusses parking policy",
                        "seendate": "20261009T050000Z",
                        "domain": "example.org",
                        "language": "English",
                        "sourcecountry": "United States",
                    },
                ]
            }
        )
    )

    results = discover_stories(
        lookback_hours=72,
        limit=10,
        session=session,
        now=datetime(2026, 10, 9, 7, 0, tzinfo=timezone.utc),
    )

    assert len(results) == 2
    assert results[0].title.startswith("New bodycam video")
    assert results[0].score > results[1].score
    assert results[0].footage_score > 0
    assert results[0].freshness_score == 30

    _, kwargs = session.calls[0]
    assert kwargs["params"]["mode"] == "artlist"
    assert kwargs["params"]["format"] == "json"
    assert kwargs["params"]["timespan"] == "72h"
    assert "bodycam" in kwargs["params"]["query"].lower()


def test_discover_stories_deduplicates_urls_and_titles():
    session = _Session(
        _Response(
            {
                "articles": [
                    {
                        "url": "https://example.com/a",
                        "title": "CCTV video shows unusual escape",
                        "seendate": "20261009T060000Z",
                    },
                    {
                        "url": "https://example.com/a",
                        "title": "Duplicate URL",
                        "seendate": "20261009T060000Z",
                    },
                    {
                        "url": "https://example.net/b",
                        "title": "CCTV video shows unusual escape",
                        "seendate": "20261009T060000Z",
                    },
                ]
            }
        )
    )

    results = discover_stories(
        limit=10,
        session=session,
        now=datetime(2026, 10, 9, 7, 0, tzinfo=timezone.utc),
    )

    assert len(results) == 1


def test_discover_stories_deduplicates_editorial_prefix_variants():
    session = _Session(
        _Response(
            {
                "articles": [
                    {
                        "url": "https://example.com/a",
                        "title": "VIDEO: PPB releases body camera footage, officer names in shooting that struck homicide suspect, K9",
                        "seendate": "20261009T060000Z",
                    },
                    {
                        "url": "https://example.net/b",
                        "title": "PPB releases body camera footage, officer names in shooting that struck homicide suspect, K9",
                        "seendate": "20261009T055900Z",
                    },
                ]
            }
        )
    )

    results = discover_stories(
        limit=10,
        session=session,
        now=datetime(2026, 10, 9, 7, 0, tzinfo=timezone.utc),
    )

    assert len(results) == 1


def test_discover_stories_clusters_different_headlines_for_same_event():
    session = _Session(
        _Response(
            {
                "articles": [
                    {
                        "url": "https://koin.example/story",
                        "title": "VIDEO: PPB releases body camera footage, officer names in shooting that struck homicide suspect, K9",
                        "seendate": "20261008T221441Z",
                        "domain": "KOIN.com",
                    },
                    {
                        "url": "https://kgw.example/story",
                        "title": "Bodycam footage captures deadly police shooting of Gresham homicide suspect, Portland police K9",
                        "seendate": "20261008T215900Z",
                        "domain": "KGW",
                    },
                    {
                        "url": "https://katu.example/story",
                        "title": "WATCH: Bodycam footage shows Portland police shooting that left suspect, K-9 officer dead",
                        "seendate": "20261009T012856Z",
                        "domain": "KATU",
                    },
                ]
            }
        )
    )

    results = discover_stories(
        limit=10,
        session=session,
        now=datetime(2026, 10, 9, 7, 0, tzinfo=timezone.utc),
    )

    assert len(results) == 1
    assert results[0].story_score == 30


def test_discover_stories_penalizes_camera_mentions_without_published_footage():
    session = _Session(
        _Response(
            {
                "articles": [
                    {
                        "url": "https://example.com/request",
                        "title": "MP challenges officials to release CCTV footage of 100 seats",
                        "seendate": "20261009T060000Z",
                    },
                    {
                        "url": "https://example.com/released",
                        "title": "Police release bodycam footage showing dramatic chase and arrest",
                        "seendate": "20261009T060000Z",
                    },
                ]
            }
        )
    )

    results = discover_stories(
        limit=10,
        session=session,
        now=datetime(2026, 10, 9, 7, 0, tzinfo=timezone.utc),
    )

    released = next(item for item in results if "bodycam" in item.title.lower())
    request = next(item for item in results if "challenges" in item.title.lower())
    assert released.footage_score > request.footage_score
    assert released.score > request.score


def test_discover_stories_camera_infrastructure_is_not_treated_as_footage():
    session = _Session(
        _Response(
            {
                "articles": [
                    {
                        "url": "https://example.com/cameras",
                        "title": "Councilmember pushes for harbor cameras after migrant boat intercepted",
                        "seendate": "20261009T060000Z",
                    }
                ]
            }
        )
    )

    result = discover_stories(
        limit=10,
        session=session,
        now=datetime(2026, 10, 9, 7, 0, tzinfo=timezone.utc),
    )[0]

    assert result.footage_score <= 4


def test_discover_stories_scores_high_stakes_story_terms():
    session = _Session(
        _Response(
            {
                "articles": [
                    {
                        "url": "https://example.com/bodycam-story",
                        "title": "Body camera footage released after shooting of homicide suspect and K9",
                        "seendate": "20261009T060000Z",
                    }
                ]
            }
        )
    )

    result = discover_stories(
        limit=10,
        session=session,
        now=datetime(2026, 10, 9, 7, 0, tzinfo=timezone.utc),
    )[0]

    assert result.story_score >= 25
    assert result.score >= 90


def test_discover_stories_uses_literal_user_topic():
    query = _build_query('airport "incident"')

    assert '"airport incident"' in query
    assert "bodycam" in query.lower()


def test_discover_stories_falls_back_when_gdelt_rate_limits():
    rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss><channel>
      <item>
        <title>Bodycam video shows dramatic rescue - Example News</title>
        <link>https://news.google.com/rss/articles/example</link>
        <pubDate>Fri, 09 Oct 2026 06:00:00 GMT</pubDate>
        <source url="https://example.com">Example News</source>
      </item>
    </channel></rss>
    """
    session = _SequenceSession(
        [
            _Response({}, status_code=429),
            _Response({}, status_code=200, text=rss),
        ]
    )

    results = discover_stories(
        lookback_hours=72,
        limit=10,
        session=session,
        now=datetime(2026, 10, 9, 7, 0, tzinfo=timezone.utc),
    )

    assert len(results) == 1
    assert results[0].publisher == "Example News"
    assert results[0].title == "Bodycam video shows dramatic rescue"
    assert results[0].footage_score > 0
    assert len(session.calls) == 2
    assert "gdeltproject.org" in session.calls[0][0]
    assert "news.google.com" in session.calls[1][0]


def test_discover_stories_rejects_bad_http_status():
    session = _Session(_Response({}, status_code=503))

    with pytest.raises(StoryDiscoveryError, match="HTTP 503"):
        discover_stories(session=session)


def test_candidate_to_source_keeps_rights_unverified():
    session = _Session(
        _Response(
            {
                "articles": [
                    {
                        "url": "https://example.com/bodycam-story",
                        "title": "Bodycam footage shows rescue",
                        "seendate": "20261009T060000Z",
                        "domain": "example.com",
                    }
                ]
            }
        )
    )
    candidate = discover_stories(
        session=session,
        now=datetime(2026, 10, 9, 7, 0, tzinfo=timezone.utc),
    )[0]

    source = candidate_to_source(candidate)

    assert source.source_type == SourceType.news
    assert source.provenance == ProvenanceType.third_party_platform
    assert source.rights_status == RightsStatus.unknown_review_required
    assert source.source_url == candidate.url
    assert source.local_path == ""
