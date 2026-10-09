from datetime import datetime, timezone

import pytest

from app.models.documentary import ProvenanceType, RightsStatus, SourceType
from app.services.documentary.story_discovery import (
    StoryDiscoveryError,
    candidate_to_source,
    discover_stories,
)


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

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


def test_discover_stories_uses_literal_user_topic():
    session = _Session(_Response({"articles": []}))

    discover_stories(
        'airport "incident"',
        session=session,
        now=datetime(2026, 10, 9, 7, 0, tzinfo=timezone.utc),
    )

    query = session.calls[0][1]["params"]["query"]
    assert '"airport incident"' in query
    assert "bodycam" in query.lower()


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
