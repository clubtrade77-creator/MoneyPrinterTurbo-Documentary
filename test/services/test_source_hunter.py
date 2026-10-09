import json

from app.models.documentary import ProvenanceType, RightsStatus, SourceType
from app.services.documentary.source_hunter import (
    _build_web_search_query,
    candidate_to_web_source,
    candidate_to_youtube_source,
    find_source_videos,
    find_web_sources,
)


class _Response:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


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


def _youtube_html(renderers):
    payload = {
        "contents": {
            "twoColumnSearchResultsRenderer": {
                "primaryContents": {
                    "sectionListRenderer": {
                        "contents": [
                            {
                                "itemSectionRenderer": {
                                    "contents": [
                                        {"videoRenderer": renderer}
                                        for renderer in renderers
                                    ]
                                }
                            }
                        ]
                    }
                }
            }
        }
    }
    return "var ytInitialData = " + json.dumps(payload) + ";</script>"


def _renderer(video_id, title, channel, published_text="2 hours ago"):
    return {
        "videoId": video_id,
        "title": {"runs": [{"text": title}]},
        "ownerText": {"runs": [{"text": channel}]},
        "publishedTimeText": {"simpleText": published_text},
        "lengthText": {"simpleText": "8:12"},
        "viewCountText": {"simpleText": "12K views"},
    }


def test_source_hunter_prefers_matching_official_channel():
    session = _Session(
        _Response(
            _youtube_html(
                [
                    _renderer(
                        "abcdefghijk",
                        "Portland shooting discussion and reaction",
                        "Random Commentary",
                    ),
                    _renderer(
                        "lmnopqrstuv",
                        "Bodycam footage: Portland police shooting involving homicide suspect and K9",
                        "Portland Police Bureau",
                    ),
                ]
            )
        )
    )

    results = find_source_videos(
        "Bodycam footage shows Portland police shooting that left suspect, K9 officer dead",
        session=session,
    )

    assert len(results) == 1
    assert results[0].video_id == "lmnopqrstuv"
    assert results[0].source_quality_score == 20
    assert results[0].video_signal_score > 0


def test_source_hunter_penalizes_old_similar_incident():
    session = _Session(
        _Response(
            _youtube_html(
                [
                    _renderer(
                        "abcdefghijk",
                        "Suspect attacks two people, injures Portland police officer",
                        "KGW News",
                        "7 years ago",
                    ),
                    _renderer(
                        "lmnopqrstuv",
                        "Portland K9 shooting bodycam footage released after homicide suspect killed",
                        "Local News",
                        "3 hours ago",
                    ),
                ]
            )
        )
    )

    results = find_source_videos(
        "Bodycam footage shows Portland police shooting that left suspect, K9 officer dead",
        session=session,
    )

    assert len(results) == 1
    assert results[0].video_id == "lmnopqrstuv"
    assert results[0].freshness_score == 20


def test_source_hunter_web_query_prioritizes_event_anchors():
    query = _build_web_search_query(
        "WATCH: Bodycam footage shows Portland police shooting that left suspect, K-9 officer dead"
    )

    assert "portland" in query.lower()
    assert "k9" in query.lower()
    assert "shooting" in query.lower()
    assert '"video released"' in query.lower()
    assert "suspect" not in query.lower()


def test_source_hunter_rejects_old_generic_same_city_shooting():
    session = _Session(
        _Response(
            _youtube_html(
                [
                    _renderer(
                        "abcdefghijk",
                        "Body cam footage released of Portland officer-involved shooting",
                        "KOIN 6",
                        "1y ago",
                    ),
                    _renderer(
                        "lmnopqrstuv",
                        "Portland K9 shooting bodycam footage released after homicide suspect killed",
                        "Local News",
                        "2 hours ago",
                    ),
                ]
            )
        )
    )

    results = find_source_videos(
        "Bodycam footage shows Portland police shooting that left suspect, K9 officer dead",
        session=session,
    )

    assert len(results) == 1
    assert results[0].video_id == "lmnopqrstuv"


def test_source_hunter_query_keeps_k9_event_anchor():
    session = _Session(_Response(_youtube_html([])))

    find_source_videos(
        "WATCH: Bodycam footage shows Portland police shooting that left suspect, K-9 officer dead",
        session=session,
    )

    query = session.calls[0][1]["params"]["search_query"]
    assert "k9" in query.lower()
    assert "shooting" in query.lower()
    assert "bodycam footage" in query.lower()


def test_source_hunter_understands_short_youtube_age_labels():
    session = _Session(
        _Response(
            _youtube_html(
                [
                    _renderer(
                        "abcdefghijk",
                        "Bodycam footage released after K9 shooting",
                        "Local News",
                        "2w ago",
                    ),
                    _renderer(
                        "lmnopqrstuv",
                        "Bodycam footage released after K9 shooting update",
                        "Local News",
                        "9d ago",
                    ),
                ]
            )
        )
    )

    results = find_source_videos(
        "Bodycam footage shows Portland police shooting that left suspect, K9 officer dead",
        session=session,
    )

    assert results
    by_id = {item.video_id: item for item in results}
    assert by_id["abcdefghijk"].freshness_score == 14
    assert by_id["lmnopqrstuv"].freshness_score == 14


def test_source_hunter_filters_news_recaps_without_source_video_signal():
    session = _Session(
        _Response(
            _youtube_html(
                [
                    _renderer(
                        "abcdefghijk",
                        "Wanted Gresham homicide suspect dead after police shooting in Portland; police K-9 killed",
                        "KATU News",
                        "2w ago",
                    ),
                    _renderer(
                        "lmnopqrstuv",
                        "Portland Police release bodycam footage from K9 shooting",
                        "Local News",
                        "2 hours ago",
                    ),
                ]
            )
        )
    )

    results = find_source_videos(
        "Bodycam footage shows Portland police shooting that left suspect, K9 officer dead",
        session=session,
    )

    assert len(results) == 1
    assert results[0].video_id == "lmnopqrstuv"
    assert results[0].video_signal_score > 0


def test_source_hunter_deduplicates_same_title_different_video_ids():
    session = _Session(
        _Response(
            _youtube_html(
                [
                    _renderer(
                        "abcdefghijk",
                        "Portland Police release bodycam footage from K9 shooting",
                        "Local News",
                    ),
                    _renderer(
                        "lmnopqrstuv",
                        "Portland Police release bodycam footage from K9 shooting",
                        "Local News",
                    ),
                ]
            )
        )
    )

    results = find_source_videos(
        "Bodycam footage shows Portland police shooting that left suspect, K9 officer dead",
        session=session,
    )

    assert len(results) == 1


def test_source_hunter_deduplicates_video_ids():
    renderer = _renderer(
        "abcdefghijk",
        "Bodycam footage shows rescue",
        "City Police Department",
    )
    session = _Session(_Response(_youtube_html([renderer, renderer])))

    results = find_source_videos(
        "Bodycam footage shows rescue",
        session=session,
    )

    assert len(results) == 1


def test_source_hunter_finds_official_web_source():
    html = """
    <html><body>
      <div class="result">
        <a class="result__a" href="https://www.portlandoregon.gov/police/article/123">
          Portland Police Bureau releases bodycam footage in K9 shooting
        </a>
        <a class="result__snippet">
          Official police bureau page with released body camera video from the shooting.
        </a>
      </div>
      <div class="result">
        <a class="result__a" href="https://example.com/commentary">
          Portland shooting commentary
        </a>
        <a class="result__snippet">
          Discussion of the incident.
        </a>
      </div>
    </body></html>
    """
    session = _Session(_Response(html))

    results = find_web_sources(
        "Bodycam footage shows Portland police shooting that left suspect, K9 officer dead",
        session=session,
    )

    assert len(results) == 1
    assert results[0].domain == "portlandoregon.gov"
    assert results[0].official_score == 35
    assert results[0].video_signal_score > 0


def test_source_hunter_filters_irrelevant_web_results():
    html = """
    <html><body>
      <div class="result">
        <a class="result__a" href="https://support.microsoft.com/printer-driver">
          Download and install printer drivers
        </a>
        <a class="result__snippet">
          Microsoft recommends installing the latest printer driver.
        </a>
      </div>
      <div class="result">
        <a class="result__a" href="https://agency.gov/bodycam-release">
          Police department releases bodycam footage after K9 shooting
        </a>
        <a class="result__snippet">
          Official body camera video from the Portland shooting.
        </a>
      </div>
    </body></html>
    """

    results = find_web_sources(
        "Bodycam footage shows Portland police shooting that left suspect, K9 officer dead",
        session=_Session(_Response(html)),
    )

    assert len(results) == 1
    assert results[0].domain == "agency.gov"


def test_source_hunter_web_search_falls_back_to_bing():
    bing_html = """
    <html><body>
      <li class="b_algo">
        <h2>
          <a href="https://www.portlandoregon.gov/police/article/123">
            Portland Police Bureau releases bodycam footage in K9 shooting
          </a>
        </h2>
        <p>Official police bureau page with released body camera video.</p>
      </li>
    </body></html>
    """
    session = _SequenceSession(
        [
            _Response("<html><body>No results</body></html>", status_code=200),
            _Response(bing_html, status_code=200),
        ]
    )

    results = find_web_sources(
        "Bodycam footage shows Portland police shooting that left suspect, K9 officer dead",
        session=session,
    )

    assert len(results) == 1
    assert results[0].domain == "portlandoregon.gov"
    assert results[0].official_score == 35
    assert len(session.calls) == 2
    assert "duckduckgo.com" in session.calls[0][0]
    assert "bing.com" in session.calls[1][0]


def test_source_hunter_parses_bing_rss_fallback():
    rss = """<?xml version="1.0"?>
    <rss><channel>
      <item>
        <title>Portland Police Bureau releases bodycam footage in K9 shooting</title>
        <link>https://www.portlandoregon.gov/police/article/123</link>
        <description>Official police bureau page with released body camera video.</description>
      </item>
    </channel></rss>
    """
    session = _SequenceSession(
        [
            _Response("<html><body>No results</body></html>", status_code=200),
            _Response(rss, status_code=200),
        ]
    )

    results = find_web_sources(
        "Bodycam footage shows Portland police shooting that left suspect, K9 officer dead",
        session=session,
    )

    assert len(results) == 1
    assert results[0].domain == "portlandoregon.gov"


def test_source_hunter_web_converter_preserves_rights_review():
    html = """
    <html><body>
      <div class="result">
        <a class="result__a" href="https://agency.gov/bodycam-release">
          Police department releases bodycam footage
        </a>
        <a class="result__snippet">
          Official body camera footage released after shooting.
        </a>
      </div>
    </body></html>
    """
    candidate = find_web_sources(
        "Police shooting bodycam footage",
        session=_Session(_Response(html)),
    )[0]

    source = candidate_to_web_source(candidate)

    assert source.provenance == ProvenanceType.official_public_source
    assert source.source_type == SourceType.bodycam
    assert source.rights_status == RightsStatus.unknown_review_required
    assert source.local_path == ""


def test_source_hunter_converts_candidate_without_granting_rights():
    session = _Session(
        _Response(
            _youtube_html(
                [
                    _renderer(
                        "abcdefghijk",
                        "Bodycam footage shows rescue",
                        "City Police Department",
                    )
                ]
            )
        )
    )
    candidate = find_source_videos(
        "Bodycam footage shows rescue",
        session=session,
    )[0]

    source = candidate_to_youtube_source(candidate)

    assert source.source_type == SourceType.youtube
    assert source.youtube_video_id == "abcdefghijk"
    assert source.rights_status == RightsStatus.unknown_review_required
    assert source.local_path == ""
