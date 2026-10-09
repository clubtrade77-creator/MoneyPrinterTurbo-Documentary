import json

from app.models.documentary import RightsStatus, SourceType
from app.services.documentary.source_hunter import (
    candidate_to_youtube_source,
    find_source_videos,
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


def _renderer(video_id, title, channel):
    return {
        "videoId": video_id,
        "title": {"runs": [{"text": title}]},
        "ownerText": {"runs": [{"text": channel}]},
        "publishedTimeText": {"simpleText": "2 hours ago"},
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

    assert len(results) == 2
    assert results[0].video_id == "lmnopqrstuv"
    assert results[0].source_quality_score == 25
    assert results[0].video_signal_score > 0
    assert results[0].score > results[1].score


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
