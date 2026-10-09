from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models.documentary import (
    ProvenanceType,
    RightsStatus,
    SourceAsset,
    SourceType,
)
from app.services.documentary import acquisition


class _DirectResponse:
    def __init__(self, body=b"video-bytes", status_code=200, headers=None):
        self.body = body
        self.status_code = status_code
        self.headers = headers or {
            "Content-Type": "video/mp4",
            "Content-Length": str(len(body)),
        }
        self.closed = False

    def iter_content(self, chunk_size=1024):
        del chunk_size
        yield self.body

    def close(self):
        self.closed = True


class _DirectSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class _FakeYDL:
    options = None

    def __init__(self, options):
        type(self).options = options
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def extract_info(self, url, download=True):
        assert download is True
        assert url == "https://www.youtube.com/watch?v=abcdefghijk"
        output = self.options["outtmpl"].replace("%(id)s", "abcdefghijk").replace(
            "%(ext)s", "mp4"
        )
        Path(output).write_bytes(b"fake-mp4")
        return {"id": "abcdefghijk"}


def _youtube_source():
    return SourceAsset(
        id="youtube_abcdefghijk",
        source_type=SourceType.youtube,
        provenance=ProvenanceType.third_party_platform,
        title="Official bodycam video",
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        rights_status=RightsStatus.unknown_review_required,
    )


def test_can_auto_acquire_youtube_without_local_copy():
    source = _youtube_source()

    assert acquisition.can_auto_acquire(source) is True


def test_can_auto_acquire_direct_official_video():
    source = SourceAsset(
        id="media_abcdefghijk1234",
        source_type=SourceType.news,
        provenance=ProvenanceType.official_public_source,
        source_url="https://agency.gov/media/bodycam.mp4",
        rights_status=RightsStatus.unknown_review_required,
    )

    assert acquisition.can_auto_acquire(source) is True


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/video.mp4",
        "http://10.0.0.5/video.mp4",
        "http://localhost/video.mp4",
        "file:///tmp/video.mp4",
    ],
)
def test_direct_acquisition_rejects_non_public_urls(url):
    source = SourceAsset(
        id="media_abcdefghijk1234",
        source_type=SourceType.news,
        provenance=ProvenanceType.official_public_source,
        source_url=url,
        rights_status=RightsStatus.unknown_review_required,
    )

    assert acquisition.can_auto_acquire(source) is False


def test_download_direct_video_streams_to_temp_file(tmp_path):
    response = _DirectResponse(body=b"abc123")
    session = _DirectSession([response])

    path = acquisition._download_direct_video(
        "https://agency.gov/media/bodycam.mp4",
        tmp_path,
        timeout_seconds=30,
        session=session,
    )

    assert path.read_bytes() == b"abc123"
    assert response.closed is True
    assert session.calls[0][1]["allow_redirects"] is False


def test_download_direct_video_validates_redirect_target(tmp_path):
    response = _DirectResponse(
        body=b"",
        status_code=302,
        headers={"Location": "http://127.0.0.1/private.mp4"},
    )
    session = _DirectSession([response])

    with pytest.raises(
        acquisition.SourceAcquisitionError,
        match="redirected to an unsafe URL",
    ):
        acquisition._download_direct_video(
            "https://agency.gov/media/bodycam.mp4",
            tmp_path,
            timeout_seconds=30,
            session=session,
        )


def test_download_youtube_uses_bounded_format_and_produces_mp4(tmp_path):
    source = _youtube_source()

    path = acquisition._download_youtube_video(
        source,
        tmp_path,
        timeout_seconds=30,
        ydl_factory=_FakeYDL,
    )

    assert path.suffix == ".mp4"
    assert path.read_bytes() == b"fake-mp4"
    assert "height<=1080" in _FakeYDL.options["format"]
    assert _FakeYDL.options["noplaylist"] is True
    assert _FakeYDL.options["ignoreconfig"] is True


def test_acquire_source_attaches_copy_without_changing_rights(monkeypatch):
    source = _youtube_source()
    project = SimpleNamespace(sources=[source])
    attached = SourceAsset(
        **source.model_dump(),
    )
    attached.local_path = "/tmp/project/source.mp4"
    attached.checksum_sha256 = "a" * 64

    monkeypatch.setattr(acquisition, "load_project", lambda project_id, root=None: project)

    captured = {}

    def fake_attach(project_id, source_id, source_path, root=None):
        captured["project_id"] = project_id
        captured["source_id"] = source_id
        captured["source_path"] = Path(source_path)
        captured["root"] = root
        return attached

    monkeypatch.setattr(acquisition, "attach_local_copy_to_source", fake_attach)

    result = acquisition.acquire_source(
        "doc_test",
        source.id,
        ydl_factory=_FakeYDL,
    )

    assert captured["project_id"] == "doc_test"
    assert captured["source_id"] == source.id
    assert captured["source_path"].suffix == ".mp4"
    assert result.rights_status == RightsStatus.unknown_review_required
