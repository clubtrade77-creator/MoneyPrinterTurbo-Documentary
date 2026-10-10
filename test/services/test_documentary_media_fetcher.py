from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.documentary import media_fetcher


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/video.mp4", True),
        ("http://example.com/video.mov", True),
        ("file:///tmp/video.mp4", False),
        ("http://localhost/video.mp4", False),
        ("http://127.0.0.1/video.mp4", False),
        ("http://10.0.0.5/video.mp4", False),
    ],
)
def test_public_media_url_filter(url: str, expected: bool):
    assert media_fetcher._is_public_http_url(url) is expected


def test_download_with_yt_dlp_returns_generated_mp4(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        media_fetcher,
        "_yt_dlp_command",
        lambda: ["yt-dlp"],
    )

    def fake_run(command, **kwargs):
        template = Path(command[command.index("-o") + 1])
        output = template.with_name("source.mp4")
        output.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(media_fetcher.subprocess, "run", fake_run)

    result = media_fetcher._download_with_yt_dlp(
        "https://www.youtube.com/watch?v=abcdefghijk",
        tmp_path,
    )

    assert result == tmp_path / "source.mp4"
    assert result.read_bytes() == b"video"


def test_download_with_yt_dlp_surfaces_downloader_error(
    monkeypatch,
    tmp_path: Path,
):
    monkeypatch.setattr(
        media_fetcher,
        "_yt_dlp_command",
        lambda: ["yt-dlp"],
    )
    monkeypatch.setattr(
        media_fetcher.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="source unavailable",
        ),
    )

    with pytest.raises(
        media_fetcher.DocumentaryMediaFetchError,
        match="source unavailable",
    ):
        media_fetcher._download_with_yt_dlp(
            "https://www.youtube.com/watch?v=abcdefghijk",
            tmp_path,
        )
