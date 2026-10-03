from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models.documentary import (
    AudioMode,
    DocumentaryScene,
    RightsStatus,
    SceneType,
    SourceAsset,
    VideoMetadata,
)
from app.services.documentary.project import (
    add_source,
    create_project,
    project_dir,
    save_project,
    sha256_file,
)
from app.services.documentary.renderer import (
    DocumentaryRenderError,
    build_documentary_render_command,
    documentary_render_path,
    render_documentary,
)


def _project_with_video_source(
    tmp_path: Path,
    *,
    has_audio: bool = True,
    duration_seconds: float = 12.0,
):
    project = create_project(
        "Renderer case",
        project_id="doc_renderer_case",
        root=tmp_path,
    )
    local_path = project_dir(project.id, tmp_path) / "sources" / "source_renderer.mp4"
    local_path.write_bytes(b"fake-video-content")

    source = SourceAsset(
        id="source_renderer",
        source_type="local_video",
        title="Renderer source",
        local_path=str(local_path.resolve()),
        checksum_sha256=sha256_file(local_path),
        video_metadata=VideoMetadata(
            duration_seconds=duration_seconds,
            width=1280,
            height=720,
            fps=30,
            has_audio=has_audio,
            video_codec="h264",
            audio_codec="aac" if has_audio else "",
            container="mov,mp4",
            file_size_bytes=local_path.stat().st_size,
        ),
        rights_status=RightsStatus.user_owned,
    )
    project = add_source(project.id, source, root=tmp_path)
    return project, source, local_path


def _save_scene(
    project,
    source,
    tmp_path: Path,
    *,
    start: float = 1.0,
    end: float = 5.0,
    audio_mode: AudioMode = AudioMode.original,
):
    project.plan.scenes = [
        DocumentaryScene(
            id="scene_renderer",
            scene_type=SceneType.original_clip,
            source_id=source.id,
            source_start=start,
            source_end=end,
            audio_mode=audio_mode,
        )
    ]
    save_project(project, tmp_path)


def test_build_render_command_uses_exact_source_range_and_original_audio(tmp_path: Path):
    project, source, local_path = _project_with_video_source(tmp_path, has_audio=True)
    _save_scene(project, source, tmp_path)

    command = build_documentary_render_command(
        project.id,
        root=tmp_path,
        width=1280,
        height=720,
        fps=30,
    )
    joined = " ".join(command)

    assert str(local_path.resolve()) in command
    assert "trim=start=1.000000:end=5.000000" in joined
    assert "atrim=start=1.000000:end=5.000000" in joined
    assert "scale=1280:720:force_original_aspect_ratio=increase" in joined
    assert "crop=1280:720" in joined
    assert "libx264" in command
    assert "aac" in command


def test_build_render_command_generates_silence_when_source_has_no_audio(tmp_path: Path):
    project, source, _ = _project_with_video_source(tmp_path, has_audio=False)
    _save_scene(project, source, tmp_path, start=0.0, end=3.5)

    command = build_documentary_render_command(project.id, root=tmp_path)
    joined = " ".join(command)

    assert "anullsrc=r=48000:cl=stereo" in joined
    assert "atrim=duration=3.500000" in joined
    assert "[0:a]" not in joined


def test_build_render_command_generates_silence_for_muted_scene(tmp_path: Path):
    project, source, _ = _project_with_video_source(tmp_path, has_audio=True)
    _save_scene(
        project,
        source,
        tmp_path,
        start=2.0,
        end=4.0,
        audio_mode=AudioMode.muted,
    )

    command = build_documentary_render_command(project.id, root=tmp_path)
    joined = " ".join(command)

    assert "anullsrc=r=48000:cl=stereo" in joined
    assert "[0:a]" not in joined


def test_build_render_command_concats_multiple_scenes_and_hashes_source_once(
    tmp_path: Path,
    monkeypatch,
):
    project, source, _ = _project_with_video_source(tmp_path, has_audio=False)
    project.plan.scenes = [
        DocumentaryScene(
            id="scene_first",
            scene_type=SceneType.original_clip,
            source_id=source.id,
            source_start=0.0,
            source_end=2.0,
            audio_mode=AudioMode.muted,
        ),
        DocumentaryScene(
            id="scene_second",
            scene_type=SceneType.original_clip,
            source_id=source.id,
            source_start=2.0,
            source_end=4.0,
            audio_mode=AudioMode.muted,
        ),
    ]
    save_project(project, tmp_path)

    real_sha256_file = sha256_file
    checksum_calls = []

    def counted_sha256_file(path):
        checksum_calls.append(Path(path))
        return real_sha256_file(path)

    monkeypatch.setattr(
        "app.services.documentary.renderer.sha256_file",
        counted_sha256_file,
    )

    command = build_documentary_render_command(project.id, root=tmp_path)
    joined = " ".join(command)

    assert joined.count("source_renderer.mp4") == 2
    assert "concat=n=2:v=1:a=1[vout][aout]" in joined
    assert command.count("[vout]") == 1
    assert command.count("[aout]") == 1
    assert len(checksum_calls) == 1


def test_build_render_command_rejects_scene_outside_source_duration(tmp_path: Path):
    project, source, _ = _project_with_video_source(
        tmp_path,
        duration_seconds=4.0,
    )
    _save_scene(project, source, tmp_path, start=3.0, end=5.0)

    with pytest.raises(DocumentaryRenderError, match="ends outside source duration"):
        build_documentary_render_command(project.id, root=tmp_path)


def test_build_render_command_rejects_changed_source_file(tmp_path: Path):
    project, source, local_path = _project_with_video_source(tmp_path)
    _save_scene(project, source, tmp_path)
    local_path.write_bytes(b"changed-after-ingestion")

    with pytest.raises(DocumentaryRenderError, match="changed after ingestion"):
        build_documentary_render_command(project.id, root=tmp_path)


def test_build_render_command_rejects_unsupported_narration_audio_mode(tmp_path: Path):
    project, source, _ = _project_with_video_source(tmp_path)
    _save_scene(
        project,
        source,
        tmp_path,
        audio_mode=AudioMode.narration,
    )

    with pytest.raises(DocumentaryRenderError, match="does not support audio mode yet"):
        build_documentary_render_command(project.id, root=tmp_path)


def test_build_render_command_rejects_project_without_scenes(tmp_path: Path):
    project, _, _ = _project_with_video_source(tmp_path)

    with pytest.raises(DocumentaryRenderError, match="has no scenes"):
        build_documentary_render_command(project.id, root=tmp_path)


def test_render_documentary_publishes_staged_output_atomically(
    tmp_path: Path,
    monkeypatch,
):
    project, source, _ = _project_with_video_source(tmp_path, has_audio=False)
    _save_scene(project, source, tmp_path, start=0.0, end=2.0)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b"rendered-master")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "app.services.documentary.renderer.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "app.services.documentary.renderer.probe_video_metadata",
        lambda path: VideoMetadata(
            duration_seconds=2.0,
            width=1280,
            height=720,
            fps=30,
            has_audio=True,
            video_codec="h264",
            audio_codec="aac",
            container="mov,mp4",
            file_size_bytes=Path(path).stat().st_size,
        ),
    )

    output = render_documentary(
        project.id,
        root=tmp_path,
        width=1280,
        height=720,
    )

    assert output == documentary_render_path(project.id, tmp_path).resolve()
    assert output.read_bytes() == b"rendered-master"
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 3600
    assert not list(output.parent.glob(".master-*.mp4"))


def test_render_documentary_rejects_wrong_output_metadata_before_publish(
    tmp_path: Path,
    monkeypatch,
):
    project, source, _ = _project_with_video_source(tmp_path, has_audio=False)
    _save_scene(project, source, tmp_path, start=0.0, end=2.0)
    output = documentary_render_path(project.id, tmp_path)
    output.write_bytes(b"previous-good-render")

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"rendered-but-wrong")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "app.services.documentary.renderer.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "app.services.documentary.renderer.probe_video_metadata",
        lambda path: VideoMetadata(
            duration_seconds=1.0,
            width=640,
            height=360,
            fps=24,
            has_audio=False,
            video_codec="h264",
            audio_codec="",
            container="mov,mp4",
            file_size_bytes=Path(path).stat().st_size,
        ),
    )

    with pytest.raises(DocumentaryRenderError, match="resolution mismatch"):
        render_documentary(
            project.id,
            root=tmp_path,
            width=1280,
            height=720,
            fps=30,
        )

    assert output.read_bytes() == b"previous-good-render"
    assert not list(output.parent.glob(".master-*.mp4"))


def test_render_documentary_rejects_missing_audio_stream_before_publish(
    tmp_path: Path,
    monkeypatch,
):
    project, source, _ = _project_with_video_source(tmp_path, has_audio=False)
    _save_scene(project, source, tmp_path, start=0.0, end=2.0)

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"rendered-no-audio")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "app.services.documentary.renderer.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "app.services.documentary.renderer.probe_video_metadata",
        lambda path: VideoMetadata(
            duration_seconds=2.0,
            width=1920,
            height=1080,
            fps=30,
            has_audio=False,
            video_codec="h264",
            audio_codec="",
            container="mov,mp4",
            file_size_bytes=Path(path).stat().st_size,
        ),
    )

    with pytest.raises(DocumentaryRenderError, match="missing its audio stream"):
        render_documentary(project.id, root=tmp_path)

    assert not documentary_render_path(project.id, tmp_path).exists()


def test_render_documentary_does_not_replace_existing_output_on_ffmpeg_failure(
    tmp_path: Path,
    monkeypatch,
):
    project, source, _ = _project_with_video_source(tmp_path, has_audio=False)
    _save_scene(project, source, tmp_path, start=0.0, end=2.0)
    output = documentary_render_path(project.id, tmp_path)
    output.write_bytes(b"previous-good-render")

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial-render")
        return SimpleNamespace(returncode=1, stdout="", stderr="encoder failed")

    monkeypatch.setattr(
        "app.services.documentary.renderer.subprocess.run",
        fake_run,
    )

    with pytest.raises(DocumentaryRenderError, match="encoder failed"):
        render_documentary(project.id, root=tmp_path)

    assert output.read_bytes() == b"previous-good-render"
    assert not list(output.parent.glob(".master-*.mp4"))
