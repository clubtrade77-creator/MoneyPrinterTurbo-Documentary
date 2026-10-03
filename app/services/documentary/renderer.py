from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from app.models.documentary import (
    AudioMode,
    DocumentaryProject,
    NarrationAudioAsset,
    SceneType,
    SourceAsset,
    VideoMetadata,
)
from app.services.documentary.audio import NarrationAudioError, load_narration_audio
from app.services.documentary.clip_selector import story_plan_fingerprint
from app.services.documentary.metadata import MediaProbeError, probe_video_metadata
from app.services.documentary.project import load_project, project_dir, sha256_file
from app.services.documentary.story_planner import load_story_plan
from app.utils import utils

DEFAULT_RENDER_WIDTH = 1920
DEFAULT_RENDER_HEIGHT = 1080
DEFAULT_RENDER_FPS = 30
DEFAULT_RENDER_TIMEOUT_SECONDS = 3600
_SOURCE_RANGE_TOLERANCE_SECONDS = 0.05
_NARRATION_DURATION_TOLERANCE_SECONDS = 0.05

_SUPPORTED_SCENE_TYPES = {
    SceneType.original_clip,
    SceneType.narration_over_source,
    SceneType.broll,
}
_SUPPORTED_AUDIO_MODES = {
    AudioMode.original,
    AudioMode.narration,
    AudioMode.mixed,
    AudioMode.muted,
}


class DocumentaryRenderError(RuntimeError):
    """Raised when a documentary timeline cannot be rendered safely."""


def documentary_render_path(
    project_id: str,
    root: str | os.PathLike | None = None,
) -> Path:
    return project_dir(project_id, root) / "renders" / "master.mp4"


def _validate_render_settings(width: int, height: int, fps: int) -> None:
    if width < 16 or height < 16:
        raise ValueError("documentary render dimensions must be at least 16x16")
    if width % 2 or height % 2:
        raise ValueError("documentary render dimensions must be even for yuv420p")
    if fps < 1 or fps > 120:
        raise ValueError("documentary render fps must be between 1 and 120")


def _resolve_source(
    project: DocumentaryProject,
    source_id: str,
    *,
    root: str | os.PathLike | None,
) -> SourceAsset:
    source = next((item for item in project.sources if item.id == source_id), None)
    if source is None:
        raise DocumentaryRenderError(
            f"documentary scene references unknown source: {source_id}"
        )
    if not source.local_path:
        raise DocumentaryRenderError(
            f"documentary source has no local copy: {source_id}"
        )

    source_path = Path(source.local_path).expanduser().resolve()
    sources_dir = (project_dir(project.id, root) / "sources").resolve()
    if sources_dir != source_path.parent and sources_dir not in source_path.parents:
        raise DocumentaryRenderError(
            f"documentary source path escapes project sources directory: {source_id}"
        )
    if not source_path.is_file():
        raise DocumentaryRenderError(
            f"documentary source local copy is missing: {source_id}"
        )
    if source.video_metadata is None:
        raise DocumentaryRenderError(
            f"documentary source lacks verified video metadata: {source_id}"
        )
    if source.checksum_sha256:
        current_checksum = sha256_file(source_path)
        if current_checksum != source.checksum_sha256:
            raise DocumentaryRenderError(
                f"documentary source changed after ingestion: {source_id}"
            )
    return source


def _validate_scene(
    project: DocumentaryProject,
    scene,
    *,
    root: str | os.PathLike | None,
    source_cache: dict[str, SourceAsset] | None = None,
) -> tuple[SourceAsset, float, NarrationAudioAsset | None]:
    if scene.scene_type not in _SUPPORTED_SCENE_TYPES:
        raise DocumentaryRenderError(
            f"documentary renderer does not support scene type yet: "
            f"{scene.scene_type.value}"
        )
    if scene.audio_mode not in _SUPPORTED_AUDIO_MODES:
        raise DocumentaryRenderError(
            f"documentary renderer does not support audio mode yet: "
            f"{scene.audio_mode.value}"
        )

    source = None
    if source_cache is not None:
        source = source_cache.get(scene.source_id)
    if source is None:
        source = _resolve_source(project, scene.source_id, root=root)
        if source_cache is not None:
            source_cache[scene.source_id] = source
    if scene.source_start is None or scene.source_end is None:
        raise DocumentaryRenderError(
            f"source-backed documentary scene has no source range: {scene.id}"
        )

    source_duration = source.video_metadata.duration_seconds
    if scene.source_start >= source_duration:
        raise DocumentaryRenderError(
            f"documentary scene starts outside source duration: {scene.id}"
        )
    if scene.source_end > source_duration + _SOURCE_RANGE_TOLERANCE_SECONDS:
        raise DocumentaryRenderError(
            f"documentary scene ends outside source duration: {scene.id}"
        )

    duration = min(scene.source_end, source_duration) - scene.source_start
    if duration <= 0:
        raise DocumentaryRenderError(
            f"documentary scene has an empty source range: {scene.id}"
        )

    narration_asset = None
    if scene.audio_mode in {AudioMode.narration, AudioMode.mixed}:
        if scene.audio_mode == AudioMode.mixed and not source.video_metadata.has_audio:
            raise DocumentaryRenderError(
                f"mixed documentary audio requires source audio: {scene.id}"
            )
        try:
            narration_asset = load_narration_audio(
                project.id,
                scene.id,
                language=project.master_language,
                root=root,
            )
        except NarrationAudioError as exc:
            raise DocumentaryRenderError(str(exc)) from exc
        if (
            narration_asset.duration_seconds
            > duration + _NARRATION_DURATION_TOLERANCE_SECONDS
        ):
            raise DocumentaryRenderError(
                f"narration audio exceeds scene duration: {scene.id}"
            )

    return source, duration, narration_asset


def _build_documentary_render_command_and_expectations(
    project_id: str,
    *,
    root: str | os.PathLike | None = None,
    output_path: str | os.PathLike | None = None,
    width: int = DEFAULT_RENDER_WIDTH,
    height: int = DEFAULT_RENDER_HEIGHT,
    fps: int = DEFAULT_RENDER_FPS,
) -> tuple[list[str], float, int, int]:
    _validate_render_settings(width, height, fps)
    project = load_project(project_id, root)
    if not project.plan.scenes:
        raise DocumentaryRenderError("documentary project has no scenes to render")

    if project.plan.story_plan_fingerprint:
        current_story_plan = load_story_plan(project_id, root=root)
        current_story_fingerprint = story_plan_fingerprint(current_story_plan)
        if current_story_fingerprint != project.plan.story_plan_fingerprint:
            raise DocumentaryRenderError(
                "documentary timeline is stale; Story Plan changed after clip selection"
            )

    resolved_scenes: list[
        tuple[object, SourceAsset, float, NarrationAudioAsset | None]
    ] = []
    source_cache: dict[str, SourceAsset] = {}
    for scene in project.plan.scenes:
        source, duration, narration_asset = _validate_scene(
            project,
            scene,
            root=root,
            source_cache=source_cache,
        )
        resolved_scenes.append((scene, source, duration, narration_asset))

    target = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else documentary_render_path(project_id, root).resolve()
    )

    command = [utils.get_ffmpeg_binary(), "-y", "-nostdin"]
    input_indices: list[tuple[int, int | None]] = []
    next_input_index = 0
    for _, source, _, narration_asset in resolved_scenes:
        source_input_index = next_input_index
        command.extend(["-i", str(Path(source.local_path).expanduser().resolve())])
        next_input_index += 1

        narration_input_index = None
        if narration_asset is not None:
            narration_input_index = next_input_index
            command.extend(
                [
                    "-i",
                    str(Path(narration_asset.local_path).expanduser().resolve()),
                ]
            )
            next_input_index += 1
        input_indices.append((source_input_index, narration_input_index))

    filters: list[str] = []
    concat_inputs: list[str] = []

    for index, (scene, source, duration, narration_asset) in enumerate(resolved_scenes):
        source_input_index, narration_input_index = input_indices[index]
        start = float(scene.source_start)
        end = start + duration

        filters.append(
            f"[{source_input_index}:v]"
            f"trim=start={start:.6f}:end={end:.6f},"
            "setpts=PTS-STARTPTS,"
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"fps={fps},setsar=1,format=yuv420p[v{index}]"
        )

        if (
            scene.audio_mode == AudioMode.original
            and bool(source.video_metadata.has_audio)
        ):
            filters.append(
                f"[{source_input_index}:a]"
                f"atrim=start={start:.6f}:end={end:.6f},"
                "asetpts=PTS-STARTPTS,"
                "aresample=48000,"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"volume={scene.original_volume:.6f}"
                f"[a{index}]"
            )
        elif scene.audio_mode == AudioMode.narration:
            filters.append(
                f"[{narration_input_index}:a]"
                "asetpts=PTS-STARTPTS,"
                "aresample=48000,"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"volume={scene.narration_volume:.6f},"
                f"apad=pad_dur={duration:.6f},"
                f"atrim=duration={duration:.6f}"
                f"[a{index}]"
            )
        elif scene.audio_mode == AudioMode.mixed:
            filters.append(
                f"[{source_input_index}:a]"
                f"atrim=start={start:.6f}:end={end:.6f},"
                "asetpts=PTS-STARTPTS,"
                "aresample=48000,"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"volume={scene.original_volume:.6f}"
                f"[ao{index}]"
            )
            filters.append(
                f"[{narration_input_index}:a]"
                "asetpts=PTS-STARTPTS,"
                "aresample=48000,"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"volume={scene.narration_volume:.6f},"
                f"apad=pad_dur={duration:.6f},"
                f"atrim=duration={duration:.6f}"
                f"[an{index}]"
            )
            filters.append(
                f"[ao{index}][an{index}]"
                "amix=inputs=2:duration=longest:dropout_transition=0:normalize=1,"
                f"atrim=duration={duration:.6f}"
                f"[a{index}]"
            )
        else:
            filters.append(
                "anullsrc=r=48000:cl=stereo,"
                f"atrim=duration={duration:.6f},"
                "asetpts=PTS-STARTPTS"
                f"[a{index}]"
            )

        concat_inputs.append(f"[v{index}][a{index}]")

    if len(resolved_scenes) == 1:
        video_label = "[v0]"
        audio_label = "[a0]"
    else:
        filters.append(
            "".join(concat_inputs)
            + f"concat=n={len(resolved_scenes)}:v=1:a=1[vout][aout]"
        )
        video_label = "[vout]"
        audio_label = "[aout]"

    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            video_label,
            "-map",
            audio_label,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    expected_duration = sum(duration for _, _, duration, _ in resolved_scenes)
    return command, expected_duration, len(resolved_scenes), project.revision


def build_documentary_render_command(
    project_id: str,
    *,
    root: str | os.PathLike | None = None,
    output_path: str | os.PathLike | None = None,
    width: int = DEFAULT_RENDER_WIDTH,
    height: int = DEFAULT_RENDER_HEIGHT,
    fps: int = DEFAULT_RENDER_FPS,
) -> list[str]:
    """Build one shell-free FFmpeg command for the current documentary timeline."""
    command, _, _, _ = _build_documentary_render_command_and_expectations(
        project_id,
        root=root,
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
    )
    return command


def _validate_rendered_output(
    metadata: VideoMetadata,
    *,
    width: int,
    height: int,
    fps: int,
    expected_duration: float,
    scene_count: int,
) -> None:
    if (metadata.width, metadata.height) != (width, height):
        raise DocumentaryRenderError(
            "documentary render output resolution mismatch: "
            f"expected {width}x{height}, got {metadata.width}x{metadata.height}"
        )

    fps_tolerance = max(0.05, fps * 0.005)
    if abs(metadata.fps - fps) > fps_tolerance:
        raise DocumentaryRenderError(
            "documentary render output fps mismatch: "
            f"expected {fps}, got {metadata.fps:g}"
        )

    # Each scene is quantized to output frames independently before concat. Allow
    # one frame per scene plus a small muxing margin, but reject meaningful
    # truncation or accidental extra content.
    duration_tolerance = max(0.25, scene_count / fps + 0.1)
    if abs(metadata.duration_seconds - expected_duration) > duration_tolerance:
        raise DocumentaryRenderError(
            "documentary render output duration mismatch: "
            f"expected about {expected_duration:.3f}s, "
            f"got {metadata.duration_seconds:.3f}s"
        )

    # The renderer always maps an audio stream. Muted/no-audio scenes receive
    # generated silence so later scenes with original sound remain concat-safe.
    if not metadata.has_audio:
        raise DocumentaryRenderError(
            "documentary render output is missing its audio stream"
        )


def render_documentary(
    project_id: str,
    *,
    root: str | os.PathLike | None = None,
    width: int = DEFAULT_RENDER_WIDTH,
    height: int = DEFAULT_RENDER_HEIGHT,
    fps: int = DEFAULT_RENDER_FPS,
    timeout_seconds: int = DEFAULT_RENDER_TIMEOUT_SECONDS,
) -> Path:
    """Render the current source-backed documentary timeline to renders/master.mp4."""
    if timeout_seconds <= 0:
        raise ValueError("documentary render timeout must be positive")

    output_path = documentary_render_path(project_id, root).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, staged_name = tempfile.mkstemp(
        prefix=".master-",
        suffix=".mp4",
        dir=output_path.parent,
    )
    os.close(descriptor)
    staged_path = Path(staged_name)

    try:
        command, expected_duration, scene_count, project_revision = (
            _build_documentary_render_command_and_expectations(
                project_id,
                root=root,
                output_path=staged_path,
                width=width,
                height=height,
                fps=fps,
            )
        )
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise DocumentaryRenderError(
                f"documentary render timed out after {timeout_seconds} seconds"
            ) from exc
        except OSError as exc:
            raise DocumentaryRenderError(
                f"failed to start documentary renderer: {exc}"
            ) from exc

        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            if len(message) > 4000:
                message = message[-4000:]
            raise DocumentaryRenderError(
                f"ffmpeg documentary render failed: {message or 'unknown error'}"
            )

        if not staged_path.is_file() or staged_path.stat().st_size <= 0:
            raise DocumentaryRenderError(
                "ffmpeg documentary render produced no output"
            )

        try:
            rendered_metadata = probe_video_metadata(staged_path)
        except (MediaProbeError, OSError, ValueError) as exc:
            raise DocumentaryRenderError(
                "ffmpeg documentary render produced an unreadable output"
            ) from exc
        _validate_rendered_output(
            rendered_metadata,
            width=width,
            height=height,
            fps=fps,
            expected_duration=expected_duration,
            scene_count=scene_count,
        )

        current_project = load_project(project_id, root)
        if current_project.revision != project_revision:
            raise DocumentaryRenderError(
                "documentary project changed while rendering; "
                "discarding stale render"
            )

        os.replace(staged_path, output_path)
        return output_path
    finally:
        staged_path.unlink(missing_ok=True)
