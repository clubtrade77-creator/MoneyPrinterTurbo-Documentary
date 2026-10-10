from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.models.documentary import AudioMode
from app.services.documentary.localization import load_localization_plan
from app.services.documentary.project import load_project, project_dir
from app.services.documentary.transcription import load_source_transcript
from app.utils import utils


class LocalizedSubtitleError(RuntimeError):
    """Raised when localized documentary subtitles cannot be built safely."""


@dataclass(frozen=True)
class LocalizedSubtitleCue:
    start_seconds: float
    end_seconds: float
    text: str
    scene_id: str
    source_id: str
    segment_id: int


def localized_subtitle_path(
    project_id: str,
    target_language: str,
    root: str | os.PathLike | None = None,
) -> Path:
    language = str(target_language or "").strip().lower()
    if not language:
        raise ValueError("localized subtitle language is required")
    safe_language = language.replace("-", "_")
    return project_dir(project_id, root) / "subtitles" / f"localized-{safe_language}.srt"


def build_localized_subtitle_cues(
    project_id: str,
    target_language: str,
    *,
    root: str | os.PathLike | None = None,
) -> list[LocalizedSubtitleCue]:
    project = load_project(project_id, root)
    plan = load_localization_plan(
        project_id,
        target_language,
        root=root,
    )
    localized_by_scene = {scene.scene_id: scene for scene in plan.scenes}
    sources_by_id = {source.id: source for source in project.sources}
    transcript_cache = {}

    cues: list[LocalizedSubtitleCue] = []
    timeline_offset = 0.0

    for scene in project.plan.scenes:
        source = sources_by_id.get(scene.source_id)
        if source is None or source.video_metadata is None:
            raise LocalizedSubtitleError(
                f"localized subtitle scene has unavailable source metadata: {scene.id}"
            )
        if scene.source_start is None or scene.source_end is None:
            raise LocalizedSubtitleError(
                f"localized subtitle scene has no source range: {scene.id}"
            )

        source_end = min(
            float(scene.source_end),
            float(source.video_metadata.duration_seconds),
        )
        scene_duration = source_end - float(scene.source_start)
        if scene_duration <= 0:
            raise LocalizedSubtitleError(
                f"localized subtitle scene has an empty source range: {scene.id}"
            )

        localized = localized_by_scene.get(scene.id)
        if localized is None:
            raise LocalizedSubtitleError(
                f"localized subtitle scene is missing from plan: {scene.id}"
            )

        if scene.audio_mode in {AudioMode.original, AudioMode.mixed}:
            if localized.subtitle_segments:
                transcript = transcript_cache.get(scene.source_id)
                if transcript is None:
                    transcript = load_source_transcript(
                        project_id,
                        scene.source_id,
                        root=root,
                    )
                    transcript_cache[scene.source_id] = transcript
                segments_by_id = {
                    segment.id: segment
                    for segment in transcript.segments
                }

                for item in localized.subtitle_segments:
                    if item.source_id != scene.source_id:
                        raise LocalizedSubtitleError(
                            f"localized subtitle source mismatch: {scene.id}"
                        )
                    segment = segments_by_id.get(item.segment_id)
                    if segment is None:
                        raise LocalizedSubtitleError(
                            f"localized subtitle segment is missing: "
                            f"{scene.id}/{item.segment_id}"
                        )

                    clipped_start = max(
                        float(segment.start_seconds),
                        float(scene.source_start),
                    )
                    clipped_end = min(
                        float(segment.end_seconds),
                        source_end,
                    )
                    text = item.text.strip()
                    if not text or clipped_end <= clipped_start:
                        continue

                    cues.append(
                        LocalizedSubtitleCue(
                            start_seconds=(
                                timeline_offset
                                + clipped_start
                                - float(scene.source_start)
                            ),
                            end_seconds=(
                                timeline_offset
                                + clipped_end
                                - float(scene.source_start)
                            ),
                            text=text,
                            scene_id=scene.id,
                            source_id=scene.source_id,
                            segment_id=item.segment_id,
                        )
                    )

        timeline_offset += scene_duration

    return cues


def build_localized_srt_text(cues: list[LocalizedSubtitleCue]) -> str:
    lines = [
        utils.text_to_srt(
            index,
            cue.text,
            cue.start_seconds,
            cue.end_seconds,
        )
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def write_localized_subtitles(
    project_id: str,
    target_language: str,
    *,
    root: str | os.PathLike | None = None,
) -> Path | None:
    cues = build_localized_subtitle_cues(
        project_id,
        target_language,
        root=root,
    )
    if not cues:
        return None

    target = localized_subtitle_path(
        project_id,
        target_language,
        root,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    content = build_localized_srt_text(cues)
    if target.is_file():
        try:
            if target.read_text(encoding="utf-8") == content:
                return target
        except OSError:
            pass

    temp = target.with_suffix(target.suffix + f".{uuid4().hex}.tmp")
    try:
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target
