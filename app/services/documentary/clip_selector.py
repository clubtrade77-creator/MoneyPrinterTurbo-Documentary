from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from app.models.documentary import (
    AudioMode,
    ClipPlan,
    ClipSelection,
    DocumentaryScene,
    SceneType,
    StoryPlan,
)
from app.services.documentary.project import load_project, project_dir, save_project
from app.services.documentary.story_planner import load_story_plan
from app.services.documentary.transcription import load_source_transcript

DEFAULT_CLIP_PADDING_SECONDS = 0.35
DEFAULT_MAX_MERGE_GAP_SECONDS = 0.75


class ClipSelectorError(RuntimeError):
    """Raised when transcript evidence cannot be converted into safe video clips."""


def clip_plan_path(
    project_id: str,
    root: str | os.PathLike | None = None,
) -> Path:
    return project_dir(project_id, root) / "plans" / "clip-plan.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _model_fingerprint(model) -> str:
    payload = model.model_dump(mode="json")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def story_plan_fingerprint(plan: StoryPlan) -> str:
    return _model_fingerprint(plan)


def _validate_selector_settings(
    padding_seconds: float,
    max_merge_gap_seconds: float,
) -> None:
    if padding_seconds < 0 or padding_seconds > 5:
        raise ValueError("clip padding must be between 0 and 5 seconds")
    if max_merge_gap_seconds < 0 or max_merge_gap_seconds > 10:
        raise ValueError("clip merge gap must be between 0 and 10 seconds")


def _group_evidence_segments(segments, max_merge_gap_seconds: float):
    ordered = sorted(segments, key=lambda segment: (segment.start_seconds, segment.id))
    if not ordered:
        return []

    groups = [[ordered[0]]]
    for segment in ordered[1:]:
        previous = groups[-1][-1]
        ids_are_consecutive = segment.id == previous.id + 1
        gap = segment.start_seconds - previous.end_seconds
        if ids_are_consecutive and gap <= max_merge_gap_seconds:
            groups[-1].append(segment)
        else:
            groups.append([segment])
    return groups


def _referenced_source_ids(story_plan: StoryPlan) -> list[str]:
    source_ids = []
    seen_source_ids = set()
    for beat in story_plan.beats:
        for evidence in beat.evidence:
            if evidence.source_id not in seen_source_ids:
                seen_source_ids.add(evidence.source_id)
                source_ids.append(evidence.source_id)
    return source_ids


def _load_current_selection_context(
    project_id: str,
    story_plan: StoryPlan,
    *,
    root: str | os.PathLike | None,
):
    project = load_project(project_id, root)
    transcripts = {
        source_id: load_source_transcript(
            project_id,
            source_id,
            root=root,
        )
        for source_id in _referenced_source_ids(story_plan)
    }
    return project, transcripts


def _build_grounded_clips(
    *,
    story_plan: StoryPlan,
    project,
    transcripts,
    padding_seconds: float,
    max_merge_gap_seconds: float,
) -> list[ClipSelection]:
    sources = {source.id: source for source in project.sources}
    clips: list[ClipSelection] = []
    clip_number = 0

    for beat in story_plan.beats:
        seen_segments_in_beat: set[tuple[str, int]] = set()

        for evidence in beat.evidence:
            source = sources.get(evidence.source_id)
            if source is None:
                raise ClipSelectorError(
                    f"story evidence references missing project source: "
                    f"{evidence.source_id}"
                )
            if not source.has_local_copy:
                raise ClipSelectorError(
                    f"story evidence source has no local copy: {source.id}"
                )
            if source.video_metadata is None:
                raise ClipSelectorError(
                    f"story evidence source is not renderable video: {source.id}"
                )

            transcript = transcripts[evidence.source_id]
            segments_by_id = {segment.id: segment for segment in transcript.segments}

            selected_segments = []
            for segment_id in evidence.segment_ids:
                key = (evidence.source_id, segment_id)
                if key in seen_segments_in_beat:
                    continue
                segment = segments_by_id.get(segment_id)
                if segment is None:
                    raise ClipSelectorError(
                        f"story evidence references missing transcript segment "
                        f"{segment_id} for {evidence.source_id}"
                    )
                seen_segments_in_beat.add(key)
                selected_segments.append(segment)

            for group in _group_evidence_segments(
                selected_segments,
                max_merge_gap_seconds,
            ):
                if not group:
                    continue

                source_duration = source.video_metadata.duration_seconds
                start = max(0.0, group[0].start_seconds - padding_seconds)
                end = min(
                    source_duration,
                    group[-1].end_seconds + padding_seconds,
                )
                if end <= start:
                    raise ClipSelectorError(
                        f"selected clip has empty source range for beat {beat.id}"
                    )

                audio_mode = (
                    AudioMode.original
                    if beat.original_audio_priority
                    and source.video_metadata.has_audio
                    else AudioMode.muted
                )

                clip_number += 1
                clips.append(
                    ClipSelection(
                        id=f"clip_{clip_number:03d}_{beat.id[:80]}",
                        story_beat_id=beat.id,
                        purpose=beat.purpose,
                        source_id=source.id,
                        segment_ids=[segment.id for segment in group],
                        source_start_seconds=start,
                        source_end_seconds=end,
                        audio_mode=audio_mode,
                    )
                )

    if not clips:
        raise ClipSelectorError("story plan produced no renderable clip selections")
    return clips


def _validate_clip_plan_grounding(
    project_id: str,
    clip_plan: ClipPlan,
    story_plan: StoryPlan,
    *,
    root: str | os.PathLike | None,
) -> None:
    _validate_selector_settings(
        clip_plan.padding_seconds,
        clip_plan.max_merge_gap_seconds,
    )
    project, transcripts = _load_current_selection_context(
        project_id,
        story_plan,
        root=root,
    )
    expected_clips = _build_grounded_clips(
        story_plan=story_plan,
        project=project,
        transcripts=transcripts,
        padding_seconds=clip_plan.padding_seconds,
        max_merge_gap_seconds=clip_plan.max_merge_gap_seconds,
    )
    actual_payload = [
        clip.model_dump(mode="json")
        for clip in clip_plan.clips
    ]
    expected_payload = [
        clip.model_dump(mode="json")
        for clip in expected_clips
    ]
    if actual_payload != expected_payload:
        raise ClipSelectorError(
            "documentary clip plan does not match grounded transcript evidence"
        )


def build_clip_plan(
    project_id: str,
    *,
    root: str | os.PathLike | None = None,
    padding_seconds: float = DEFAULT_CLIP_PADDING_SECONDS,
    max_merge_gap_seconds: float = DEFAULT_MAX_MERGE_GAP_SECONDS,
) -> ClipPlan:
    """Convert grounded Story Plan evidence into exact renderable source ranges."""
    _validate_selector_settings(padding_seconds, max_merge_gap_seconds)

    story_plan = load_story_plan(project_id, root=root)
    project, transcripts = _load_current_selection_context(
        project_id,
        story_plan,
        root=root,
    )
    clips = _build_grounded_clips(
        story_plan=story_plan,
        project=project,
        transcripts=transcripts,
        padding_seconds=padding_seconds,
        max_merge_gap_seconds=max_merge_gap_seconds,
    )

    return ClipPlan(
        padding_seconds=padding_seconds,
        max_merge_gap_seconds=max_merge_gap_seconds,
        story_plan_fingerprint=story_plan_fingerprint(story_plan),
        transcript_fingerprints=dict(story_plan.transcript_fingerprints),
        clips=clips,
    )


def load_clip_plan(
    project_id: str,
    *,
    root: str | os.PathLike | None = None,
) -> ClipPlan:
    path = clip_plan_path(project_id, root)
    if not path.is_file():
        raise FileNotFoundError(f"documentary clip plan not found: {project_id}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        clip_plan = ClipPlan.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ClipSelectorError(f"invalid documentary clip plan: {project_id}") from exc

    story_plan = load_story_plan(project_id, root=root)
    if clip_plan.story_plan_fingerprint != story_plan_fingerprint(story_plan):
        raise ClipSelectorError(
            "documentary clip plan is stale; story plan changed"
        )
    if clip_plan.transcript_fingerprints != story_plan.transcript_fingerprints:
        raise ClipSelectorError(
            "documentary clip plan is stale; transcript evidence changed"
        )
    _validate_clip_plan_grounding(
        project_id,
        clip_plan,
        story_plan,
        root=root,
    )
    return clip_plan


def _scene_from_clip(clip: ClipSelection) -> DocumentaryScene:
    scene_type = (
        SceneType.original_clip
        if clip.audio_mode == AudioMode.original
        else SceneType.narration_over_source
    )
    return DocumentaryScene(
        id=f"scene_{clip.id[:114]}",
        scene_type=scene_type,
        source_id=clip.source_id,
        source_start=clip.source_start_seconds,
        source_end=clip.source_end_seconds,
        audio_mode=clip.audio_mode,
        purpose=clip.purpose,
        story_beat_id=clip.story_beat_id,
        transcript_segment_ids=list(clip.segment_ids),
    )


def apply_clip_plan(
    project_id: str,
    clip_plan: ClipPlan | None = None,
    *,
    root: str | os.PathLike | None = None,
):
    """Replace the documentary scene timeline with one validated clip plan."""
    plan = clip_plan or load_clip_plan(project_id, root=root)
    current_story_plan = load_story_plan(project_id, root=root)

    if plan.story_plan_fingerprint != story_plan_fingerprint(current_story_plan):
        raise ClipSelectorError(
            "documentary clip plan is stale; story plan changed"
        )
    if plan.transcript_fingerprints != current_story_plan.transcript_fingerprints:
        raise ClipSelectorError(
            "documentary clip plan is stale; transcript evidence changed"
        )
    _validate_clip_plan_grounding(
        project_id,
        plan,
        current_story_plan,
        root=root,
    )

    project = load_project(project_id, root)
    sources = {source.id: source for source in project.sources}
    scenes = []

    for clip in plan.clips:
        source = sources.get(clip.source_id)
        if source is None:
            raise ClipSelectorError(
                f"clip plan references missing project source: {clip.source_id}"
            )
        if not source.has_local_copy or source.video_metadata is None:
            raise ClipSelectorError(
                f"clip plan source is no longer renderable: {clip.source_id}"
            )
        if clip.source_end_seconds > source.video_metadata.duration_seconds + 0.05:
            raise ClipSelectorError(
                f"clip plan range exceeds source duration: {clip.id}"
            )
        scenes.append(_scene_from_clip(clip))

    # Re-read immediately before the project mutation so a regenerated Story Plan
    # is very unlikely to be silently paired with clips selected from the old one.
    latest_story_plan = load_story_plan(project_id, root=root)
    if plan.story_plan_fingerprint != story_plan_fingerprint(latest_story_plan):
        raise ClipSelectorError(
            "documentary clip plan became stale before it could be applied"
        )

    project.plan.story_plan_fingerprint = plan.story_plan_fingerprint
    project.plan.scenes = scenes
    save_project(project, root)
    return project


def select_clips(
    project_id: str,
    *,
    root: str | os.PathLike | None = None,
    padding_seconds: float = DEFAULT_CLIP_PADDING_SECONDS,
    max_merge_gap_seconds: float = DEFAULT_MAX_MERGE_GAP_SECONDS,
) -> ClipPlan:
    """Build, persist, validate, and apply the deterministic transcript clip plan."""
    clip_plan = build_clip_plan(
        project_id,
        root=root,
        padding_seconds=padding_seconds,
        max_merge_gap_seconds=max_merge_gap_seconds,
    )
    apply_clip_plan(project_id, clip_plan, root=root)
    _atomic_write_json(
        clip_plan_path(project_id, root),
        clip_plan.model_dump(mode="json"),
    )
    return clip_plan
