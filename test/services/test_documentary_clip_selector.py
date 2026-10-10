import json
from pathlib import Path

import pytest

from app.models.documentary import (
    AudioMode,
    DocumentaryTranscript,
    RightsStatus,
    SceneType,
    SourceAsset,
    SourceType,
    TranscriptSegment,
    VideoMetadata,
)
from app.services.documentary.clip_selector import (
    ClipSelectorError,
    apply_clip_plan,
    build_clip_plan,
    clip_plan_path,
    load_clip_plan,
    select_clips,
)
from app.services.documentary.project import (
    add_source,
    create_project,
    load_project,
    project_dir,
    sha256_file,
)
from app.services.documentary.story_planner import plan_story, story_plan_path
from app.services.documentary.transcription import transcript_path


def _approve_review(prompt: str) -> str:
    return json.dumps({"supported": True, "issues": []})


def _register_video_transcript(
    tmp_path: Path,
    *,
    has_video_metadata: bool = True,
    has_audio: bool = True,
):
    project = create_project(
        "Clip selector case",
        project_id="doc_clip_selector",
        root=tmp_path,
    )
    local_path = project_dir(project.id, tmp_path) / "sources" / "source_clip.mp4"
    local_path.write_bytes(b"fake-video-source")

    metadata = None
    if has_video_metadata:
        metadata = VideoMetadata(
            duration_seconds=12.0,
            width=1280,
            height=720,
            fps=30,
            has_audio=has_audio,
            video_codec="h264",
            audio_codec="aac" if has_audio else "",
            container="mov,mp4",
            file_size_bytes=local_path.stat().st_size,
        )

    source = SourceAsset(
        id="source_clip",
        source_type=SourceType.local_video,
        title="Clip source",
        local_path=str(local_path.resolve()),
        checksum_sha256=sha256_file(local_path),
        video_metadata=metadata,
        rights_status=RightsStatus.user_owned,
    )
    project = add_source(project.id, source, root=tmp_path)

    transcript = DocumentaryTranscript(
        source_id=source.id,
        source_checksum_sha256=source.checksum_sha256,
        language="en",
        language_mode="forced",
        requested_language="en",
        media_duration_seconds=12.0,
        model_size="small",
        full_text="First moment. Second moment. Third moment.",
        segments=[
            TranscriptSegment(
                id=0,
                start_seconds=0.5,
                end_seconds=1.5,
                text="First moment.",
            ),
            TranscriptSegment(
                id=1,
                start_seconds=1.8,
                end_seconds=2.8,
                text="Second moment.",
            ),
            TranscriptSegment(
                id=2,
                start_seconds=5.0,
                end_seconds=6.0,
                text="Third moment.",
            ),
        ],
    )
    transcript_path(project.id, source.id, tmp_path).write_text(
        json.dumps(transcript.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return project, source, transcript


def _create_story_plan(
    project_id: str,
    source_id: str,
    tmp_path: Path,
    *,
    segment_ids: list[int] | None = None,
    original_audio_priority: bool = True,
):
    payload = {
        "version": 1,
        "title": "Grounded clip story",
        "angle": "A factual sequence from the transcript.",
        "hook": "The recorded sequence begins.",
        "target_duration_seconds": 60,
        "beats": [
            {
                "id": "beat_clip_01",
                "purpose": "hook",
                "title": "Recorded sequence",
                "summary": "The cited transcript moments are shown.",
                "target_duration_seconds": 60,
                "narration_goal": "Present only the cited transcript moments.",
                "original_audio_priority": original_audio_priority,
                "evidence": [
                    {
                        "source_id": source_id,
                        "segment_ids": segment_ids or [0, 1, 2],
                        "note": "Direct transcript evidence.",
                    }
                ],
            }
        ],
    }
    return plan_story(
        project_id,
        source_ids=[source_id],
        target_duration_seconds=60,
        root=tmp_path,
        generate_fn=lambda prompt: json.dumps(payload),
        review_fn=_approve_review,
    )


def test_build_clip_plan_uses_transcript_timecodes_and_merges_adjacent_segments(
    tmp_path: Path,
):
    project, source, _ = _register_video_transcript(tmp_path)
    _create_story_plan(project.id, source.id, tmp_path)

    plan = build_clip_plan(
        project.id,
        root=tmp_path,
        padding_seconds=0.25,
        max_merge_gap_seconds=0.5,
    )

    assert plan.padding_seconds == pytest.approx(0.25)
    assert plan.max_merge_gap_seconds == pytest.approx(0.5)
    assert len(plan.clips) == 2
    first, second = plan.clips

    assert first.story_beat_id == "beat_clip_01"
    assert first.source_id == source.id
    assert first.segment_ids == [0, 1]
    assert first.source_start_seconds == pytest.approx(0.25)
    assert first.source_end_seconds == pytest.approx(3.05)
    assert first.audio_mode == AudioMode.original

    assert second.segment_ids == [2]
    assert second.source_start_seconds == pytest.approx(4.75)
    assert second.source_end_seconds == pytest.approx(6.25)


def test_clip_selector_clamps_padding_to_source_boundaries(tmp_path: Path):
    project, source, transcript = _register_video_transcript(tmp_path)
    transcript.segments[0].start_seconds = 0.1
    transcript.segments[0].end_seconds = 0.4
    transcript.segments[1].start_seconds = 11.4
    transcript.segments[1].end_seconds = 11.9
    transcript.segments = transcript.segments[:2]
    transcript.full_text = " ".join(segment.text for segment in transcript.segments)
    transcript_path(project.id, source.id, tmp_path).write_text(
        json.dumps(transcript.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _create_story_plan(
        project.id,
        source.id,
        tmp_path,
        segment_ids=[0, 1],
    )

    plan = build_clip_plan(
        project.id,
        root=tmp_path,
        padding_seconds=1.0,
        max_merge_gap_seconds=0.0,
    )

    assert plan.clips[0].source_start_seconds == 0.0
    assert plan.clips[-1].source_end_seconds == 12.0


def test_select_clips_persists_plan_and_applies_traceable_scenes(tmp_path: Path):
    project, source, _ = _register_video_transcript(tmp_path)
    story = _create_story_plan(project.id, source.id, tmp_path)

    clip_plan = select_clips(
        project.id,
        root=tmp_path,
        padding_seconds=0.25,
        max_merge_gap_seconds=0.5,
    )

    assert clip_plan_path(project.id, tmp_path).is_file()
    assert clip_plan.transcript_fingerprints == story.transcript_fingerprints

    updated = load_project(project.id, tmp_path)
    assert len(updated.plan.scenes) == 2
    assert updated.plan.story_plan_fingerprint == clip_plan.story_plan_fingerprint
    assert updated.plan.scenes[0].scene_type == SceneType.original_clip
    assert updated.plan.scenes[0].audio_mode == AudioMode.original
    assert updated.plan.scenes[0].story_beat_id == "beat_clip_01"
    assert updated.plan.scenes[0].transcript_segment_ids == [0, 1]


def test_select_clips_does_not_persist_plan_if_apply_fails(
    tmp_path: Path,
    monkeypatch,
):
    project, source, _ = _register_video_transcript(tmp_path)
    _create_story_plan(project.id, source.id, tmp_path)

    def fail_apply(*args, **kwargs):
        raise ClipSelectorError("apply failed")

    monkeypatch.setattr(
        "app.services.documentary.clip_selector.apply_clip_plan",
        fail_apply,
    )

    with pytest.raises(ClipSelectorError, match="apply failed"):
        select_clips(project.id, root=tmp_path)

    assert not clip_plan_path(project.id, tmp_path).exists()


def test_select_clips_uses_narration_audio_when_original_audio_not_priority(
    tmp_path: Path,
):
    project, source, _ = _register_video_transcript(tmp_path)
    _create_story_plan(
        project.id,
        source.id,
        tmp_path,
        original_audio_priority=False,
    )

    select_clips(project.id, root=tmp_path)
    updated = load_project(project.id, tmp_path)

    assert updated.plan.scenes
    assert all(
        scene.scene_type == SceneType.narration_over_source
        for scene in updated.plan.scenes
    )
    assert all(scene.audio_mode == AudioMode.narration for scene in updated.plan.scenes)


def test_select_clips_falls_back_to_narration_when_source_has_no_audio(tmp_path: Path):
    project, source, _ = _register_video_transcript(tmp_path, has_audio=False)
    _create_story_plan(
        project.id,
        source.id,
        tmp_path,
        original_audio_priority=True,
    )

    clip_plan = build_clip_plan(project.id, root=tmp_path)

    assert all(clip.audio_mode == AudioMode.narration for clip in clip_plan.clips)


def test_build_clip_plan_rejects_evidence_source_without_video_metadata(tmp_path: Path):
    project, source, _ = _register_video_transcript(
        tmp_path,
        has_video_metadata=False,
    )
    _create_story_plan(project.id, source.id, tmp_path)

    with pytest.raises(ClipSelectorError, match="not renderable video"):
        build_clip_plan(project.id, root=tmp_path)


def test_load_clip_plan_rejects_story_plan_changed_after_selection(tmp_path: Path):
    project, source, _ = _register_video_transcript(tmp_path)
    _create_story_plan(project.id, source.id, tmp_path)
    select_clips(project.id, root=tmp_path)

    story_path = story_plan_path(project.id, tmp_path)
    payload = json.loads(story_path.read_text(encoding="utf-8"))
    payload["title"] = "Changed grounded title"
    story_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ClipSelectorError, match="story plan changed"):
        load_clip_plan(project.id, root=tmp_path)


def test_load_clip_plan_rejects_tampered_clip_range(tmp_path: Path):
    project, source, _ = _register_video_transcript(tmp_path)
    _create_story_plan(project.id, source.id, tmp_path)
    select_clips(
        project.id,
        root=tmp_path,
        padding_seconds=0.25,
        max_merge_gap_seconds=0.5,
    )

    path = clip_plan_path(project.id, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["clips"][0]["source_start_seconds"] = 0.0
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(ClipSelectorError, match="grounded transcript evidence"):
        load_clip_plan(project.id, root=tmp_path)


def test_apply_clip_plan_rejects_tampered_in_memory_clip(tmp_path: Path):
    project, source, _ = _register_video_transcript(tmp_path)
    _create_story_plan(project.id, source.id, tmp_path)

    plan = build_clip_plan(
        project.id,
        root=tmp_path,
        padding_seconds=0.25,
        max_merge_gap_seconds=0.5,
    )
    tampered = plan.model_copy(deep=True)
    tampered.clips[0].source_end_seconds += 1.0

    with pytest.raises(ClipSelectorError, match="grounded transcript evidence"):
        apply_clip_plan(project.id, tampered, root=tmp_path)


def test_build_clip_plan_rejects_invalid_selector_settings(tmp_path: Path):
    project, source, _ = _register_video_transcript(tmp_path)
    _create_story_plan(project.id, source.id, tmp_path)

    with pytest.raises(ValueError, match="padding"):
        build_clip_plan(
            project.id,
            root=tmp_path,
            padding_seconds=-0.1,
        )

    with pytest.raises(ValueError, match="merge gap"):
        build_clip_plan(
            project.id,
            root=tmp_path,
            max_merge_gap_seconds=11.0,
        )
