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
from app.services.documentary.clip_selector import select_clips
from app.services.documentary.narration import (
    NarrationWriterError,
    build_narration_prompt,
    write_narration,
)
from app.services.documentary.project import (
    add_source,
    create_project,
    load_project,
    project_dir,
    sha256_file,
)
from app.services.documentary.story_planner import plan_story
from app.services.documentary.transcription import transcript_path


def _approve_review(prompt: str) -> str:
    return json.dumps({"supported": True, "issues": []})


def _setup_case(tmp_path: Path, *, original_audio_priority: bool):
    project = create_project(
        "Narration case",
        project_id=(
            "doc_narration_original"
            if original_audio_priority
            else "doc_narration_needed"
        ),
        root=tmp_path,
    )
    local_path = project_dir(project.id, tmp_path) / "sources" / "source.mp4"
    local_path.write_bytes(b"fake-video")
    source = SourceAsset(
        id="source_narration",
        source_type=SourceType.local_video,
        title="Narration source",
        local_path=str(local_path.resolve()),
        checksum_sha256=sha256_file(local_path),
        video_metadata=VideoMetadata(
            duration_seconds=8.0,
            width=1280,
            height=720,
            fps=30,
            has_audio=True,
            video_codec="h264",
            audio_codec="aac",
            container="mov,mp4",
            file_size_bytes=local_path.stat().st_size,
        ),
        rights_status=RightsStatus.user_owned,
    )
    add_source(project.id, source, root=tmp_path)

    transcript = DocumentaryTranscript(
        source_id=source.id,
        source_checksum_sha256=source.checksum_sha256,
        language="en",
        language_mode="forced",
        requested_language="en",
        media_duration_seconds=8.0,
        model_size="small",
        full_text=(
            "The officer approaches the vehicle. "
            "The driver looks toward the officer. "
            "The officer remains beside the vehicle."
        ),
        segments=[
            TranscriptSegment(
                id=0,
                start_seconds=0.5,
                end_seconds=1.5,
                text="The officer approaches the vehicle.",
            ),
            TranscriptSegment(
                id=1,
                start_seconds=1.8,
                end_seconds=2.8,
                text="The driver looks toward the officer.",
            ),
            TranscriptSegment(
                id=2,
                start_seconds=3.1,
                end_seconds=4.0,
                text="The officer remains beside the vehicle.",
            ),
        ],
    )
    transcript_path(project.id, source.id, tmp_path).write_text(
        json.dumps(transcript.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    payload = {
        "version": 1,
        "title": "Recorded moment",
        "angle": "The officer approaches the vehicle.",
        "hook": "The officer approaches the vehicle.",
        "target_duration_seconds": 60,
        "beats": [
            {
                "id": "beat_narration_01",
                "purpose": "hook",
                "title": "The approach",
                "summary": "The officer approaches the vehicle.",
                "target_duration_seconds": 60,
                "narration_goal": "State only that the officer approaches the vehicle.",
                "original_audio_priority": original_audio_priority,
                "evidence": [
                    {
                        "source_id": source.id,
                        "segment_ids": [0, 1, 2],
                        "note": "Direct transcript evidence.",
                    }
                ],
            }
        ],
    }
    plan_story(
        project.id,
        source_ids=[source.id],
        target_duration_seconds=60,
        root=tmp_path,
        generate_fn=lambda prompt: json.dumps(payload),
        review_fn=_approve_review,
    )
    select_clips(project.id, root=tmp_path)
    return project, source


def test_narration_prompt_requires_grounded_concise_text():
    prompt = build_narration_prompt(
        project_title="Case",
        scenes=[
            {
                "scene_id": "scene_1",
                "purpose": "context",
                "beat_title": "Context",
                "beat_summary": "The officer approaches the vehicle.",
                "narration_goal": "Explain only the recorded action.",
                "evidence": [
                    {
                        "segment_id": 0,
                        "start_seconds": 0.0,
                        "end_seconds": 3.0,
                        "text": "The officer approaches the vehicle.",
                    }
                ],
            }
        ],
    )

    assert "Use ONLY facts stated in the supplied transcript evidence" in prompt
    assert "Do not claim that one event caused another" in prompt
    assert "scene_1" in prompt


def test_write_narration_skips_original_audio_priority_scene(tmp_path: Path):
    project, _ = _setup_case(tmp_path, original_audio_priority=True)
    before = load_project(project.id, tmp_path)

    generation_calls = []
    result = write_narration(
        project.id,
        root=tmp_path,
        generate_fn=lambda prompt: generation_calls.append(prompt) or "{}",
        review_fn=_approve_review,
    )

    after = load_project(project.id, tmp_path)
    assert generation_calls == []
    assert result.revision == before.revision
    assert after.plan.scenes[0].audio_mode == AudioMode.original
    assert after.plan.scenes[0].narration_text == ""


def test_write_narration_generates_reviewed_text_for_muted_scene(tmp_path: Path):
    project, _ = _setup_case(tmp_path, original_audio_priority=False)
    before = load_project(project.id, tmp_path)
    scene_id = before.plan.scenes[0].id

    result = write_narration(
        project.id,
        root=tmp_path,
        generate_fn=lambda prompt: json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": scene_id,
                        "narration_text": "The officer approaches the vehicle.",
                    }
                ]
            }
        ),
        review_fn=_approve_review,
    )

    scene = result.plan.scenes[0]
    assert result.revision == before.revision + 1
    assert scene.narration_text == "The officer approaches the vehicle."
    assert scene.audio_mode == AudioMode.narration
    assert scene.scene_type == SceneType.narration_over_source


def test_write_narration_requires_reviewer_for_custom_generator(tmp_path: Path):
    project, _ = _setup_case(tmp_path, original_audio_priority=False)

    with pytest.raises(ValueError, match="review_fn is required"):
        write_narration(
            project.id,
            root=tmp_path,
            generate_fn=lambda prompt: "{}",
        )


def test_write_narration_rejects_semantic_additions(tmp_path: Path):
    project, _ = _setup_case(tmp_path, original_audio_priority=False)
    scene_id = load_project(project.id, tmp_path).plan.scenes[0].id

    responses = [
        json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": scene_id,
                        "narration_text": "A tense traffic stop begins.",
                    }
                ]
            }
        )
    ] * 3

    def reject_review(prompt: str) -> str:
        return json.dumps(
            {
                "supported": False,
                "issues": ["The transcript does not establish a tense traffic stop."],
            }
        )

    with pytest.raises(NarrationWriterError, match="semantic narration review failed"):
        write_narration(
            project.id,
            root=tmp_path,
            generate_fn=lambda prompt: responses.pop(0),
            review_fn=reject_review,
        )
