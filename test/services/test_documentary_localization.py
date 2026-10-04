import json
from pathlib import Path

import pytest

from app.models.documentary import (
    AudioMode,
    DocumentaryScene,
    DocumentaryTranscript,
    RightsStatus,
    SceneType,
    SourceAsset,
    SourceType,
    TranscriptSegment,
    VideoMetadata,
)
from app.services.documentary.localization import (
    CURRENT_LOCALIZATION_REVIEW_VERSION,
    LocalizationError,
    load_localization_plan,
    localization_plan_path,
    localize_project,
)
from app.services.documentary.project import (
    add_source,
    create_project,
    load_project,
    project_dir,
    save_project,
    sha256_file,
)
from app.services.documentary.transcription import transcript_path


def _project_with_localizable_scene(tmp_path: Path):
    project = create_project(
        "Localization case",
        project_id="doc_localization_case",
        master_language="en",
        root=tmp_path,
    )
    local_path = (
        project_dir(project.id, tmp_path)
        / "sources"
        / "source_localization.mp4"
    )
    local_path.write_bytes(b"fake-localization-video")

    source = SourceAsset(
        id="source_localization",
        source_type=SourceType.local_video,
        title="Localization source",
        local_path=str(local_path.resolve()),
        checksum_sha256=sha256_file(local_path),
        video_metadata=VideoMetadata(
            duration_seconds=12.0,
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
    project = add_source(project.id, source, root=tmp_path)

    transcript = DocumentaryTranscript(
        source_id=source.id,
        source_checksum_sha256=source.checksum_sha256,
        language="en",
        language_mode="forced",
        requested_language="en",
        media_duration_seconds=12.0,
        model_size="small",
        full_text="The officer approached. Everything changed.",
        segments=[
            TranscriptSegment(
                id=0,
                start_seconds=0.5,
                end_seconds=1.8,
                text="The officer approached.",
            ),
            TranscriptSegment(
                id=1,
                start_seconds=2.0,
                end_seconds=3.4,
                text="Everything changed.",
            ),
        ],
    )
    transcript_path(project.id, source.id, tmp_path).write_text(
        json.dumps(
            transcript.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    project = load_project(project.id, tmp_path)
    project.plan.scenes = [
        DocumentaryScene(
            id="scene_localize",
            scene_type=SceneType.original_clip,
            source_id=source.id,
            source_start=0.0,
            source_end=4.0,
            audio_mode=AudioMode.original,
            narration_text="The officer approaches the vehicle.",
            on_screen_text="Recorded encounter",
            transcript_segment_ids=[0, 1],
        )
    ]
    save_project(project, tmp_path)
    return load_project(project.id, tmp_path), source


def _approved_review(prompt: str) -> str:
    return json.dumps({"supported": True, "issues": []})


def _russian_response() -> str:
    return json.dumps(
        {
            "scenes": [
                {
                    "scene_id": "scene_localize",
                    "narration_text": "Офицер подходит к автомобилю.",
                    "on_screen_text": "Записанная встреча",
                    "subtitle_segments": [
                        {
                            "source_id": "source_localization",
                            "segment_id": 0,
                            "text": "Офицер подошёл.",
                        },
                        {
                            "source_id": "source_localization",
                            "segment_id": 1,
                            "text": "Всё изменилось.",
                        },
                    ],
                }
            ]
        },
        ensure_ascii=False,
    )


def test_localize_project_persists_grounded_scene_text_and_subtitle_refs(
    tmp_path: Path,
):
    project, source = _project_with_localizable_scene(tmp_path)

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=_approved_review,
    )

    assert plan.source_language == "en"
    assert plan.target_language == "ru"
    assert plan.semantic_reviewed is True
    assert plan.semantic_review_version == CURRENT_LOCALIZATION_REVIEW_VERSION
    assert len(plan.reviewed_content_fingerprint) == 64
    assert len(plan.master_plan_fingerprint) == 64
    assert list(plan.transcript_fingerprints) == [source.id]
    assert plan.scenes[0].scene_id == "scene_localize"
    assert plan.scenes[0].narration_text == "Офицер подходит к автомобилю."
    assert [
        (item.source_id, item.segment_id)
        for item in plan.scenes[0].subtitle_segments
    ] == [
        (source.id, 0),
        (source.id, 1),
    ]
    assert localization_plan_path(project.id, "ru", tmp_path).is_file()

    loaded = load_localization_plan(project.id, "ru", root=tmp_path)
    assert loaded == plan


def test_localize_project_requires_reviewer_for_custom_generator(tmp_path: Path):
    project, _ = _project_with_localizable_scene(tmp_path)

    with pytest.raises(ValueError, match="review_fn is required"):
        localize_project(
            project.id,
            "ru",
            root=tmp_path,
            generate_fn=lambda prompt: _russian_response(),
        )


def test_localize_project_rejects_changed_scene_ids_after_retries(tmp_path: Path):
    project, _ = _project_with_localizable_scene(tmp_path)
    bad_payload = json.loads(_russian_response())
    bad_payload["scenes"][0]["scene_id"] = "scene_changed"
    attempts = []

    def generate(prompt: str) -> str:
        attempts.append(prompt)
        return json.dumps(bad_payload, ensure_ascii=False)

    with pytest.raises(LocalizationError, match="scene order or scene ids"):
        localize_project(
            project.id,
            "ru",
            root=tmp_path,
            generate_fn=generate,
            review_fn=_approved_review,
        )

    assert len(attempts) == 3
    assert "previous response was rejected" in attempts[-1]


def test_localize_project_retries_semantically_rejected_translation(
    tmp_path: Path,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    review_calls = []

    def review(prompt: str) -> str:
        review_calls.append(prompt)
        if len(review_calls) == 1:
            return json.dumps(
                {
                    "supported": False,
                    "issues": ["The translation changed a factual claim."],
                }
            )
        return json.dumps({"supported": True, "issues": []})

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=review,
    )

    assert plan.target_language == "ru"
    assert len(review_calls) == 2


def test_localize_project_preserves_empty_master_text_fields(tmp_path: Path):
    project, _ = _project_with_localizable_scene(tmp_path)
    project = load_project(project.id, tmp_path)
    project.plan.scenes[0].narration_text = ""
    project.plan.scenes[0].on_screen_text = ""
    save_project(project, tmp_path)

    payload = json.loads(_russian_response())
    payload["scenes"][0]["narration_text"] = ""
    payload["scenes"][0]["on_screen_text"] = ""

    plan = localize_project(
        project.id,
        "es",
        root=tmp_path,
        generate_fn=lambda prompt: json.dumps(payload, ensure_ascii=False),
        review_fn=_approved_review,
    )

    assert plan.scenes[0].narration_text == ""
    assert plan.scenes[0].on_screen_text == ""


def test_load_localization_plan_rejects_changed_master_timeline(tmp_path: Path):
    project, _ = _project_with_localizable_scene(tmp_path)
    localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=_approved_review,
    )

    changed = load_project(project.id, tmp_path)
    changed.plan.scenes[0].narration_text = "The narration changed."
    save_project(changed, tmp_path)

    with pytest.raises(LocalizationError, match="master timeline changed"):
        load_localization_plan(project.id, "ru", root=tmp_path)


def test_load_localization_plan_rejects_changed_transcript_evidence(tmp_path: Path):
    project, source = _project_with_localizable_scene(tmp_path)
    localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=_approved_review,
    )

    path = transcript_path(project.id, source.id, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["segments"][0]["text"] = "The source transcript changed."
    payload["full_text"] = "The source transcript changed. Everything changed."
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(LocalizationError, match="transcript evidence changed"):
        load_localization_plan(project.id, "ru", root=tmp_path)


def test_load_localization_plan_rejects_text_changed_after_review(
    tmp_path: Path,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=_approved_review,
    )

    path = localization_plan_path(project.id, "ru", tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scenes"][0]["narration_text"] = "Подменённый перевод."
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(LocalizationError, match="changed after semantic review"):
        load_localization_plan(project.id, "ru", root=tmp_path)


def test_load_localization_plan_rejects_old_review_policy(tmp_path: Path):
    project, _ = _project_with_localizable_scene(tmp_path)
    localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=_approved_review,
    )

    path = localization_plan_path(project.id, "ru", tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["semantic_review_version"] = 0
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(LocalizationError, match="review policy is stale"):
        load_localization_plan(project.id, "ru", root=tmp_path)


def test_localize_project_rejects_master_language_as_target(tmp_path: Path):
    project, _ = _project_with_localizable_scene(tmp_path)

    with pytest.raises(ValueError, match="must differ"):
        localize_project(
            project.id,
            "en",
            root=tmp_path,
            generate_fn=lambda prompt: _russian_response(),
            review_fn=_approved_review,
        )
