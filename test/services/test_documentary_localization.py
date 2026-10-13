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
    localization_work_path,
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


def test_localize_project_repairs_wrapped_scene_response(tmp_path: Path):
    project, source = _project_with_localizable_scene(tmp_path)

    wrapped = {
        "target_language": "ru",
        "result": {
            "scenes": [
                {
                    "id": "scene_localize",
                    "narration": "Офицер подходит к автомобилю.",
                    "onScreenText": "Записанная встреча",
                    "subtitles": [
                        {
                            "sourceId": source.id,
                            "segmentId": 0,
                            "translation": "Офицер подошёл.",
                            "confidence": 0.99,
                        },
                        {
                            "sourceId": source.id,
                            "segmentId": 1,
                            "translation": "Всё изменилось.",
                        },
                    ],
                    "notes": "wrapper metadata must be ignored",
                }
            ]
        },
        "model": "test",
    }

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: json.dumps(wrapped, ensure_ascii=False),
        review_fn=_approved_review,
    )

    assert plan.scenes[0].scene_id == "scene_localize"
    assert plan.scenes[0].narration_text == "Офицер подходит к автомобилю."
    assert plan.scenes[0].on_screen_text == "Записанная встреча"
    assert [
        (item.source_id, item.segment_id, item.text)
        for item in plan.scenes[0].subtitle_segments
    ] == [
        (source.id, 0, "Офицер подошёл."),
        (source.id, 1, "Всё изменилось."),
    ]


def test_localization_reviewer_accepts_json_surrounded_by_prose(tmp_path: Path):
    project, _ = _project_with_localizable_scene(tmp_path)

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=lambda prompt: (
            'Review complete. {"supported": true, "issues": []} Thank you.'
        ),
    )

    assert plan.semantic_reviewed is True


def test_semantic_review_repairs_only_rejected_batch(
    tmp_path: Path,
    monkeypatch,
):
    project, source = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_BATCH_ITEMS",
        2,
    )
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_BATCH_CHARS",
        100000,
    )
    generate_calls = []
    rejected_once = {"value": False}

    bad_response = json.loads(_russian_response())
    bad_response["scenes"][0]["subtitle_segments"][0]["text"] = "ПЛОХО."
    bad_response = json.dumps(bad_response, ensure_ascii=False)

    def generate(prompt: str) -> str:
        generate_calls.append(prompt)
        if "Repair ONLY this rejected documentary translation item" not in prompt:
            return bad_response

        item_id = int(prompt.split("CURRENT ITEM ID:\n", 1)[1].splitlines()[0])
        translated = {
            2: "Офицер подошёл.",
            3: "Всё изменилось.",
        }
        return json.dumps(
            {
                "items": [
                    {"i": item_id, "text": translated[item_id]},
                ]
            },
            ensure_ascii=False,
        )

    def review(prompt: str) -> str:
        items = json.loads(prompt.split("ITEMS:\n", 1)[1])
        ids = [item["i"] for item in items]
        if ids == [2, 3] and not rejected_once["value"]:
            rejected_once["value"] = True
            return json.dumps(
                {
                    "supported": False,
                    "issues": ["item 2 is mistranslated"],
                }
            )
        return json.dumps({"supported": True, "issues": []})

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=generate,
        review_fn=review,
    )

    assert plan.semantic_reviewed is True
    assert len(generate_calls) == 3
    assert all(
        "Repair ONLY this rejected documentary translation item" in prompt
        for prompt in generate_calls[1:]
    )
    assert "CURRENT ITEM ID:\n2" in generate_calls[1]
    assert "CURRENT ITEM ID:\n3" in generate_calls[2]
    assert plan.scenes[0].narration_text == "Офицер подходит к автомобилю."
    assert plan.scenes[0].on_screen_text == "Записанная встреча"
    assert [
        (item.source_id, item.segment_id, item.text)
        for item in plan.scenes[0].subtitle_segments
    ] == [
        (source.id, 0, "Офицер подошёл."),
        (source.id, 1, "Всё изменилось."),
    ]


def test_unresolved_semantic_batch_does_not_retry_whole_language(
    tmp_path: Path,
    monkeypatch,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_BATCH_ITEMS",
        2,
    )
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_BATCH_CHARS",
        100000,
    )
    generate_calls = []

    def generate(prompt: str) -> str:
        generate_calls.append(prompt)
        if "Repair ONLY this rejected documentary translation item" in prompt:
            item_id = int(
                prompt.split("CURRENT ITEM ID:\n", 1)[1].splitlines()[0]
            )
            return json.dumps(
                {
                    "items": [
                        {"i": item_id, "text": "Всё ещё неверно."}
                    ]
                },
                ensure_ascii=False,
            )
        return _russian_response()

    def review(prompt: str) -> str:
        items = json.loads(prompt.split("ITEMS:\n", 1)[1])
        ids = [item["i"] for item in items]
        if ids == [2, 3]:
            return json.dumps(
                {
                    "supported": False,
                    "issues": ["items 2-3 still fail semantic review"],
                }
            )
        return json.dumps({"supported": True, "issues": []})

    with pytest.raises(
        LocalizationError,
        match="semantic localization review failed",
    ):
        localize_project(
            project.id,
            "ru",
            root=tmp_path,
            generate_fn=generate,
            review_fn=review,
        )

    full_generation_calls = [
        prompt
        for prompt in generate_calls
        if "Repair ONLY this rejected documentary translation item" not in prompt
    ]
    repair_calls = [
        prompt
        for prompt in generate_calls
        if "Repair ONLY this rejected documentary translation item" in prompt
    ]
    assert len(full_generation_calls) == 1
    assert len(repair_calls) == 4


def test_compact_review_accepts_minimal_pass_after_unclear_json(
    tmp_path: Path,
    monkeypatch,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_THRESHOLD_CHARS",
        1,
    )
    calls = []

    def review(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "The translation looks faithful, but here is my explanation."
        return "PASS"

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=review,
    )

    assert plan.semantic_reviewed is True
    assert len(calls) == 2
    assert "Return exactly PASS" in calls[1]


def test_compact_review_splits_batch_after_two_unclear_verdicts(
    tmp_path: Path,
    monkeypatch,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_THRESHOLD_CHARS",
        1,
    )
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_BATCH_ITEMS",
        99,
    )
    calls = []

    def review(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) <= 2:
            return "unclear"
        return "PASS"

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=review,
    )

    assert plan.semantic_reviewed is True
    assert len(calls) > 2


def test_compact_review_retries_only_unclear_batch(
    tmp_path: Path,
    monkeypatch,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_THRESHOLD_CHARS",
        1,
    )
    review_calls = []

    def review(prompt: str) -> str:
        review_calls.append(prompt)
        if len(review_calls) == 1:
            return "I cannot format the verdict."
        assert "Return exactly PASS" in prompt
        return "PASS"

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=review,
    )

    assert plan.semantic_reviewed is True
    assert len(review_calls) == 2


def test_localization_reviewer_accepts_plain_pass_verdict(tmp_path: Path):
    project, _ = _project_with_localizable_scene(tmp_path)

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=lambda prompt: "PASS",
    )

    assert plan.semantic_reviewed is True


def test_localization_reviewer_rechecks_unclear_verdict_in_compact_batch(
    tmp_path: Path,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    review_calls = []

    def review(prompt: str) -> str:
        review_calls.append(prompt)
        if len(review_calls) == 1:
            return "The translation appears faithful and complete."
        assert "Return exactly PASS" in prompt
        return "PASS"

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=review,
    )

    assert plan.semantic_reviewed is True
    assert len(review_calls) == 2


def test_large_localization_review_uses_compact_batches(
    tmp_path: Path,
    monkeypatch,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_THRESHOLD_CHARS",
        1,
    )
    review_calls = []

    def review(prompt: str) -> str:
        review_calls.append(prompt)
        assert "SMALL batch of documentary translations" in prompt
        return json.dumps({"supported": True, "issues": []})

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=review,
    )

    assert plan.semantic_reviewed is True
    assert len(review_calls) == 1


def test_localization_reviewer_normalizes_alias_verdict_fields(tmp_path: Path):
    project, _ = _project_with_localizable_scene(tmp_path)

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=lambda prompt: json.dumps(
            {
                "verdict": "approved",
                "reason": "",
                "confidence": 0.98,
            }
        ),
    )

    assert plan.semantic_reviewed is True


def test_localization_reviewer_allows_harmless_metadata_fields(tmp_path: Path):
    project, _ = _project_with_localizable_scene(tmp_path)

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: _russian_response(),
        review_fn=lambda prompt: json.dumps(
            {
                "supported": True,
                "issues": [],
                "confidence": 0.99,
                "notes": "faithful",
            }
        ),
    )

    assert plan.semantic_reviewed is True


def test_localize_project_uses_compact_batches_for_large_payload(
    tmp_path: Path,
    monkeypatch,
):
    project, source = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_LOCALIZATION_THRESHOLD_CHARS",
        1,
    )
    prompts = []

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        assert "compact batch" in prompt.lower()
        return json.dumps(
            {
                "items": [
                    {"i": 0, "text": "Офицер подходит к автомобилю."},
                    {"i": 1, "text": "Записанная встреча"},
                    {"i": 2, "text": "Офицер подошёл."},
                    {"i": 3, "text": "Всё изменилось."},
                ]
            },
            ensure_ascii=False,
        )

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=generate,
        review_fn=_approved_review,
    )

    assert len(prompts) == 1
    assert plan.scenes[0].narration_text == "Офицер подходит к автомобилю."
    assert plan.scenes[0].on_screen_text == "Записанная встреча"
    assert [
        (item.source_id, item.segment_id, item.text)
        for item in plan.scenes[0].subtitle_segments
    ] == [
        (source.id, 0, "Офицер подошёл."),
        (source.id, 1, "Всё изменилось."),
    ]


def test_compact_localization_reports_real_batch_progress(
    tmp_path: Path,
    monkeypatch,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_LOCALIZATION_THRESHOLD_CHARS",
        1,
    )
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_LOCALIZATION_BATCH_ITEMS",
        1,
    )
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_THRESHOLD_CHARS",
        1,
    )
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_BATCH_ITEMS",
        1,
    )
    progress = []
    status_events = []

    def generate(prompt: str) -> str:
        items = json.loads(prompt.split("INPUT ITEMS:\n", 1)[1])
        return json.dumps(
            {
                "items": [
                    {
                        "i": item["i"],
                        "text": f"Перевод {item['i']}",
                    }
                    for item in items
                ]
            },
            ensure_ascii=False,
        )

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=generate,
        review_fn=_approved_review,
        progress=progress.append,
        status=lambda phase, current, total, attempt, attempts: status_events.append(
            (phase, current, total, attempt, attempts)
        ),
    )

    assert plan.semantic_reviewed is True
    assert progress[0] == 0.0
    assert progress[-1] == 1.0
    assert progress == sorted(progress)
    assert any(0.0 < value < 0.70 for value in progress)
    assert any(0.70 < value < 1.0 for value in progress)
    assert any(event[0] == "translation_batch" for event in status_events)
    assert any(event[0] == "review_batch" for event in status_events)


def test_compact_localization_retries_only_missing_items(
    tmp_path: Path,
    monkeypatch,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_LOCALIZATION_THRESHOLD_CHARS",
        1,
    )
    calls = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(
                {
                    "items": [
                        {"i": 0, "text": "Офицер подходит к автомобилю."},
                        {"i": 1, "text": "Записанная встреча"},
                        {"i": 3, "text": "Всё изменилось."},
                    ]
                },
                ensure_ascii=False,
            )
        input_items = prompt.split("INPUT ITEMS:\n", 1)[1]
        assert '"i":2' in input_items
        assert '"i":0' not in input_items
        assert '"i":1' not in input_items
        assert '"i":3' not in input_items
        return json.dumps(
            {"items": [{"i": 2, "text": "Офицер подошёл."}]},
            ensure_ascii=False,
        )

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=generate,
        review_fn=_approved_review,
    )

    assert len(calls) == 2
    assert plan.scenes[0].narration_text == "Офицер подходит к автомобилю."
    assert [
        item.text
        for item in plan.scenes[0].subtitle_segments
    ] == ["Офицер подошёл.", "Всё изменилось."]


def test_compact_localization_splits_unidentifiable_partial_string_batch(
    tmp_path: Path,
    monkeypatch,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_LOCALIZATION_THRESHOLD_CHARS",
        1,
    )
    calls = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(["частичный перевод"])
        if '"i":0' in prompt and '"i":1' in prompt:
            return json.dumps(
                {
                    "items": [
                        {"i": 0, "text": "Офицер подходит к автомобилю."},
                        {"i": 1, "text": "Записанная встреча"},
                    ]
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "items": [
                    {"i": 2, "text": "Офицер подошёл."},
                    {"i": 3, "text": "Всё изменилось."},
                ]
            },
            ensure_ascii=False,
        )

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=generate,
        review_fn=_approved_review,
    )

    assert len(calls) == 3
    assert plan.target_language == "ru"
    assert plan.scenes[0].on_screen_text == "Записанная встреча"


def test_localize_project_falls_back_to_compact_after_invalid_json(
    tmp_path: Path,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    calls = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        if "compact batch" not in prompt.lower():
            return '{"scenes": ['
        return json.dumps(
            {
                "items": [
                    {"i": 0, "text": "Офицер подходит к автомобилю."},
                    {"i": 1, "text": "Записанная встреча"},
                    {"i": 2, "text": "Офицер подошёл."},
                    {"i": 3, "text": "Всё изменилось."},
                ]
            },
            ensure_ascii=False,
        )

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=generate,
        review_fn=_approved_review,
    )

    assert len(calls) == 2
    assert "compact batch" in calls[1].lower()
    assert plan.semantic_reviewed is True


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


def test_localize_project_repairs_semantically_rejected_translation(
    tmp_path: Path,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    review_calls = []
    generate_calls = []

    def generate(prompt: str) -> str:
        generate_calls.append(prompt)
        if "Repair ONLY this rejected documentary translation item" in prompt:
            item_id = int(
                prompt.split("CURRENT ITEM ID:\n", 1)[1].splitlines()[0]
            )
            translated = {
                0: "Офицер подходит к автомобилю.",
                1: "Записанная встреча",
                2: "Офицер подошёл.",
                3: "Всё изменилось.",
            }
            return json.dumps(
                {
                    "items": [
                        {"i": item_id, "text": translated[item_id]}
                    ]
                },
                ensure_ascii=False,
            )
        return _russian_response()

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
        generate_fn=generate,
        review_fn=review,
    )

    assert plan.target_language == "ru"
    assert len(review_calls) == 2
    assert len(generate_calls) == 5
    assert all(
        "Repair ONLY this rejected documentary translation item" in prompt
        for prompt in generate_calls[1:]
    )



def test_repair_items_use_neighbor_context_without_shifting_content(
    tmp_path: Path,
    monkeypatch,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_BATCH_ITEMS",
        2,
    )
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_BATCH_CHARS",
        100000,
    )
    prompts = []
    rejected_once = {"value": False}

    def generate(prompt: str) -> str:
        prompts.append(prompt)
        if "Repair ONLY this rejected documentary translation item" in prompt:
            item_id = int(
                prompt.split("CURRENT ITEM ID:\n", 1)[1].splitlines()[0]
            )
            expected = {
                2: "Офицер подошёл.",
                3: "Всё изменилось.",
            }
            if item_id == 2:
                assert "CURRENT SOURCE:\nThe officer approached." in prompt
                assert "NEXT SOURCE:\nEverything changed." in prompt
            if item_id == 3:
                assert "PREVIOUS SOURCE:\nThe officer approached." in prompt
                assert "CURRENT SOURCE:\nEverything changed." in prompt
            return json.dumps(
                {"items": [{"i": item_id, "text": expected[item_id]}]},
                ensure_ascii=False,
            )
        return _russian_response()

    def review(prompt: str) -> str:
        items = json.loads(prompt.split("ITEMS:\n", 1)[1])
        ids = [item["i"] for item in items]
        if ids == [2, 3] and not rejected_once["value"]:
            rejected_once["value"] = True
            return json.dumps(
                {
                    "supported": False,
                    "issues": ["item 2 shifted into neighboring item"],
                }
            )
        return json.dumps({"supported": True, "issues": []})

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=generate,
        review_fn=review,
    )

    assert plan.semantic_reviewed is True
    repair_prompts = [
        prompt
        for prompt in prompts
        if "Repair ONLY this rejected documentary translation item" in prompt
    ]
    assert len(repair_prompts) == 2


def test_localization_resume_reuses_translated_batches_after_review_crash(
    tmp_path: Path,
    monkeypatch,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_LOCALIZATION_THRESHOLD_CHARS",
        1,
    )
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_LOCALIZATION_BATCH_ITEMS",
        1,
    )
    first_generate_calls = []

    def first_generate(prompt: str) -> str:
        first_generate_calls.append(prompt)
        items = json.loads(prompt.split("INPUT ITEMS:\n", 1)[1])
        translated = {
            0: "Офицер подходит к автомобилю.",
            1: "Записанная встреча",
            2: "Офицер подошёл.",
            3: "Всё изменилось.",
        }
        return json.dumps(
            {
                "items": [
                    {"i": item["i"], "text": translated[item["i"]]}
                    for item in items
                ]
            },
            ensure_ascii=False,
        )

    def crashing_review(prompt: str) -> str:
        raise RuntimeError("review provider interrupted")

    with pytest.raises(RuntimeError, match="review provider interrupted"):
        localize_project(
            project.id,
            "ru",
            root=tmp_path,
            generate_fn=first_generate,
            review_fn=crashing_review,
        )

    assert len(first_generate_calls) == 4
    assert localization_work_path(project.id, "ru", tmp_path).is_file()

    def must_not_translate(prompt: str) -> str:
        raise AssertionError("saved translation batches must be reused")

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=must_not_translate,
        review_fn=_approved_review,
    )

    assert plan.semantic_reviewed is True
    assert not localization_work_path(project.id, "ru", tmp_path).exists()


def test_localization_resume_skips_reviewed_batches_after_review_crash(
    tmp_path: Path,
    monkeypatch,
):
    project, _ = _project_with_localizable_scene(tmp_path)
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_LOCALIZATION_THRESHOLD_CHARS",
        1,
    )
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_BATCH_ITEMS",
        2,
    )
    monkeypatch.setattr(
        "app.services.documentary.localization.COMPACT_REVIEW_BATCH_CHARS",
        100000,
    )

    translated = {
        0: "Офицер подходит к автомобилю.",
        1: "Записанная встреча",
        2: "Офицер подошёл.",
        3: "Всё изменилось.",
    }

    def generate(prompt: str) -> str:
        items = json.loads(prompt.split("INPUT ITEMS:\n", 1)[1])
        return json.dumps(
            {
                "items": [
                    {"i": item["i"], "text": translated[item["i"]]}
                    for item in items
                ]
            },
            ensure_ascii=False,
        )

    first_reviewed = []

    def first_review(prompt: str) -> str:
        items = json.loads(prompt.split("ITEMS:\n", 1)[1])
        ids = [item["i"] for item in items]
        first_reviewed.append(ids)
        if ids == [2, 3]:
            raise RuntimeError("second review batch interrupted")
        return json.dumps({"supported": True, "issues": []})

    with pytest.raises(RuntimeError, match="second review batch interrupted"):
        localize_project(
            project.id,
            "ru",
            root=tmp_path,
            generate_fn=generate,
            review_fn=first_review,
        )

    assert first_reviewed == [[0, 1], [2, 3]]

    resumed_reviewed = []

    def resumed_review(prompt: str) -> str:
        items = json.loads(prompt.split("ITEMS:\n", 1)[1])
        resumed_reviewed.append([item["i"] for item in items])
        return json.dumps({"supported": True, "issues": []})

    def must_not_translate(prompt: str) -> str:
        raise AssertionError("saved translation batches must be reused")

    plan = localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=must_not_translate,
        review_fn=resumed_review,
    )

    assert plan.semantic_reviewed is True
    assert resumed_reviewed == [[2, 3]]


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


def test_localize_project_discards_result_if_master_changes_during_run(
    tmp_path: Path,
):
    project, _ = _project_with_localizable_scene(tmp_path)

    def review(prompt: str) -> str:
        changed = load_project(project.id, tmp_path)
        changed.plan.scenes[0].on_screen_text = "Changed during localization"
        save_project(changed, tmp_path)
        return json.dumps({"supported": True, "issues": []})

    with pytest.raises(
        LocalizationError,
        match="master timeline changed while localization was running",
    ):
        localize_project(
            project.id,
            "ru",
            root=tmp_path,
            generate_fn=lambda prompt: _russian_response(),
            review_fn=review,
        )

    assert not localization_plan_path(project.id, "ru", tmp_path).exists()


def test_localize_project_discards_result_if_transcript_changes_during_run(
    tmp_path: Path,
):
    project, source = _project_with_localizable_scene(tmp_path)

    def review(prompt: str) -> str:
        path = transcript_path(project.id, source.id, tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["segments"][0]["text"] = "Changed during localization."
        payload["full_text"] = "Changed during localization. Everything changed."
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return json.dumps({"supported": True, "issues": []})

    with pytest.raises(
        LocalizationError,
        match="transcript evidence changed while localization was running",
    ):
        localize_project(
            project.id,
            "ru",
            root=tmp_path,
            generate_fn=lambda prompt: _russian_response(),
            review_fn=review,
        )

    assert not localization_plan_path(project.id, "ru", tmp_path).exists()


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
