import json
from pathlib import Path

import pytest

from app.models.documentary import (
    AudioMode,
    DocumentaryScene,
    NarrativePurpose,
    RightsStatus,
    SceneType,
    SourceAsset,
    SourceType,
    VideoMetadata,
)
from app.services.documentary.project import (
    add_source,
    create_project,
    load_project,
    save_project,
)
from app.services.documentary.retention_audit import (
    RetentionAuditError,
    audit_retention,
    load_retention_audit,
    retention_audit_path,
)


def _project_with_timeline(tmp_path: Path):
    project = create_project(
        "Retention audit case",
        project_id="doc_retention_case",
        root=tmp_path,
    )
    source = SourceAsset(
        id="source_retention",
        source_type=SourceType.local_video,
        title="Retention source",
        video_metadata=VideoMetadata(
            duration_seconds=30.0,
            width=1280,
            height=720,
            fps=30,
            has_audio=True,
            video_codec="h264",
            audio_codec="aac",
            container="mov,mp4",
        ),
        rights_status=RightsStatus.user_owned,
    )
    project = add_source(project.id, source, root=tmp_path)
    project = load_project(project.id, tmp_path)
    project.plan.scenes = [
        DocumentaryScene(
            id="scene_open",
            scene_type=SceneType.original_clip,
            source_id=source.id,
            source_start=0.0,
            source_end=5.0,
            audio_mode=AudioMode.original,
            purpose=NarrativePurpose.hook,
            narration_text="A quiet opening.",
        ),
        DocumentaryScene(
            id="scene_reveal",
            scene_type=SceneType.original_clip,
            source_id=source.id,
            source_start=5.0,
            source_end=11.0,
            audio_mode=AudioMode.original,
            purpose=NarrativePurpose.reveal,
            narration_text="The key moment appears.",
        ),
    ]
    save_project(project, tmp_path)
    return load_project(project.id, tmp_path)


def _approved_review(prompt: str) -> str:
    return json.dumps({"supported": True, "issues": []})


def _valid_response() -> str:
    return json.dumps(
        {
            "strongest_opening_scene_id": "scene_reveal",
            "open_loop": "What changed between the opening and the reveal?",
            "reveal_payoff_notes": "The reveal lands in the second scene.",
            "diagnostics": [
                {
                    "kind": "weak_opening",
                    "scene_ids": ["scene_open"],
                    "beat_ids": [],
                    "explanation": "The first scene is quieter than the later reveal.",
                    "recommendation": "Consider opening closer to scene_reveal.",
                }
            ],
            "short_candidates": [
                {
                    "start_scene_id": "scene_open",
                    "end_scene_id": "scene_reveal",
                    "hook": "A quiet start before the key moment.",
                    "reason": "The two-scene range has a setup and reveal.",
                }
            ],
        }
    )


def test_audit_retention_persists_and_loads_grounded_diagnostics(tmp_path: Path):
    project = _project_with_timeline(tmp_path)

    audit = audit_retention(
        project.id,
        root=tmp_path,
        generate_fn=lambda prompt: _valid_response(),
        review_fn=_approved_review,
    )

    assert audit.strongest_opening_scene_id == "scene_reveal"
    assert audit.diagnostics[0].scene_ids == ["scene_open"]
    assert audit.short_candidates[0].end_scene_id == "scene_reveal"
    assert len(audit.master_plan_fingerprint) == 64
    assert retention_audit_path(project.id, tmp_path).is_file()
    assert load_retention_audit(project.id, root=tmp_path) == audit


def test_audit_retention_retries_unknown_opening_scene(tmp_path: Path):
    project = _project_with_timeline(tmp_path)
    payload = json.loads(_valid_response())
    payload["strongest_opening_scene_id"] = "scene_missing"
    calls = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(payload)

    with pytest.raises(RetentionAuditError, match="unknown strongest opening"):
        audit_retention(
            project.id,
            root=tmp_path,
            generate_fn=generate,
            review_fn=_approved_review,
        )

    assert len(calls) == 3
    assert "previous response was rejected" in calls[-1]


def test_audit_retention_rejects_unknown_diagnostic_scene(tmp_path: Path):
    project = _project_with_timeline(tmp_path)
    payload = json.loads(_valid_response())
    payload["diagnostics"][0]["scene_ids"] = ["scene_missing"]

    with pytest.raises(RetentionAuditError, match="diagnostic references an unknown scene"):
        audit_retention(
            project.id,
            root=tmp_path,
            generate_fn=lambda prompt: json.dumps(payload),
            review_fn=_approved_review,
        )


def test_audit_retention_rejects_reversed_short_range(tmp_path: Path):
    project = _project_with_timeline(tmp_path)
    payload = json.loads(_valid_response())
    payload["short_candidates"][0]["start_scene_id"] = "scene_reveal"
    payload["short_candidates"][0]["end_scene_id"] = "scene_open"

    with pytest.raises(RetentionAuditError, match="reversed scene order"):
        audit_retention(
            project.id,
            root=tmp_path,
            generate_fn=lambda prompt: json.dumps(payload),
            review_fn=_approved_review,
        )


def test_audit_retention_rejects_numerical_score_field(tmp_path: Path):
    project = _project_with_timeline(tmp_path)
    payload = json.loads(_valid_response())
    payload["score"] = 87

    with pytest.raises(RetentionAuditError, match="unexpected fields"):
        audit_retention(
            project.id,
            root=tmp_path,
            generate_fn=lambda prompt: json.dumps(payload),
            review_fn=_approved_review,
        )


def test_audit_retention_does_not_retry_provider_error(tmp_path: Path):
    project = _project_with_timeline(tmp_path)
    calls = []

    def generate(prompt: str) -> str:
        calls.append(prompt)
        return "Error: provider unavailable"

    with pytest.raises(RetentionAuditError, match="provider unavailable"):
        audit_retention(
            project.id,
            root=tmp_path,
            generate_fn=generate,
            review_fn=_approved_review,
        )

    assert len(calls) == 1


def test_load_retention_audit_rejects_changed_timeline(tmp_path: Path):
    project = _project_with_timeline(tmp_path)
    audit_retention(
        project.id,
        root=tmp_path,
        generate_fn=lambda prompt: _valid_response(),
        review_fn=_approved_review,
    )

    changed = load_project(project.id, tmp_path)
    changed.plan.scenes[0].narration_text = "Timeline changed."
    save_project(changed, tmp_path)

    with pytest.raises(RetentionAuditError, match="timeline changed"):
        load_retention_audit(project.id, root=tmp_path)


def test_audit_retention_discards_result_if_timeline_changes_during_run(
    tmp_path: Path,
):
    project = _project_with_timeline(tmp_path)

    def generate(prompt: str) -> str:
        changed = load_project(project.id, tmp_path)
        changed.plan.scenes[0].on_screen_text = "Changed during audit"
        save_project(changed, tmp_path)
        return _valid_response()

    with pytest.raises(
        RetentionAuditError,
        match="timeline changed while retention audit was running",
    ):
        audit_retention(
            project.id,
            root=tmp_path,
            generate_fn=generate,
            review_fn=_approved_review,
        )

    assert not retention_audit_path(project.id, tmp_path).exists()


def test_audit_retention_requires_timeline_scenes(tmp_path: Path):
    project = create_project(
        "Empty retention audit",
        project_id="doc_retention_empty",
        root=tmp_path,
    )

    with pytest.raises(RetentionAuditError, match="requires at least one scene"):
        audit_retention(
            project.id,
            root=tmp_path,
            generate_fn=lambda prompt: _valid_response(),
            review_fn=_approved_review,
        )
