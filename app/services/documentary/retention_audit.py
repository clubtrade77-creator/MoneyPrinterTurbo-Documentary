from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable
from uuid import uuid4

from app.models.documentary import DocumentaryProject, RetentionAudit
from app.services import llm as llm_service
from app.services.documentary.clip_selector import story_plan_fingerprint
from app.services.documentary.project import load_project, project_dir
from app.services.documentary.story_planner import load_story_plan

MAX_RETENTION_AUDIT_ATTEMPTS = 3
MAX_RETENTION_AUDIT_PROMPT_CHARS = 120_000
MAX_RETENTION_REVIEW_PROMPT_CHARS = 180_000
CURRENT_RETENTION_REVIEW_VERSION = 2
MIN_NARRATION_STRETCH_SECONDS = 8.0
MIN_SOURCE_STAGNATION_SECONDS = 12.0


class RetentionAuditError(RuntimeError):
    """Raised when a grounded documentary retention audit cannot be produced safely."""


class _NonRetryableRetentionAuditError(RetentionAuditError):
    """Raised for provider/runtime failures that retries cannot repair."""


def retention_audit_path(
    project_id: str,
    root: str | os.PathLike | None = None,
) -> Path:
    return project_dir(project_id, root) / "plans" / "retention-audit.json"


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


def master_plan_fingerprint(project: DocumentaryProject) -> str:
    payload = project.plan.model_dump(mode="json")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _audit_content_fingerprint(audit: RetentionAudit) -> str:
    payload = {
        "strongest_opening_scene_id": audit.strongest_opening_scene_id,
        "open_loop": audit.open_loop,
        "reveal_payoff_notes": audit.reveal_payoff_notes,
        "diagnostics": [
            item.model_dump(mode="json")
            for item in audit.diagnostics
        ],
        "short_candidates": [
            item.model_dump(mode="json")
            for item in audit.short_candidates
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _strip_code_fence(text: str) -> str:
    value = (text or "").strip()
    fence = chr(96) * 3
    if not value.startswith(fence):
        return value
    lines = value.splitlines()
    if lines and lines[0].strip().startswith(fence):
        lines = lines[1:]
    if lines and lines[-1].strip() == fence:
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _load_context(
    project_id: str,
    *,
    root: str | os.PathLike | None,
):
    project = load_project(project_id, root)
    if not project.plan.scenes:
        raise RetentionAuditError(
            "documentary retention audit requires at least one scene"
        )

    story_plan = None
    current_story_fingerprint = ""
    if project.plan.story_plan_fingerprint:
        story_plan = load_story_plan(project_id, root=root)
        current_story_fingerprint = story_plan_fingerprint(story_plan)
        if current_story_fingerprint != project.plan.story_plan_fingerprint:
            raise RetentionAuditError(
                "documentary timeline is stale; Story Plan changed before retention audit"
            )
    return project, story_plan, current_story_fingerprint


def _scene_duration_seconds(scene) -> float:
    if scene.source_start is None or scene.source_end is None:
        return 0.0
    return max(float(scene.source_end) - float(scene.source_start), 0.0)


def timeline_payload(project: DocumentaryProject) -> list[dict]:
    payload = []
    elapsed = 0.0
    previous_source_id = None

    for index, scene in enumerate(project.plan.scenes):
        duration = _scene_duration_seconds(scene)
        payload.append(
            {
                "index": index,
                "scene_id": scene.id,
                "timeline_start_seconds": round(elapsed, 3),
                "duration_seconds": round(duration, 3),
                "purpose": scene.purpose.value,
                "scene_type": scene.scene_type.value,
                "source_id": scene.source_id,
                "source_changed": bool(
                    index > 0 and scene.source_id != previous_source_id
                ),
                "audio_mode": scene.audio_mode.value,
                "narration_text": scene.narration_text[:2000],
                "narration_characters": len(scene.narration_text),
                "on_screen_text": scene.on_screen_text[:1000],
                "story_beat_id": scene.story_beat_id,
                "transcript_segment_ids": scene.transcript_segment_ids,
            }
        )
        elapsed += duration
        previous_source_id = scene.source_id

    return payload


def build_retention_audit_prompt(
    project: DocumentaryProject,
    story_plan,
) -> str:
    story_payload = None
    if story_plan is not None:
        story_payload = {
            "title": story_plan.title,
            "angle": story_plan.angle,
            "hook": story_plan.hook,
            "beats": [
                {
                    "id": beat.id,
                    "purpose": beat.purpose.value,
                    "title": beat.title,
                    "summary": beat.summary,
                    "target_duration_seconds": beat.target_duration_seconds,
                }
                for beat in story_plan.beats
            ],
        }

    source = {
        "story_plan": story_payload,
        "timeline": timeline_payload(project),
    }

    prompt = (
        "Audit this documentary timeline for viewer retention. "
        "Do not produce a numerical score. Return concrete editorial diagnostics only. "
        "Use only scene_id and beat_id values present in the input. "
        "Do not invent incident facts or claims about what unseen footage contains. "
        "Do not infer qualities of imagery, sound, mood, or camera behavior unless those "
        "qualities are explicit in the supplied text or metadata. "
        "Identify the strongest candidate opening scene, any unresolved question/open loop, "
        "stretches with too much narration and no new evidence, long stretches without a "
        "source change or new information, reveal/payoff placement, and useful candidate "
        "cut ranges for Shorts. A Shorts candidate must use existing start/end scene ids "
        "in forward timeline order. open_loop and short candidate hook must be natural-language "
        "editorial text, never a raw scene_id or beat_id. Only flag narration_stretch when the "
        f"referenced stretch is at least {MIN_NARRATION_STRETCH_SECONDS:g} seconds. Only flag "
        f"source_stagnation when the same-source stretch is at least "
        f"{MIN_SOURCE_STAGNATION_SECONDS:g} seconds. It is valid to return no diagnostics or no "
        "short candidates when the input does not support them. "
        "Return JSON only with exactly these top-level keys: "
        "strongest_opening_scene_id, open_loop, reveal_payoff_notes, diagnostics, "
        "short_candidates. "
        "Each diagnostic must contain exactly: kind, scene_ids, beat_ids, explanation, "
        "recommendation. kind must be one of weak_opening, open_loop, narration_stretch, "
        "source_stagnation, reveal_payoff, other. "
        "Each short candidate must contain exactly: start_scene_id, end_scene_id, hook, "
        "reason.\n\nINPUT:\n"
        + json.dumps(source, ensure_ascii=False, separators=(",", ":"))
    )
    if len(prompt) > MAX_RETENTION_AUDIT_PROMPT_CHARS:
        raise RetentionAuditError(
            "documentary retention audit prompt exceeds safe size; split the timeline"
        )
    return prompt


def _validate_references(
    audit: RetentionAudit,
    project: DocumentaryProject,
    story_plan,
) -> None:
    scene_ids = [scene.id for scene in project.plan.scenes]
    scene_index = {scene_id: index for index, scene_id in enumerate(scene_ids)}
    known_scene_ids = set(scene_ids)

    if audit.strongest_opening_scene_id not in known_scene_ids:
        raise RetentionAuditError(
            "documentary retention audit referenced an unknown strongest opening scene"
        )

    known_beat_ids = (
        {beat.id for beat in story_plan.beats}
        if story_plan is not None
        else set()
    )

    known_text_ids = known_scene_ids | known_beat_ids
    if audit.open_loop and audit.open_loop in known_text_ids:
        raise RetentionAuditError(
            "documentary retention open_loop must be editorial text, not a raw id"
        )

    scenes_by_id = {scene.id: scene for scene in project.plan.scenes}

    def referenced_scenes_are_contiguous(scene_ids_to_check: list[str]) -> bool:
        positions = [scene_index[scene_id] for scene_id in scene_ids_to_check]
        return positions == list(range(positions[0], positions[0] + len(positions)))

    for diagnostic in audit.diagnostics:
        if any(scene_id not in known_scene_ids for scene_id in diagnostic.scene_ids):
            raise RetentionAuditError(
                "documentary retention diagnostic references an unknown scene"
            )
        if story_plan is None and diagnostic.beat_ids:
            raise RetentionAuditError(
                "documentary retention diagnostic references beats without a Story Plan"
            )
        if any(beat_id not in known_beat_ids for beat_id in diagnostic.beat_ids):
            raise RetentionAuditError(
                "documentary retention diagnostic references an unknown beat"
            )

        referenced_scenes = [
            scenes_by_id[scene_id]
            for scene_id in diagnostic.scene_ids
        ]
        referenced_duration = sum(
            _scene_duration_seconds(scene)
            for scene in referenced_scenes
        )
        if diagnostic.kind == "narration_stretch":
            if not referenced_scenes_are_contiguous(diagnostic.scene_ids):
                raise RetentionAuditError(
                    "documentary narration_stretch diagnostic must reference contiguous scenes"
                )
            if referenced_duration < MIN_NARRATION_STRETCH_SECONDS:
                raise RetentionAuditError(
                    "documentary narration_stretch diagnostic is below minimum duration"
                )
            if any(not scene.narration_text.strip() for scene in referenced_scenes):
                raise RetentionAuditError(
                    "documentary narration_stretch diagnostic includes a scene without narration"
                )
        if diagnostic.kind == "source_stagnation":
            if not referenced_scenes_are_contiguous(diagnostic.scene_ids):
                raise RetentionAuditError(
                    "documentary source_stagnation diagnostic must reference contiguous scenes"
                )
            if referenced_duration < MIN_SOURCE_STAGNATION_SECONDS:
                raise RetentionAuditError(
                    "documentary source_stagnation diagnostic is below minimum duration"
                )
            source_ids = {scene.source_id for scene in referenced_scenes}
            if len(source_ids) != 1:
                raise RetentionAuditError(
                    "documentary source_stagnation diagnostic spans multiple sources"
                )

    for candidate in audit.short_candidates:
        if (
            candidate.start_scene_id not in known_scene_ids
            or candidate.end_scene_id not in known_scene_ids
        ):
            raise RetentionAuditError(
                "documentary short candidate references an unknown scene"
            )
        if scene_index[candidate.start_scene_id] > scene_index[candidate.end_scene_id]:
            raise RetentionAuditError(
                "documentary short candidate uses reversed scene order"
            )
        if candidate.hook in known_text_ids:
            raise RetentionAuditError(
                "documentary short candidate hook must be editorial text, not a raw id"
            )


def _parse_candidate(
    response_text: str,
    project: DocumentaryProject,
    story_plan,
) -> RetentionAudit:
    raw = _strip_code_fence(response_text)
    if not raw:
        raise RetentionAuditError(
            "documentary retention auditor returned an empty response"
        )
    if raw.startswith("Error:"):
        raise _NonRetryableRetentionAuditError(
            raw.removeprefix("Error:").strip() or raw
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RetentionAuditError(
            "documentary retention auditor did not return valid JSON"
        ) from exc

    expected_keys = {
        "strongest_opening_scene_id",
        "open_loop",
        "reveal_payoff_notes",
        "diagnostics",
        "short_candidates",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise RetentionAuditError(
            "documentary retention auditor response has unexpected fields"
        )

    try:
        audit = RetentionAudit(
            master_plan_fingerprint=master_plan_fingerprint(project),
            story_plan_fingerprint=(
                story_plan_fingerprint(story_plan)
                if story_plan is not None
                else ""
            ),
            **payload,
        )
    except ValueError as exc:
        raise RetentionAuditError(
            f"invalid documentary retention audit: {exc}"
        ) from exc

    _validate_references(audit, project, story_plan)
    return audit


def build_retention_review_prompt(
    project: DocumentaryProject,
    story_plan,
    audit: RetentionAudit,
) -> str:
    payload = {
        "timeline": timeline_payload(project),
        "story_plan": (
            {
                "title": story_plan.title,
                "angle": story_plan.angle,
                "hook": story_plan.hook,
                "beats": [
                    {
                        "id": beat.id,
                        "purpose": beat.purpose.value,
                        "title": beat.title,
                        "summary": beat.summary,
                    }
                    for beat in story_plan.beats
                ],
            }
            if story_plan is not None
            else None
        ),
        "audit": {
            "strongest_opening_scene_id": audit.strongest_opening_scene_id,
            "open_loop": audit.open_loop,
            "reveal_payoff_notes": audit.reveal_payoff_notes,
            "diagnostics": [
                item.model_dump(mode="json")
                for item in audit.diagnostics
            ],
            "short_candidates": [
                item.model_dump(mode="json")
                for item in audit.short_candidates
            ],
        },
    }
    prompt = (
        "Review this documentary retention audit for internal consistency and grounding. "
        "Reject unsupported incident facts, raw ids used as editorial copy, contradictory "
        "opening recommendations, diagnostics that are not supported by the supplied timeline, "
        "or Shorts hooks/reasons that merely repeat ids instead of useful editorial text. "
        "Also reject claims about imagery, sound, mood, or camera behavior that are not explicit "
        "in the supplied text or metadata. Do not judge stylistic taste unless it creates a "
        "factual or logical contradiction. "
        'Return JSON only: {"supported": true, "issues": []} when acceptable, or '
        '{"supported": false, "issues": ["specific issue"]} when not.\n\n'
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    if len(prompt) > MAX_RETENTION_REVIEW_PROMPT_CHARS:
        raise RetentionAuditError(
            "documentary retention review prompt exceeds safe size"
        )
    return prompt


def build_retention_grounding_review_prompt(
    project: DocumentaryProject,
    story_plan,
    audit: RetentionAudit,
) -> str:
    payload = {
        "timeline": timeline_payload(project),
        "story_plan": (
            {
                "title": story_plan.title,
                "angle": story_plan.angle,
                "hook": story_plan.hook,
                "beats": [
                    {
                        "id": beat.id,
                        "purpose": beat.purpose.value,
                        "title": beat.title,
                        "summary": beat.summary,
                    }
                    for beat in story_plan.beats
                ],
            }
            if story_plan is not None
            else None
        ),
        "audit": {
            "strongest_opening_scene_id": audit.strongest_opening_scene_id,
            "open_loop": audit.open_loop,
            "reveal_payoff_notes": audit.reveal_payoff_notes,
            "diagnostics": [
                item.model_dump(mode="json")
                for item in audit.diagnostics
            ],
            "short_candidates": [
                item.model_dump(mode="json")
                for item in audit.short_candidates
            ],
        },
    }
    prompt = (
        "Review every descriptive claim in this retention audit against the supplied input. "
        "Timing, scene order, narration presence, source repetition, purpose labels, and the "
        "supplied text may be used. Reject descriptive claims about imagery, sound, mood, "
        "camera behavior, or incident details when those claims are not explicit in the input. "
        "Pay special attention to adjectives and Shorts hooks. "
        'Return JSON only: {"supported": true, "issues": []} when grounded, or '
        '{"supported": false, "issues": ["specific issue"]} when not.\n\n'
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    if len(prompt) > MAX_RETENTION_REVIEW_PROMPT_CHARS:
        raise RetentionAuditError(
            "documentary retention grounding review prompt exceeds safe size"
        )
    return prompt


def _parse_review(response_text: str) -> list[str]:
    raw = _strip_code_fence(response_text)
    if not raw:
        raise _NonRetryableRetentionAuditError(
            "documentary retention reviewer returned an empty response"
        )
    if raw.startswith("Error:"):
        raise _NonRetryableRetentionAuditError(
            raw.removeprefix("Error:").strip() or raw
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RetentionAuditError(
            "documentary retention reviewer did not return valid JSON"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {"supported", "issues"}:
        raise RetentionAuditError(
            "documentary retention reviewer response has unexpected fields"
        )
    if not isinstance(payload["supported"], bool) or not isinstance(
        payload["issues"], list
    ):
        raise RetentionAuditError(
            "documentary retention reviewer response has invalid field types"
        )
    issues = [
        str(issue).strip()
        for issue in payload["issues"]
        if str(issue).strip()
    ]
    if payload["supported"] and issues:
        raise RetentionAuditError(
            "documentary retention reviewer approved while reporting issues"
        )
    if not payload["supported"] and not issues:
        raise RetentionAuditError(
            "documentary retention reviewer rejected without an issue"
        )
    return [] if payload["supported"] else issues


def _retry_prompt(base_prompt: str, error: Exception, attempt: int) -> str:
    return (
        base_prompt
        + "\n\nYour previous response was rejected. "
        + f"Attempt {attempt} issue: {error}. "
        + "Return corrected JSON only."
    )


def audit_retention(
    project_id: str,
    *,
    root: str | os.PathLike | None = None,
    generate_fn: Callable[[str], str] | None = None,
    review_fn: Callable[[str], str] | None = None,
) -> RetentionAudit:
    project, story_plan, story_fingerprint = _load_context(project_id, root=root)
    prompt = build_retention_audit_prompt(project, story_plan)
    generator = generate_fn or llm_service.generate_text
    if generate_fn is not None and review_fn is None:
        raise ValueError(
            "review_fn is required when generate_fn is supplied so custom "
            "retention audits cannot bypass semantic review"
        )
    reviewer = review_fn or llm_service.generate_text

    current_prompt = prompt
    audit = None
    last_error = None

    for attempt in range(1, MAX_RETENTION_AUDIT_ATTEMPTS + 1):
        try:
            audit = _parse_candidate(
                generator(current_prompt),
                project,
                story_plan,
            )
            review_issues = _parse_review(
                reviewer(
                    build_retention_review_prompt(
                        project,
                        story_plan,
                        audit,
                    )
                )
            )
            if review_issues:
                raise RetentionAuditError(
                    "semantic retention review failed: "
                    + "; ".join(review_issues)
                )
            grounding_issues = _parse_review(
                reviewer(
                    build_retention_grounding_review_prompt(
                        project,
                        story_plan,
                        audit,
                    )
                )
            )
            if grounding_issues:
                raise RetentionAuditError(
                    "grounding retention review failed: "
                    + "; ".join(grounding_issues)
                )
            audit.semantic_reviewed = True
            audit.semantic_review_version = CURRENT_RETENTION_REVIEW_VERSION
            audit.reviewed_content_fingerprint = _audit_content_fingerprint(audit)
            break
        except _NonRetryableRetentionAuditError:
            raise
        except RetentionAuditError as exc:
            last_error = exc
            if attempt >= MAX_RETENTION_AUDIT_ATTEMPTS:
                raise
            current_prompt = _retry_prompt(prompt, exc, attempt)

    if audit is None:
        raise last_error or RetentionAuditError(
            "documentary retention audit failed without a result"
        )

    latest_project, _, latest_story_fingerprint = _load_context(
        project_id,
        root=root,
    )
    if (
        master_plan_fingerprint(latest_project)
        != audit.master_plan_fingerprint
    ):
        raise RetentionAuditError(
            "documentary timeline changed while retention audit was running"
        )
    if latest_story_fingerprint != story_fingerprint:
        raise RetentionAuditError(
            "documentary Story Plan changed while retention audit was running"
        )

    _atomic_write_json(
        retention_audit_path(project_id, root),
        audit.model_dump(mode="json"),
    )
    return audit


def load_retention_audit(
    project_id: str,
    *,
    root: str | os.PathLike | None = None,
) -> RetentionAudit:
    path = retention_audit_path(project_id, root)
    if not path.is_file():
        raise FileNotFoundError(
            f"documentary retention audit not found: {project_id}"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit = RetentionAudit.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise RetentionAuditError(
            f"invalid documentary retention audit: {project_id}"
        ) from exc

    if not audit.semantic_reviewed:
        raise RetentionAuditError(
            "documentary retention audit has not passed semantic review"
        )
    if audit.semantic_review_version != CURRENT_RETENTION_REVIEW_VERSION:
        raise RetentionAuditError(
            "documentary retention audit review policy is stale"
        )
    if audit.reviewed_content_fingerprint != _audit_content_fingerprint(audit):
        raise RetentionAuditError(
            "documentary retention audit changed after semantic review"
        )

    project, story_plan, story_fingerprint = _load_context(
        project_id,
        root=root,
    )
    if audit.master_plan_fingerprint != master_plan_fingerprint(project):
        raise RetentionAuditError(
            "documentary retention audit is stale; timeline changed"
        )
    if audit.story_plan_fingerprint != story_fingerprint:
        raise RetentionAuditError(
            "documentary retention audit is stale; Story Plan changed"
        )

    _validate_references(audit, project, story_plan)
    return audit
