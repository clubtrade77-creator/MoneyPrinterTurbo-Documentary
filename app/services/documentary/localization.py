from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable
from uuid import uuid4

from app.models.documentary import (
    DocumentaryProject,
    DocumentaryTranscript,
    LocalizationPlan,
    LocalizedSceneText,
)
from app.services import llm as llm_service
from app.services.documentary.project import load_project, project_dir
from app.services.documentary.transcription import load_source_transcript

MAX_LOCALIZATION_ATTEMPTS = 3
MAX_LOCALIZATION_PROMPT_CHARS = 120_000
MAX_LOCALIZATION_REVIEW_PROMPT_CHARS = 180_000
CURRENT_LOCALIZATION_REVIEW_VERSION = 1


class LocalizationError(RuntimeError):
    """Raised when a faithful documentary localization cannot be produced safely."""


class _NonRetryableLocalizationError(LocalizationError):
    """Raised for provider/runtime failures that retries cannot repair."""


def _normalize_language(language: str) -> str:
    value = (language or "").strip().lower()
    parts = value.split("-")
    if not 1 <= len(parts) <= 4:
        raise ValueError("invalid documentary localization language")
    if not (2 <= len(parts[0]) <= 8 and parts[0].isalpha()):
        raise ValueError("invalid documentary localization language")
    if any(not (1 <= len(part) <= 8 and part.isalnum()) for part in parts[1:]):
        raise ValueError("invalid documentary localization language")
    return value


def localization_plan_path(
    project_id: str,
    target_language: str,
    root: str | os.PathLike | None = None,
) -> Path:
    language = _normalize_language(target_language)
    safe_language = language.replace("-", "_")
    return project_dir(project_id, root) / "plans" / f"localization-{safe_language}.json"


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


def _transcript_fingerprint(transcript: DocumentaryTranscript) -> str:
    return _model_fingerprint(transcript)


def _localized_scenes_fingerprint(
    scenes: list[LocalizedSceneText],
) -> str:
    payload = [
        scene.model_dump(mode="json")
        for scene in scenes
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def master_plan_fingerprint(project: DocumentaryProject) -> str:
    return _model_fingerprint(project.plan)


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


def _load_localization_context(
    project_id: str,
    *,
    root: str | os.PathLike | None,
) -> tuple[DocumentaryProject, dict[str, DocumentaryTranscript]]:
    project = load_project(project_id, root)
    if not project.plan.scenes:
        raise LocalizationError("documentary localization requires at least one scene")

    referenced_source_ids = []
    seen_source_ids = set()
    for scene in project.plan.scenes:
        if not scene.transcript_segment_ids:
            continue
        if not scene.source_id:
            raise LocalizationError(
                f"scene {scene.id} has transcript segment ids without a source"
            )
        if scene.source_id not in seen_source_ids:
            seen_source_ids.add(scene.source_id)
            referenced_source_ids.append(scene.source_id)

    transcripts = {
        source_id: load_source_transcript(
            project_id,
            source_id,
            root=root,
        )
        for source_id in referenced_source_ids
    }
    return project, transcripts


def _source_payload(
    project: DocumentaryProject,
    transcripts: dict[str, DocumentaryTranscript],
) -> list[dict]:
    payload = []
    for scene in project.plan.scenes:
        subtitle_segments = []
        if scene.transcript_segment_ids:
            transcript = transcripts.get(scene.source_id)
            if transcript is None:
                raise LocalizationError(
                    f"scene {scene.id} references an unavailable transcript"
                )
            segments_by_id = {segment.id: segment for segment in transcript.segments}
            for segment_id in scene.transcript_segment_ids:
                segment = segments_by_id.get(segment_id)
                if segment is None:
                    raise LocalizationError(
                        f"scene {scene.id} references unknown transcript segment "
                        f"{segment_id} for {scene.source_id}"
                    )
                subtitle_segments.append(
                    {
                        "source_id": scene.source_id,
                        "segment_id": segment.id,
                        "text": segment.text,
                    }
                )

        payload.append(
            {
                "scene_id": scene.id,
                "narration_text": scene.narration_text,
                "on_screen_text": scene.on_screen_text,
                "subtitle_segments": subtitle_segments,
            }
        )
    return payload


def build_localization_prompt(
    *,
    source_language: str,
    target_language: str,
    source_scenes: list[dict],
) -> str:
    source_language = _normalize_language(source_language)
    target_language = _normalize_language(target_language)
    if source_language == target_language:
        raise ValueError("localization target language must differ from source language")

    source_json = json.dumps(
        {"scenes": source_scenes},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prompt = f"""
You are localizing a documentary timeline from {source_language} to {target_language}.

Translate only the human-readable text. Do not change scene_id, source_id, or segment_id.
Do not add facts, explanations, dates, names, numbers, allegations, legal conclusions,
or emotional framing that are absent from the source.
Do not omit factual qualifications or uncertainty.
Preserve names, numbers, dates, quoted claims, and attribution faithfully.
For narration_text, keep the translation concise and close to the source's likely
spoken duration. Prefer natural compact phrasing over expansion, while preserving
every material fact, qualification, attribution, and uncertainty.
An empty narration_text or on_screen_text must remain an empty string.
Every source scene must appear exactly once and in the same order.
Every subtitle segment must appear exactly once and in the same order.
Return JSON only, with exactly this shape:
{{
  "scenes": [
    {{
      "scene_id": "same id",
      "narration_text": "translated text or empty string",
      "on_screen_text": "translated text or empty string",
      "subtitle_segments": [
        {{
          "source_id": "same source id",
          "segment_id": 0,
          "text": "translated subtitle"
        }}
      ]
    }}
  ]
}}

SOURCE:
{source_json}
""".strip()
    if len(prompt) > MAX_LOCALIZATION_PROMPT_CHARS:
        raise LocalizationError(
            "documentary localization prompt exceeds safe size; split the project"
        )
    return prompt


def _parse_candidate(response_text: str) -> list[LocalizedSceneText]:
    raw = _strip_code_fence(response_text)
    if not raw:
        raise LocalizationError("documentary localizer returned an empty response")
    if raw.startswith("Error:"):
        raise _NonRetryableLocalizationError(
            raw.removeprefix("Error:").strip() or raw
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalizationError("documentary localizer did not return valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"scenes"}:
        raise LocalizationError(
            "documentary localizer response must contain only a scenes array"
        )
    if not isinstance(payload["scenes"], list):
        raise LocalizationError("documentary localizer scenes must be an array")

    try:
        return [
            LocalizedSceneText.model_validate(item)
            for item in payload["scenes"]
        ]
    except ValueError as exc:
        raise LocalizationError(f"invalid documentary localization: {exc}") from exc


def _validate_candidate_against_source(
    localized_scenes: list[LocalizedSceneText],
    source_scenes: list[dict],
) -> None:
    if len(localized_scenes) != len(source_scenes):
        raise LocalizationError(
            "documentary localization must contain every source scene exactly once"
        )

    for localized, source in zip(localized_scenes, source_scenes):
        if localized.scene_id != source["scene_id"]:
            raise LocalizationError(
                "documentary localization changed scene order or scene ids"
            )

        source_narration = str(source["narration_text"] or "").strip()
        source_on_screen = str(source["on_screen_text"] or "").strip()
        if bool(source_narration) != bool(localized.narration_text):
            raise LocalizationError(
                f"documentary localization changed narration presence: "
                f"{localized.scene_id}"
            )
        if bool(source_on_screen) != bool(localized.on_screen_text):
            raise LocalizationError(
                f"documentary localization changed on-screen text presence: "
                f"{localized.scene_id}"
            )

        expected_refs = [
            (item["source_id"], item["segment_id"])
            for item in source["subtitle_segments"]
        ]
        actual_refs = [
            (item.source_id, item.segment_id)
            for item in localized.subtitle_segments
        ]
        if actual_refs != expected_refs:
            raise LocalizationError(
                f"documentary localization changed subtitle evidence refs: "
                f"{localized.scene_id}"
            )


def build_localization_review_prompt(
    *,
    source_language: str,
    target_language: str,
    source_scenes: list[dict],
    localized_scenes: list[LocalizedSceneText],
) -> str:
    payload = {
        "source_language": source_language,
        "target_language": target_language,
        "source": {"scenes": source_scenes},
        "localized": {
            "scenes": [
                scene.model_dump(mode="json")
                for scene in localized_scenes
            ]
        },
    }
    prompt = (
        "Review this documentary translation for factual fidelity and target-language "
        "completeness. Reject additions, omissions, changed attribution, changed "
        "uncertainty, changed names/numbers/dates, meaning changes, or untranslated "
        "source-language prose that should have been translated. Proper names and "
        "conventional identical forms may remain unchanged. Do not critique style. "
        'Return JSON only: {"supported": true, "issues": []} when faithful, or '
        '{"supported": false, "issues": ["specific issue"]} when not.\n\n'
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    if len(prompt) > MAX_LOCALIZATION_REVIEW_PROMPT_CHARS:
        raise LocalizationError(
            "documentary localization review prompt exceeds safe size; split the project"
        )
    return prompt


def _parse_review(response_text: str) -> list[str]:
    raw = _strip_code_fence(response_text)
    if not raw:
        raise _NonRetryableLocalizationError(
            "documentary localization reviewer returned an empty response"
        )
    if raw.startswith("Error:"):
        raise _NonRetryableLocalizationError(
            raw.removeprefix("Error:").strip() or raw
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalizationError(
            "documentary localization reviewer did not return valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise LocalizationError(
            "documentary localization reviewer response must be one JSON object"
        )
    if set(payload) != {"supported", "issues"}:
        raise LocalizationError(
            "documentary localization reviewer response has unexpected fields"
        )
    if not isinstance(payload["supported"], bool) or not isinstance(
        payload["issues"], list
    ):
        raise LocalizationError(
            "documentary localization reviewer response has invalid field types"
        )
    issues = [
        str(issue).strip()
        for issue in payload["issues"]
        if str(issue).strip()
    ]
    if payload["supported"] and issues:
        raise LocalizationError(
            "documentary localization reviewer approved while reporting issues"
        )
    if not payload["supported"] and not issues:
        raise LocalizationError(
            "documentary localization reviewer rejected without an issue"
        )
    return [] if payload["supported"] else issues


def _retry_prompt(base_prompt: str, error: Exception, attempt: int) -> str:
    return (
        base_prompt
        + "\n\nYour previous response was rejected. "
        + f"Attempt {attempt} issue: {error}. "
        + "Return a corrected JSON object only."
    )


def localize_project(
    project_id: str,
    target_language: str,
    *,
    root: str | os.PathLike | None = None,
    generate_fn: Callable[[str], str] | None = None,
    review_fn: Callable[[str], str] | None = None,
) -> LocalizationPlan:
    project, transcripts = _load_localization_context(project_id, root=root)
    source_language = _normalize_language(project.master_language)
    target_language = _normalize_language(target_language)
    if source_language == target_language:
        raise ValueError("localization target language must differ from source language")

    source_scenes = _source_payload(project, transcripts)
    prompt = build_localization_prompt(
        source_language=source_language,
        target_language=target_language,
        source_scenes=source_scenes,
    )

    generator = generate_fn or llm_service.generate_text
    if generate_fn is not None and review_fn is None:
        raise ValueError(
            "review_fn is required when generate_fn is supplied so custom "
            "localization cannot bypass semantic review"
        )
    reviewer = review_fn or llm_service.generate_text

    current_prompt = prompt
    localized_scenes = None
    last_error = None

    for attempt in range(1, MAX_LOCALIZATION_ATTEMPTS + 1):
        try:
            candidate = _parse_candidate(generator(current_prompt))
            _validate_candidate_against_source(candidate, source_scenes)
            review_issues = _parse_review(
                reviewer(
                    build_localization_review_prompt(
                        source_language=source_language,
                        target_language=target_language,
                        source_scenes=source_scenes,
                        localized_scenes=candidate,
                    )
                )
            )
            if review_issues:
                raise LocalizationError(
                    "semantic localization review failed: "
                    + "; ".join(review_issues)
                )
            localized_scenes = candidate
            break
        except _NonRetryableLocalizationError:
            raise
        except LocalizationError as exc:
            last_error = exc
            if attempt >= MAX_LOCALIZATION_ATTEMPTS:
                raise
            current_prompt = _retry_prompt(prompt, exc, attempt)

    if localized_scenes is None:
        raise last_error or LocalizationError(
            "documentary localization failed without a result"
        )

    source_master_fingerprint = master_plan_fingerprint(project)
    source_transcript_fingerprints = {
        source_id: _transcript_fingerprint(transcript)
        for source_id, transcript in transcripts.items()
    }

    latest_project, latest_transcripts = _load_localization_context(
        project_id,
        root=root,
    )
    if master_plan_fingerprint(latest_project) != source_master_fingerprint:
        raise LocalizationError(
            "documentary master timeline changed while localization was running"
        )
    latest_transcript_fingerprints = {
        source_id: _transcript_fingerprint(transcript)
        for source_id, transcript in latest_transcripts.items()
    }
    if latest_transcript_fingerprints != source_transcript_fingerprints:
        raise LocalizationError(
            "documentary transcript evidence changed while localization was running"
        )

    plan = LocalizationPlan(
        source_language=source_language,
        target_language=target_language,
        master_plan_fingerprint=source_master_fingerprint,
        transcript_fingerprints=source_transcript_fingerprints,
        semantic_reviewed=True,
        semantic_review_version=CURRENT_LOCALIZATION_REVIEW_VERSION,
        reviewed_content_fingerprint=_localized_scenes_fingerprint(
            localized_scenes
        ),
        scenes=localized_scenes,
    )
    _atomic_write_json(
        localization_plan_path(project_id, target_language, root),
        plan.model_dump(mode="json"),
    )
    return plan


def load_localization_plan(
    project_id: str,
    target_language: str,
    *,
    root: str | os.PathLike | None = None,
) -> LocalizationPlan:
    path = localization_plan_path(project_id, target_language, root)
    if not path.is_file():
        raise FileNotFoundError(
            f"documentary localization plan not found: "
            f"{project_id}/{_normalize_language(target_language)}"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        plan = LocalizationPlan.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise LocalizationError(
            f"invalid documentary localization plan: {project_id}"
        ) from exc

    requested_language = _normalize_language(target_language)
    if plan.target_language != requested_language:
        raise LocalizationError(
            "documentary localization target language does not match requested language"
        )
    if not plan.semantic_reviewed:
        raise LocalizationError(
            "documentary localization has not passed semantic review"
        )
    if plan.semantic_review_version != CURRENT_LOCALIZATION_REVIEW_VERSION:
        raise LocalizationError(
            "documentary localization review policy is stale"
        )
    if (
        plan.reviewed_content_fingerprint
        != _localized_scenes_fingerprint(plan.scenes)
    ):
        raise LocalizationError(
            "documentary localization changed after semantic review"
        )

    project, transcripts = _load_localization_context(project_id, root=root)
    if plan.source_language != _normalize_language(project.master_language):
        raise LocalizationError(
            "documentary localization is stale; master language changed"
        )
    if plan.master_plan_fingerprint != master_plan_fingerprint(project):
        raise LocalizationError(
            "documentary localization is stale; master timeline changed"
        )

    current_transcript_fingerprints = {
        source_id: _transcript_fingerprint(transcript)
        for source_id, transcript in transcripts.items()
    }
    if plan.transcript_fingerprints != current_transcript_fingerprints:
        raise LocalizationError(
            "documentary localization is stale; transcript evidence changed"
        )

    source_scenes = _source_payload(project, transcripts)
    _validate_candidate_against_source(plan.scenes, source_scenes)
    return plan
