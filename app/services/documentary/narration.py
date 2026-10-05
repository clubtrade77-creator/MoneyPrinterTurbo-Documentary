from __future__ import annotations

import json
import os
from typing import Callable

from app.models.documentary import AudioMode, SceneType
from app.services import llm as llm_service
from app.services.documentary.clip_selector import story_plan_fingerprint
from app.services.documentary.project import load_project, save_project
from app.services.documentary.story_planner import load_story_plan
from app.services.documentary.transcription import load_source_transcript

MAX_NARRATION_PROMPT_CHARS = 120_000
MAX_NARRATION_ATTEMPTS = 3


class NarrationWriterError(RuntimeError):
    """Raised when grounded documentary narration cannot be produced safely."""


class _NonRetryableNarrationWriterError(NarrationWriterError):
    pass


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


def _scene_context(project_id: str, root: str | os.PathLike | None = None):
    project = load_project(project_id, root)
    story_plan = load_story_plan(project_id, root=root)
    fingerprint = story_plan_fingerprint(story_plan)
    if project.plan.story_plan_fingerprint != fingerprint:
        raise NarrationWriterError(
            "documentary timeline is stale; rebuild clips before writing narration"
        )

    beats = {beat.id: beat for beat in story_plan.beats}
    transcripts = {}
    required = []

    for scene in project.plan.scenes:
        beat = beats.get(scene.story_beat_id)
        if beat is None:
            raise NarrationWriterError(
                f"scene references unknown story beat: {scene.id}"
            )

        if beat.original_audio_priority and scene.audio_mode == AudioMode.original:
            continue

        transcript = transcripts.get(scene.source_id)
        if transcript is None:
            transcript = load_source_transcript(
                project_id,
                scene.source_id,
                root=root,
            )
            transcripts[scene.source_id] = transcript

        segments_by_id = {segment.id: segment for segment in transcript.segments}
        evidence = []
        for segment_id in scene.transcript_segment_ids:
            segment = segments_by_id.get(segment_id)
            if segment is None:
                raise NarrationWriterError(
                    f"scene {scene.id} references missing transcript segment "
                    f"{segment_id}"
                )
            evidence.append(
                {
                    "segment_id": segment.id,
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                    "text": segment.text,
                }
            )

        required.append(
            {
                "scene_id": scene.id,
                "purpose": getattr(scene.purpose, "value", str(scene.purpose)),
                "beat_title": beat.title,
                "beat_summary": beat.summary,
                "narration_goal": beat.narration_goal,
                "evidence": evidence,
            }
        )

    return project, story_plan, required


def build_narration_prompt(*, project_title: str, scenes: list[dict]) -> str:
    prompt = f"""
You are writing factual documentary narration.

PROJECT:
{project_title}

TASK:
Write concise voice-over narration only for the scenes supplied below.

RULES:
- Use ONLY facts stated in the supplied transcript evidence.
- Do not invent motives, identities, incident types, danger, emotions, causes,
  consequences, dates, locations, or outcomes.
- Do not claim that one event caused another unless the evidence explicitly says so.
- Preserve uncertainty and vague source wording instead of filling gaps.
- Do not repeat source dialogue word-for-word unless a brief phrase is necessary.
- Narration should complement the footage, not overwrite useful original dialogue.
- Keep each scene's narration short enough for its source time range.
- Return every scene exactly once and preserve scene_id exactly.
- Return JSON only.

OUTPUT:
{{
  "scenes": [
    {{
      "scene_id": "same scene id",
      "narration_text": "grounded narration"
    }}
  ]
}}

SCENES:
{json.dumps(scenes, ensure_ascii=False)}
""".strip()
    if len(prompt) > MAX_NARRATION_PROMPT_CHARS:
        raise NarrationWriterError(
            "narration evidence is too large for one safe prompt"
        )
    return prompt


def build_narration_review_prompt(*, scenes: list[dict], candidate: list[dict]) -> str:
    prompt = (
        "Review the documentary narration for factual grounding. Reject any added "
        "specificity, causality, incident labels, motive, emotion, danger, consequence, "
        "or factual context not present in the transcript evidence. Exact or faithful "
        "neutral paraphrases are allowed. Return JSON only as "
        '{"supported": true, "issues": []} or '
        '{"supported": false, "issues": ["specific issue"]}.\n\n'
        + json.dumps(
            {"source_scenes": scenes, "candidate": candidate},
            ensure_ascii=False,
        )
    )
    if len(prompt) > MAX_NARRATION_PROMPT_CHARS:
        raise NarrationWriterError(
            "narration review evidence is too large for one safe prompt"
        )
    return prompt


def _parse_candidate(response_text: str, expected_scene_ids: list[str]) -> list[dict]:
    raw = _strip_code_fence(response_text)
    if not raw:
        raise NarrationWriterError("narration writer returned an empty response")
    if raw.startswith("Error:"):
        raise _NonRetryableNarrationWriterError(
            raw.removeprefix("Error:").strip() or raw
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NarrationWriterError("narration writer did not return valid JSON") from exc

    if not isinstance(payload, dict) or set(payload) != {"scenes"}:
        raise NarrationWriterError(
            "narration writer response must contain only a scenes array"
        )
    scenes = payload["scenes"]
    if not isinstance(scenes, list):
        raise NarrationWriterError("narration writer scenes must be an array")

    parsed = []
    for item in scenes:
        if not isinstance(item, dict) or set(item) != {"scene_id", "narration_text"}:
            raise NarrationWriterError("invalid narration scene object")
        scene_id = str(item["scene_id"] or "").strip()
        narration_text = str(item["narration_text"] or "").strip()
        if not scene_id or not narration_text:
            raise NarrationWriterError("narration scene requires id and text")
        if len(narration_text) > 8000:
            raise NarrationWriterError("narration scene text is too long")
        parsed.append(
            {"scene_id": scene_id, "narration_text": narration_text}
        )

    actual_ids = [item["scene_id"] for item in parsed]
    if actual_ids != expected_scene_ids:
        raise NarrationWriterError(
            "narration writer changed scene order or scene ids"
        )
    return parsed


def _parse_review(response_text: str) -> list[str]:
    raw = _strip_code_fence(response_text)
    if not raw:
        raise _NonRetryableNarrationWriterError(
            "narration reviewer returned an empty response"
        )
    if raw.startswith("Error:"):
        raise _NonRetryableNarrationWriterError(
            raw.removeprefix("Error:").strip() or raw
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise NarrationWriterError("narration reviewer did not return valid JSON") from exc
    if not isinstance(payload, dict):
        raise NarrationWriterError("invalid narration review response")
    supported = payload.get("supported")
    issues = payload.get("issues")
    if not isinstance(supported, bool) or not isinstance(issues, list):
        raise NarrationWriterError("invalid narration review response")
    normalized = [str(issue).strip() for issue in issues if str(issue).strip()]
    if supported and normalized:
        raise NarrationWriterError(
            "narration reviewer approved while reporting issues"
        )
    if not supported and not normalized:
        raise NarrationWriterError(
            "narration reviewer rejected without an issue"
        )
    return [] if supported else normalized


def _retry_prompt(base_prompt: str, error: Exception, attempt: int) -> str:
    return (
        base_prompt
        + "\n\nCORRECTION REQUIRED:\n"
        + f"Attempt {attempt} was rejected: {error}. "
        + "Return a corrected JSON object only. Remove unsupported wording rather "
          "than replacing it with dramatic synonyms."
    )


def write_narration(
    project_id: str,
    *,
    root: str | os.PathLike | None = None,
    generate_fn: Callable[[str], str] | None = None,
    review_fn: Callable[[str], str] | None = None,
):
    """Write reviewed narration for scenes that do not preserve original audio."""
    project, story_plan, scenes = _scene_context(project_id, root)

    if not scenes:
        return project

    generator = generate_fn or llm_service.generate_text
    if generate_fn is not None and review_fn is None:
        raise ValueError(
            "review_fn is required when generate_fn is supplied so custom narration "
            "cannot bypass semantic review"
        )
    reviewer = review_fn or llm_service.generate_text

    base_prompt = build_narration_prompt(
        project_title=project.title,
        scenes=scenes,
    )
    current_prompt = base_prompt
    expected_ids = [scene["scene_id"] for scene in scenes]
    candidate = None
    last_error = None

    for attempt in range(1, MAX_NARRATION_ATTEMPTS + 1):
        try:
            parsed = _parse_candidate(generator(current_prompt), expected_ids)
            issues = _parse_review(
                reviewer(
                    build_narration_review_prompt(
                        scenes=scenes,
                        candidate=parsed,
                    )
                )
            )
            if issues:
                raise NarrationWriterError(
                    "semantic narration review failed: " + "; ".join(issues)
                )
            candidate = parsed
            break
        except _NonRetryableNarrationWriterError:
            raise
        except NarrationWriterError as exc:
            last_error = exc
            if attempt >= MAX_NARRATION_ATTEMPTS:
                raise
            current_prompt = _retry_prompt(base_prompt, exc, attempt)

    if candidate is None:
        raise last_error or NarrationWriterError(
            "narration writer failed without a result"
        )

    latest = load_project(project_id, root)
    if latest.revision != project.revision:
        raise NarrationWriterError(
            "documentary project changed while narration was being written"
        )
    if latest.plan.story_plan_fingerprint != story_plan_fingerprint(story_plan):
        raise NarrationWriterError(
            "documentary story timeline changed while narration was being written"
        )

    by_id = {item["scene_id"]: item["narration_text"] for item in candidate}
    for scene in latest.plan.scenes:
        if scene.id not in by_id:
            continue
        scene.narration_text = by_id[scene.id]
        scene.scene_type = SceneType.narration_over_source
        scene.audio_mode = AudioMode.narration

    save_project(latest, root)
    return latest
