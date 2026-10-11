from __future__ import annotations

import json
import math
import os
import re
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


class _NarrationSchemaError(NarrationWriterError):
    """Raised when generated narration JSON does not match the expected schema."""


class _NarrationShapeError(_NarrationSchemaError):
    """Raised for recognizable response-shape drift that should not be retried."""


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


def scene_requires_narration(scene, beat=None) -> bool:
    if scene.audio_mode in {AudioMode.narration, AudioMode.mixed}:
        return True
    return (
        scene.audio_mode == AudioMode.muted
        and scene.scene_type == SceneType.narration_over_source
        and beat is not None
        and not beat.original_audio_priority
    )


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

        if not scene_requires_narration(scene, beat):
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
                "available_seconds": max(
                    0.0,
                    float(scene.source_end or 0) - float(scene.source_start or 0),
                ),
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
- The top-level JSON object MUST contain exactly one field named "scenes".
- Do not return top-level "title", "language", "narration", "result", "items",
  "metadata", or any explanatory fields.
- Each scene object must contain exactly "scene_id" and "narration_text".
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


def _extract_narration_items(payload) -> list | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None

    scenes = payload.get("scenes")
    if isinstance(scenes, list):
        return scenes

    for alias in ("narration", "items", "results", "result", "output"):
        value = payload.get(alias)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("scenes"), list):
            return value["scenes"]

    if any(
        key in payload
        for key in ("scene_id", "sceneId", "scene", "id")
    ) and any(
        key in payload
        for key in (
            "narration_text",
            "narration",
            "text",
            "voiceover",
            "voice_over",
            "content",
            "script",
        )
    ):
        return [payload]
    return None


def _repair_narration_items(
    items: list,
    expected_scene_ids: list[str],
) -> list[dict]:
    if len(items) != len(expected_scene_ids):
        raise _NarrationShapeError(
            "narration writer changed the number of scenes"
        )

    repaired = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise _NarrationShapeError("invalid narration scene object")

        scene_id = ""
        for key in ("scene_id", "sceneId", "scene", "id"):
            value = str(item.get(key) or "").strip()
            if value:
                scene_id = value
                break
        if not scene_id:
            # Ordered fallback is safe because the prompt requires exact scene order;
            # semantic review still validates the generated text against each scene.
            scene_id = expected_scene_ids[index]

        narration_text = ""
        for key in (
            "narration_text",
            "narration",
            "text",
            "voiceover",
            "voice_over",
            "content",
            "script",
        ):
            value = str(item.get(key) or "").strip()
            if value:
                narration_text = value
                break

        if not narration_text:
            raise _NarrationShapeError("narration scene requires text")
        if len(narration_text) > 8000:
            raise _NarrationSchemaError("narration scene text is too long")

        repaired.append(
            {
                "scene_id": scene_id,
                "narration_text": narration_text,
            }
        )

    actual_ids = [item["scene_id"] for item in repaired]
    if actual_ids != expected_scene_ids:
        raise _NarrationShapeError(
            "narration writer changed scene order or scene ids"
        )
    return repaired


def _parse_candidate(response_text: str, expected_scene_ids: list[str]) -> list[dict]:
    raw = _strip_code_fence(response_text)
    if not raw:
        raise _NarrationSchemaError("narration writer returned an empty response")
    if raw.startswith("Error:"):
        raise _NonRetryableNarrationWriterError(
            raw.removeprefix("Error:").strip() or raw
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _NarrationSchemaError(
            "narration writer did not return valid JSON"
        ) from exc

    items = _extract_narration_items(payload)
    if items is None:
        raise _NarrationShapeError(
            "narration writer response does not contain a usable scenes array"
        )
    return _repair_narration_items(items, expected_scene_ids)


def _truncate_to_word_budget(text: str, word_budget: int) -> str:
    words = re.findall(r"\S+", str(text or "").strip())
    if not words:
        return ""
    if len(words) <= word_budget:
        return " ".join(words)
    clipped = " ".join(words[:word_budget]).rstrip(" ,;:-")
    if clipped and clipped[-1] not in ".!?…":
        clipped += "."
    return clipped


def _deterministic_narration(scenes: list[dict]) -> list[dict]:
    candidate = []
    for scene in scenes:
        available = max(1.0, float(scene.get("available_seconds") or 0))
        # Conservative English-equivalent speaking budget. TTS fitting can safely
        # absorb small timing variance later, but fallback should already be short.
        word_budget = max(4, math.floor(available * 1.7))

        evidence_texts = [
            str(item.get("text") or "").strip()
            for item in scene.get("evidence", [])
            if isinstance(item, dict) and str(item.get("text") or "").strip()
        ]
        grounded_text = str(scene.get("beat_summary") or "").strip()
        if not grounded_text:
            grounded_text = " ".join(evidence_texts)
        if not grounded_text and evidence_texts:
            grounded_text = evidence_texts[0]

        narration_text = _truncate_to_word_budget(
            grounded_text,
            word_budget,
        )
        if not narration_text:
            raise NarrationWriterError(
                f"scene {scene.get('scene_id')} has no grounded text for fallback"
            )
        candidate.append(
            {
                "scene_id": scene["scene_id"],
                "narration_text": narration_text,
            }
        )
    return candidate



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
        except _NarrationShapeError:
            # Recognizable structure drift is not worth retrying. Build concise
            # narration from already-reviewed Story Plan text / transcript evidence.
            candidate = _deterministic_narration(scenes)
            break
        except _NonRetryableNarrationWriterError:
            raise
        except _NarrationSchemaError as exc:
            last_error = exc
            if attempt >= MAX_NARRATION_ATTEMPTS:
                candidate = _deterministic_narration(scenes)
                break
            current_prompt = _retry_prompt(base_prompt, exc, attempt)
        except NarrationWriterError as exc:
            # Semantic grounding failures are substantive and must never be hidden
            # behind deterministic fallback.
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
        if scene.audio_mode == AudioMode.muted:
            scene.audio_mode = AudioMode.narration

    save_project(latest, root)
    return latest
