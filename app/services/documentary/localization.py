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
COMPACT_LOCALIZATION_THRESHOLD_CHARS = 4_000
COMPACT_LOCALIZATION_BATCH_CHARS = 3_500
COMPACT_LOCALIZATION_BATCH_ITEMS = 24
CURRENT_LOCALIZATION_REVIEW_VERSION = 1


class LocalizationError(RuntimeError):
    """Raised when a faithful documentary localization cannot be produced safely."""


class _NonRetryableLocalizationError(LocalizationError):
    """Raised for provider/runtime failures that retries cannot repair."""


class _LocalizationSchemaError(LocalizationError):
    """Raised when generated localization JSON has a repairable schema drift."""


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
The top-level JSON object must contain one "scenes" array. Do not wrap it in
"translation", "translations", "result", "output", or add explanatory prose.
Inside each scene, preserve the exact field names shown below.
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


def _decode_localization_json(response_text: str):
    raw = _strip_code_fence(response_text)
    if not raw:
        raise _LocalizationSchemaError(
            "documentary localizer returned an empty response"
        )
    if raw.startswith("Error:"):
        raise _NonRetryableLocalizationError(
            raw.removeprefix("Error:").strip() or raw
        )

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Providers sometimes surround an otherwise valid JSON value with a short
    # sentence. Extract the first complete JSON object/array deterministically.
    decoder = json.JSONDecoder()
    for position, char in enumerate(raw):
        if char not in "[{":
            continue
        try:
            payload, _ = decoder.raw_decode(raw[position:])
        except json.JSONDecodeError:
            continue
        return payload

    raise _LocalizationSchemaError(
        "documentary localizer did not return valid JSON"
    )


def _localization_text_units(source_scenes: list[dict]) -> list[dict]:
    units = []
    unit_index = 0
    for scene_index, scene in enumerate(source_scenes):
        narration = str(scene.get("narration_text") or "").strip()
        if narration:
            units.append(
                {
                    "i": unit_index,
                    "scene_index": scene_index,
                    "kind": "narration",
                    "text": narration,
                }
            )
            unit_index += 1

        on_screen = str(scene.get("on_screen_text") or "").strip()
        if on_screen:
            units.append(
                {
                    "i": unit_index,
                    "scene_index": scene_index,
                    "kind": "on_screen",
                    "text": on_screen,
                }
            )
            unit_index += 1

        for subtitle_index, subtitle in enumerate(scene.get("subtitle_segments", [])):
            text = str(subtitle.get("text") or "").strip()
            if not text:
                continue
            units.append(
                {
                    "i": unit_index,
                    "scene_index": scene_index,
                    "kind": "subtitle",
                    "subtitle_index": subtitle_index,
                    "text": text,
                }
            )
            unit_index += 1
    return units


def _should_use_compact_localization(source_scenes: list[dict]) -> bool:
    units = _localization_text_units(source_scenes)
    return (
        sum(len(unit["text"]) for unit in units)
        > COMPACT_LOCALIZATION_THRESHOLD_CHARS
        or len(units) > COMPACT_LOCALIZATION_BATCH_ITEMS
    )


def _compact_localization_batches(units: list[dict]) -> list[list[dict]]:
    batches = []
    current = []
    current_chars = 0
    for unit in units:
        unit_chars = len(unit["text"])
        if current and (
            len(current) >= COMPACT_LOCALIZATION_BATCH_ITEMS
            or current_chars + unit_chars > COMPACT_LOCALIZATION_BATCH_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(unit)
        current_chars += unit_chars
    if current:
        batches.append(current)
    return batches


def _build_compact_localization_prompt(
    *,
    source_language: str,
    target_language: str,
    batch: list[dict],
) -> str:
    payload = [
        {"i": unit["i"], "text": unit["text"]}
        for unit in batch
    ]
    return f"""
You are translating a compact batch of factual documentary text from
{source_language} to {target_language}.

Translate ONLY the text. Preserve names, numbers, dates, attribution, uncertainty,
and factual meaning. Do not add explanations or facts. Keep translations concise.

Return exactly one JSON object:
{{"items":[{{"i":0,"text":"translated text"}}]}}

Rules:
- Return every input item exactly once and in the same order.
- Preserve each integer "i" exactly.
- "text" must contain the translation only.
- No markdown and no explanatory prose.

INPUT ITEMS:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _parse_compact_localization_response(
    response_text: str,
    batch: list[dict],
) -> dict[int, str]:
    payload = _decode_localization_json(response_text)
    items = None
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("items", "translations", "results", "output"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
            if isinstance(value, dict) and isinstance(value.get("items"), list):
                items = value["items"]
                break
    if not isinstance(items, list):
        raise _LocalizationSchemaError(
            "compact documentary localization did not return an items array"
        )

    expected_ids = {unit["i"] for unit in batch}
    translated: dict[int, str] = {}

    # A plain string array has no stable ids. It is safe only when the model
    # returned exactly one string for every requested item in order.
    if items and all(isinstance(item, str) for item in items):
        if len(items) != len(batch):
            raise _LocalizationSchemaError(
                "compact documentary localization returned unidentifiable partial items"
            )
        for expected, item in zip(batch, items):
            text = item.strip()
            if not text:
                continue
            translated[expected["i"]] = text
        return translated

    for item in items:
        if not isinstance(item, dict):
            raise _LocalizationSchemaError(
                "compact documentary localization item must be text or object"
            )

        raw_index = item.get("i", item.get("index", item.get("id")))
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise _LocalizationSchemaError(
                "compact documentary localization item has no valid index"
            ) from exc

        if index not in expected_ids:
            raise _LocalizationSchemaError(
                "compact documentary localization returned an unexpected item id"
            )
        if index in translated:
            raise _LocalizationSchemaError(
                "compact documentary localization returned a duplicate item id"
            )

        text = str(
            item.get("text")
            or item.get("translation")
            or item.get("translated_text")
            or item.get("content")
            or ""
        ).strip()
        if not text:
            # Treat empty text as missing so only this item is retried.
            continue
        translated[index] = text

    return translated


def _translate_compact_batch(
    *,
    source_language: str,
    target_language: str,
    batch: list[dict],
    generator: Callable[[str], str],
) -> dict[int, str]:
    if not batch:
        return {}

    try:
        response = generator(
            _build_compact_localization_prompt(
                source_language=source_language,
                target_language=target_language,
                batch=batch,
            )
        )
        translated = _parse_compact_localization_response(
            response,
            batch,
        )
    except _NonRetryableLocalizationError:
        raise
    except _LocalizationSchemaError:
        if len(batch) == 1:
            raise
        midpoint = max(1, len(batch) // 2)
        left = _translate_compact_batch(
            source_language=source_language,
            target_language=target_language,
            batch=batch[:midpoint],
            generator=generator,
        )
        right = _translate_compact_batch(
            source_language=source_language,
            target_language=target_language,
            batch=batch[midpoint:],
            generator=generator,
        )
        return {**left, **right}

    missing = [
        unit
        for unit in batch
        if unit["i"] not in translated
    ]
    if not missing:
        return translated

    # Keep every valid translation already received and request only missing ids.
    if len(missing) < len(batch):
        translated.update(
            _translate_compact_batch(
                source_language=source_language,
                target_language=target_language,
                batch=missing,
                generator=generator,
            )
        )
        return translated

    # Nothing in this response was usable. Split the batch to make the model's
    # output contract easier. A one-item failure is surfaced rather than looping.
    if len(batch) == 1:
        raise _LocalizationSchemaError(
            "compact documentary localization returned no usable translation"
        )

    midpoint = max(1, len(batch) // 2)
    left = _translate_compact_batch(
        source_language=source_language,
        target_language=target_language,
        batch=batch[:midpoint],
        generator=generator,
    )
    right = _translate_compact_batch(
        source_language=source_language,
        target_language=target_language,
        batch=batch[midpoint:],
        generator=generator,
    )
    return {**left, **right}



def _assemble_compact_localization(
    source_scenes: list[dict],
    units: list[dict],
    translations: dict[int, str],
) -> list[LocalizedSceneText]:
    repaired = []
    for scene in source_scenes:
        repaired.append(
            {
                "scene_id": scene["scene_id"],
                "narration_text": "",
                "on_screen_text": "",
                "subtitle_segments": [
                    {
                        "source_id": subtitle["source_id"],
                        "segment_id": subtitle["segment_id"],
                        "text": "",
                    }
                    for subtitle in scene["subtitle_segments"]
                ],
            }
        )

    for unit in units:
        translated = translations.get(unit["i"], "").strip()
        if not translated:
            raise _LocalizationSchemaError(
                "compact documentary localization is missing translated text"
            )
        scene = repaired[unit["scene_index"]]
        if unit["kind"] == "narration":
            scene["narration_text"] = translated
        elif unit["kind"] == "on_screen":
            scene["on_screen_text"] = translated
        else:
            scene["subtitle_segments"][unit["subtitle_index"]]["text"] = translated

    return [
        LocalizedSceneText.model_validate(scene)
        for scene in repaired
    ]


def _localize_compact(
    *,
    source_language: str,
    target_language: str,
    source_scenes: list[dict],
    generator: Callable[[str], str],
) -> list[LocalizedSceneText]:
    units = _localization_text_units(source_scenes)
    if not units:
        return [
            LocalizedSceneText.model_validate(
                {
                    "scene_id": scene["scene_id"],
                    "narration_text": "",
                    "on_screen_text": "",
                    "subtitle_segments": [],
                }
            )
            for scene in source_scenes
        ]

    translations: dict[int, str] = {}
    for batch in _compact_localization_batches(units):
        translations.update(
            _translate_compact_batch(
                source_language=source_language,
                target_language=target_language,
                batch=batch,
                generator=generator,
            )
        )
    return _assemble_compact_localization(
        source_scenes,
        units,
        translations,
    )


def _extract_localization_scene_items(payload) -> list | None:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None

    if isinstance(payload.get("scenes"), list):
        return payload["scenes"]

    for key in (
        "translation",
        "translations",
        "localized",
        "localization",
        "result",
        "results",
        "output",
        "data",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            if isinstance(value.get("scenes"), list):
                return value["scenes"]
            for nested_key in ("translations", "items", "results"):
                if isinstance(value.get(nested_key), list):
                    return value[nested_key]

    # Last-resort structural repair: accept exactly one list-valued field when
    # its items look like scene objects. This ignores wrapper metadata but never
    # invents or translates text.
    list_values = [
        value
        for value in payload.values()
        if isinstance(value, list)
        and value
        and all(isinstance(item, dict) for item in value)
    ]
    if len(list_values) == 1:
        return list_values[0]
    return None


def _repair_subtitle_segments(items, source_segments: list[dict]) -> list[dict]:
    if items is None:
        items = []
    if not isinstance(items, list):
        raise _LocalizationSchemaError(
            "documentary localized subtitle segments must be an array"
        )
    if len(items) != len(source_segments):
        raise _LocalizationSchemaError(
            "documentary localization changed the number of subtitle segments"
        )

    repaired = []
    for index, (item, source) in enumerate(zip(items, source_segments)):
        if not isinstance(item, dict):
            raise _LocalizationSchemaError(
                "documentary localized subtitle item must be an object"
            )

        source_id = str(
            item.get("source_id")
            or item.get("sourceId")
            or source["source_id"]
        ).strip()
        raw_segment_id = item.get("segment_id", item.get("segmentId"))
        segment_id = source["segment_id"] if raw_segment_id is None else raw_segment_id
        text = str(
            item.get("text")
            or item.get("translation")
            or item.get("translated_text")
            or item.get("content")
            or ""
        ).strip()

        if not text and str(source.get("text") or "").strip():
            raise _LocalizationSchemaError(
                f"localized subtitle segment {index} is missing translated text"
            )

        repaired.append(
            {
                "source_id": source_id,
                "segment_id": segment_id,
                "text": text,
            }
        )
    return repaired


def _repair_localization_scene_items(
    items: list,
    source_scenes: list[dict],
) -> list[dict]:
    if len(items) != len(source_scenes):
        raise _LocalizationSchemaError(
            "documentary localization must contain every source scene exactly once"
        )

    repaired = []
    for index, (item, source) in enumerate(zip(items, source_scenes)):
        if not isinstance(item, dict):
            raise _LocalizationSchemaError(
                "documentary localized scene must be an object"
            )

        scene_id = str(
            item.get("scene_id")
            or item.get("sceneId")
            or item.get("id")
            or source["scene_id"]
        ).strip()

        narration_text = str(
            item.get("narration_text")
            if item.get("narration_text") is not None
            else item.get("narration")
            if item.get("narration") is not None
            else item.get("voiceover")
            if item.get("voiceover") is not None
            else item.get("voice_over")
            if item.get("voice_over") is not None
            else ""
        ).strip()

        on_screen_text = str(
            item.get("on_screen_text")
            if item.get("on_screen_text") is not None
            else item.get("onScreenText")
            if item.get("onScreenText") is not None
            else item.get("onscreen_text")
            if item.get("onscreen_text") is not None
            else item.get("title")
            if item.get("title") is not None
            else ""
        ).strip()

        subtitle_items = None
        for key in (
            "subtitle_segments",
            "subtitles",
            "subtitleSegments",
            "captions",
        ):
            if key in item:
                subtitle_items = item[key]
                break

        repaired.append(
            {
                "scene_id": scene_id,
                "narration_text": narration_text,
                "on_screen_text": on_screen_text,
                "subtitle_segments": _repair_subtitle_segments(
                    subtitle_items,
                    source["subtitle_segments"],
                ),
            }
        )
    return repaired


def _parse_candidate(
    response_text: str,
    source_scenes: list[dict],
) -> list[LocalizedSceneText]:
    payload = _decode_localization_json(response_text)

    items = _extract_localization_scene_items(payload)
    if items is None:
        raise _LocalizationSchemaError(
            "documentary localizer response does not contain a usable scenes array"
        )

    repaired = _repair_localization_scene_items(items, source_scenes)
    try:
        return [
            LocalizedSceneText.model_validate(item)
            for item in repaired
        ]
    except ValueError as exc:
        raise _LocalizationSchemaError(
            f"invalid documentary localization: {exc}"
        ) from exc



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
    if "supported" not in payload or "issues" not in payload:
        raise LocalizationError(
            "documentary localization reviewer response is missing verdict fields"
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

    if _should_use_compact_localization(source_scenes):
        localized_scenes = _localize_compact(
            source_language=source_language,
            target_language=target_language,
            source_scenes=source_scenes,
            generator=generator,
        )
        _validate_candidate_against_source(localized_scenes, source_scenes)
        review_issues = _parse_review(
            reviewer(
                build_localization_review_prompt(
                    source_language=source_language,
                    target_language=target_language,
                    source_scenes=source_scenes,
                    localized_scenes=localized_scenes,
                )
            )
        )
        if review_issues:
            raise LocalizationError(
                "semantic localization review failed: "
                + "; ".join(review_issues)
            )

    for attempt in range(1, MAX_LOCALIZATION_ATTEMPTS + 1):
        if localized_scenes is not None:
            break
        try:
            candidate = _parse_candidate(
                generator(current_prompt),
                source_scenes,
            )
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
        except _LocalizationSchemaError as exc:
            last_error = exc
            localized_scenes = _localize_compact(
                source_language=source_language,
                target_language=target_language,
                source_scenes=source_scenes,
                generator=generator,
            )
            _validate_candidate_against_source(localized_scenes, source_scenes)
            review_issues = _parse_review(
                reviewer(
                    build_localization_review_prompt(
                        source_language=source_language,
                        target_language=target_language,
                        source_scenes=source_scenes,
                        localized_scenes=localized_scenes,
                    )
                )
            )
            if review_issues:
                raise LocalizationError(
                    "semantic localization review failed: "
                    + "; ".join(review_issues)
                )
            break
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
