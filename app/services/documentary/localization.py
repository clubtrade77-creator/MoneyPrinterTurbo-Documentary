from __future__ import annotations

import hashlib
import json
import os
import re
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
MAX_LOCALIZATION_REPAIR_PASSES = 2
MAX_LOCALIZATION_PROMPT_CHARS = 120_000
MAX_LOCALIZATION_REVIEW_PROMPT_CHARS = 180_000
COMPACT_LOCALIZATION_THRESHOLD_CHARS = 4_000
COMPACT_LOCALIZATION_BATCH_CHARS = 3_500
COMPACT_LOCALIZATION_BATCH_ITEMS = 24
COMPACT_REVIEW_THRESHOLD_CHARS = 5_000
COMPACT_REVIEW_BATCH_CHARS = 4_000
COMPACT_REVIEW_BATCH_ITEMS = 24
CURRENT_LOCALIZATION_REVIEW_VERSION = 1

LocalizationProgressCallback = Callable[[float], None]


def _report_localization_progress(
    callback: LocalizationProgressCallback | None,
    fraction: float,
) -> None:
    if callback is not None:
        callback(max(0.0, min(1.0, float(fraction))))


class LocalizationError(RuntimeError):
    """Raised when a faithful documentary localization cannot be produced safely."""


class _NonRetryableLocalizationError(LocalizationError):
    """Raised for provider/runtime failures that retries cannot repair."""


class _LocalizationSchemaError(LocalizationError):
    """Raised when generated localization JSON has a repairable schema drift."""


class _LocalizationReviewSchemaError(LocalizationError):
    """Raised when a reviewer verdict cannot be parsed safely."""


class _LocalizationSemanticError(LocalizationError):
    """Raised when targeted semantic repair still leaves review failures."""


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
    progress: LocalizationProgressCallback | None = None,
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
    batches = _compact_localization_batches(units)
    _report_localization_progress(progress, 0.0)
    for index, batch in enumerate(batches, start=1):
        translations.update(
            _translate_compact_batch(
                source_language=source_language,
                target_language=target_language,
                batch=batch,
                generator=generator,
            )
        )
        _report_localization_progress(
            progress,
            index / max(1, len(batches)),
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


def _normalize_review_boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {
        "true",
        "yes",
        "y",
        "pass",
        "passed",
        "approved",
        "supported",
        "faithful",
        "ok",
        "valid",
    }:
        return True
    if text in {
        "false",
        "no",
        "n",
        "fail",
        "failed",
        "rejected",
        "unsupported",
        "unfaithful",
        "invalid",
    }:
        return False
    return None


def _normalize_review_issues(value) -> list[str] | None:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [
            str(issue).strip()
            for issue in value
            if str(issue).strip()
        ]
    return None


def _review_from_mapping(payload: dict) -> list[str]:
    supported = None
    for key in (
        "supported",
        "approved",
        "faithful",
        "valid",
        "ok",
        "pass",
        "passed",
        "verdict",
        "status",
    ):
        if key in payload:
            supported = _normalize_review_boolean(payload[key])
            if supported is not None:
                break

    issues = None
    for key in (
        "issues",
        "errors",
        "problems",
        "concerns",
        "reasons",
        "reason",
    ):
        if key in payload:
            issues = _normalize_review_issues(payload[key])
            if issues is not None:
                break

    if supported is None and issues is not None:
        supported = not bool(issues)

    if supported is None:
        raise _LocalizationReviewSchemaError(
            "documentary localization reviewer response is missing a recognizable verdict"
        )
    if issues is None:
        issues = []

    if supported and issues:
        raise _LocalizationReviewSchemaError(
            "documentary localization reviewer approved while reporting issues"
        )
    if not supported and not issues:
        notes = str(
            payload.get("notes")
            or payload.get("explanation")
            or payload.get("message")
            or ""
        ).strip()
        issues = [notes or "reviewer rejected localization"]
    return [] if supported else issues


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
        payload = _decode_localization_json(raw)
    except _LocalizationSchemaError:
        payload = None

    if isinstance(payload, dict):
        return _review_from_mapping(payload)
    if isinstance(payload, list):
        issues = _normalize_review_issues(payload)
        if issues is None:
            raise _LocalizationReviewSchemaError(
                "documentary localization reviewer returned an invalid issue list"
            )
        return issues

    normalized = re.sub(r"\s+", " ", raw).strip()
    verdict = _normalize_review_boolean(normalized.rstrip(".!"))
    if verdict is True:
        return []
    if verdict is False:
        return ["reviewer rejected localization"]

    verdict_match = re.search(
        r"\b(?:supported|approved|faithful|valid|verdict|status)\s*[:=]\s*"
        r"(true|false|yes|no|pass(?:ed)?|approved|rejected|supported|unsupported|ok|fail(?:ed)?)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if verdict_match:
        verdict = _normalize_review_boolean(verdict_match.group(1))
        if verdict is True:
            return []
        if verdict is False:
            reason_match = re.search(
                r"(?:reason|issue|problem|explanation)\s*[:=]\s*(.+)$",
                normalized,
                flags=re.IGNORECASE,
            )
            reason = (
                reason_match.group(1).strip()
                if reason_match
                else "reviewer rejected localization"
            )
            return [reason]

    upper = normalized.upper()
    for prefix in ("PASS", "APPROVED", "SUPPORTED", "FAITHFUL", "OK"):
        if upper == prefix or upper.startswith(prefix + ":") or upper.startswith(prefix + " -"):
            return []
    for prefix in ("FAIL", "FAILED", "REJECTED", "UNSUPPORTED"):
        if upper == prefix or upper.startswith(prefix + ":") or upper.startswith(prefix + " -"):
            reason = normalized[len(prefix):].lstrip(" :-").strip()
            return [reason or "reviewer rejected localization"]

    raise _LocalizationReviewSchemaError(
        "documentary localization reviewer did not return a recognizable verdict"
    )


def _localization_review_units(
    source_scenes: list[dict],
    localized_scenes: list[LocalizedSceneText],
) -> list[dict]:
    units = []
    unit_index = 0
    for source_scene, localized_scene in zip(source_scenes, localized_scenes):
        pairs = [
            (
                str(source_scene.get("narration_text") or "").strip(),
                localized_scene.narration_text.strip(),
                "narration",
            ),
            (
                str(source_scene.get("on_screen_text") or "").strip(),
                localized_scene.on_screen_text.strip(),
                "on_screen",
            ),
        ]
        for source_text, localized_text, kind in pairs:
            if source_text or localized_text:
                units.append(
                    {
                        "i": unit_index,
                        "kind": kind,
                        "source": source_text,
                        "translation": localized_text,
                    }
                )
                unit_index += 1

        for source_subtitle, localized_subtitle in zip(
            source_scene.get("subtitle_segments", []),
            localized_scene.subtitle_segments,
        ):
            source_text = str(source_subtitle.get("text") or "").strip()
            localized_text = localized_subtitle.text.strip()
            if not source_text and not localized_text:
                continue
            units.append(
                {
                    "i": unit_index,
                    "kind": "subtitle",
                    "source": source_text,
                    "translation": localized_text,
                }
            )
            unit_index += 1
    return units


def _compact_review_batches(units: list[dict]) -> list[list[dict]]:
    batches = []
    current = []
    current_chars = 0
    for unit in units:
        unit_chars = len(unit["source"]) + len(unit["translation"])
        if current and (
            len(current) >= COMPACT_REVIEW_BATCH_ITEMS
            or current_chars + unit_chars > COMPACT_REVIEW_BATCH_CHARS
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(unit)
        current_chars += unit_chars
    if current:
        batches.append(current)
    return batches


def _build_compact_review_prompt(
    *,
    source_language: str,
    target_language: str,
    batch: list[dict],
) -> str:
    payload = [
        {
            "i": unit["i"],
            "source": unit["source"],
            "translation": unit["translation"],
        }
        for unit in batch
    ]
    return f"""
Review this SMALL batch of documentary translations from
{source_language} to {target_language}.

Check ONLY factual fidelity and translation completeness:
- reject additions or omissions of material facts;
- reject changed names, numbers, dates, attribution, uncertainty, or meaning;
- reject untranslated source-language prose that should have been translated;
- proper names and conventional identical forms may remain unchanged;
- do not critique style.

Return exactly one JSON object and nothing else:
{{"supported":true,"issues":[]}}
or
{{"supported":false,"issues":["specific issue with item i"]}}

ITEMS:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _run_compact_review_batch(
    *,
    reviewer: Callable[[str], str],
    source_language: str,
    target_language: str,
    batch: list[dict],
) -> list[str]:
    prompt = _build_compact_review_prompt(
        source_language=source_language,
        target_language=target_language,
        batch=batch,
    )
    first_response = reviewer(prompt)
    try:
        return _parse_review(first_response)
    except _LocalizationReviewSchemaError:
        pass

    # Ollama models sometimes explain a correct verdict instead of following the
    # JSON schema. Ask for the smallest possible contract before doing more work.
    pass_fail_prompt = (
        "Review the SAME documentary translation items below for factual fidelity. "
        "Return exactly PASS if every translation is faithful and complete. "
        "Otherwise return exactly FAIL: followed by one short specific reason. "
        "Do not explain anything else.\n\n"
        + json.dumps(
            [
                {
                    "i": unit["i"],
                    "source": unit["source"],
                    "translation": unit["translation"],
                }
                for unit in batch
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    second_response = reviewer(pass_fail_prompt)
    try:
        return _parse_review(second_response)
    except _LocalizationReviewSchemaError:
        pass

    # If a multi-item batch is still ambiguous, isolate the problematic model
    # interaction to smaller batches rather than failing the whole language.
    if len(batch) > 1:
        midpoint = max(1, len(batch) // 2)
        left = _run_compact_review_batch(
            reviewer=reviewer,
            source_language=source_language,
            target_language=target_language,
            batch=batch[:midpoint],
        )
        right = _run_compact_review_batch(
            reviewer=reviewer,
            source_language=source_language,
            target_language=target_language,
            batch=batch[midpoint:],
        )
        return [*left, *right]

    # One final ultra-minimal attempt for a single translation unit. We still do
    # not auto-approve an unclear review: a genuine lack of verdict remains safe.
    unit = batch[0]
    final_prompt = (
        "Decide whether this translation preserves the source meaning exactly "
        "enough for a factual documentary. Reply with only PASS or FAIL.\n"
        f"SOURCE: {unit['source']}\n"
        f"TRANSLATION: {unit['translation']}"
    )
    final_response = reviewer(final_prompt)
    try:
        return _parse_review(final_response)
    except _LocalizationReviewSchemaError as exc:
        raise _LocalizationReviewSchemaError(
            "documentary localization reviewer could not produce PASS or FAIL "
            f"for translation item {unit['i']}"
        ) from exc


def _localized_translation_map(
    source_scenes: list[dict],
    localized_scenes: list[LocalizedSceneText],
) -> tuple[list[dict], dict[int, str]]:
    units = _localization_text_units(source_scenes)
    translations: dict[int, str] = {}
    for unit in units:
        scene = localized_scenes[unit["scene_index"]]
        if unit["kind"] == "narration":
            text = scene.narration_text
        elif unit["kind"] == "on_screen":
            text = scene.on_screen_text
        else:
            text = scene.subtitle_segments[unit["subtitle_index"]].text
        translations[unit["i"]] = str(text or "").strip()
    return units, translations


def _build_compact_repair_prompt(
    *,
    source_language: str,
    target_language: str,
    batch: list[dict],
    translations: dict[int, str],
    issues: list[str],
) -> str:
    issue_text = "; ".join(str(issue) for issue in issues)
    if len(issue_text) > 4000:
        issue_text = issue_text[:4000]

    payload = [
        {
            "i": unit["i"],
            "source": unit["text"],
            "current_translation": translations.get(unit["i"], ""),
        }
        for unit in batch
    ]
    return f"""
Repair ONLY this rejected documentary translation batch from
{source_language} to {target_language}.

The semantic reviewer found one or more fidelity problems in this batch.
Translate each SOURCE again from scratch. Use current_translation only to understand
what went wrong; do not preserve wording that conflicts with the source.

Requirements:
- preserve exact item ids;
- preserve names, numbers, dates, gender, attribution, uncertainty, and meaning;
- do not omit or add material information;
- do not shift a translation to a neighboring item;
- keep every item complete even when adjacent source items overlap;
- return every input item exactly once.

REVIEW ISSUES:
{issue_text}

Return exactly:
{{"items":[{{"i":0,"text":"corrected translation"}}]}}

ITEMS:
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def _repair_compact_review_batch(
    *,
    generator: Callable[[str], str],
    source_language: str,
    target_language: str,
    batch: list[dict],
    translations: dict[int, str],
    issues: list[str],
) -> dict[int, str]:
    response = generator(
        _build_compact_repair_prompt(
            source_language=source_language,
            target_language=target_language,
            batch=batch,
            translations=translations,
            issues=issues,
        )
    )
    try:
        repaired = _parse_compact_localization_response(response, batch)
    except _LocalizationSchemaError:
        repaired = {}

    missing = [
        unit
        for unit in batch
        if unit["i"] not in repaired
    ]
    if missing:
        repaired.update(
            _translate_compact_batch(
                source_language=source_language,
                target_language=target_language,
                batch=missing,
                generator=generator,
            )
        )
    return repaired


def _review_and_repair_localization(
    *,
    generator: Callable[[str], str],
    reviewer: Callable[[str], str],
    source_language: str,
    target_language: str,
    source_scenes: list[dict],
    localized_scenes: list[LocalizedSceneText],
    progress: LocalizationProgressCallback | None = None,
) -> tuple[list[LocalizedSceneText], list[str]]:
    source_units, translations = _localized_translation_map(
        source_scenes,
        localized_scenes,
    )
    source_units_by_id = {
        unit["i"]: unit
        for unit in source_units
    }

    review_units = _localization_review_units(
        source_scenes,
        localized_scenes,
    )
    batches = _compact_review_batches(review_units)
    unresolved_issues: list[str] = []
    _report_localization_progress(progress, 0.0)

    for batch_index, review_batch in enumerate(batches, start=1):
        batch_issues = _run_compact_review_batch(
            reviewer=reviewer,
            source_language=source_language,
            target_language=target_language,
            batch=review_batch,
        )

        if batch_issues:
            batch_ids = {
                unit["i"]
                for unit in review_batch
            }
            repair_batch = [
                source_units_by_id[item_id]
                for item_id in sorted(batch_ids)
                if item_id in source_units_by_id
            ]

            for _ in range(MAX_LOCALIZATION_REPAIR_PASSES):
                repaired = _repair_compact_review_batch(
                    generator=generator,
                    source_language=source_language,
                    target_language=target_language,
                    batch=repair_batch,
                    translations=translations,
                    issues=batch_issues,
                )
                translations.update(repaired)
                localized_scenes = _assemble_compact_localization(
                    source_scenes,
                    source_units,
                    translations,
                )
                _validate_candidate_against_source(
                    localized_scenes,
                    source_scenes,
                )

                updated_review_units = {
                    unit["i"]: unit
                    for unit in _localization_review_units(
                        source_scenes,
                        localized_scenes,
                    )
                }
                updated_batch = [
                    updated_review_units[item_id]
                    for item_id in sorted(batch_ids)
                    if item_id in updated_review_units
                ]
                batch_issues = _run_compact_review_batch(
                    reviewer=reviewer,
                    source_language=source_language,
                    target_language=target_language,
                    batch=updated_batch,
                )
                if not batch_issues:
                    break

            if batch_issues:
                unresolved_issues.extend(batch_issues)

        _report_localization_progress(
            progress,
            batch_index / max(1, len(batches)),
        )

    return localized_scenes, unresolved_issues


def _run_compact_localization_review(
    *,
    reviewer: Callable[[str], str],
    source_language: str,
    target_language: str,
    source_scenes: list[dict],
    localized_scenes: list[LocalizedSceneText],
    progress: LocalizationProgressCallback | None = None,
) -> list[str]:
    units = _localization_review_units(source_scenes, localized_scenes)
    issues = []
    batches = _compact_review_batches(units)
    _report_localization_progress(progress, 0.0)
    for index, batch in enumerate(batches, start=1):
        issues.extend(
            _run_compact_review_batch(
                reviewer=reviewer,
                source_language=source_language,
                target_language=target_language,
                batch=batch,
            )
        )
        _report_localization_progress(
            progress,
            index / max(1, len(batches)),
        )
    return issues


def _run_localization_review(
    *,
    reviewer: Callable[[str], str],
    source_language: str,
    target_language: str,
    source_scenes: list[dict],
    localized_scenes: list[LocalizedSceneText],
    progress: LocalizationProgressCallback | None = None,
) -> list[str]:
    units = _localization_review_units(source_scenes, localized_scenes)
    total_chars = sum(
        len(unit["source"]) + len(unit["translation"])
        for unit in units
    )
    if (
        total_chars > COMPACT_REVIEW_THRESHOLD_CHARS
        or len(units) > COMPACT_REVIEW_BATCH_ITEMS
    ):
        return _run_compact_localization_review(
            reviewer=reviewer,
            source_language=source_language,
            target_language=target_language,
            source_scenes=source_scenes,
            localized_scenes=localized_scenes,
            progress=progress,
        )

    response = reviewer(
        build_localization_review_prompt(
            source_language=source_language,
            target_language=target_language,
            source_scenes=source_scenes,
            localized_scenes=localized_scenes,
        )
    )
    try:
        issues = _parse_review(response)
        _report_localization_progress(progress, 1.0)
        return issues
    except _LocalizationReviewSchemaError:
        # The translation itself may still be fine; review it again in a small,
        # tightly constrained batch instead of converting an unclear verdict into
        # a semantic rejection.
        return _run_compact_localization_review(
            reviewer=reviewer,
            source_language=source_language,
            target_language=target_language,
            source_scenes=source_scenes,
            localized_scenes=localized_scenes,
            progress=progress,
        )



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
    progress: LocalizationProgressCallback | None = None,
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
    _report_localization_progress(progress, 0.0)

    current_prompt = prompt
    localized_scenes = None
    last_error = None

    if _should_use_compact_localization(source_scenes):
        localized_scenes = _localize_compact(
            source_language=source_language,
            target_language=target_language,
            source_scenes=source_scenes,
            generator=generator,
            progress=lambda fraction: _report_localization_progress(
                progress,
                0.70 * fraction,
            ),
        )
        _validate_candidate_against_source(localized_scenes, source_scenes)
        localized_scenes, review_issues = _review_and_repair_localization(
            generator=generator,
            reviewer=reviewer,
            source_language=source_language,
            target_language=target_language,
            source_scenes=source_scenes,
            localized_scenes=localized_scenes,
            progress=lambda fraction: _report_localization_progress(
                progress,
                0.70 + 0.30 * fraction,
            ),
        )
        if review_issues:
            raise _LocalizationSemanticError(
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
            _report_localization_progress(progress, 0.70)
            candidate, review_issues = _review_and_repair_localization(
                generator=generator,
                reviewer=reviewer,
                source_language=source_language,
                target_language=target_language,
                source_scenes=source_scenes,
                localized_scenes=candidate,
                progress=lambda fraction: _report_localization_progress(
                    progress,
                    0.70 + 0.30 * fraction,
                ),
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
                progress=lambda fraction: _report_localization_progress(
                    progress,
                    0.70 * fraction,
                ),
            )
            _validate_candidate_against_source(localized_scenes, source_scenes)
            localized_scenes, review_issues = _review_and_repair_localization(
                generator=generator,
                reviewer=reviewer,
                source_language=source_language,
                target_language=target_language,
                source_scenes=source_scenes,
                localized_scenes=localized_scenes,
                progress=lambda fraction: _report_localization_progress(
                    progress,
                    0.70 + 0.30 * fraction,
                ),
            )
            if review_issues:
                raise LocalizationError(
                    "semantic localization review failed: "
                    + "; ".join(review_issues)
                )
            break
        except _LocalizationSemanticError:
            # Targeted batch repair has already been attempted. Do not throw away
            # good translations by regenerating the entire language.
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
    _report_localization_progress(progress, 1.0)
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
