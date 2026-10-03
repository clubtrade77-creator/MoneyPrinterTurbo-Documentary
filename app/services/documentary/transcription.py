from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.models.documentary import (
    DocumentaryTranscript,
    TranscriptSegment,
    TranscriptWord,
)
from app.services import subtitle as subtitle_service
from app.services.documentary.project import load_project, project_dir, sha256_file

_TRANSCRIPT_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,160}$")


class TranscriptionError(RuntimeError):
    """Raised when a documentary source cannot be transcribed safely."""


def _validate_source_id(source_id: str) -> str:
    if not _TRANSCRIPT_SOURCE_ID_RE.fullmatch(source_id or ""):
        raise ValueError("invalid documentary source id")
    return source_id


def transcript_path(
    project_id: str,
    source_id: str,
    root: str | os.PathLike | None = None,
) -> Path:
    return project_dir(project_id, root) / "transcripts" / (
        f"{_validate_source_id(source_id)}.json"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _probability(value: Any) -> float | None:
    parsed = _finite_float(value)
    if parsed is None or parsed < 0.0 or parsed > 1.0:
        return None
    return parsed


def _time_range(start: Any, end: Any) -> tuple[float, float] | None:
    start_value = _finite_float(start)
    end_value = _finite_float(end)
    if (
        start_value is None
        or end_value is None
        or start_value < 0
        or end_value <= start_value
    ):
        return None
    return start_value, end_value


def _build_word(word: Any) -> TranscriptWord | None:
    timing = _time_range(getattr(word, "start", None), getattr(word, "end", None))
    text = str(getattr(word, "word", "") or "").strip()
    if timing is None or not text:
        return None
    return TranscriptWord(
        start_seconds=timing[0],
        end_seconds=timing[1],
        text=text,
        probability=_probability(getattr(word, "probability", None)),
    )


def transcribe_media(
    file_path: str | os.PathLike,
    *,
    source_id: str,
    source_checksum_sha256: str = "",
    media_duration_seconds: float | None = None,
    language: str | None = None,
    model_override: Any = None,
    model_name: str | None = None,
) -> DocumentaryTranscript:
    """Transcribe one local media file into stable segment/word timecodes."""
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"media file not found: {path}")
    _validate_source_id(source_id)

    model = model_override or subtitle_service.get_whisper_model()
    transcribe_kwargs: dict[str, Any] = {
        "beam_size": 5,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    if language:
        transcribe_kwargs["language"] = language
    if subtitle_service.initial_prompt:
        transcribe_kwargs["initial_prompt"] = subtitle_service.initial_prompt

    try:
        raw_segments, info = model.transcribe(str(path), **transcribe_kwargs)
        segments: list[TranscriptSegment] = []
        for index, raw_segment in enumerate(raw_segments):
            timing = _time_range(
                getattr(raw_segment, "start", None),
                getattr(raw_segment, "end", None),
            )
            text = str(getattr(raw_segment, "text", "") or "").strip()
            if timing is None or not text:
                continue

            words = []
            for raw_word in getattr(raw_segment, "words", None) or []:
                word = _build_word(raw_word)
                if word is not None:
                    words.append(word)

            segments.append(
                TranscriptSegment(
                    id=index,
                    start_seconds=timing[0],
                    end_seconds=timing[1],
                    text=text,
                    words=words,
                    avg_logprob=_finite_float(
                        getattr(raw_segment, "avg_logprob", None)
                    ),
                    no_speech_probability=_probability(
                        getattr(raw_segment, "no_speech_prob", None)
                    ),
                )
            )
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(f"Whisper transcription failed for {path.name}: {exc}") from exc

    detected_language = str(getattr(info, "language", "") or "")
    language_probability = _probability(
        getattr(info, "language_probability", None)
    )
    full_text = " ".join(segment.text for segment in segments).strip()

    return DocumentaryTranscript(
        source_id=source_id,
        source_checksum_sha256=source_checksum_sha256,
        language=detected_language,
        language_probability=language_probability,
        media_duration_seconds=media_duration_seconds,
        model_size=model_name or str(subtitle_service.model_size),
        full_text=full_text,
        segments=segments,
    )


def transcribe_source(
    project_id: str,
    source_id: str,
    *,
    root: str | os.PathLike | None = None,
    language: str | None = None,
    model_override: Any = None,
    model_name: str | None = None,
) -> DocumentaryTranscript:
    """Transcribe one registered source and atomically persist its JSON transcript."""
    project = load_project(project_id, root)
    source = next((item for item in project.sources if item.id == source_id), None)
    if source is None:
        raise ValueError(f"source not found in project: {source_id}")
    if not source.has_local_copy:
        raise TranscriptionError(f"source has no local media copy: {source_id}")
    if source.video_metadata is not None and not source.video_metadata.has_audio:
        raise TranscriptionError(f"source has no audio stream: {source_id}")

    checksum = source.checksum_sha256 or sha256_file(source.local_path)
    media_duration = (
        source.video_metadata.duration_seconds
        if source.video_metadata is not None
        else None
    )
    transcript = transcribe_media(
        source.local_path,
        source_id=source.id,
        source_checksum_sha256=checksum,
        media_duration_seconds=media_duration,
        language=language,
        model_override=model_override,
        model_name=model_name,
    )
    _atomic_write_json(
        transcript_path(project_id, source.id, root),
        transcript.model_dump(mode="json"),
    )
    return transcript


def load_source_transcript(
    project_id: str,
    source_id: str,
    *,
    root: str | os.PathLike | None = None,
) -> DocumentaryTranscript:
    """Load a transcript and reject it if the registered source has since changed."""
    project = load_project(project_id, root)
    source = next((item for item in project.sources if item.id == source_id), None)
    if source is None:
        raise ValueError(f"source not found in project: {source_id}")

    path = transcript_path(project_id, source.id, root)
    if not path.is_file():
        raise FileNotFoundError(f"documentary transcript not found: {source_id}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        transcript = DocumentaryTranscript.model_validate(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        raise TranscriptionError(f"invalid documentary transcript: {source_id}") from exc

    if transcript.source_id != source.id:
        raise TranscriptionError(
            f"transcript source mismatch: expected {source.id}, got {transcript.source_id}"
        )
    if (
        source.checksum_sha256
        and transcript.source_checksum_sha256 != source.checksum_sha256
    ):
        raise TranscriptionError(
            f"transcript is stale for source {source.id}; source checksum changed"
        )
    return transcript
