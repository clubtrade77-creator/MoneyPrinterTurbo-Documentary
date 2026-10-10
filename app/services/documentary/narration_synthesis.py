from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.models.documentary import AudioMode, NarrationAudioAsset
from app.services import voice as voice_service
from app.services.documentary.audio import (
    NarrationAudioError,
    attach_narration_audio,
    load_narration_audio,
)
from app.services.documentary.localization import (
    LocalizationError,
    load_localization_plan,
)
from app.services.documentary.project import load_project, set_narrator_voice
from app.utils import utils


_NARRATION_FIT_TOLERANCE_SECONDS = 0.05
_MAX_LOCAL_TIME_COMPRESSION_RATIO = 1.12


class NarrationSynthesisError(RuntimeError):
    pass


@dataclass(frozen=True)
class NarrationSynthesisResult:
    generated: tuple[NarrationAudioAsset, ...]
    reused: tuple[NarrationAudioAsset, ...]
    voice_name: str
    language: str


def _normalize_language(value: str) -> str:
    language = str(value or "").strip().lower()
    parts = language.split("-")
    if not (
        1 <= len(parts) <= 4
        and 2 <= len(parts[0]) <= 8
        and parts[0].isalpha()
        and all(
            1 <= len(part) <= 8 and part.isalnum()
            for part in parts[1:]
        )
    ):
        raise ValueError("invalid documentary narration language")
    return language


def _narration_text_context(
    project,
    language: str,
    *,
    root: str | os.PathLike | None,
) -> tuple[dict[str, str], str]:
    master_language = _normalize_language(project.master_language)
    if language == master_language:
        return (
            {
                scene.id: scene.narration_text
                for scene in project.plan.scenes
            },
            "",
        )

    try:
        plan = load_localization_plan(
            project.id,
            language,
            root=root,
        )
    except (FileNotFoundError, LocalizationError) as exc:
        raise NarrationSynthesisError(
            f"localization plan is unavailable for language {language}"
        ) from exc

    return (
        {
            scene.scene_id: scene.narration_text
            for scene in plan.scenes
        },
        plan.reviewed_content_fingerprint,
    )


def _assert_narration_context_unchanged(
    project_id: str,
    snapshot,
    *,
    language: str,
    expected_text: str,
    localization_fingerprint: str,
    root: str | os.PathLike | None,
) -> None:
    _assert_scene_unchanged(
        project_id,
        snapshot,
        root=root,
    )

    current_project = load_project(project_id, root)
    if language == _normalize_language(current_project.master_language):
        if snapshot.narration_text != expected_text:
            raise NarrationSynthesisError(
                f"narration text changed during synthesis: {snapshot.id}"
            )
        return

    try:
        plan = load_localization_plan(
            project_id,
            language,
            root=root,
        )
    except (FileNotFoundError, LocalizationError) as exc:
        raise NarrationSynthesisError(
            f"localization changed during narration synthesis: {snapshot.id}"
        ) from exc

    if plan.reviewed_content_fingerprint != localization_fingerprint:
        raise NarrationSynthesisError(
            f"localization changed during narration synthesis: {snapshot.id}"
        )
    localized = next(
        (item for item in plan.scenes if item.scene_id == snapshot.id),
        None,
    )
    if localized is None or localized.narration_text != expected_text:
        raise NarrationSynthesisError(
            f"localization changed during narration synthesis: {snapshot.id}"
        )


def _fit_narration_audio_to_scene(
    path: Path,
    *,
    duration: float,
    available: float,
) -> tuple[Path, float]:
    if duration <= available + _NARRATION_FIT_TOLERANCE_SECONDS:
        return path, duration

    target_duration = max(0.05, available - _NARRATION_FIT_TOLERANCE_SECONDS)
    ratio = duration / target_duration
    if ratio > _MAX_LOCAL_TIME_COMPRESSION_RATIO:
        raise NarrationSynthesisError(
            f"narration exceeds the scene by too much to fit safely: "
            f"{duration:.2f}s > {available:.2f}s"
        )

    fitted = path.with_name(f"{path.stem}-fit{path.suffix}")
    command = [
        utils.get_ffmpeg_binary(),
        "-y",
        "-nostdin",
        "-i",
        str(path),
        "-filter:a",
        f"atempo={ratio:.6f}",
        "-vn",
        str(fitted),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NarrationSynthesisError(
            "could not time-fit narration audio"
        ) from exc

    if result.returncode != 0 or not fitted.is_file() or fitted.stat().st_size <= 0:
        message = (result.stderr or result.stdout or "").strip()
        if len(message) > 1200:
            message = message[-1200:]
        raise NarrationSynthesisError(
            "could not time-fit narration audio"
            + (f": {message}" if message else "")
        )

    try:
        fitted_duration = float(
            voice_service.get_audio_duration(str(fitted)) or 0
        )
    except Exception as exc:
        raise NarrationSynthesisError(
            "could not inspect time-fitted narration audio"
        ) from exc
    if (
        fitted_duration <= 0
        or fitted_duration > available + _NARRATION_FIT_TOLERANCE_SECONDS
    ):
        raise NarrationSynthesisError(
            f"time-fitted narration still does not fit scene: "
            f"{fitted_duration:.2f}s > {available:.2f}s"
        )
    return fitted, fitted_duration


def _assert_scene_unchanged(
    project_id: str,
    snapshot,
    *,
    root: str | os.PathLike | None,
) -> None:
    current_project = load_project(project_id, root)
    current = next(
        (scene for scene in current_project.plan.scenes if scene.id == snapshot.id),
        None,
    )
    if current is None:
        raise NarrationSynthesisError(
            f"scene changed during narration synthesis: {snapshot.id}"
        )

    if (
        current.narration_text != snapshot.narration_text
        or current.audio_mode != snapshot.audio_mode
        or current.source_start != snapshot.source_start
        or current.source_end != snapshot.source_end
    ):
        raise NarrationSynthesisError(
            f"scene changed during narration synthesis: {snapshot.id}"
        )


def synthesize_narration(
    project_id: str,
    voice_name: str,
    *,
    language: str | None = None,
    root: str | os.PathLike | None = None,
) -> NarrationSynthesisResult:
    project = load_project(project_id, root)
    language = _normalize_language(language or project.master_language)
    voice_name = str(voice_name or "").strip()
    if not language or not voice_name:
        raise ValueError("narration language and voice are required")

    narration_texts, localization_fingerprint = _narration_text_context(
        project,
        language,
        root=root,
    )
    scenes = [
        scene
        for scene in project.plan.scenes
        if scene.audio_mode in {AudioMode.narration, AudioMode.mixed}
        and narration_texts.get(scene.id, "").strip()
    ]
    if not scenes:
        raise NarrationSynthesisError("no narrated scenes are available")

    reused: list[NarrationAudioAsset] = []
    pending = []
    for scene in scenes:
        try:
            current = load_narration_audio(
                project_id,
                scene.id,
                language=language,
                root=root,
            )
        except (FileNotFoundError, NarrationAudioError):
            current = None
        if current is not None and current.voice_name == voice_name:
            reused.append(current)
        else:
            pending.append(scene)

    generated: list[NarrationAudioAsset] = []
    with tempfile.TemporaryDirectory(prefix="documentary-narration-") as temp_dir:
        for scene in pending:
            narration_text = narration_texts[scene.id]
            _assert_narration_context_unchanged(
                project_id,
                scene,
                language=language,
                expected_text=narration_text,
                localization_fingerprint=localization_fingerprint,
                root=root,
            )
            output = Path(temp_dir) / f"{scene.id}.mp3"
            try:
                result = voice_service.tts(
                    text=narration_text,
                    voice_name=voice_name,
                    voice_rate=1.0,
                    voice_file=str(output),
                    voice_volume=1.0,
                )
            except Exception as exc:
                raise NarrationSynthesisError(
                    f"TTS failed for scene {scene.id}: {exc}"
                ) from exc
            if result is None or not output.is_file() or output.stat().st_size <= 0:
                raise NarrationSynthesisError(
                    f"narration generation failed for scene {scene.id}"
                )
            try:
                duration = float(
                    voice_service.get_audio_duration(str(output)) or 0
                )
            except Exception as exc:
                raise NarrationSynthesisError(
                    f"could not inspect narration duration for scene {scene.id}: {exc}"
                ) from exc
            available = float(scene.source_end or 0) - float(scene.source_start or 0)
            if duration <= 0:
                raise NarrationSynthesisError(
                    f"invalid narration duration for scene {scene.id}"
                )
            if available <= 0:
                raise NarrationSynthesisError(
                    f"narration scene has invalid duration: {scene.id}"
                )
            output, duration = _fit_narration_audio_to_scene(
                output,
                duration=duration,
                available=available,
            )
            _assert_narration_context_unchanged(
                project_id,
                scene,
                language=language,
                expected_text=narration_text,
                localization_fingerprint=localization_fingerprint,
                root=root,
            )
            try:
                generated.append(
                    attach_narration_audio(
                        project_id,
                        scene.id,
                        output,
                        language=language,
                        voice_name=voice_name,
                        root=root,
                    )
                )
            except Exception as exc:
                raise NarrationSynthesisError(
                    f"could not register narration audio for scene {scene.id}: {exc}"
                ) from exc

    try:
        set_narrator_voice(project_id, language, voice_name, root=root)
    except Exception as exc:
        raise NarrationSynthesisError(
            f"could not persist narrator voice selection: {exc}"
        ) from exc
    return NarrationSynthesisResult(
        generated=tuple(generated),
        reused=tuple(reused),
        voice_name=voice_name,
        language=language,
    )
