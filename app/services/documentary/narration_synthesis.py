from __future__ import annotations

import os
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
from app.services.documentary.project import load_project, save_project, set_narrator_voice


class NarrationSynthesisError(RuntimeError):
    pass


@dataclass(frozen=True)
class NarrationSynthesisResult:
    generated: tuple[NarrationAudioAsset, ...]
    reused: tuple[NarrationAudioAsset, ...]
    voice_name: str
    language: str


def _record_asset_voice(
    project_id: str,
    scene_id: str,
    *,
    language: str,
    voice_name: str,
    root: str | os.PathLike | None,
) -> NarrationAudioAsset:
    project = load_project(project_id, root)
    asset = next(
        (
            item
            for item in project.narration_audio
            if item.scene_id == scene_id and item.language == language
        ),
        None,
    )
    if asset is None:
        raise NarrationSynthesisError(
            f"registered narration audio is missing for scene {scene_id}"
        )
    asset.voice_name = voice_name
    project.narrator_voices[language] = voice_name
    save_project(project, root)
    return asset


def synthesize_narration(
    project_id: str,
    voice_name: str,
    *,
    language: str | None = None,
    root: str | os.PathLike | None = None,
) -> NarrationSynthesisResult:
    project = load_project(project_id, root)
    language = str(language or project.master_language).strip().lower()
    voice_name = str(voice_name or "").strip()
    if not language or not voice_name:
        raise ValueError("narration language and voice are required")

    scenes = [
        scene
        for scene in project.plan.scenes
        if scene.audio_mode in {AudioMode.narration, AudioMode.mixed}
        and scene.narration_text.strip()
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
            output = Path(temp_dir) / f"{scene.id}.mp3"
            try:
                result = voice_service.tts(
                    text=scene.narration_text,
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
            if available <= 0 or duration > available + 0.05:
                raise NarrationSynthesisError(
                    f"narration does not fit scene {scene.id}"
                )
            try:
                attach_narration_audio(
                    project_id,
                    scene.id,
                    output,
                    language=language,
                    root=root,
                )
                generated.append(
                    _record_asset_voice(
                        project_id,
                        scene.id,
                        language=language,
                        voice_name=voice_name,
                        root=root,
                    )
                )
            except NarrationSynthesisError:
                raise
            except Exception as exc:
                raise NarrationSynthesisError(
                    f"could not register narration audio for scene {scene.id}: {exc}"
                ) from exc

    set_narrator_voice(project_id, language, voice_name, root=root)
    return NarrationSynthesisResult(
        generated=tuple(generated),
        reused=tuple(reused),
        voice_name=voice_name,
        language=language,
    )
