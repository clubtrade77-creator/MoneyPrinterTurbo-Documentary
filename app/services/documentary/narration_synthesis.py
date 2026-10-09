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
from app.services.documentary.project import load_project, set_narrator_voice


class NarrationSynthesisError(RuntimeError):
    pass


@dataclass(frozen=True)
class NarrationSynthesisResult:
    generated: tuple[NarrationAudioAsset, ...]
    reused: tuple[NarrationAudioAsset, ...]
    voice_name: str
    language: str


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

    same_voice = project.narrator_voices.get(language) == voice_name
    reused: list[NarrationAudioAsset] = []
    pending = []
    for scene in scenes:
        current = None
        if same_voice:
            try:
                current = load_narration_audio(
                    project_id,
                    scene.id,
                    language=language,
                    root=root,
                )
            except (FileNotFoundError, NarrationAudioError):
                current = None
        if current is None:
            pending.append(scene)
        else:
            reused.append(current)

    generated: list[NarrationAudioAsset] = []
    with tempfile.TemporaryDirectory(prefix="documentary-narration-") as temp_dir:
        staged: list[tuple[object, Path]] = []
        for scene in pending:
            output = Path(temp_dir) / f"{scene.id}.mp3"
            result = voice_service.tts(
                text=scene.narration_text,
                voice_name=voice_name,
                voice_rate=1.0,
                voice_file=str(output),
                voice_volume=1.0,
            )
            if result is None or not output.is_file() or output.stat().st_size <= 0:
                raise NarrationSynthesisError(
                    f"narration generation failed for scene {scene.id}"
                )
            duration = float(voice_service.get_audio_duration(str(output)) or 0)
            available = float(scene.source_end or 0) - float(scene.source_start or 0)
            if duration <= 0:
                raise NarrationSynthesisError(
                    f"invalid narration duration for scene {scene.id}"
                )
            if available <= 0 or duration > available + 0.05:
                raise NarrationSynthesisError(
                    f"narration does not fit scene {scene.id}"
                )
            staged.append((scene, output))

        for scene, output in staged:
            generated.append(
                attach_narration_audio(
                    project_id,
                    scene.id,
                    output,
                    language=language,
                    root=root,
                )
            )

    set_narrator_voice(project_id, language, voice_name, root=root)
    return NarrationSynthesisResult(
        generated=tuple(generated),
        reused=tuple(reused),
        voice_name=voice_name,
        language=language,
    )
