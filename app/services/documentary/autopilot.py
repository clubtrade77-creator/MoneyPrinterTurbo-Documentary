from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from loguru import logger

from app.models.documentary import AudioMode
from app.services import subtitle as subtitle_service
from app.services import voice as voice_service
from app.services.documentary.clip_selector import select_clips
from app.services.documentary.localization import localize_project
from app.services.documentary.media_fetcher import fetch_source_local_copy
from app.services.documentary.narration import write_narration
from app.services.documentary.narration_synthesis import synthesize_narration
from app.services.documentary.project import (
    add_source,
    create_project,
    load_project,
    save_project,
)
from app.services.documentary.renderer import render_documentary
from app.services.documentary.source_hunter import (
    SourceHunterError,
    candidate_to_web_source,
    candidate_to_youtube_source,
    embedded_media_to_source,
    find_embedded_media,
    find_source_videos,
    find_web_sources,
    inspect_story_article,
)
from app.services.documentary.story_discovery import (
    StoryCandidate,
    candidate_to_source,
    discover_stories,
)
from app.services.documentary.story_planner import plan_story
from app.services.documentary.transcription import transcribe_source


ProgressCallback = Callable[["AutopilotProgress"], None]


@dataclass(frozen=True)
class AutopilotProfile:
    topic: str = ""
    master_language: str = ""
    output_languages: tuple[str, ...] = ("ru", "en", "es")
    target_duration_seconds: int = 600
    lookback_hours: int = 72
    discovery_limit: int = 12
    source_limit: int = 6
    voice_name: str = ""


@dataclass(frozen=True)
class AutopilotProgress:
    stage: str
    message: str
    fraction: float


@dataclass(frozen=True)
class AutopilotResult:
    project_id: str
    title: str
    source_id: str
    preview_paths: dict[str, str]
    rights_review_required: tuple[str, ...]
    final_master_created: bool


class AutopilotError(RuntimeError):
    def __init__(self, stage: str, message: str, *, project_id: str = ""):
        super().__init__(message)
        self.stage = stage
        self.project_id = project_id


def _emit(
    callback: ProgressCallback | None,
    stage: str,
    message: str,
    fraction: float,
) -> None:
    if callback is None:
        return
    callback(
        AutopilotProgress(
            stage=stage,
            message=message,
            fraction=max(0.0, min(1.0, float(fraction))),
        )
    )


def _normalize_language(value: str) -> str:
    language = str(value or "").strip().lower()
    if language not in {"ru", "en", "es"}:
        raise ValueError("autopilot language must be ru, en, or es")
    return language


def _resolve_voice(language: str, explicit_voice: str = "") -> str:
    explicit = str(explicit_voice or "").strip()
    if explicit:
        return explicit

    if voice_service.get_cartesia_api_key():
        voices = voice_service.get_cartesia_voices(language=language)
        if voices:
            return voices[0]

    return {
        "ru": "ru-RU-DmitryNeural-Male",
        "en": "en-US-ChristopherNeural-Male",
        "es": "es-US-AlonsoNeural-Male",
    }[language]


def _choose_story(candidates: list[StoryCandidate]) -> StoryCandidate:
    if not candidates:
        raise AutopilotError("story", "autopilot found no usable story candidates")
    return max(
        candidates,
        key=lambda item: (
            item.score,
            item.footage_score,
            item.freshness_score,
            item.story_score,
        ),
    )


def _hunt_source(
    project_id: str,
    story: StoryCandidate,
    *,
    limit: int,
    root=None,
):
    agency_hints = ()
    official_urls = ()
    try:
        hints = inspect_story_article(story.url, timeout_seconds=5.0)
    except (OSError, ValueError, SourceHunterError):
        hints = None
    if hints is not None:
        agency_hints = hints.agency_names
        official_urls = hints.official_urls

    try:
        web_candidates = find_web_sources(
            story.title,
            source_country=story.source_country,
            search_hints=agency_hints,
            official_urls=official_urls,
            limit=limit,
        )
    except (OSError, ValueError, SourceHunterError):
        web_candidates = []

    # Embedded media on an official source page has the strongest provenance signal,
    # so prefer it before a generic platform search.
    for web_candidate in web_candidates[:3]:
        if web_candidate.official_score < 35:
            continue
        try:
            embedded = find_embedded_media(
                web_candidate.url,
                limit=4,
                timeout_seconds=6.0,
            )
        except (OSError, ValueError, SourceHunterError):
            continue
        if not embedded:
            continue
        web_source = candidate_to_web_source(web_candidate)
        add_source(project_id, web_source, root=root)
        media_source = embedded_media_to_source(
            embedded[0],
            title=web_candidate.title,
        )
        add_source(project_id, media_source, root=root)
        return media_source

    try:
        video_candidates = find_source_videos(
            story.title,
            search_hints=agency_hints,
            limit=limit,
        )
    except (OSError, ValueError, SourceHunterError) as exc:
        raise AutopilotError(
            "sources",
            f"autopilot could not find source video: {exc}",
            project_id=project_id,
        ) from exc

    usable = [
        candidate
        for candidate in video_candidates
        if candidate.source_quality_score > 0 or candidate.score >= 50
    ]
    if not usable:
        raise AutopilotError(
            "sources",
            "autopilot found no source video with enough confidence",
            project_id=project_id,
        )

    selected = max(
        usable,
        key=lambda item: (
            item.source_quality_score > 0,
            item.score,
            item.source_quality_score,
            item.video_signal_score,
            item.freshness_score,
        ),
    )
    source = candidate_to_youtube_source(selected)
    add_source(project_id, source, root=root)
    return source


def _project_has_narrated_scenes(project_id: str, *, root=None) -> bool:
    project = load_project(project_id, root)
    return any(
        scene.audio_mode in {AudioMode.narration, AudioMode.mixed}
        and scene.narration_text.strip()
        for scene in project.plan.scenes
    )


def _rights_review_source_ids(project_id: str, *, root=None) -> tuple[str, ...]:
    project = load_project(project_id, root)
    referenced_source_ids = {
        scene.source_id for scene in project.plan.scenes if scene.source_id
    }
    return tuple(
        source.id
        for source in project.sources
        if (
            source.id in referenced_source_ids
            and not source.rights_cleared_for_publish
        )
    )


def run_autopilot(
    profile: AutopilotProfile | None = None,
    *,
    root: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> AutopilotResult:
    """Create a documentary technical preview from a topic or automatic discovery.

    The run intentionally produces technical previews when publication rights are
    still unverified. Final masters stay gated by the existing renderer rights checks.
    """
    profile = profile or AutopilotProfile()
    initial_master_language = (
        _normalize_language(profile.master_language)
        if str(profile.master_language or "").strip()
        else "en"
    )
    output_languages = tuple(
        dict.fromkeys(
            _normalize_language(item)
            for item in profile.output_languages
        )
    )
    project_id = ""
    stage = "story"

    try:
        _emit(progress, "story", "Searching for the best story", 0.04)
        stories = discover_stories(
            profile.topic,
            lookback_hours=profile.lookback_hours,
            limit=profile.discovery_limit,
        )
        story = _choose_story(stories)

        project = create_project(
            story.title,
            master_language=initial_master_language,
            root=root,
        )
        project_id = project.id
        add_source(project.id, candidate_to_source(story), root=root)

        stage = "sources"
        _emit(progress, stage, "Finding original or authoritative video", 0.13)
        source = _hunt_source(
            project.id,
            story,
            limit=profile.source_limit,
            root=root,
        )

        stage = "media"
        _emit(progress, stage, "Downloading a technical review copy", 0.23)
        fetch_source_local_copy(project.id, source.id, root=root)

        stage = "transcription"
        _emit(progress, stage, "Analyzing speech and timecodes", 0.34)
        model = subtitle_service.get_whisper_model("small")
        transcript = transcribe_source(
            project.id,
            source.id,
            root=root,
            language=None,
            model_override=model,
            model_name="small",
        )
        detected_language = str(transcript.language or "").split("-", 1)[0].lower()
        master_language = (
            detected_language
            if detected_language in {"ru", "en", "es"}
            else initial_master_language
        )
        if master_language != initial_master_language:
            current_project = load_project(project.id, root)
            current_project.master_language = master_language
            save_project(current_project, root)

        target_languages = tuple(
            language
            for language in output_languages
            if language != master_language
        )

        stage = "story_plan"
        _emit(progress, stage, "Building the documentary story", 0.46)
        plan_story(
            project.id,
            source_ids=[source.id],
            target_duration_seconds=profile.target_duration_seconds,
            root=root,
        )

        stage = "clips"
        _emit(progress, stage, "Selecting source clips", 0.57)
        select_clips(project.id, root=root)

        stage = "narration"
        _emit(progress, stage, "Writing grounded narration", 0.66)
        write_narration(project.id, root=root)

        stage = "voice"
        if _project_has_narrated_scenes(project.id, root=root):
            _emit(progress, stage, "Generating narrator audio", 0.75)
            synthesize_narration(
                project.id,
                _resolve_voice(master_language, profile.voice_name),
                language=master_language,
                root=root,
            )

        stage = "preview"
        _emit(progress, stage, "Rendering master-language preview", 0.83)
        master_preview = render_documentary(
            project.id,
            preview=True,
            root=root,
        )
        previews = {master_language: str(master_preview)}

        if target_languages:
            step = 0.14 / max(1, len(target_languages))
            for index, language in enumerate(target_languages):
                stage = f"localize_{language}"
                base_fraction = 0.84 + step * index
                _emit(
                    progress,
                    stage,
                    f"Creating {language.upper()} version",
                    base_fraction,
                )
                localize_project(
                    project.id,
                    language,
                    root=root,
                )
                if _project_has_narrated_scenes(project.id, root=root):
                    synthesize_narration(
                        project.id,
                        _resolve_voice(language),
                        language=language,
                        root=root,
                    )
                localized_preview = render_documentary(
                    project.id,
                    language=language,
                    preview=True,
                    root=root,
                )
                previews[language] = str(localized_preview)

        rights_review_required = _rights_review_source_ids(
            project.id,
            root=root,
        )
        final_master_created = False
        if not rights_review_required:
            stage = "final"
            _emit(progress, stage, "Rendering publication-ready master", 0.98)
            render_documentary(project.id, root=root)
            final_master_created = True

        _emit(progress, "done", "Documentary is ready", 1.0)
        return AutopilotResult(
            project_id=project.id,
            title=story.title,
            source_id=source.id,
            preview_paths=previews,
            rights_review_required=rights_review_required,
            final_master_created=final_master_created,
        )
    except AutopilotError:
        raise
    except Exception as exc:
        logger.exception(
            f"Documentary Autopilot failed: stage={stage}, project_id={project_id}"
        )
        raise AutopilotError(
            stage,
            str(exc) or exc.__class__.__name__,
            project_id=project_id,
        ) from exc
