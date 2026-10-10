from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from loguru import logger

from app.models.documentary import AudioMode, SourceAsset
from app.services import subtitle as subtitle_service
from app.services import voice as voice_service
from app.services.documentary.clip_selector import select_clips
from app.services.documentary.localization import localize_project
from app.services.documentary.media_fetcher import (
    DocumentaryMediaFetchError,
    fetch_source_local_copy,
)
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
    find_public_page_media,
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
from app.services.documentary.transcription import TranscriptionError, transcribe_source


ProgressCallback = Callable[["AutopilotProgress"], None]


@dataclass(frozen=True)
class AutopilotProfile:
    topic: str = ""
    master_language: str = ""
    output_languages: tuple[str, ...] = ("ru", "en", "es")
    target_duration_seconds: int = 600
    lookback_hours: int = 72
    discovery_limit: int = 30
    source_limit: int = 8
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


@dataclass(frozen=True)
class _SourceBundle:
    media_source: SourceAsset
    supporting_sources: tuple[SourceAsset, ...] = ()


def _discover_source_bundle(
    story: StoryCandidate,
    *,
    limit: int,
) -> _SourceBundle:
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

    # First inspect official pages, then any other high-confidence public
    # Source-Hunter result for embedded/direct/social video media.
    for web_candidate in web_candidates[:5]:
        try:
            if web_candidate.official_score >= 35:
                embedded = find_embedded_media(
                    web_candidate.url,
                    limit=4,
                    timeout_seconds=6.0,
                )
            else:
                embedded = find_public_page_media(
                    web_candidate.url,
                    limit=4,
                    timeout_seconds=6.0,
                )
        except (OSError, ValueError, SourceHunterError):
            continue
        if not embedded:
            continue
        return _SourceBundle(
            media_source=embedded_media_to_source(
                embedded[0],
                title=web_candidate.title,
            ),
            supporting_sources=(candidate_to_web_source(web_candidate),),
        )

    # Many news sites use players whose actual media URL is generated by JavaScript.
    # yt-dlp can often extract that media directly from the article page, so keep
    # the strongest video-signalled web result as a media candidate before YouTube.
    page_media_candidates = [
        candidate
        for candidate in web_candidates
        if candidate.video_signal_score > 0
    ]
    if page_media_candidates:
        return _SourceBundle(
            media_source=candidate_to_web_source(page_media_candidates[0]),
        )

    try:
        video_candidates = find_source_videos(
            story.title,
            search_hints=agency_hints,
            limit=limit,
        )
    except (OSError, ValueError, SourceHunterError) as exc:
        logger.info(
            "Documentary Autopilot YouTube search unavailable; "
            f"continuing without YouTube: {exc}"
        )
        video_candidates = []

    # find_source_videos() already performs strict event matching and rejects
    # ordinary news recaps. Do not apply a second arbitrary score threshold here.
    if not video_candidates:
        raise AutopilotError(
            "sources",
            "autopilot found no downloadable web or platform video for this story",
        )

    selected = max(
        video_candidates,
        key=lambda item: (
            item.source_quality_score > 0,
            item.score,
            item.source_quality_score,
            item.video_signal_score,
            item.freshness_score,
            item.title_overlap_score,
        ),
    )
    return _SourceBundle(
        media_source=candidate_to_youtube_source(selected),
    )


def _persist_source_bundle(
    project_id: str,
    bundle: _SourceBundle,
    *,
    root=None,
) -> SourceAsset:
    for source in (*bundle.supporting_sources, bundle.media_source):
        try:
            add_source(project_id, source, root=root)
        except ValueError as exc:
            if "source already exists in project" not in str(exc).lower():
                raise
    return bundle.media_source


def _hunt_source(
    project_id: str,
    story: StoryCandidate,
    *,
    limit: int,
    root=None,
):
    bundle = _discover_source_bundle(story, limit=limit)
    return _persist_source_bundle(project_id, bundle, root=root)


def _rank_stories(candidates: list[StoryCandidate]) -> list[StoryCandidate]:
    return sorted(
        candidates,
        key=lambda item: (
            item.score,
            item.footage_score,
            item.freshness_score,
            item.story_score,
        ),
        reverse=True,
    )


def _published_footage_stories(
    candidates: list[StoryCandidate],
) -> list[StoryCandidate]:
    # Story Discovery gives a small score to any title that merely mentions a
    # camera/video. Autopilot is stricter: 28+ requires a published-footage signal
    # such as "video shows", "released footage", "bodycam footage", etc.
    return [
        story
        for story in _rank_stories(candidates)
        if story.footage_score >= 28
    ]


def _discover_autopilot_stories(
    profile: AutopilotProfile,
) -> list[StoryCandidate]:
    windows = [profile.lookback_hours]
    if not str(profile.topic or "").strip() and profile.lookback_hours < 24 * 7:
        windows.append(24 * 7)

    last_candidates: list[StoryCandidate] = []
    for lookback_hours in dict.fromkeys(windows):
        candidates = discover_stories(
            profile.topic,
            lookback_hours=lookback_hours,
            limit=profile.discovery_limit,
        )
        last_candidates = candidates
        viable = _published_footage_stories(candidates)
        if viable:
            return viable

    if last_candidates:
        raise AutopilotError(
            "story",
            "autopilot found recent stories, but none had a strong signal that "
            "source footage was already published",
        )
    raise AutopilotError(
        "story",
        "autopilot found no usable story candidates",
    )


def _choose_story_with_source(
    candidates: list[StoryCandidate],
    *,
    source_limit: int,
    max_story_attempts: int | None = None,
) -> tuple[StoryCandidate, _SourceBundle]:
    errors = []
    ranked = _published_footage_stories(candidates)
    if max_story_attempts is not None:
        ranked = ranked[:max_story_attempts]
    for story in ranked:
        try:
            bundle = _discover_source_bundle(story, limit=source_limit)
            return story, bundle
        except AutopilotError as exc:
            errors.append(f"{story.title}: {exc}")
            logger.info(
                "Documentary Autopilot skipped story without source video: "
                f"title={story.title!r}, reason={exc}"
            )

    detail = " | ".join(errors)
    if len(detail) > 3000:
        detail = detail[-3000:]
    if not ranked:
        raise AutopilotError(
            "sources",
            "autopilot had no stories with a published-footage signal to inspect",
        )
    raise AutopilotError(
        "sources",
        "autopilot checked every published-footage story but could not find "
        "downloadable source video"
        + (f": {detail}" if detail else ""),
    )



def _fallback_video_sources(
    story: StoryCandidate,
    *,
    exclude_source_id: str,
    limit: int,
):
    sources = []
    seen_ids = {exclude_source_id}

    try:
        web_candidates = find_web_sources(
            story.title,
            source_country=story.source_country,
            limit=limit,
        )
    except (OSError, ValueError, SourceHunterError):
        web_candidates = []

    for candidate in web_candidates:
        if candidate.video_signal_score <= 0:
            continue
        source = candidate_to_web_source(candidate)
        if source.id in seen_ids:
            continue
        seen_ids.add(source.id)
        sources.append(source)

    try:
        candidates = find_source_videos(
            story.title,
            limit=max(2, limit),
        )
    except (OSError, ValueError, SourceHunterError):
        candidates = []

    candidates = list(candidates)
    candidates.sort(
        key=lambda item: (
            item.source_quality_score > 0,
            item.score,
            item.source_quality_score,
            item.video_signal_score,
            item.freshness_score,
        ),
        reverse=True,
    )
    for candidate in candidates:
        source = candidate_to_youtube_source(candidate)
        if source.id in seen_ids:
            continue
        seen_ids.add(source.id)
        sources.append(source)
    return sources



def _fetch_first_available_source(
    project_id: str,
    story: StoryCandidate,
    primary_source,
    *,
    source_limit: int,
    root=None,
):
    errors = []
    try:
        fetch_source_local_copy(
            project_id,
            primary_source.id,
            root=root,
        )
        return primary_source
    except DocumentaryMediaFetchError as exc:
        errors.append(f"{primary_source.id}: {exc}")
        logger.warning(
            "Documentary Autopilot primary source download failed; "
            f"trying fallbacks: source={primary_source.id}, error={exc}"
        )

    for source in _fallback_video_sources(
        story,
        exclude_source_id=primary_source.id,
        limit=source_limit,
    ):
        try:
            add_source(project_id, source, root=root)
        except ValueError as exc:
            if "source already exists in project" not in str(exc).lower():
                raise
        try:
            fetch_source_local_copy(
                project_id,
                source.id,
                root=root,
            )
            return source
        except DocumentaryMediaFetchError as exc:
            errors.append(f"{source.id}: {exc}")
            logger.warning(
                "Documentary Autopilot fallback source download failed; "
                f"trying next source: source={source.id}, error={exc}"
            )

    detail = " | ".join(errors)
    if len(detail) > 3000:
        detail = detail[-3000:]
    raise AutopilotError(
        "media",
        "autopilot could not download any suitable source video"
        + (f": {detail}" if detail else ""),
        project_id=project_id,
    )


def _fetch_and_transcribe_first_available_source(
    project_id: str,
    story: StoryCandidate,
    primary_source: SourceAsset,
    *,
    source_limit: int,
    model,
    root=None,
):
    errors = []
    seen_ids = set()

    def try_source(source: SourceAsset):
        if source.id in seen_ids:
            return None
        seen_ids.add(source.id)

        try:
            add_source(project_id, source, root=root)
        except ValueError as exc:
            if "source already exists in project" not in str(exc).lower():
                raise

        try:
            fetched = fetch_source_local_copy(
                project_id,
                source.id,
                root=root,
            )
        except DocumentaryMediaFetchError as exc:
            errors.append(f"{source.id}: download failed: {exc}")
            logger.warning(
                "Documentary Autopilot source download failed; trying next source: "
                f"source={source.id}, error={exc}"
            )
            return None

        try:
            transcript = transcribe_source(
                project_id,
                fetched.id,
                root=root,
                language=None,
                model_override=model,
                model_name="small",
            )
        except TranscriptionError as exc:
            errors.append(f"{source.id}: transcription failed: {exc}")
            logger.warning(
                "Documentary Autopilot source has no usable speech; "
                f"trying next source: source={source.id}, error={exc}"
            )
            return None

        return fetched, transcript

    result = try_source(primary_source)
    if result is not None:
        return result

    for source in _fallback_video_sources(
        story,
        exclude_source_id=primary_source.id,
        limit=source_limit,
    ):
        result = try_source(source)
        if result is not None:
            return result

    detail = " | ".join(errors)
    if len(detail) > 3000:
        detail = detail[-3000:]
    raise AutopilotError(
        "transcription",
        "autopilot could not find a downloadable source with recognizable speech"
        + (f": {detail}" if detail else ""),
        project_id=project_id,
    )



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
        stories = _discover_autopilot_stories(profile)

        stage = "sources"
        _emit(progress, stage, "Finding a story with source video", 0.13)
        story, source_bundle = _choose_story_with_source(
            stories,
            source_limit=profile.source_limit,
        )

        project = create_project(
            story.title,
            master_language=initial_master_language,
            root=root,
        )
        project_id = project.id
        add_source(project.id, candidate_to_source(story), root=root)
        source = _persist_source_bundle(
            project.id,
            source_bundle,
            root=root,
        )

        stage = "transcription"
        _emit(
            progress,
            stage,
            "Downloading and validating source speech",
            0.23,
        )
        model = subtitle_service.get_whisper_model("small")

        story_attempts = [story] + [
            candidate
            for candidate in _published_footage_stories(stories)
            if candidate is not story
        ]
        transcription_errors = []
        transcript = None

        for index, candidate_story in enumerate(story_attempts):
            candidate_source = source
            if index:
                try:
                    candidate_bundle = _discover_source_bundle(
                        candidate_story,
                        limit=profile.source_limit,
                    )
                except AutopilotError as exc:
                    transcription_errors.append(
                        f"{candidate_story.title}: source search failed: {exc}"
                    )
                    continue

                try:
                    add_source(
                        project.id,
                        candidate_to_source(candidate_story),
                        root=root,
                    )
                except ValueError as exc:
                    if "source already exists in project" not in str(exc).lower():
                        raise
                candidate_source = _persist_source_bundle(
                    project.id,
                    candidate_bundle,
                    root=root,
                )

            try:
                candidate_source, candidate_transcript = (
                    _fetch_and_transcribe_first_available_source(
                        project.id,
                        candidate_story,
                        candidate_source,
                        source_limit=profile.source_limit,
                        model=model,
                        root=root,
                    )
                )
            except AutopilotError as exc:
                transcription_errors.append(
                    f"{candidate_story.title}: {exc}"
                )
                logger.info(
                    "Documentary Autopilot skipped story without usable speech: "
                    f"title={candidate_story.title!r}, reason={exc}"
                )
                continue

            story = candidate_story
            source = candidate_source
            transcript = candidate_transcript
            if project.title != story.title:
                current_project = load_project(project.id, root)
                current_project.title = story.title
                save_project(current_project, root)
            break

        if transcript is None:
            detail = " | ".join(transcription_errors)
            if len(detail) > 3000:
                detail = detail[-3000:]
            raise AutopilotError(
                "transcription",
                "autopilot checked available stories but found no downloadable "
                "source with recognizable speech"
                + (f": {detail}" if detail else ""),
                project_id=project.id,
            )

        _emit(progress, stage, "Analyzing speech and timecodes", 0.34)
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
