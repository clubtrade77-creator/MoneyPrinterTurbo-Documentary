from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

import streamlit as st

from app.config import config
from app.models.documentary import AudioMode, RightsStatus, SourceType
from app.services import subtitle as subtitle_service
from app.services import voice as voice_service
from app.services.documentary.autopilot import (
    AutopilotError,
    AutopilotProfile,
    run_autopilot,
)
from app.services.documentary.clip_selector import (
    ClipSelectorError,
    clip_plan_path,
    load_clip_plan,
    select_clips,
)
from app.services.documentary.project import (
    add_source,
    attach_local_copy_to_source,
    attach_local_video,
    create_project,
    list_projects,
    load_project,
    project_dir,
    update_scene_narration,
    update_source_rights,
)
from app.services.documentary.localization import (
    LocalizationError,
    load_localization_plan,
    localization_plan_path,
    localize_project,
)
from app.services.documentary.narration_synthesis import (
    NarrationSynthesisError,
    synthesize_narration,
)
from app.services.documentary.narration import (
    NarrationWriterError,
    scene_requires_narration,
    write_narration,
)
from app.services.documentary.story_discovery import (
    StoryDiscoveryError,
    candidate_to_source,
    discover_stories,
)
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
from app.services.documentary.story_planner import (
    StoryPlannerError,
    load_story_plan,
    plan_story,
    story_plan_path,
)
from app.services.documentary.renderer import (
    DocumentaryRenderError,
    documentary_preview_path,
    documentary_render_path,
    documentary_render_readiness_issues,
    render_documentary,
)
from app.services.documentary.transcription import (
    TranscriptionError,
    load_source_transcript,
    transcribe_source,
)

Tr = Callable[[str], str]

_SOURCE_TYPES = (
    SourceType.local_video,
    SourceType.bodycam,
    SourceType.cctv,
    SourceType.court,
    SourceType.interview,
    SourceType.news,
    SourceType.broll,
)
_RIGHTS_STATUSES = (
    RightsStatus.unknown_review_required,
    RightsStatus.user_owned,
    RightsStatus.licensed,
    RightsStatus.permission_confirmed,
    RightsStatus.public_domain,
)

_SOURCE_TYPE_LABELS = {
    SourceType.local_video: "Documentary Source Local Video",
    SourceType.youtube: "Documentary Source YouTube",
    SourceType.bodycam: "Documentary Source Bodycam",
    SourceType.cctv: "Documentary Source CCTV",
    SourceType.court: "Documentary Source Court",
    SourceType.interview: "Documentary Source Interview",
    SourceType.news: "Documentary Source News",
    SourceType.broll: "Documentary Source Broll",
}
_RIGHTS_LABELS = {
    RightsStatus.unknown_review_required: "Documentary Rights Review Required",
    RightsStatus.user_owned: "Documentary Rights User Owned",
    RightsStatus.licensed: "Documentary Rights Licensed",
    RightsStatus.permission_confirmed: "Documentary Rights Permission Confirmed",
    RightsStatus.public_domain: "Documentary Rights Public Domain",
}

_SIMPLE_SECTIONS = (
    ("story", "Documentary Simple Section Story"),
    ("materials", "Documentary Simple Section Materials"),
    ("production", "Documentary Simple Section Production"),
    ("video", "Documentary Simple Section Video"),
)


def _source_duration(source) -> str:
    metadata = source.video_metadata
    if metadata is None:
        return "-"
    return f"{metadata.duration_seconds:.1f}s"


def _format_transcript_time(seconds: float) -> str:
    total_centiseconds = max(0, round(float(seconds) * 100))
    hours, remainder = divmod(total_centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"
    return f"{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def _format_transcript_segment(segment) -> str:
    start = _format_transcript_time(segment.start_seconds)
    end = _format_transcript_time(segment.end_seconds)
    return f"[{start}–{end}] {segment.text.strip()}"


def _story_evidence_timecode(transcript, segment_ids: list[int]) -> str:
    segments_by_id = {segment.id: segment for segment in transcript.segments}
    selected = [
        segments_by_id[segment_id]
        for segment_id in segment_ids
        if segment_id in segments_by_id
    ]
    if not selected:
        return ""
    start = min(segment.start_seconds for segment in selected)
    end = max(segment.end_seconds for segment in selected)
    return f"[{_format_transcript_time(start)}–{_format_transcript_time(end)}]"


def _load_story_plan_if_available(project_id: str):
    try:
        return load_story_plan(project_id)
    except FileNotFoundError:
        return None


def _load_clip_plan_if_available(project_id: str):
    try:
        return load_clip_plan(project_id)
    except FileNotFoundError:
        return None


def _render_output_is_current(
    project_id: str,
    *,
    language: str | None = None,
    preview: bool = False,
) -> bool:
    if language is None:
        default_output = (
            documentary_preview_path(project_id)
            if preview
            else documentary_render_path(project_id)
        )
        if not default_output.is_file():
            return False

    project = load_project(project_id)
    master_language = project.master_language.lower()
    render_language = (language or master_language).lower()
    localized = render_language != master_language

    if localized:
        try:
            load_localization_plan(
                project_id,
                render_language,
            )
        except (FileNotFoundError, LocalizationError):
            return False

    output = (
        documentary_preview_path(
            project_id,
            language=render_language if localized else None,
        )
        if preview
        else documentary_render_path(
            project_id,
            language=render_language if localized else None,
        )
    )
    if not output.is_file():
        return False

    metadata_dependencies = [
        story_plan_path(project_id),
        clip_plan_path(project_id),
    ]
    if localized:
        metadata_dependencies.append(
            localization_plan_path(project_id, render_language)
        )

    referenced_source_ids = {
        scene.source_id
        for scene in project.plan.scenes
        if scene.source_id
    }
    if not preview:
        referenced_sources = [
            source
            for source in project.sources
            if source.id in referenced_source_ids
        ]
        if any(
            not source.rights_cleared_for_publish
            for source in referenced_sources
        ):
            return False
    narration_scene_ids = {
        scene.id
        for scene in project.plan.scenes
        if scene.audio_mode in {AudioMode.narration, AudioMode.mixed}
    }

    media_dependencies = [
        Path(source.local_path)
        for source in project.sources
        if source.local_path and source.id in referenced_source_ids
    ]
    media_dependencies.extend(
        Path(asset.local_path)
        for asset in project.narration_audio
        if (
            asset.local_path
            and asset.language == render_language
            and asset.scene_id in narration_scene_ids
        )
    )

    if any(not dependency.is_file() for dependency in media_dependencies):
        return False

    output_mtime = output.stat().st_mtime
    return all(
        not dependency.is_file() or output_mtime >= dependency.stat().st_mtime
        for dependency in metadata_dependencies
    ) and all(
        output_mtime >= dependency.stat().st_mtime
        for dependency in media_dependencies
    )


def _default_story_target_seconds(total_media_seconds: float) -> int:
    if total_media_seconds <= 0:
        return 60
    if total_media_seconds < 60:
        return max(5, min(60, round(total_media_seconds)))
    return 600


def _documentary_voice_options(language: str) -> list[tuple[str, str]]:
    language = (language or "en").lower()
    edge_by_language = {
        "en": [
            ("en-US-ChristopherNeural-Male", "Edge · Christopher"),
            ("en-US-JennyNeural-Female", "Edge · Jenny"),
        ],
        "ru": [
            ("ru-RU-DmitryNeural-Male", "Edge · Dmitry"),
            ("ru-RU-SvetlanaNeural-Female", "Edge · Svetlana"),
        ],
        "es": [
            ("es-US-AlonsoNeural-Male", "Edge · Alonso"),
            ("es-US-PalomaNeural-Female", "Edge · Paloma"),
        ],
    }
    options: list[tuple[str, str]] = []

    if voice_service.get_cartesia_api_key():
        cartesia_voice_id = str(
            config.cartesia.get("voice_id", "")
            or voice_service._CARTESIA_DEFAULT_VOICE
        ).strip()
        cartesia_model = str(
            config.cartesia.get("model_id", "")
            or voice_service._CARTESIA_DEFAULT_MODEL
        ).strip()
        cartesia_catalog = voice_service.list_cartesia_voices(
            language=language,
            limit=100,
        )
        if cartesia_catalog:
            ordered_catalog = sorted(
                cartesia_catalog,
                key=lambda item: (
                    str(item.get("id") or "").strip() != cartesia_voice_id,
                    str(item.get("name") or "").lower(),
                ),
            )
            seen_voice_ids = set()
            for item in ordered_catalog:
                voice_id = str(item.get("id") or "").strip()
                voice_label = str(item.get("name") or "").strip()
                if not voice_id or not voice_label or voice_id in seen_voice_ids:
                    continue
                seen_voice_ids.add(voice_id)
                gender = str(item.get("gender") or "").strip()
                label = f"Cartesia · {voice_label}"
                if gender:
                    label += f" · {gender}"
                options.append(
                    (
                        f"cartesia:{voice_id}:{language}",
                        label,
                    )
                )
        else:
            options.append(
                (
                    f"cartesia:{cartesia_voice_id}:{language}",
                    f"Cartesia · {cartesia_model}",
                )
            )

    if config.app.get("gemini_api_key", ""):
        options.extend(
            [
                ("gemini:Charon-Informative", "Gemini · Charon · Informative"),
                ("gemini:Gacrux-Mature", "Gemini · Gacrux · Mature"),
                ("gemini:Sulafat-Warm", "Gemini · Sulafat · Warm"),
            ]
        )
    if language == "en" and voice_service.get_minimax_tts_api_key():
        options.append(
            ("minimax:English_expressive_narrator", "MiniMax · Expressive Narrator")
        )
    if voice_service.get_fish_audio_api_key():
        options.extend(
            [
                (
                    "fish_audio:7b6131ba75ba47c98a46c847db729ab6:Clear Male-Male",
                    "Fish Audio · Clear Male",
                ),
                (
                    "fish_audio:2324c907b9a94c64ab4afb941e5b3408:Clear Female-Female",
                    "Fish Audio · Clear Female",
                ),
            ]
        )

    options.extend(edge_by_language.get(language, edge_by_language["en"]))
    return options


def _documentary_voice_preview_text(language: str) -> str:
    return {
        "ru": "Это пример голоса рассказчика для документального видео.",
        "es": "Esta es una muestra de la voz del narrador para un documental.",
    }.get(
        (language or "en").lower(),
        "This is a sample of the narrator voice for a documentary.",
    )


def _generate_documentary_voice_preview(
    project_id: str,
    *,
    voice_name: str,
    text: str,
) -> Path:
    preview_dir = project_dir(project_id) / "audio" / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    fd, preview_name = tempfile.mkstemp(
        prefix="voice-preview-",
        suffix=".mp3",
        dir=preview_dir,
    )
    os.close(fd)
    preview_path = Path(preview_name)
    preview_path.unlink(missing_ok=True)
    result = voice_service.tts(
        text=text,
        voice_name=voice_name,
        voice_rate=1.0,
        voice_file=str(preview_path),
        voice_volume=1.0,
    )
    if result is None or not preview_path.is_file() or preview_path.stat().st_size <= 0:
        preview_path.unlink(missing_ok=True)
        raise RuntimeError("voice preview generation failed")
    return preview_path


def _write_uploaded_video_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name or "").suffix.lower()
    if suffix not in {".mp4", ".mov"}:
        raise ValueError("unsupported documentary video extension")

    fd, temp_name = tempfile.mkstemp(prefix="mpt-documentary-upload-", suffix=suffix)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            uploaded_file.seek(0)
            while chunk := uploaded_file.read(1024 * 1024):
                handle.write(chunk)
        return temp_path
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _render_story_discovery(tr: Tr) -> None:
    with st.expander(tr("Documentary Story Discovery"), expanded=True):
        st.caption(tr("Documentary Story Discovery Description"))

        cols = st.columns([2, 1, 1])
        query = cols[0].text_input(
            tr("Documentary Story Discovery Topic"),
            placeholder=tr("Documentary Story Discovery Topic Placeholder"),
            key="documentary_story_discovery_topic",
        )
        lookback_hours = cols[1].selectbox(
            tr("Documentary Story Discovery Window"),
            options=(24, 72, 168),
            index=1,
            format_func=lambda hours: {
                24: tr("Documentary Story Discovery 24h"),
                72: tr("Documentary Story Discovery 72h"),
                168: tr("Documentary Story Discovery 7d"),
            }[hours],
            key="documentary_story_discovery_window",
        )
        project_language = cols[2].selectbox(
            tr("Documentary Story Discovery Project Language"),
            options=("ru", "en", "es"),
            index=0,
            format_func=lambda code: {
                "ru": "Русский",
                "en": "English",
                "es": "Español",
            }[code],
            key="documentary_story_discovery_language",
        )

        if st.button(
            tr("Documentary Story Discovery Search"),
            type="primary",
            width="stretch",
            key="documentary_story_discovery_search",
        ):
            try:
                with st.spinner(tr("Documentary Story Discovery Searching")):
                    candidates = discover_stories(
                        query,
                        lookback_hours=lookback_hours,
                        limit=12,
                    )
            except (OSError, ValueError, StoryDiscoveryError) as exc:
                st.error(
                    tr("Documentary Story Discovery Failed").format(error=str(exc))
                )
            else:
                st.session_state["documentary_story_candidates"] = candidates

        candidates = st.session_state.get("documentary_story_candidates", [])
        if not candidates:
            return

        st.markdown(
            f"**{tr('Documentary Story Discovery Results').format(count=len(candidates))}**"
        )
        for index, candidate in enumerate(candidates, start=1):
            with st.container(border=True):
                st.markdown(f"### {index}. {candidate.title}")
                metadata = " · ".join(
                    item
                    for item in (
                        candidate.publisher,
                        candidate.source_country,
                        candidate.language,
                        candidate.published_at,
                    )
                    if item
                )
                if metadata:
                    st.caption(metadata)

                score_cols = st.columns(4)
                score_cols[0].metric(
                    tr("Documentary Story Discovery Score"),
                    f"{candidate.score}/100",
                )
                score_cols[1].metric(
                    tr("Documentary Story Discovery Footage"),
                    candidate.footage_score,
                )
                score_cols[2].metric(
                    tr("Documentary Story Discovery Freshness"),
                    candidate.freshness_score,
                )
                score_cols[3].metric(
                    tr("Documentary Story Discovery Story"),
                    candidate.story_score,
                )

                if candidate.reasons:
                    st.caption(" · ".join(candidate.reasons))

                actions = st.columns([1, 1])
                actions[0].link_button(
                    tr("Documentary Story Discovery Open Source"),
                    candidate.url,
                    width="stretch",
                )

                hunter_key = f"documentary_source_hunter_{candidate.id}"
                if actions[1].button(
                    tr("Documentary Source Hunter Search"),
                    type="primary",
                    width="stretch",
                    key=f"documentary_source_hunter_search_{candidate.id}",
                ):
                    bundle = {
                        "web": [],
                        "videos": [],
                        "embedded": {},
                        "article_hints": None,
                        "errors": [],
                    }
                    with st.spinner(tr("Documentary Source Hunter Searching")):
                        try:
                            bundle["article_hints"] = inspect_story_article(
                                candidate.url,
                                timeout_seconds=5.0,
                            )
                        except (OSError, ValueError, SourceHunterError):
                            bundle["article_hints"] = None

                        article_hints = bundle["article_hints"]
                        agency_hints = (
                            article_hints.agency_names
                            if article_hints is not None
                            else ()
                        )
                        official_urls = (
                            article_hints.official_urls
                            if article_hints is not None
                            else ()
                        )

                        try:
                            bundle["web"] = find_web_sources(
                                candidate.title,
                                source_country=candidate.source_country,
                                search_hints=agency_hints,
                                official_urls=official_urls,
                                limit=6,
                            )
                        except (OSError, ValueError, SourceHunterError) as exc:
                            bundle["errors"].append(f"web: {exc}")
                        else:
                            article_hints = bundle["article_hints"]
                            if (
                                article_hints is None
                                or (
                                    not article_hints.agency_names
                                    and not article_hints.official_urls
                                )
                            ):
                                for web_candidate in bundle["web"][:2]:
                                    if web_candidate.official_score >= 35:
                                        continue
                                    try:
                                        refined_hints = inspect_story_article(
                                            web_candidate.url,
                                            timeout_seconds=5.0,
                                        )
                                    except (
                                        OSError,
                                        ValueError,
                                        SourceHunterError,
                                    ):
                                        continue
                                    if (
                                        refined_hints.agency_names
                                        or refined_hints.official_urls
                                    ):
                                        bundle["article_hints"] = refined_hints
                                        article_hints = refined_hints
                                        agency_hints = refined_hints.agency_names
                                        official_urls = refined_hints.official_urls
                                        try:
                                            refined_web = find_web_sources(
                                                candidate.title,
                                                source_country=(
                                                    candidate.source_country
                                                ),
                                                search_hints=agency_hints,
                                                official_urls=official_urls,
                                                limit=6,
                                            )
                                        except (
                                            OSError,
                                            ValueError,
                                            SourceHunterError,
                                        ):
                                            refined_web = []
                                        if refined_web:
                                            merged = {}
                                            for item in (
                                                bundle["web"] + refined_web
                                            ):
                                                current = merged.get(item.url)
                                                if (
                                                    current is None
                                                    or item.score > current.score
                                                ):
                                                    merged[item.url] = item
                                            bundle["web"] = sorted(
                                                merged.values(),
                                                key=lambda item: (
                                                    item.score,
                                                    item.official_score,
                                                    item.video_signal_score,
                                                    item.title_overlap_score,
                                                    item.title.lower(),
                                                ),
                                                reverse=True,
                                            )[:6]
                                        break

                            inspected_official_pages = 0
                            for official_candidate in bundle["web"]:
                                if official_candidate.official_score < 35:
                                    continue
                                if inspected_official_pages >= 3:
                                    break
                                inspected_official_pages += 1
                                try:
                                    media = find_embedded_media(
                                        official_candidate.url,
                                        limit=4,
                                        timeout_seconds=6.0,
                                    )
                                except (
                                    OSError,
                                    ValueError,
                                    SourceHunterError,
                                ) as exc:
                                    bundle["errors"].append(
                                        f"official media: {exc}"
                                    )
                                    continue

                                if media:
                                    bundle["embedded"][
                                        official_candidate.url
                                    ] = media
                                    break

                        try:
                            bundle["videos"] = find_source_videos(
                                candidate.title,
                                search_hints=agency_hints,
                                limit=6,
                            )
                        except (OSError, ValueError, SourceHunterError) as exc:
                            bundle["errors"].append(f"YouTube: {exc}")

                    if not bundle["web"] and not bundle["videos"]:
                        st.error(
                            tr("Documentary Source Hunter Failed").format(
                                error="; ".join(bundle["errors"])
                            )
                        )
                    else:
                        st.session_state[hunter_key] = bundle

                source_bundle = st.session_state.get(
                    hunter_key,
                    {
                        "web": [],
                        "videos": [],
                        "embedded": {},
                        "errors": [],
                    },
                )
                web_candidates = source_bundle.get("web", [])
                source_candidates = source_bundle.get("videos", [])
                embedded_by_page = source_bundle.get("embedded", {})
                article_hints = source_bundle.get("article_hints")
                source_errors = source_bundle.get("errors", [])

                if article_hints is not None and (
                    article_hints.agency_names or article_hints.official_urls
                ):
                    details = []
                    if article_hints.agency_names:
                        details.append(
                            tr("Documentary Source Hunter Article Agencies").format(
                                agencies=", ".join(article_hints.agency_names)
                            )
                        )
                    if article_hints.official_urls:
                        details.append(
                            tr("Documentary Source Hunter Article Official Links").format(
                                count=len(article_hints.official_urls)
                            )
                        )
                    st.caption(" · ".join(details))

                if source_errors:
                    st.warning(
                        tr("Documentary Source Hunter Partial").format(
                            errors="; ".join(source_errors)
                        )
                    )

                if web_candidates:
                    st.markdown(
                        f"**{tr('Documentary Source Hunter Web Results').format(count=len(web_candidates))}**"
                    )

                for web_index, web_candidate in enumerate(
                    web_candidates,
                    start=1,
                ):
                    with st.container(border=True):
                        st.markdown(
                            f"**{web_index}. {web_candidate.title}**"
                        )
                        st.caption(web_candidate.domain)
                        if web_candidate.snippet:
                            st.caption(web_candidate.snippet)

                        web_scores = st.columns(4)
                        web_scores[0].metric(
                            tr("Documentary Source Hunter Score"),
                            f"{web_candidate.score}/100",
                        )
                        web_scores[1].metric(
                            tr("Documentary Source Hunter Match"),
                            web_candidate.title_overlap_score,
                        )
                        web_scores[2].metric(
                            tr("Documentary Source Hunter Official"),
                            web_candidate.official_score,
                        )
                        web_scores[3].metric(
                            tr("Documentary Source Hunter Video Signal"),
                            web_candidate.video_signal_score,
                        )

                        if web_candidate.reasons:
                            st.caption(" · ".join(web_candidate.reasons))

                        embedded_media = embedded_by_page.get(
                            web_candidate.url,
                            [],
                        )
                        if embedded_media:
                            st.markdown(
                                f"**{tr('Documentary Source Hunter Embedded Media').format(count=len(embedded_media))}**"
                            )
                            for media_index, media_candidate in enumerate(
                                embedded_media,
                                start=1,
                            ):
                                media_actions = st.columns([1, 1])
                                platform_label = {
                                    "youtube": "YouTube",
                                    "vimeo": "Vimeo",
                                    "direct_video": tr(
                                        "Documentary Source Hunter Direct Video"
                                    ),
                                }.get(
                                    media_candidate.platform,
                                    media_candidate.platform,
                                )
                                media_actions[0].link_button(
                                    f"{platform_label}: {media_candidate.label}",
                                    media_candidate.url,
                                    width="stretch",
                                )
                                if media_actions[1].button(
                                    tr(
                                        "Documentary Source Hunter Create Embedded Project"
                                    ),
                                    type="primary",
                                    width="stretch",
                                    key=(
                                        "documentary_source_hunter_embedded_create_"
                                        f"{candidate.id}_{web_index}_{media_index}"
                                    ),
                                ):
                                    try:
                                        project = create_project(
                                            candidate.title,
                                            master_language=project_language,
                                        )
                                        add_source(
                                            project.id,
                                            candidate_to_source(candidate),
                                        )
                                        add_source(
                                            project.id,
                                            candidate_to_web_source(
                                                web_candidate
                                            ),
                                        )
                                        add_source(
                                            project.id,
                                            embedded_media_to_source(
                                                media_candidate,
                                                title=web_candidate.title,
                                            ),
                                        )
                                    except (OSError, ValueError) as exc:
                                        st.error(
                                            tr(
                                                "Documentary Source Hunter Create Failed"
                                            ).format(error=str(exc))
                                        )
                                    else:
                                        st.session_state[
                                            "documentary_project_id"
                                        ] = project.id
                                        st.success(
                                            tr(
                                                "Documentary Source Hunter Project Created"
                                            )
                                        )
                                        st.rerun()

                        web_actions = st.columns([1, 1])
                        web_actions[0].link_button(
                            tr("Documentary Source Hunter Open Web Source"),
                            web_candidate.url,
                            width="stretch",
                        )
                        web_create_allowed = (
                            web_candidate.official_score >= 35
                            or web_candidate.score >= 50
                        )
                        if not web_create_allowed:
                            st.caption(
                                tr("Documentary Source Hunter Low Confidence")
                            )
                        if web_actions[1].button(
                            tr("Documentary Source Hunter Create Web Project"),
                            type="primary",
                            width="stretch",
                            disabled=not web_create_allowed,
                            key=(
                                "documentary_source_hunter_web_create_"
                                f"{candidate.id}_{web_index}"
                            ),
                        ):
                            try:
                                project = create_project(
                                    candidate.title,
                                    master_language=project_language,
                                )
                                add_source(
                                    project.id,
                                    candidate_to_source(candidate),
                                )
                                add_source(
                                    project.id,
                                    candidate_to_web_source(web_candidate),
                                )
                            except (OSError, ValueError) as exc:
                                st.error(
                                    tr(
                                        "Documentary Source Hunter Create Failed"
                                    ).format(error=str(exc))
                                )
                            else:
                                st.session_state[
                                    "documentary_project_id"
                                ] = project.id
                                st.success(
                                    tr(
                                        "Documentary Source Hunter Project Created"
                                    )
                                )
                                st.rerun()

                if source_candidates:
                    st.markdown(
                        f"**{tr('Documentary Source Hunter YouTube Results').format(count=len(source_candidates))}**"
                    )

                for source_index, source_candidate in enumerate(
                    source_candidates,
                    start=1,
                ):
                    with st.container(border=True):
                        st.markdown(
                            f"**{source_index}. {source_candidate.title}**"
                        )
                        source_meta = " · ".join(
                            item
                            for item in (
                                source_candidate.channel,
                                source_candidate.published_text,
                                source_candidate.duration_text,
                                source_candidate.view_count_text,
                            )
                            if item
                        )
                        if source_meta:
                            st.caption(source_meta)

                        source_scores = st.columns(5)
                        source_scores[0].metric(
                            tr("Documentary Source Hunter Score"),
                            f"{source_candidate.score}/100",
                        )
                        source_scores[1].metric(
                            tr("Documentary Source Hunter Match"),
                            source_candidate.title_overlap_score,
                        )
                        source_scores[2].metric(
                            tr("Documentary Source Hunter Freshness"),
                            source_candidate.freshness_score,
                        )
                        source_scores[3].metric(
                            tr("Documentary Source Hunter Official"),
                            source_candidate.source_quality_score,
                        )
                        source_scores[4].metric(
                            tr("Documentary Source Hunter Video Signal"),
                            source_candidate.video_signal_score,
                        )

                        if source_candidate.reasons:
                            st.caption(" · ".join(source_candidate.reasons))

                        source_actions = st.columns([1, 1])
                        source_actions[0].link_button(
                            tr("Documentary Source Hunter Open Video"),
                            source_candidate.url,
                            width="stretch",
                        )
                        video_create_allowed = (
                            source_candidate.source_quality_score > 0
                            or source_candidate.score >= 50
                        )
                        if not video_create_allowed:
                            st.caption(
                                tr("Documentary Source Hunter Low Confidence")
                            )
                        if source_actions[1].button(
                            tr("Documentary Source Hunter Create Project"),
                            type="primary",
                            width="stretch",
                            disabled=not video_create_allowed,
                            key=(
                                "documentary_source_hunter_create_"
                                f"{candidate.id}_{source_candidate.video_id}"
                            ),
                        ):
                            try:
                                project = create_project(
                                    candidate.title,
                                    master_language=project_language,
                                )
                                add_source(
                                    project.id,
                                    candidate_to_source(candidate),
                                )
                                add_source(
                                    project.id,
                                    candidate_to_youtube_source(
                                        source_candidate
                                    ),
                                )
                            except (OSError, ValueError) as exc:
                                st.error(
                                    tr(
                                        "Documentary Source Hunter Create Failed"
                                    ).format(error=str(exc))
                                )
                            else:
                                st.session_state[
                                    "documentary_project_id"
                                ] = project.id
                                st.success(
                                    tr(
                                        "Documentary Source Hunter Project Created"
                                    )
                                )
                                st.rerun()


def _autopilot_progress_label(stage: str, tr: Tr) -> str:
    if stage.startswith("localize_"):
        language = stage.removeprefix("localize_").upper()
        return tr("Documentary Autopilot Stage Localization").format(
            language=language
        )
    labels = {
        "story": "Documentary Autopilot Stage Story",
        "sources": "Documentary Autopilot Stage Sources",
        "media": "Documentary Autopilot Stage Media",
        "transcription": "Documentary Autopilot Stage Transcription",
        "story_plan": "Documentary Autopilot Stage Script",
        "clips": "Documentary Autopilot Stage Clips",
        "narration": "Documentary Autopilot Stage Narration",
        "voice": "Documentary Autopilot Stage Voice",
        "preview": "Documentary Autopilot Stage Render",
        "final": "Documentary Autopilot Stage Final",
        "done": "Documentary Autopilot Stage Done",
    }
    return tr(labels.get(stage, "Documentary Autopilot Running"))


def _render_autopilot(tr: Tr) -> None:
    with st.container(border=True):
        st.markdown(f"## {tr('Documentary Autopilot Title')}")
        st.caption(tr("Documentary Autopilot Help"))
        topic = st.text_input(
            tr("Documentary Autopilot Topic"),
            placeholder=tr("Documentary Autopilot Topic Placeholder"),
            key="documentary_autopilot_topic",
            help=tr("Documentary Autopilot Topic Help"),
        )
        st.caption(tr("Documentary Autopilot Defaults"))

        if not st.button(
            tr("Documentary Autopilot Start"),
            type="primary",
            width="stretch",
            key="documentary_autopilot_start",
        ):
            return

        progress_bar = st.progress(0.0)
        status = st.status(
            tr("Documentary Autopilot Starting"),
            expanded=True,
        )

        def on_progress(event):
            progress_bar.progress(event.fraction)
            label = _autopilot_progress_label(event.stage, tr)
            status.update(label=label, state="running", expanded=True)
            status.write(label)

        try:
            result = run_autopilot(
                AutopilotProfile(topic=topic.strip()),
                progress=on_progress,
            )
        except AutopilotError as exc:
            progress_bar.empty()
            status.update(
                label=tr("Documentary Autopilot Failed").format(
                    stage=_autopilot_progress_label(exc.stage, tr),
                ),
                state="error",
                expanded=True,
            )
            st.error(str(exc))
            if exc.project_id:
                st.session_state["documentary_project_id"] = exc.project_id
            return

        progress_bar.progress(1.0)
        status.update(
            label=tr("Documentary Autopilot Complete"),
            state="complete",
            expanded=False,
        )
        st.session_state["documentary_project_id"] = result.project_id
        st.success(
            tr("Documentary Autopilot Created").format(
                title=result.title,
                languages=", ".join(
                    language.upper() for language in result.preview_paths
                ),
            )
        )
        if result.rights_review_required:
            st.warning(
                tr("Documentary Autopilot Rights Review").format(
                    count=len(result.rights_review_required),
                )
            )


def _render_create_project(tr: Tr) -> None:
    with st.expander(tr("Documentary Create Project"), expanded=False):
        with st.form("documentary_create_project_form", clear_on_submit=True):
            title = st.text_input(
                tr("Documentary Project Title"),
                placeholder=tr("Documentary Project Title Placeholder"),
            )
            master_language = st.selectbox(
                tr("Documentary Master Language"),
                options=("en", "ru", "es"),
                format_func=lambda code: {
                    "en": "English",
                    "ru": "Русский",
                    "es": "Español",
                }[code],
            )
            submitted = st.form_submit_button(
                tr("Documentary Create"),
                type="primary",
                width="stretch",
            )

        if submitted:
            try:
                project = create_project(
                    title,
                    master_language=master_language,
                )
            except (OSError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state["documentary_project_id"] = project.id
                st.success(tr("Documentary Project Created"))
                st.rerun()


def _documentary_guided_step(project) -> str:
    if not project.sources:
        return "story"

    local_sources = [source for source in project.sources if source.has_local_copy]
    if not local_sources:
        return "materials"

    transcripts = {}
    for source in local_sources:
        try:
            transcript = _load_transcript_if_available(project.id, source.id)
        except TranscriptionError:
            transcript = None
        if transcript is not None and transcript.segments:
            transcripts[source.id] = transcript
    if not transcripts:
        return "materials"

    try:
        story_plan = _load_story_plan_if_available(project.id)
    except StoryPlannerError:
        story_plan = None
    if story_plan is None or not project.plan.scenes:
        return "script"

    narrated_scenes = [
        scene
        for scene in project.plan.scenes
        if scene.audio_mode in {AudioMode.narration, AudioMode.mixed}
    ]
    if any(not scene.narration_text.strip() for scene in narrated_scenes):
        return "script"

    master_language = project.master_language.lower()
    available_audio = {
        (asset.scene_id, asset.language.lower())
        for asset in project.narration_audio
        if asset.local_path
    }
    if narrated_scenes and any(
        (scene.id, master_language) not in available_audio
        for scene in narrated_scenes
    ):
        return "voice"

    if not _render_output_is_current(project.id, preview=True):
        return "preview"
    return "ready"


def _documentary_simple_section(project) -> str:
    guided_step = _documentary_guided_step(project)
    if guided_step == "story":
        return "story"
    if guided_step == "materials":
        return "materials"
    if guided_step in {"script", "voice"}:
        return "production"
    return "video"


def _documentary_section_status(project, section_id: str, tr: Tr) -> str:
    if section_id == "story":
        return tr("Documentary Simple Story Selected")

    if section_id == "materials":
        media_sources = [
            source
            for source in project.sources
            if not source.id.startswith("story_")
        ]
        local_count = sum(1 for source in media_sources if source.has_local_copy)
        if not media_sources:
            return tr("Documentary Simple Materials Missing")
        return tr("Documentary Simple Materials Status").format(
            ready=local_count,
            total=len(media_sources),
        )

    if section_id == "production":
        if not project.plan.scenes:
            return tr("Documentary Simple Script Missing")
        narrated_scenes = [
            scene
            for scene in project.plan.scenes
            if scene.audio_mode in {AudioMode.narration, AudioMode.mixed}
        ]
        if any(not scene.narration_text.strip() for scene in narrated_scenes):
            return tr("Documentary Simple Script Draft")
        master_language = project.master_language.lower()
        available_audio = {
            (asset.scene_id, asset.language.lower())
            for asset in project.narration_audio
            if asset.local_path
        }
        if narrated_scenes and any(
            (scene.id, master_language) not in available_audio
            for scene in narrated_scenes
        ):
            return tr("Documentary Simple Voice Missing")
        return tr("Documentary Simple Production Ready")

    if _render_output_is_current(project.id):
        return tr("Documentary Simple Final Ready")
    if _render_output_is_current(project.id, preview=True):
        return tr("Documentary Simple Preview Ready")
    return tr("Documentary Simple Video Missing")


def _render_simple_dashboard(project, tr: Tr) -> str:
    recommended = _documentary_simple_section(project)
    state_key = f"documentary_simple_section_{project.id}"
    if st.session_state.get(state_key) not in {
        section_id for section_id, _ in _SIMPLE_SECTIONS
    }:
        st.session_state[state_key] = recommended

    cols = st.columns(4)
    for index, (section_id, label_key) in enumerate(_SIMPLE_SECTIONS):
        with cols[index]:
            with st.container(border=True):
                st.markdown(f"### {tr(label_key)}")
                st.caption(_documentary_section_status(project, section_id, tr))
                button_label = (
                    tr("Documentary Simple Continue")
                    if section_id == recommended
                    else tr("Documentary Simple Open")
                )
                if st.button(
                    button_label,
                    type="primary" if section_id == recommended else "secondary",
                    width="stretch",
                    key=f"documentary_simple_open_{project.id}_{section_id}",
                ):
                    st.session_state[state_key] = section_id
                    st.rerun()

    return st.session_state[state_key]


def _render_guided_project_summary(project, tr: Tr) -> None:
    st.subheader(project.title)

def _render_guided_materials(project, tr: Tr) -> None:
    st.markdown(f"### {tr('Documentary Guided Materials Title')}")
    st.caption(tr("Documentary Guided Materials Help"))
    for source in project.sources:
        title = source.title or source.original_filename or tr("Documentary Source Other")
        with st.container(border=True):
            st.markdown(f"**{title}**")
            cols = st.columns(3)
            cols[0].caption(
                tr(
                    _SOURCE_TYPE_LABELS.get(
                        source.source_type,
                        "Documentary Source Other",
                    )
                )
            )
            research_lead = source.id.startswith("story_")
            if research_lead:
                cols[1].write("ℹ️ " + tr("Documentary Guided Research Source"))
                cols[2].write("—")
            else:
                cols[1].write(
                    "✅ " + tr("Documentary Guided Video Ready")
                    if source.has_local_copy
                    else "⬜ " + tr("Documentary Guided Video Missing")
                )
                cols[2].write(
                    "✅ " + tr("Documentary Guided Rights Ready")
                    if source.rights_cleared_for_publish
                    else "⚠️ " + tr("Documentary Guided Rights Review")
                )


def _render_project_overview(project, tr: Tr) -> None:
    st.subheader(project.title)
    cols = st.columns(4)
    cols[0].metric(tr("Documentary Project ID"), project.id)
    cols[1].metric(tr("Documentary Master Language"), project.master_language.upper())
    cols[2].metric(tr("Documentary Sources"), len(project.sources))
    cols[3].metric(tr("Documentary Scenes"), len(project.plan.scenes))

    if not project.sources:
        st.info(tr("Documentary No Sources"))
        return

    rows = []
    for source in project.sources:
        rows.append(
            {
                tr("Documentary Source ID"): source.id,
                tr("Documentary Source Title"): source.title or source.original_filename,
                tr("Documentary Source Type"): tr(
                    _SOURCE_TYPE_LABELS.get(
                        source.source_type,
                        "Documentary Source Other",
                    )
                ),
                tr("Documentary Duration"): _source_duration(source),
                tr("Documentary Local Copy"): (
                    tr("Documentary Yes")
                    if source.has_local_copy
                    else tr("Documentary No")
                ),
                tr("Documentary Rights"): tr(
                    _RIGHTS_LABELS.get(
                        source.rights_status,
                        "Documentary Rights Review Required",
                    )
                ),
                tr("Documentary Publishable"): (
                    tr("Documentary Yes")
                    if source.is_publishable
                    else tr("Documentary No")
                ),
            }
        )

    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
    )


def _render_rights_review(
    project,
    tr: Tr,
    *,
    simple: bool = False,
) -> None:
    if not project.sources:
        return

    referenced_source_ids = {
        scene.source_id
        for scene in project.plan.scenes
        if scene.source_id
    }
    review_sources = [
        source
        for source in project.sources
        if (
            source.id in referenced_source_ids
            or source.rights_status == RightsStatus.unknown_review_required
        )
        and (not simple or not source.id.startswith("story_"))
    ]
    if not review_sources:
        return

    with st.expander(
        tr("Documentary Rights Review"),
        expanded=any(
            source.rights_status == RightsStatus.unknown_review_required
            for source in review_sources
        ),
    ):
        st.caption(tr("Documentary Rights Review Help"))
        source_by_id = {source.id: source for source in review_sources}
        source_id = st.selectbox(
            tr("Documentary Rights Review Source"),
            options=list(source_by_id),
            format_func=lambda value: (
                source_by_id[value].title
                or source_by_id[value].original_filename
                or value
            ),
            key=f"documentary_rights_review_source_{project.id}",
        )
        source = source_by_id[source_id]

        if source.source_url:
            st.link_button(
                tr("Documentary Rights Review Open Source"),
                source.source_url,
                width="stretch",
            )

        statuses = list(_RIGHTS_STATUSES)
        current_index = (
            statuses.index(source.rights_status)
            if source.rights_status in statuses
            else 0
        )
        rights_status = st.selectbox(
            tr("Documentary Rights"),
            options=statuses,
            index=current_index,
            format_func=lambda value: tr(_RIGHTS_LABELS[value]),
            key=(
                "documentary_rights_review_status_"
                f"{project.id}_{source.id}_{project.revision}"
            ),
        )
        rights_note = st.text_area(
            tr("Documentary Rights Note"),
            value=source.rights_note,
            height=100,
            key=(
                "documentary_rights_review_note_"
                f"{project.id}_{source.id}_{project.revision}"
            ),
        )

        changed = (
            rights_status != source.rights_status
            or rights_note.strip() != source.rights_note
        )
        note_required = rights_status in {
            RightsStatus.licensed,
            RightsStatus.permission_confirmed,
            RightsStatus.public_domain,
        }
        note_missing = note_required and len(rights_note.strip()) < 5
        if rights_status != RightsStatus.unknown_review_required:
            st.warning(tr("Documentary Rights Confirmation Warning"))
        if note_missing:
            st.info(tr("Documentary Rights Note Required"))

        if st.button(
            tr("Documentary Rights Save"),
            type="primary",
            width="stretch",
            disabled=(not changed) or note_missing,
            key=f"documentary_rights_review_save_{project.id}_{source.id}",
        ):
            try:
                update_source_rights(
                    project.id,
                    source.id,
                    rights_status,
                    rights_note=rights_note,
                )
            except (OSError, ValueError) as exc:
                st.error(
                    tr("Documentary Rights Save Failed").format(
                        error=str(exc)
                    )
                )
            else:
                st.success(tr("Documentary Rights Saved"))
                st.rerun()


def _render_external_source_local_copy(project, tr: Tr) -> None:
    external_video_sources = [
        source
        for source in project.sources
        if source.source_url
        and not source.id.startswith("story_")
        and not source.has_local_copy
        and source.source_type
        in {
            SourceType.youtube,
            SourceType.bodycam,
            SourceType.cctv,
            SourceType.court,
            SourceType.interview,
            SourceType.news,
            SourceType.broll,
        }
    ]
    if not external_video_sources:
        return

    with st.expander(
        tr("Documentary Attach Local Copy"),
        expanded=True,
    ):
        st.caption(tr("Documentary Attach Local Copy Help"))
        source_by_id = {source.id: source for source in external_video_sources}
        source_id = st.selectbox(
            tr("Documentary Attach Local Copy Source"),
            options=list(source_by_id),
            format_func=lambda value: (
                source_by_id[value].title
                or source_by_id[value].publisher
                or value
            ),
            key=f"documentary_attach_copy_source_{project.id}",
        )
        source = source_by_id[source_id]
        if source.source_url:
            st.link_button(
                tr("Documentary Attach Local Copy Open Source"),
                source.source_url,
                width="stretch",
            )

        upload = st.file_uploader(
            tr("Documentary Attach Local Copy File"),
            type=["mp4", "mov"],
            accept_multiple_files=False,
            key=f"documentary_attach_copy_file_{project.id}_{source_id}",
        )
        if st.button(
            tr("Documentary Attach Local Copy Button"),
            type="primary",
            width="stretch",
            disabled=upload is None,
            key=f"documentary_attach_copy_button_{project.id}_{source_id}",
        ):
            temp_path = None
            try:
                temp_path = _write_uploaded_video_to_temp(upload)
                attached = attach_local_copy_to_source(
                    project.id,
                    source_id,
                    temp_path,
                    original_filename=upload.name,
                )
            except Exception as exc:
                st.error(
                    tr("Documentary Attach Local Copy Failed").format(
                        error=str(exc)
                    )
                )
            else:
                st.success(
                    tr("Documentary Attach Local Copy Complete").format(
                        source_id=attached.id
                    )
                )
                st.rerun()
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)


def _load_transcript_if_available(project_id: str, source_id: str):
    try:
        return load_source_transcript(project_id, source_id)
    except FileNotFoundError:
        return None


def _render_transcription(project, tr: Tr, *, simple: bool = False) -> None:
    if not project.sources:
        return

    with st.expander(tr("Documentary Transcription"), expanded=True):
        candidates = [
            source
            for source in project.sources
            if source.has_local_copy
            and (
                source.video_metadata is None
                or source.video_metadata.has_audio
            )
        ]
        if not candidates:
            st.info(tr("Documentary No Transcribable Sources"))
            return

        source_by_id = {source.id: source for source in candidates}
        selected_source_id = st.selectbox(
            tr("Documentary Transcription Source"),
            options=list(source_by_id),
            key=f"documentary_transcription_source_{project.id}",
            format_func=lambda source_id: (
                source_by_id[source_id].title
                or source_by_id[source_id].original_filename
                or source_id
            ),
        )
        source = source_by_id[selected_source_id]

        try:
            transcript = _load_transcript_if_available(project.id, source.id)
        except TranscriptionError as exc:
            transcript = None
            st.warning(
                tr("Documentary Transcript Stale").format(error=str(exc))
            )

        if transcript is None:
            st.caption(tr("Documentary Transcript Missing"))
        else:
            cols = st.columns(2 if simple else 3)
            cols[0].metric(
                tr("Documentary Transcript Language"),
                (transcript.language or "-").upper(),
            )
            cols[1].metric(
                tr("Documentary Transcript Segments"),
                len(transcript.segments),
            )
            if not simple:
                cols[2].metric(
                    tr("Documentary Transcript Model"),
                    transcript.model_size or "-",
                )
            if transcript.full_text:
                st.markdown(f"**{tr('Documentary Transcript Preview')}**")
                for segment in transcript.segments:
                    st.write(_format_transcript_segment(segment))

        if simple:
            language_mode = "auto"
            whisper_model = "small"
            st.caption(tr("Documentary Guided Analysis Help"))
        else:
            language_mode = st.selectbox(
                tr("Documentary Transcription Language"),
                options=("auto", "en", "ru", "es"),
                format_func=lambda code: {
                    "auto": tr("Documentary Language Auto"),
                    "en": "English",
                    "ru": "Русский",
                    "es": "Español",
                }[code],
                key=f"documentary_transcription_language_{project.id}_{source.id}",
            )
            whisper_model = st.selectbox(
                tr("Documentary Whisper Model"),
                options=("small", "large-v3"),
                index=0,
                format_func=lambda value: (
                    tr("Documentary Whisper Small")
                    if value == "small"
                    else tr("Documentary Whisper Large")
                ),
                key=f"documentary_whisper_model_{project.id}_{source.id}",
                help=tr("Documentary Whisper Model Help"),
            )

        button_label = (
            tr("Documentary Retranscribe")
            if transcript is not None
            else tr("Documentary Transcribe")
        )
        if st.button(
            button_label,
            type="primary",
            width="stretch",
            key=f"documentary_transcribe_{project.id}_{source.id}",
        ):
            requested_language = None if language_mode == "auto" else language_mode
            try:
                with st.spinner(tr("Documentary Transcribing")):
                    model = subtitle_service.get_whisper_model(whisper_model)
                    result = transcribe_source(
                        project.id,
                        source.id,
                        language=requested_language,
                        model_override=model,
                        model_name=whisper_model,
                    )
            except (OSError, ValueError, TranscriptionError) as exc:
                st.error(
                    tr("Documentary Transcription Failed").format(error=str(exc))
                )
            else:
                st.success(
                    tr("Documentary Transcription Complete").format(
                        segments=len(result.segments),
                    )
                )
                st.rerun()


def _render_story_planner(project, tr: Tr) -> None:
    if not project.sources:
        return

    transcripts = {}
    stale_transcript_errors = []
    for source in project.sources:
        try:
            transcript = _load_transcript_if_available(project.id, source.id)
        except TranscriptionError as exc:
            stale_transcript_errors.append(str(exc))
            continue
        if transcript is not None and transcript.segments:
            transcripts[source.id] = transcript

    if not transcripts:
        return

    source_by_id = {source.id: source for source in project.sources}
    with st.expander(tr("Documentary Story Planner"), expanded=True):
        if stale_transcript_errors:
            st.warning(tr("Documentary Story Transcript Warning"))

        try:
            story_plan = _load_story_plan_if_available(project.id)
        except StoryPlannerError as exc:
            story_plan = None
            st.warning(
                tr("Documentary Story Plan Stale").format(error=str(exc))
            )

        if story_plan is not None:
            st.markdown(f"### {story_plan.title}")
            st.markdown(
                f"**{tr('Documentary Story Angle')}:** {story_plan.angle}"
            )
            st.markdown(
                f"**{tr('Documentary Story Hook')}:** {story_plan.hook}"
            )
            if story_plan.target_duration_seconds < 60:
                st.caption(
                    tr("Documentary Story Summary Seconds").format(
                        beats=len(story_plan.beats),
                        seconds=story_plan.target_duration_seconds,
                    )
                )
            else:
                st.caption(
                    tr("Documentary Story Summary").format(
                        beats=len(story_plan.beats),
                        minutes=story_plan.target_duration_seconds / 60,
                    )
                )

            for index, beat in enumerate(story_plan.beats, start=1):
                purpose = getattr(beat.purpose, "value", str(beat.purpose))
                with st.expander(
                    f"{index}. {beat.title} · {purpose} · "
                    f"{beat.target_duration_seconds:.0f}s",
                    expanded=index == 1,
                ):
                    st.write(beat.summary)
                    if beat.narration_goal:
                        st.markdown(
                            f"**{tr('Documentary Story Narration Goal')}:** "
                            f"{beat.narration_goal}"
                        )
                    st.markdown(
                        f"**{tr('Documentary Story Original Audio')}:** "
                        + (
                            tr("Documentary Yes")
                            if beat.original_audio_priority
                            else tr("Documentary No")
                        )
                    )
                    st.markdown(f"**{tr('Documentary Story Evidence')}:**")
                    for evidence in beat.evidence:
                        transcript = transcripts.get(evidence.source_id)
                        timecode = (
                            _story_evidence_timecode(
                                transcript,
                                evidence.segment_ids,
                            )
                            if transcript is not None
                            else ""
                        )
                        source = source_by_id.get(evidence.source_id)
                        source_label = (
                            source.title
                            if source is not None and source.title
                            else (
                                source.original_filename
                                if source is not None
                                else evidence.source_id
                            )
                        )
                        segment_label = ", ".join(
                            str(segment_id)
                            for segment_id in evidence.segment_ids
                        )
                        st.write(
                            f"{timecode} {source_label} · "
                            f"{tr('Documentary Story Segments')} {segment_label}"
                        )
                        if evidence.note:
                            st.caption(evidence.note)

        source_ids = list(transcripts)
        selected_source_ids = st.multiselect(
            tr("Documentary Story Sources"),
            options=source_ids,
            default=source_ids,
            format_func=lambda source_id: (
                source_by_id[source_id].title
                or source_by_id[source_id].original_filename
                or source_id
            ),
            key=f"documentary_story_sources_{project.id}",
        )

        total_media_seconds = sum(
            transcript.media_duration_seconds or 0
            for transcript in transcripts.values()
        )
        if total_media_seconds < 60:
            target_duration_seconds = float(
                st.number_input(
                    tr("Documentary Story Target Seconds"),
                    min_value=5,
                    max_value=60,
                    value=_default_story_target_seconds(total_media_seconds),
                    step=1,
                    key=f"documentary_story_target_seconds_{project.id}",
                    help=tr("Documentary Story Target Seconds Help"),
                )
            )
        else:
            target_minutes = st.number_input(
                tr("Documentary Story Target Minutes"),
                min_value=1,
                max_value=30,
                value=10,
                step=1,
                key=f"documentary_story_target_minutes_{project.id}",
                help=tr("Documentary Story Target Help"),
            )
            target_duration_seconds = float(target_minutes) * 60

        button_label = (
            tr("Documentary Story Regenerate")
            if story_plan is not None
            else tr("Documentary Story Generate")
        )
        if st.button(
            button_label,
            type="primary",
            width="stretch",
            disabled=not selected_source_ids,
            key=f"documentary_story_generate_{project.id}",
        ):
            try:
                with st.spinner(tr("Documentary Story Generating")):
                    result = plan_story(
                        project.id,
                        source_ids=selected_source_ids,
                        target_duration_seconds=target_duration_seconds,
                    )
            except (OSError, ValueError, StoryPlannerError) as exc:
                st.error(
                    tr("Documentary Story Failed").format(error=str(exc))
                )
            else:
                st.success(
                    tr("Documentary Story Complete").format(
                        beats=len(result.beats)
                    )
                )
                st.rerun()


def _render_clip_selector(project, tr: Tr) -> None:
    try:
        story_plan = _load_story_plan_if_available(project.id)
    except StoryPlannerError:
        return
    if story_plan is None:
        return

    with st.expander(tr("Documentary Clip Selector"), expanded=True):
        try:
            clip_plan = _load_clip_plan_if_available(project.id)
        except (ClipSelectorError, StoryPlannerError, TranscriptionError) as exc:
            clip_plan = None
            st.warning(
                tr("Documentary Clip Plan Stale").format(error=str(exc))
            )

        if clip_plan is not None:
            st.caption(
                tr("Documentary Clip Summary").format(
                    clips=len(clip_plan.clips),
                    padding=clip_plan.padding_seconds,
                    gap=clip_plan.max_merge_gap_seconds,
                )
            )
            source_by_id = {source.id: source for source in project.sources}
            for index, clip in enumerate(clip_plan.clips, start=1):
                source = source_by_id.get(clip.source_id)
                source_label = (
                    source.title
                    if source is not None and source.title
                    else (
                        source.original_filename
                        if source is not None
                        else clip.source_id
                    )
                )
                start = _format_transcript_time(clip.source_start_seconds)
                end = _format_transcript_time(clip.source_end_seconds)
                purpose = getattr(clip.purpose, "value", str(clip.purpose))
                audio_mode = getattr(clip.audio_mode, "value", str(clip.audio_mode))
                st.write(
                    f"{index}. [{start}–{end}] {source_label} · "
                    f"{purpose} · {audio_mode}"
                )

        button_label = (
            tr("Documentary Clip Rebuild")
            if clip_plan is not None
            else tr("Documentary Clip Build")
        )
        if st.button(
            button_label,
            type="primary",
            width="stretch",
            key=f"documentary_clip_select_{project.id}",
        ):
            try:
                with st.spinner(tr("Documentary Clip Selecting")):
                    result = select_clips(project.id)
            except (
                OSError,
                ValueError,
                ClipSelectorError,
                StoryPlannerError,
                TranscriptionError,
            ) as exc:
                st.error(
                    tr("Documentary Clip Failed").format(error=str(exc))
                )
            else:
                st.success(
                    tr("Documentary Clip Complete").format(
                        clips=len(result.clips)
                    )
                )
                st.rerun()


def _render_narration_writer(
    project,
    tr: Tr,
    *,
    show_voice: bool = True,
) -> None:
    if not project.plan.scenes:
        return

    try:
        story_plan = _load_story_plan_if_available(project.id)
    except StoryPlannerError:
        return
    if story_plan is None:
        return

    beats = {beat.id: beat for beat in story_plan.beats}
    with st.expander(tr("Documentary Narration"), expanded=True):
        narration_required = 0
        narration_needed = 0
        narration_updates: dict[str, str] = {}
        for index, scene in enumerate(project.plan.scenes, start=1):
            beat = beats.get(scene.story_beat_id)
            requires_narration = scene_requires_narration(scene, beat)
            if requires_narration:
                narration_required += 1

            if not requires_narration:
                if scene.audio_mode == AudioMode.original:
                    st.write(
                        tr("Documentary Narration Original").format(index=index)
                    )
                else:
                    st.write(
                        tr("Documentary Narration Not Required").format(index=index)
                    )
                continue

            if scene.narration_text:
                st.markdown(
                    f"**{tr('Documentary Narration Scene').format(index=index)}**"
                )
                edited_text = st.text_area(
                    tr("Documentary Narration Edit"),
                    value=scene.narration_text,
                    height=110,
                    key=(
                        "documentary_narration_edit_"
                        f"{project.id}_{scene.id}_{project.revision}"
                    ),
                ).strip()
                narration_updates[scene.id] = edited_text
            else:
                narration_needed += 1
                st.write(
                    tr("Documentary Narration Missing").format(index=index)
                )

        changed_narration = {
            scene_id: text
            for scene_id, text in narration_updates.items()
            if next(
                scene
                for scene in project.plan.scenes
                if scene.id == scene_id
            ).narration_text
            != text
        }
        if changed_narration:
            st.info(tr("Documentary Narration Unsaved"))
            if st.button(
                tr("Documentary Narration Save"),
                type="primary",
                width="stretch",
                key=f"documentary_narration_save_{project.id}",
            ):
                try:
                    update_scene_narration(
                        project.id,
                        changed_narration,
                    )
                except (OSError, ValueError) as exc:
                    st.error(
                        tr("Documentary Narration Save Failed").format(
                            error=str(exc)
                        )
                    )
                else:
                    st.success(tr("Documentary Narration Saved"))
                    st.rerun()

        if narration_needed:
            if st.button(
                tr("Documentary Narration Generate"),
                type="primary",
                width="stretch",
                key=f"documentary_narration_generate_{project.id}",
            ):
                try:
                    with st.spinner(tr("Documentary Narration Generating")):
                        result = write_narration(project.id)
                except (OSError, ValueError, NarrationWriterError) as exc:
                    st.error(
                        tr("Documentary Narration Failed").format(error=str(exc))
                    )
                else:
                    generated = sum(
                        1 for scene in result.plan.scenes if scene.narration_text
                    )
                    st.success(
                        tr("Documentary Narration Complete").format(
                            scenes=generated
                        )
                    )
                    st.rerun()
        elif narration_required:
            st.success(tr("Documentary Narration Ready"))
        else:
            st.success(tr("Documentary Narration Not Needed"))
            return

        if not show_voice:
            return

        st.markdown(f"**{tr('Documentary Voice Preview')}**")
        voice_options = _documentary_voice_options(project.master_language)
        voice_by_id = dict(voice_options)
        voice_ids = list(voice_by_id)
        persisted_voice = project.narrator_voices.get(
            project.master_language.lower(),
            "",
        )
        default_voice_index = (
            voice_ids.index(persisted_voice)
            if persisted_voice in voice_ids
            else 0
        )
        selected_voice = st.selectbox(
            tr("Documentary Voice"),
            options=voice_ids,
            index=default_voice_index,
            format_func=lambda value: voice_by_id[value],
            key=f"documentary_voice_preview_voice_{project.id}",
            help=tr("Documentary Voice Help"),
        )
        preview_text = st.text_input(
            tr("Documentary Voice Preview Text"),
            value=_documentary_voice_preview_text(project.master_language),
            key=f"documentary_voice_preview_text_{project.id}",
        )

        preview_state_key = f"documentary_voice_preview_path_{project.id}"
        if st.button(
            tr("Documentary Voice Preview Button"),
            key=f"documentary_voice_preview_button_{project.id}",
        ):
            try:
                with st.spinner(tr("Documentary Voice Preview Generating")):
                    preview_path = _generate_documentary_voice_preview(
                        project.id,
                        voice_name=selected_voice,
                        text=preview_text,
                    )
            except Exception as exc:
                st.error(
                    tr("Documentary Voice Preview Failed").format(error=str(exc))
                )
            else:
                st.session_state[preview_state_key] = str(preview_path)

        preview_value = st.session_state.get(preview_state_key)
        if preview_value and Path(preview_value).is_file():
            st.audio(preview_value)

        narrated_scenes = [
            scene
            for scene in project.plan.scenes
            if scene.audio_mode in {AudioMode.narration, AudioMode.mixed}
            and scene.narration_text.strip()
        ]
        if narrated_scenes and narration_needed == 0:
            st.caption(tr("Documentary Narration Audio Help"))
            if st.button(
                tr("Documentary Narration Audio Generate"),
                type="primary",
                width="stretch",
                disabled=bool(changed_narration),
                key=f"documentary_narration_audio_generate_{project.id}",
            ):
                try:
                    with st.spinner(
                        tr("Documentary Narration Audio Generating")
                    ):
                        synthesis = synthesize_narration(
                            project.id,
                            selected_voice,
                            language=project.master_language,
                        )
                except (
                    OSError,
                    ValueError,
                    NarrationSynthesisError,
                ) as exc:
                    st.error(
                        tr("Documentary Narration Audio Failed").format(
                            error=str(exc)
                        )
                    )
                else:
                    st.success(
                        tr("Documentary Narration Audio Complete").format(
                            generated=len(synthesis.generated),
                            reused=len(synthesis.reused),
                        )
                    )
                    st.rerun()


def _render_master_video(
    project,
    tr: Tr,
    *,
    mode: str = "both",
) -> None:
    if not project.plan.scenes:
        return

    with st.expander(tr("Documentary Render"), expanded=True):
        output_path = documentary_render_path(project.id)
        preview_path = documentary_preview_path(project.id)
        current_render = _render_output_is_current(project.id)
        current_preview = _render_output_is_current(
            project.id,
            preview=True,
        )

        if mode in {"both", "final"} and output_path.is_file():
            if current_render:
                st.success(tr("Documentary Render Current"))
                st.video(str(output_path))
            else:
                st.warning(tr("Documentary Render Stale"))

        if mode in {"both", "preview"} and preview_path.is_file():
            if current_preview:
                st.info(tr("Documentary Preview Current"))
                st.video(str(preview_path))
            else:
                st.warning(tr("Documentary Preview Stale"))

        st.caption(
            tr("Documentary Render Settings").format(
                width=1920,
                height=1080,
                fps=30,
                scenes=len(project.plan.scenes),
            )
        )

        final_issues = documentary_render_readiness_issues(project.id)
        preview_issues = documentary_render_readiness_issues(
            project.id,
            require_publishable_rights=False,
        )

        if mode in {"both", "final"} and final_issues:
            st.warning(tr("Documentary Render Not Ready"))
            for issue in final_issues:
                st.caption(f"• {issue}")
        if mode == "preview" and preview_issues:
            st.warning(tr("Documentary Render Not Ready"))
            for issue in preview_issues:
                st.caption(f"• {issue}")

        if mode in {"both", "preview"}:
            if st.button(
                (
                    tr("Documentary Preview Again")
                    if preview_path.is_file()
                    else tr("Documentary Preview Build")
                ),
                width="stretch",
                disabled=bool(preview_issues),
                key=f"documentary_preview_{project.id}",
            ):
                try:
                    with st.spinner(tr("Documentary Rendering")):
                        result = render_documentary(
                            project.id,
                            preview=True,
                        )
                except (OSError, ValueError, DocumentaryRenderError) as exc:
                    st.error(
                        tr("Documentary Render Failed").format(error=str(exc))
                    )
                else:
                    st.success(
                        tr("Documentary Preview Complete").format(
                            filename=result.name
                        )
                    )
                    st.rerun()

        if mode in {"both", "final"}:
            button_label = (
                tr("Documentary Render Again")
                if output_path.is_file()
                else tr("Documentary Render Build")
            )
            if st.button(
                button_label,
                type="primary",
                width="stretch",
                disabled=bool(final_issues),
                key=f"documentary_render_{project.id}",
            ):
                try:
                    with st.spinner(tr("Documentary Rendering")):
                        result = render_documentary(project.id)
                except (OSError, ValueError, DocumentaryRenderError) as exc:
                    st.error(
                        tr("Documentary Render Failed").format(error=str(exc))
                    )
                else:
                    st.success(
                        tr("Documentary Render Complete").format(
                            filename=result.name
                        )
                    )
                    st.rerun()


def _documentary_language_label(language: str) -> str:
    return {
        "ru": "Русский",
        "en": "English",
        "es": "Español",
    }.get((language or "").lower(), (language or "").upper())


def _render_localized_versions(project, tr: Tr) -> None:
    if not project.plan.scenes:
        return

    master_language = project.master_language.lower()
    master_base_language = master_language.split("-", 1)[0]
    target_languages = [
        language
        for language in ("ru", "en", "es")
        if language != master_base_language
    ]
    if not target_languages:
        return

    with st.expander(
        tr("Documentary Localization"),
        expanded=False,
    ):
        st.caption(tr("Documentary Localization Help"))

        missing_master_narration = [
            scene.id
            for scene in project.plan.scenes
            if (
                scene.audio_mode in {AudioMode.narration, AudioMode.mixed}
                and not scene.narration_text.strip()
            )
        ]
        if missing_master_narration:
            st.warning(
                tr("Documentary Localization Master Narration Missing").format(
                    count=len(missing_master_narration)
                )
            )
            return

        target_language = st.selectbox(
            tr("Documentary Localization Target"),
            options=target_languages,
            format_func=_documentary_language_label,
            key=f"documentary_localization_target_{project.id}",
        )

        localization = None
        localization_error = ""
        try:
            localization = load_localization_plan(
                project.id,
                target_language,
            )
        except FileNotFoundError:
            localization = None
        except LocalizationError as exc:
            localization_error = str(exc)

        if localization is not None:
            st.success(
                tr("Documentary Localization Current").format(
                    language=_documentary_language_label(target_language),
                    scenes=len(localization.scenes),
                )
            )
        elif localization_error:
            st.warning(
                tr("Documentary Localization Stale").format(
                    error=localization_error
                )
            )
        else:
            st.info(
                tr("Documentary Localization Missing").format(
                    language=_documentary_language_label(target_language)
                )
            )

        localization_button_label = (
            tr("Documentary Localization Rebuild")
            if localization is not None or localization_error
            else tr("Documentary Localization Generate")
        )
        if st.button(
            localization_button_label,
            type="primary" if localization is None else "secondary",
            width="stretch",
            key=(
                "documentary_localization_generate_"
                f"{project.id}_{target_language}"
            ),
        ):
            try:
                with st.spinner(tr("Documentary Localization Generating")):
                    localization = localize_project(
                        project.id,
                        target_language,
                    )
            except (OSError, ValueError, LocalizationError) as exc:
                st.error(
                    tr("Documentary Localization Failed").format(
                        error=str(exc)
                    )
                )
            else:
                st.success(
                    tr("Documentary Localization Complete").format(
                        language=_documentary_language_label(target_language)
                    )
                )
                st.rerun()

        if localization is None:
            return

        localized_by_scene = {
            scene.scene_id: scene
            for scene in localization.scenes
        }
        narrated_scenes = [
            scene
            for scene in project.plan.scenes
            if (
                scene.audio_mode in {AudioMode.narration, AudioMode.mixed}
                and localized_by_scene.get(scene.id) is not None
                and localized_by_scene[scene.id].narration_text.strip()
            )
        ]

        if narrated_scenes:
            st.markdown(
                f"**{tr('Documentary Localization Voice')}**"
            )
            voice_options = _documentary_voice_options(target_language)
            voice_by_id = dict(voice_options)
            voice_ids = list(voice_by_id)
            persisted_voice = project.narrator_voices.get(
                target_language,
                "",
            )
            default_voice_index = (
                voice_ids.index(persisted_voice)
                if persisted_voice in voice_ids
                else 0
            )
            selected_voice = st.selectbox(
                tr("Documentary Voice"),
                options=voice_ids,
                index=default_voice_index,
                format_func=lambda value: voice_by_id[value],
                key=(
                    "documentary_localization_voice_"
                    f"{project.id}_{target_language}"
                ),
                help=tr("Documentary Voice Help"),
            )

            preview_state_key = (
                "documentary_localization_voice_preview_path_"
                f"{project.id}_{target_language}"
            )
            if st.button(
                tr("Documentary Voice Preview Button"),
                key=(
                    "documentary_localization_voice_preview_"
                    f"{project.id}_{target_language}"
                ),
            ):
                try:
                    with st.spinner(
                        tr("Documentary Voice Preview Generating")
                    ):
                        preview_path = _generate_documentary_voice_preview(
                            project.id,
                            voice_name=selected_voice,
                            text=_documentary_voice_preview_text(
                                target_language
                            ),
                        )
                except Exception as exc:
                    st.error(
                        tr("Documentary Voice Preview Failed").format(
                            error=str(exc)
                        )
                    )
                else:
                    st.session_state[preview_state_key] = str(preview_path)

            preview_value = st.session_state.get(preview_state_key)
            if preview_value and Path(preview_value).is_file():
                st.audio(preview_value)

            if st.button(
                tr("Documentary Localization Audio Generate"),
                type="primary",
                width="stretch",
                key=(
                    "documentary_localization_audio_"
                    f"{project.id}_{target_language}"
                ),
            ):
                try:
                    with st.spinner(
                        tr("Documentary Narration Audio Generating")
                    ):
                        synthesis = synthesize_narration(
                            project.id,
                            selected_voice,
                            language=target_language,
                        )
                except (
                    OSError,
                    ValueError,
                    NarrationSynthesisError,
                ) as exc:
                    st.error(
                        tr("Documentary Narration Audio Failed").format(
                            error=str(exc)
                        )
                    )
                else:
                    st.success(
                        tr("Documentary Narration Audio Complete").format(
                            generated=len(synthesis.generated),
                            reused=len(synthesis.reused),
                        )
                    )
                    st.rerun()
        else:
            st.info(tr("Documentary Localization No Narration"))

        output_path = documentary_render_path(
            project.id,
            language=target_language,
        )
        current_render = _render_output_is_current(
            project.id,
            language=target_language,
        )
        if output_path.is_file():
            if current_render:
                st.success(
                    tr("Documentary Localization Render Current").format(
                        language=_documentary_language_label(target_language)
                    )
                )
                st.video(str(output_path))
            else:
                st.warning(tr("Documentary Render Stale"))

        readiness_issues = documentary_render_readiness_issues(
            project.id,
            language=target_language,
        )
        if readiness_issues:
            st.warning(tr("Documentary Render Not Ready"))
            for issue in readiness_issues:
                st.caption(f"• {issue}")

        render_button_label = (
            tr("Documentary Render Again")
            if output_path.is_file()
            else tr("Documentary Localization Render Build")
        )
        if st.button(
            render_button_label,
            type="primary",
            width="stretch",
            disabled=bool(readiness_issues),
            key=(
                "documentary_localization_render_"
                f"{project.id}_{target_language}"
            ),
        ):
            try:
                with st.spinner(tr("Documentary Rendering")):
                    result = render_documentary(
                        project.id,
                        language=target_language,
                    )
            except (OSError, ValueError, DocumentaryRenderError) as exc:
                st.error(
                    tr("Documentary Render Failed").format(
                        error=str(exc)
                    )
                )
            else:
                st.success(
                    tr("Documentary Render Complete").format(
                        filename=result.name
                    )
                )
                st.rerun()


def _render_source_upload(project, tr: Tr) -> None:
    with st.expander(tr("Documentary Add Source"), expanded=not project.sources):
        upload_nonce_key = f"documentary_upload_nonce_{project.id}"
        upload_nonce = int(st.session_state.get(upload_nonce_key, 0) or 0)
        uploaded_file = st.file_uploader(
            tr("Documentary Upload Video"),
            type=["mp4", "mov"],
            accept_multiple_files=False,
            key=f"documentary_upload_{project.id}_{upload_nonce}",
            help=tr("Documentary Upload Video Help"),
        )
        title = st.text_input(
            tr("Documentary Source Title"),
            key=f"documentary_source_title_{project.id}",
        )
        source_type = st.selectbox(
            tr("Documentary Source Type"),
            options=_SOURCE_TYPES,
            format_func=lambda value: tr(_SOURCE_TYPE_LABELS[value]),
            key=f"documentary_source_type_{project.id}",
        )
        rights_status = st.selectbox(
            tr("Documentary Rights"),
            options=_RIGHTS_STATUSES,
            index=0,
            format_func=lambda value: tr(_RIGHTS_LABELS[value]),
            key=f"documentary_rights_{project.id}",
            help=tr("Documentary Rights Help"),
        )
        rights_note = st.text_area(
            tr("Documentary Rights Note"),
            key=f"documentary_rights_note_{project.id}",
            height=80,
        )

        attach_clicked = st.button(
            tr("Documentary Attach Source"),
            type="primary",
            width="stretch",
            disabled=uploaded_file is None,
            key=f"documentary_attach_source_{project.id}",
        )

        if not attach_clicked:
            return

        temp_path = None
        try:
            temp_path = _write_uploaded_video_to_temp(uploaded_file)
            source = attach_local_video(
                project.id,
                temp_path,
                title=title or uploaded_file.name,
                source_type=source_type,
                rights_status=rights_status,
                rights_note=rights_note,
                original_filename=uploaded_file.name,
            )
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state[upload_nonce_key] = upload_nonce + 1
            st.success(
                tr("Documentary Source Attached").format(
                    source_id=source.id,
                )
            )
            st.rerun()
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def render_documentary_application(tr: Tr) -> None:
    st.header(tr("Documentary Mode"))
    st.caption(tr("Documentary Mode Description"))

    _render_autopilot(tr)

    with st.expander(tr("Documentary Simple Advanced"), expanded=False):
        developer_mode = st.checkbox(
            tr("Documentary Developer Mode"),
            value=False,
            help=tr("Documentary Developer Mode Help"),
        )

    projects = list_projects()
    if not projects:
        with st.expander(tr("Documentary Manual Mode"), expanded=False):
            _render_story_discovery(tr)
            _render_create_project(tr)
        st.info(tr("Documentary No Projects"))
        return

    project_by_id = {project.id: project for project in projects}
    project_ids = list(project_by_id)

    if st.session_state.get("documentary_project_id") not in project_by_id:
        st.session_state["documentary_project_id"] = project_ids[0]

    selected_project_id = st.selectbox(
        tr("Documentary Select Project"),
        options=project_ids,
        key="documentary_project_id",
        format_func=lambda project_id: (
            f"{project_by_id[project_id].title}"
            if not developer_mode
            else f"{project_by_id[project_id].title} · {project_id}"
        ),
    )
    project = load_project(selected_project_id)

    if developer_mode:
        _render_story_discovery(tr)
        _render_create_project(tr)
        _render_project_overview(project, tr)
        _render_rights_review(project, tr)
        _render_external_source_local_copy(project, tr)
        _render_source_upload(project, tr)
        _render_transcription(project, tr)
        _render_story_planner(project, tr)
        _render_clip_selector(project, tr)
        project = load_project(selected_project_id)
        _render_narration_writer(project, tr)
        project = load_project(selected_project_id)
        _render_master_video(project, tr)
        project = load_project(selected_project_id)
        _render_localized_versions(project, tr)
        return

    selected_step = _render_simple_dashboard(project, tr)

    if selected_step == "story":
        st.markdown(f"### {tr('Documentary Guided Story Title')}")
        st.caption(tr("Documentary Guided Story Help"))
        _render_story_discovery(tr)
        _render_create_project(tr)
        return

    if selected_step == "materials":
        _render_guided_materials(project, tr)
        _render_rights_review(project, tr, simple=True)
        _render_external_source_local_copy(project, tr)
        _render_source_upload(project, tr)
        _render_transcription(project, tr, simple=True)
        return

    if selected_step == "production":
        st.markdown(f"### {tr('Documentary Simple Section Production')}")
        st.caption(tr("Documentary Simple Production Help"))
        _render_story_planner(project, tr)
        _render_clip_selector(project, tr)
        project = load_project(selected_project_id)
        _render_narration_writer(project, tr, show_voice=True)
        return

    st.markdown(f"### {tr('Documentary Simple Section Video')}")
    st.caption(tr("Documentary Simple Video Help"))
    _render_master_video(project, tr)
    project = load_project(selected_project_id)
    _render_localized_versions(project, tr)
