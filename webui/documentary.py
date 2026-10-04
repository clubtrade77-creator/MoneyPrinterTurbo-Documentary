from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

import streamlit as st

from app.models.documentary import RightsStatus, SourceType
from app.services import subtitle as subtitle_service
from app.services.documentary.project import (
    attach_local_video,
    create_project,
    list_projects,
    load_project,
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


def _source_duration(source) -> str:
    metadata = source.video_metadata
    if metadata is None:
        return "-"
    return f"{metadata.duration_seconds:.1f}s"


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
                use_container_width=True,
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
        use_container_width=True,
        hide_index=True,
    )


def _load_transcript_if_available(project_id: str, source_id: str):
    try:
        return load_source_transcript(project_id, source_id)
    except FileNotFoundError:
        return None


def _render_transcription(project, tr: Tr) -> None:
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
            cols = st.columns(3)
            cols[0].metric(
                tr("Documentary Transcript Language"),
                (transcript.language or "-").upper(),
            )
            cols[1].metric(
                tr("Documentary Transcript Segments"),
                len(transcript.segments),
            )
            cols[2].metric(
                tr("Documentary Transcript Model"),
                transcript.model_size or "-",
            )
            if transcript.full_text:
                st.text_area(
                    tr("Documentary Transcript Preview"),
                    value=transcript.full_text,
                    height=120,
                    disabled=True,
                    key=f"documentary_transcript_preview_{project.id}_{source.id}",
                )

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
            use_container_width=True,
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
            use_container_width=True,
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

    _render_create_project(tr)

    projects = list_projects()
    if not projects:
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
            f"{project_by_id[project_id].title} · {project_id}"
        ),
    )
    project = load_project(selected_project_id)

    _render_project_overview(project, tr)
    _render_source_upload(project, tr)
    _render_transcription(project, tr)
