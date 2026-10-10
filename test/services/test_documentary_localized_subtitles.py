import json
from pathlib import Path

from app.models.documentary import (
    AudioMode,
    DocumentaryScene,
    DocumentaryTranscript,
    RightsStatus,
    SceneType,
    SourceAsset,
    SourceType,
    TranscriptSegment,
    VideoMetadata,
)
from app.services.documentary.localization import localize_project
from app.services.documentary.localized_subtitles import (
    build_localized_subtitle_cues,
    write_localized_subtitles,
)
from app.services.documentary.project import (
    add_source,
    create_project,
    load_project,
    project_dir,
    save_project,
    sha256_file,
)
from app.services.documentary.transcription import transcript_path


def _localized_project(tmp_path: Path):
    project = create_project(
        "Localized subtitles",
        project_id="doc_localized_subtitles",
        master_language="en",
        root=tmp_path,
    )
    local_path = (
        project_dir(project.id, tmp_path)
        / "sources"
        / "source.mp4"
    )
    local_path.write_bytes(b"video")
    source = SourceAsset(
        id="source_subtitles",
        source_type=SourceType.local_video,
        local_path=str(local_path.resolve()),
        checksum_sha256=sha256_file(local_path),
        video_metadata=VideoMetadata(
            duration_seconds=10.0,
            width=1280,
            height=720,
            fps=30,
            has_audio=True,
            video_codec="h264",
            audio_codec="aac",
            container="mov,mp4",
            file_size_bytes=local_path.stat().st_size,
        ),
        rights_status=RightsStatus.user_owned,
    )
    add_source(project.id, source, root=tmp_path)

    transcript = DocumentaryTranscript(
        source_id=source.id,
        source_checksum_sha256=source.checksum_sha256,
        language="en",
        language_mode="forced",
        requested_language="en",
        media_duration_seconds=10.0,
        model_size="small",
        full_text="First line. Second line.",
        segments=[
            TranscriptSegment(
                id=0,
                start_seconds=0.5,
                end_seconds=1.5,
                text="First line.",
            ),
            TranscriptSegment(
                id=1,
                start_seconds=5.5,
                end_seconds=6.5,
                text="Second line.",
            ),
        ],
    )
    transcript_path(project.id, source.id, tmp_path).write_text(
        json.dumps(transcript.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    project = load_project(project.id, tmp_path)
    project.plan.scenes = [
        DocumentaryScene(
            id="scene_first",
            scene_type=SceneType.original_clip,
            source_id=source.id,
            source_start=0.0,
            source_end=2.0,
            audio_mode=AudioMode.original,
            transcript_segment_ids=[0],
        ),
        DocumentaryScene(
            id="scene_second",
            scene_type=SceneType.original_clip,
            source_id=source.id,
            source_start=5.0,
            source_end=7.0,
            audio_mode=AudioMode.original,
            transcript_segment_ids=[1],
        ),
    ]
    save_project(project, tmp_path)

    localize_project(
        project.id,
        "ru",
        root=tmp_path,
        generate_fn=lambda prompt: json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": "scene_first",
                        "narration_text": "",
                        "on_screen_text": "",
                        "subtitle_segments": [
                            {
                                "source_id": source.id,
                                "segment_id": 0,
                                "text": "Первая строка.",
                            }
                        ],
                    },
                    {
                        "scene_id": "scene_second",
                        "narration_text": "",
                        "on_screen_text": "",
                        "subtitle_segments": [
                            {
                                "source_id": source.id,
                                "segment_id": 1,
                                "text": "Вторая строка.",
                            }
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        review_fn=lambda prompt: json.dumps(
            {"supported": True, "issues": []}
        ),
    )
    return project


def test_localized_subtitle_cues_use_edited_timeline_time(tmp_path: Path):
    project = _localized_project(tmp_path)

    cues = build_localized_subtitle_cues(
        project.id,
        "ru",
        root=tmp_path,
    )

    assert len(cues) == 2
    assert cues[0].start_seconds == 0.5
    assert cues[0].end_seconds == 1.5
    assert cues[1].start_seconds == 2.5
    assert cues[1].end_seconds == 3.5
    assert cues[1].text == "Вторая строка."


def test_write_localized_subtitles_is_stable_when_content_is_unchanged(
    tmp_path: Path,
):
    project = _localized_project(tmp_path)

    path = write_localized_subtitles(
        project.id,
        "ru",
        root=tmp_path,
    )
    assert path is not None
    original_mtime = path.stat().st_mtime

    second = write_localized_subtitles(
        project.id,
        "ru",
        root=tmp_path,
    )

    assert second == path
    assert path.stat().st_mtime == original_mtime
    text = path.read_text(encoding="utf-8")
    assert "Первая строка." in text
    assert "Вторая строка." in text
