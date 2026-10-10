from pathlib import Path
from types import SimpleNamespace

from app.models.documentary import AudioMode, RightsStatus, SourceAsset, SourceType
from app.services.documentary import autopilot


def _project_snapshot(source: SourceAsset):
    scene = SimpleNamespace(
        source_id=source.id,
        audio_mode=AudioMode.narration,
        narration_text="Grounded narration",
    )
    return SimpleNamespace(
        master_language="ru",
        plan=SimpleNamespace(scenes=[scene]),
        sources=[source],
    )


def _checkpoint_stub(project_id, state, *, stage, status="running", root=None, **updates):
    result = dict(state)
    result.update(updates)
    result["stage"] = stage
    result["status"] = status
    return result


def test_run_autopilot_builds_three_language_previews(monkeypatch, tmp_path: Path):
    events = []
    calls = []
    story = SimpleNamespace(title="Autopilot story")
    created = SimpleNamespace(id="doc_autopilot", title=story.title)
    source = SourceAsset(
        id="youtube_autopilot",
        source_type=SourceType.youtube,
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        rights_status=RightsStatus.unknown_review_required,
    )
    project_snapshot = _project_snapshot(source)

    monkeypatch.setattr(autopilot, "_checkpoint_autopilot", _checkpoint_stub)
    source_bundle = autopilot._SourceBundle(media_source=source)
    monkeypatch.setattr(
        autopilot,
        "_discover_autopilot_stories",
        lambda *a, **k: [story],
    )
    monkeypatch.setattr(
        autopilot,
        "_choose_story_with_source",
        lambda *a, **k: (story, source_bundle),
    )
    monkeypatch.setattr(
        autopilot,
        "create_project",
        lambda *a, **k: created,
    )
    monkeypatch.setattr(
        autopilot,
        "candidate_to_source",
        lambda candidate: SourceAsset(
            id="story_autopilot",
            source_type=SourceType.news,
            source_url="https://example.com/story",
        ),
    )
    monkeypatch.setattr(
        autopilot,
        "add_source",
        lambda *a, **k: calls.append("add_source"),
    )
    monkeypatch.setattr(
        autopilot,
        "fetch_source_local_copy",
        lambda *a, **k: (calls.append("fetch") or source),
    )
    monkeypatch.setattr(
        autopilot.subtitle_service,
        "get_whisper_model",
        lambda size: object(),
    )
    monkeypatch.setattr(
        autopilot,
        "transcribe_source",
        lambda *a, **k: (
            calls.append("transcribe")
            or SimpleNamespace(language="en", full_text="Matching story transcript.")
        ),
    )
    monkeypatch.setattr(
        autopilot,
        "_story_transcript_matches",
        lambda *a, **k: (True, "test match"),
    )
    monkeypatch.setattr(
        autopilot,
        "plan_story",
        lambda *a, **k: calls.append("plan_story"),
    )
    monkeypatch.setattr(
        autopilot,
        "select_clips",
        lambda *a, **k: calls.append("select_clips"),
    )
    monkeypatch.setattr(
        autopilot,
        "write_narration",
        lambda *a, **k: calls.append("write_narration"),
    )
    monkeypatch.setattr(
        autopilot,
        "load_project",
        lambda *a, **k: project_snapshot,
    )
    monkeypatch.setattr(
        autopilot,
        "_resolve_voice",
        lambda language, explicit_voice="": f"voice:{language}",
    )
    monkeypatch.setattr(
        autopilot,
        "synthesize_narration",
        lambda *a, **k: calls.append(f"tts:{k['language']}"),
    )
    monkeypatch.setattr(
        autopilot,
        "localize_project",
        lambda *a, **k: calls.append(f"localize:{a[1]}"),
    )

    def fake_render(project_id, *, language=None, preview=False, root=None):
        # The source transcript is English in this test, so Autopilot promotes
        # English to the master language and passes language=None for that preview.
        label = language or "en"
        calls.append(f"render:{label}:{preview}")
        path = tmp_path / f"{label}-preview.mp4"
        path.write_bytes(b"preview")
        return path

    monkeypatch.setattr(autopilot, "render_documentary", fake_render)

    result = autopilot.run_autopilot(
        autopilot.AutopilotProfile(
            topic="test",
            output_languages=("ru", "en", "es"),
        ),
        root=tmp_path,
        progress=events.append,
    )

    assert result.project_id == "doc_autopilot"
    assert set(result.preview_paths) == {"ru", "en", "es"}
    assert result.rights_review_required == ("youtube_autopilot",)
    assert result.final_master_created is False
    assert "fetch" in calls
    assert "transcribe" in calls
    assert "plan_story" in calls
    assert "select_clips" in calls
    assert "write_narration" in calls
    assert "tts:ru" in calls
    assert "tts:en" in calls
    assert "tts:es" in calls
    assert "render:ru:True" in calls
    assert "render:en:True" in calls
    assert "render:es:True" in calls
    assert events[-1].stage == "done"
    assert events[-1].fraction == 1.0


def test_run_autopilot_renders_final_when_rights_are_cleared(
    monkeypatch,
    tmp_path: Path,
):
    story = SimpleNamespace(title="Cleared story")
    created = SimpleNamespace(id="doc_cleared", title=story.title)
    source = SourceAsset(
        id="source_cleared",
        source_type=SourceType.youtube,
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        rights_status=RightsStatus.user_owned,
    )
    project_snapshot = _project_snapshot(source)
    renders = []

    monkeypatch.setattr(autopilot, "_checkpoint_autopilot", _checkpoint_stub)
    source_bundle = autopilot._SourceBundle(media_source=source)
    monkeypatch.setattr(
        autopilot,
        "_discover_autopilot_stories",
        lambda *a, **k: [story],
    )
    monkeypatch.setattr(
        autopilot,
        "_choose_story_with_source",
        lambda *a, **k: (story, source_bundle),
    )
    monkeypatch.setattr(autopilot, "create_project", lambda *a, **k: created)
    monkeypatch.setattr(
        autopilot,
        "candidate_to_source",
        lambda candidate: SourceAsset(
            id="story_cleared",
            source_type=SourceType.news,
            source_url="https://example.com/story",
        ),
    )
    monkeypatch.setattr(autopilot, "add_source", lambda *a, **k: None)
    monkeypatch.setattr(
        autopilot,
        "fetch_source_local_copy",
        lambda *a, **k: source,
    )
    monkeypatch.setattr(
        autopilot.subtitle_service,
        "get_whisper_model",
        lambda size: object(),
    )
    monkeypatch.setattr(
        autopilot,
        "transcribe_source",
        lambda *a, **k: SimpleNamespace(
            language="ru",
            full_text="Matching story transcript.",
        ),
    )
    monkeypatch.setattr(
        autopilot,
        "_story_transcript_matches",
        lambda *a, **k: (True, "test match"),
    )
    monkeypatch.setattr(autopilot, "plan_story", lambda *a, **k: None)
    monkeypatch.setattr(autopilot, "select_clips", lambda *a, **k: None)
    monkeypatch.setattr(autopilot, "write_narration", lambda *a, **k: None)
    monkeypatch.setattr(autopilot, "load_project", lambda *a, **k: project_snapshot)
    monkeypatch.setattr(autopilot, "save_project", lambda *a, **k: None)
    monkeypatch.setattr(autopilot, "_resolve_voice", lambda *a, **k: "voice:ru")
    monkeypatch.setattr(autopilot, "synthesize_narration", lambda *a, **k: None)

    def fake_render(project_id, *, language=None, preview=False, root=None):
        renders.append(preview)
        path = tmp_path / ("preview.mp4" if preview else "master.mp4")
        path.write_bytes(b"video")
        return path

    monkeypatch.setattr(autopilot, "render_documentary", fake_render)

    result = autopilot.run_autopilot(
        autopilot.AutopilotProfile(output_languages=("ru",)),
        root=tmp_path,
    )

    assert result.rights_review_required == ()
    assert result.final_master_created is True
    assert renders == [True, False]


def test_fetch_first_available_source_uses_fallback_after_download_failure(
    monkeypatch,
):
    story = SimpleNamespace(title="Fallback story")
    primary = SourceAsset(
        id="youtube_primary",
        source_type=SourceType.youtube,
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
    )
    fallback = SourceAsset(
        id="youtube_fallback",
        source_type=SourceType.youtube,
        source_url="https://www.youtube.com/watch?v=lmnopqrstuv",
    )
    attempts = []
    added = []

    def fake_fetch(project_id, source_id, *, root=None):
        attempts.append(source_id)
        if source_id == primary.id:
            raise autopilot.DocumentaryMediaFetchError("format unavailable")
        return fallback

    monkeypatch.setattr(autopilot, "fetch_source_local_copy", fake_fetch)
    monkeypatch.setattr(
        autopilot,
        "_fallback_video_sources",
        lambda *a, **k: [fallback],
    )
    monkeypatch.setattr(
        autopilot,
        "add_source",
        lambda project_id, source, *, root=None: added.append(source.id),
    )

    result = autopilot._fetch_first_available_source(
        "doc_fallback",
        story,
        primary,
        source_limit=4,
    )

    assert result.id == fallback.id
    assert attempts == [primary.id, fallback.id]
    assert added == [fallback.id]


def test_published_footage_story_filter_rejects_camera_mentions():
    mention_only = SimpleNamespace(
        title="Police seek dashcam footage after shooting",
        score=80,
        footage_score=16,
        freshness_score=30,
        story_score=30,
    )
    published = SimpleNamespace(
        title="Bodycam footage shows police shooting",
        score=95,
        footage_score=32,
        freshness_score=30,
        story_score=30,
    )

    result = autopilot._published_footage_stories([mention_only, published])

    assert result == [published]


def test_discover_autopilot_stories_expands_to_seven_days(monkeypatch):
    calls = []
    weak = SimpleNamespace(
        title="Police seek dashcam footage",
        score=60,
        footage_score=16,
        freshness_score=30,
        story_score=20,
    )
    strong = SimpleNamespace(
        title="Bodycam footage shows dramatic rescue",
        score=90,
        footage_score=32,
        freshness_score=18,
        story_score=30,
    )

    def fake_discover(topic, *, lookback_hours, limit):
        calls.append(lookback_hours)
        return [weak] if lookback_hours == 72 else [strong]

    monkeypatch.setattr(autopilot, "discover_stories", fake_discover)

    result = autopilot._discover_autopilot_stories(
        autopilot.AutopilotProfile(
            lookback_hours=72,
            discovery_limit=30,
        )
    )

    assert result == [strong]
    assert calls == [72, 168]


def test_fetch_and_transcribe_skips_source_without_recognizable_speech(
    monkeypatch,
):
    story = SimpleNamespace(title="Speech fallback story")
    primary = SourceAsset(
        id="youtube_silent",
        source_type=SourceType.youtube,
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
    )
    fallback = SourceAsset(
        id="youtube_spoken",
        source_type=SourceType.youtube,
        source_url="https://www.youtube.com/watch?v=lmnopqrstuv",
    )
    attempts = []
    added = []

    monkeypatch.setattr(
        autopilot,
        "_fallback_video_sources",
        lambda *a, **k: [fallback],
    )
    monkeypatch.setattr(
        autopilot,
        "add_source",
        lambda project_id, source, *, root=None: added.append(source.id),
    )
    monkeypatch.setattr(
        autopilot,
        "fetch_source_local_copy",
        lambda project_id, source_id, *, root=None: (
            primary if source_id == primary.id else fallback
        ),
    )

    def fake_transcribe(project_id, source_id, **kwargs):
        attempts.append(source_id)
        if source_id == primary.id:
            raise autopilot.TranscriptionError(
                "Whisper detected no recognizable speech"
            )
        return SimpleNamespace(language="en", full_text="Recognized speech.")

    monkeypatch.setattr(autopilot, "transcribe_source", fake_transcribe)
    monkeypatch.setattr(
        autopilot,
        "_story_transcript_matches",
        lambda *a, **k: (True, "test match"),
    )

    source, transcript = autopilot._fetch_and_transcribe_first_available_source(
        "doc_speech_fallback",
        story,
        primary,
        source_limit=4,
        model=object(),
    )

    assert source.id == fallback.id
    assert transcript.full_text == "Recognized speech."
    assert attempts == [primary.id, fallback.id]
    assert added == [primary.id, fallback.id]


def test_story_transcript_verifier_rejects_different_venue():
    story = SimpleNamespace(
        title="Surveillance video shows alleged abuse at therapy center for kids"
    )
    transcript = SimpleNamespace(
        full_text=(
            "The report concerns a home daycare. Investigators reviewed video "
            "from the daycare and described the incident."
        )
    )

    matched, reason = autopilot._story_transcript_matches(
        story,
        transcript,
        verify_fn=lambda prompt: (
            '{"same_event": false, "reason": '
            '"headline says therapy center but transcript says home daycare"}'
        ),
    )

    assert matched is False
    assert "home daycare" in reason


def test_fetch_and_transcribe_skips_mismatched_story_source(monkeypatch):
    story = SimpleNamespace(title="Therapy center surveillance story")
    primary = SourceAsset(
        id="youtube_wrong",
        source_type=SourceType.youtube,
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
    )
    fallback = SourceAsset(
        id="youtube_right",
        source_type=SourceType.youtube,
        source_url="https://www.youtube.com/watch?v=lmnopqrstuv",
    )
    attempts = []

    monkeypatch.setattr(
        autopilot,
        "_fallback_video_sources",
        lambda *a, **k: [fallback],
    )
    monkeypatch.setattr(autopilot, "add_source", lambda *a, **k: None)
    monkeypatch.setattr(
        autopilot,
        "fetch_source_local_copy",
        lambda project_id, source_id, *, root=None: (
            primary if source_id == primary.id else fallback
        ),
    )
    monkeypatch.setattr(
        autopilot,
        "transcribe_source",
        lambda project_id, source_id, **kwargs: SimpleNamespace(
            language="en",
            full_text=(
                "This is a home daycare report."
                if source_id == primary.id
                else "This report concerns the therapy center."
            ),
        ),
    )

    def fake_match(story_arg, transcript_arg, **kwargs):
        attempts.append(transcript_arg.full_text)
        return (
            ("therapy center" in transcript_arg.full_text.lower()),
            "venue check",
        )

    monkeypatch.setattr(autopilot, "_story_transcript_matches", fake_match)

    source, transcript = autopilot._fetch_and_transcribe_first_available_source(
        "doc_story_match",
        story,
        primary,
        source_limit=4,
        model=object(),
    )

    assert source.id == fallback.id
    assert "therapy center" in transcript.full_text.lower()
    assert len(attempts) == 2


def test_run_autopilot_resume_reuses_saved_transcript_and_story_plan(
    monkeypatch,
    tmp_path: Path,
):
    source = SourceAsset(
        id="youtube_resume",
        source_type=SourceType.youtube,
        source_url="https://www.youtube.com/watch?v=abcdefghijk",
        rights_status=RightsStatus.unknown_review_required,
    )
    scene = SimpleNamespace(
        id="scene_resume",
        source_id=source.id,
        story_beat_id="beat_01",
        audio_mode=AudioMode.original,
        narration_text="",
    )
    project = SimpleNamespace(
        id="doc_resume",
        title="Resume story",
        master_language="en",
        sources=[source],
        plan=SimpleNamespace(
            scenes=[scene],
            story_plan_fingerprint="fingerprint",
        ),
    )
    transcript = SimpleNamespace(
        language="en",
        full_text="This transcript matches the resume story.",
    )
    story = autopilot.StoryCandidate(
        id="story_resume",
        title=project.title,
        url="https://example.com/story",
        publisher="Example",
        published_at="",
        language="en",
        source_country="United States",
        image_url="",
        discovery_query="",
        score=90,
        footage_score=40,
        freshness_score=30,
        story_score=20,
        reasons=(),
    )
    state = {
        "version": 1,
        "status": "failed",
        "stage": "story_plan",
        "topic": "",
        "story": autopilot._story_state_payload(story),
        "source_id": source.id,
        "source_verified": True,
    }
    calls = []

    monkeypatch.setattr(autopilot, "_load_autopilot_state", lambda *a, **k: state)
    monkeypatch.setattr(autopilot, "_checkpoint_autopilot", _checkpoint_stub)
    monkeypatch.setattr(autopilot, "load_project", lambda *a, **k: project)
    monkeypatch.setattr(
        autopilot,
        "load_source_transcript",
        lambda *a, **k: transcript,
    )
    monkeypatch.setattr(
        autopilot,
        "discover_stories",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("resume must not rediscover stories")
        ),
    )
    monkeypatch.setattr(
        autopilot,
        "transcribe_source",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("resume must not rerun Whisper")
        ),
    )
    monkeypatch.setattr(
        autopilot,
        "load_story_plan",
        lambda *a, **k: SimpleNamespace(
            beats=[
                SimpleNamespace(
                    id="beat_01",
                    original_audio_priority=True,
                )
            ]
        ),
    )
    monkeypatch.setattr(
        autopilot,
        "load_clip_plan",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr(
        autopilot,
        "_master_narration_is_complete",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        autopilot,
        "_project_has_narrated_scenes",
        lambda *a, **k: False,
    )

    def fake_render(project_id, *, language=None, preview=False, root=None):
        calls.append((language, preview))
        path = tmp_path / f"{language or 'master'}-{preview}.mp4"
        path.write_bytes(b"video")
        return path

    monkeypatch.setattr(autopilot, "render_documentary", fake_render)
    monkeypatch.setattr(
        autopilot,
        "load_localization_plan",
        lambda *a, **k: SimpleNamespace(),
    )
    monkeypatch.setattr(
        autopilot,
        "_rights_review_source_ids",
        lambda *a, **k: (source.id,),
    )

    result = autopilot.run_autopilot(
        autopilot.AutopilotProfile(output_languages=("en",)),
        root=tmp_path,
        resume_project_id=project.id,
    )

    assert result.project_id == project.id
    assert result.source_id == source.id
    assert result.title == project.title
    assert calls == [(None, True)]
