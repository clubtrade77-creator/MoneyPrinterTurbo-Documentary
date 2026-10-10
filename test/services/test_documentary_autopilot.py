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

    monkeypatch.setattr(autopilot, "discover_stories", lambda *a, **k: [story])
    monkeypatch.setattr(autopilot, "_choose_story", lambda candidates: story)
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
        "_hunt_source",
        lambda *a, **k: source,
    )
    monkeypatch.setattr(
        autopilot,
        "fetch_source_local_copy",
        lambda *a, **k: calls.append("fetch"),
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
            or SimpleNamespace(language="en")
        ),
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

    monkeypatch.setattr(autopilot, "discover_stories", lambda *a, **k: [story])
    monkeypatch.setattr(autopilot, "_choose_story", lambda candidates: story)
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
    monkeypatch.setattr(autopilot, "_hunt_source", lambda *a, **k: source)
    monkeypatch.setattr(autopilot, "fetch_source_local_copy", lambda *a, **k: None)
    monkeypatch.setattr(
        autopilot.subtitle_service,
        "get_whisper_model",
        lambda size: object(),
    )
    monkeypatch.setattr(
        autopilot,
        "transcribe_source",
        lambda *a, **k: SimpleNamespace(language="ru"),
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
