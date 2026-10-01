# MoneyPrinterTurbo Documentary — Day 1 Technical Audit

Date: 2026-10-01
Branch: `documentary-dev`
Goal: add a documentary workflow based on real source footage without breaking the existing short-video pipeline.

## 1. Architecture decision

Do not rewrite the existing MoneyPrinterTurbo pipeline and do not force documentary semantics into `app/services/task.py` or `combine_videos()`.

Keep the current short-video workflow intact and add a parallel documentary package:

```text
app/services/documentary/
  __init__.py
  pipeline.py
  youtube_source.py
  transcription.py
  story_planner.py
  clip_selector.py
  viral_editor.py
  localization.py
  renderer.py
  rights.py
```

Add documentary-specific models in:

```text
app/models/documentary.py
```

The first MVP should use the existing task directory/storage conventions and reuse existing LLM, TTS, Whisper, subtitle, FFmpeg/MoviePy, state and artifact services.

## 2. Existing modules — keep / adapt / add

### `app/services/video.py`
Keep:
- MoviePy/FFmpeg rendering infrastructure
- codec selection and hardware fallback
- canvas fit/crop (`cover` / `contain`)
- subtitle rendering helpers
- BGM mixing
- render heartbeats and file/resource cleanup
- `SubClippedVideoClip` concept with explicit start/end timestamps

Do not use `combine_videos()` as the documentary timeline engine. It currently breaks source footage into uniform clips, then arranges them using random/sequential logic and assumes narration duration determines the visual timeline.

Documentary requirement:
- scene-driven source selection
- exact `source_start` / `source_end`
- original clip audio when required
- narration over footage when required
- mixed transitions between narration and original sound
- documents/images/title cards as first-class scene types

Add a separate `documentary/renderer.py` that reuses lower-level video helpers.

### `app/services/material.py`
Keep for:
- optional stock B-roll
- provider download infrastructure
- provenance/source metadata patterns
- local material support

Do not treat stock/generated material as the primary documentary source.

Extend the provenance concept with documentary source metadata:
- source URL
- publisher / agency / YouTube channel
- source type (bodycam, CCTV, court, interview, document, photo, YouTube, B-roll)
- incident/case ID
- publication date
- rights status
- rights/license note
- original filename
- local filename
- optional checksum

### YouTube source ingestion
YouTube is a first-class Documentary source, not merely a discovery aid.

Add `app/services/documentary/youtube_source.py`.

MVP behavior:
- accept one or many YouTube URLs pasted by the user
- resolve and store public metadata needed for editorial traceability: video ID, title, channel, source URL and publication metadata when available
- register every YouTube item as a `SourceAsset`
- support a local media copy when the user has permission/rights to download or otherwise provides the file
- when direct acquisition is not permitted or not available, retain the YouTube URL as an external source reference and require an authorized/local copy before rendering that footage
- transcribe the local/authorized media copy through the same Whisper pipeline as any other source
- allow the Clip Selector to return exact YouTube-source time ranges once a usable media copy exists

Recommended YouTube provenance fields:

```text
source_type: youtube
youtube_video_id
youtube_url
youtube_title
youtube_channel
youtube_published_at
rights_status
rights_note
local_file
```

The system must not mark a YouTube video as reusable merely because it is publicly viewable. Copyright/license status remains separate from discoverability.

### `app/services/task.py`
Keep unchanged for legacy short-video tasks.

The current pipeline is:
`script -> search terms -> narration -> subtitles -> materials -> render -> optional publish`.

Documentary pipeline must instead be:
`sources (local + YouTube + other originals) -> transcription/index -> research facts -> story plan -> scene timeline -> narration/localization -> render -> QA`.

Add `documentary/pipeline.py` rather than adding many documentary branches to `_run_pipeline()`.

Reuse task state/progress/error conventions where practical.

### `app/services/llm.py`
Keep the provider abstraction and transport layer. It already supports multiple LLM providers and a common text-generation path.

Add documentary-specific structured prompts/functions without replacing the existing `generate_script()`:
- `plan_documentary_story()`
- `select_candidate_clips()`
- `audit_retention()`
- `localize_documentary_script()`
- `generate_publish_package()`

For MVP, model output should be strict JSON validated by Pydantic documentary models.

### `app/services/voice.py`
Keep. It already has a broad TTS provider layer including ElevenLabs and multiple other providers.

Add only a documentary language/voice profile abstraction so each language can have a stable narrator configuration:
- `en`
- `ru`
- `es`

Narration should be generated per narration scene or per narration block, not necessarily as one monolithic audio file.

### `app/services/subtitle.py`
Keep Whisper model loading, word timestamps, SRT writer and correction utilities.

Refactor/add a reusable structured transcription function that returns JSON segments directly, for example:

```json
{
  "language": "en",
  "segments": [
    {"start": 12.4, "end": 16.8, "text": "..."}
  ]
}
```

Current `create()` writes subtitles; documentary mode also needs the segments as searchable source data.

### `app/services/twelvelabs.py`
Optional enhancement, not required for MVP.

Potential future use:
- visual understanding when transcript alone is insufficient
- clip QA
- semantic search over source video

MVP should work with local Whisper + timestamps first to avoid unnecessary API cost.

### `webui/Main.py`
Current WebUI is a large Streamlit monolith. Avoid embedding the whole documentary workflow directly into the existing short-video form.

Add a separate documentary UI module/component and expose it as a distinct mode/page. MVP UI should have only:
- create/select documentary project
- upload source files
- paste YouTube source URL(s)
- list sources and provenance status
- run transcription
- show/edit story plan
- show/edit scene timeline
- render master
- later: generate EN/RU/ES variants

Existing i18n already includes EN/RU/ES, so UI localization can reuse the current localization mechanism.

## 3. New documentary data model

Recommended core types:

```text
DocumentaryProject
SourceAsset
TranscriptSegment
ResearchFact
DocumentaryPlan
DocumentaryScene
LanguageVariant
RenderManifest
```

Recommended scene types:

```text
original_clip
narration_over_source
narration_over_image
document
broll
title_card
```

Recommended audio modes:

```text
original
narration
mixed
muted
```

Minimum `DocumentaryScene` fields:

```text
id
scene_type
source_id
source_start
source_end
audio_mode
narration_text
original_volume
narration_volume
subtitle_mode
on_screen_text
purpose
```

`purpose` can use narrative labels such as:

```text
hook
context
conflict
escalation
reveal
payoff
transition
```

## 4. MVP file layout

Use the existing task/project storage root, with documentary subfolders:

```text
<project>/
  sources/
  transcripts/
  research/
  plans/
  audio/
    en/
    ru/
    es/
  subtitles/
  renders/
  manifests/
```

Do not copy the same source video three times for the three languages.

## 5. Rendering rule

The renderer must never decide story order by random clip selection.

It consumes an approved timeline. Example:

```json
{
  "scene_type": "original_clip",
  "source_id": "bodycam_01",
  "source_start": 221.4,
  "source_end": 231.8,
  "audio_mode": "original",
  "purpose": "reveal"
}
```

For `narration_over_source`, source audio is ducked/muted and narration is mixed over it.
For `mixed`, narration and source sound can have separate gain envelopes.

YouTube-backed sources use the same scene model after a usable local/authorized media copy has been attached to the `SourceAsset`; the timeline continues to preserve the original YouTube URL and video ID for traceability.

## 6. Viral/retention layer

Do not use an opaque numerical “viral score” as the product decision.

The Viral Editor should return concrete diagnostics:
- strongest candidate opening moment
- unresolved question/open loop
- stretches with too much narration and no new evidence
- long stretches without a source change or new information
- reveal/payoff placement
- candidate cut points for Shorts

Human can approve/reject the proposed timeline before rendering.

## 7. Rights/source traceability

Every source asset must have a provenance record before final render.
The system should distinguish:
- official/public source
- licensed source
- user-owned source
- unknown/review required

YouTube is a distribution platform, not a rights status. A public YouTube URL does not by itself establish permission to reuse the footage.

The final render manifest should record which source file and source time range was used in every scene, plus the originating YouTube URL/video ID where applicable.

## 8. What is already strong enough to reuse

- Python 3.11+ project
- MoviePy 2.2.1
- FFmpeg integration and codec fallback
- Streamlit UI
- FastAPI
- OpenAI-compatible / multi-provider LLM layer
- faster-whisper 1.1.0
- broad TTS provider layer
- subtitles/SRT rendering
- local media upload/preprocessing
- task state and progress tracking
- source metadata concept
- extensive automated test suite

## 9. What is missing for Documentary MVP

1. documentary-specific project/schema
2. YouTube SourceAsset ingestion and provenance adapter
3. structured source transcription JSON
4. source-first documentary pipeline
5. story planner producing validated scene JSON
6. exact clip selector using timestamps
7. renderer that preserves original source audio
8. per-scene audio mixing
9. rights/provenance manifest
10. dedicated Documentary UI
11. documentary tests

## 10. Implementation order after baseline run

After confirming the unmodified fork runs correctly on the target Mac:

1. add documentary Pydantic models
2. add SourceAsset manifest including local and YouTube sources
3. add YouTube URL ingestion/metadata adapter
4. add structured transcription service
5. add deterministic scene renderer for manually supplied timeline JSON
6. test ORIGINAL / NARRATION / MIXED audio modes
7. add Story Planner
8. add Clip Selector
9. add Viral Editor
10. add EN/RU/ES localization
11. add UI workflow

## 11. Day 1 conclusion

MoneyPrinterTurbo is a suitable base. The main rendering, TTS, subtitle, LLM and task infrastructure does not need to be rebuilt.

The core design rule is isolation: keep the existing short-video pipeline working and build Documentary Mode as a parallel pipeline using shared low-level services.

Documentary Mode source priority should be:
1. original/official footage for the story, including authorized YouTube sources
2. other verified source footage/documents/photos
3. stock B-roll where the story has no direct visual
4. generated visuals only when needed and clearly separated from original evidence footage

Day 2 should be a baseline installation/run of the existing fork on the target Mac before feature code is added. This gives a known-good baseline and prevents documentary changes from being blamed for pre-existing environment/setup problems.
