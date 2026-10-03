from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SourceType(str, Enum):
    local_video = "local_video"
    youtube = "youtube"
    bodycam = "bodycam"
    cctv = "cctv"
    court = "court"
    interview = "interview"
    news = "news"
    audio = "audio"
    document = "document"
    photo = "photo"
    broll = "broll"
    other = "other"


class ProvenanceType(str, Enum):
    """Where the material came from; this is intentionally separate from rights."""

    user_provided = "user_provided"
    third_party_platform = "third_party_platform"
    official_public_source = "official_public_source"
    public_record = "public_record"
    stock_provider = "stock_provider"
    other = "other"


class RightsStatus(str, Enum):
    """Reuse/publication rights status, independent from source provenance."""

    user_owned = "user_owned"
    licensed = "licensed"
    permission_confirmed = "permission_confirmed"
    public_domain = "public_domain"
    unknown_review_required = "unknown_review_required"


_PUBLISHABLE_RIGHTS = {
    RightsStatus.user_owned,
    RightsStatus.licensed,
    RightsStatus.permission_confirmed,
    RightsStatus.public_domain,
}


class SceneType(str, Enum):
    original_clip = "original_clip"
    narration_over_source = "narration_over_source"
    narration_over_image = "narration_over_image"
    document = "document"
    broll = "broll"
    title_card = "title_card"


class AudioMode(str, Enum):
    original = "original"
    narration = "narration"
    mixed = "mixed"
    muted = "muted"


class NarrativePurpose(str, Enum):
    hook = "hook"
    context = "context"
    conflict = "conflict"
    question = "question"
    escalation = "escalation"
    reveal = "reveal"
    twist = "twist"
    payoff = "payoff"
    next_hook = "next_hook"
    transition = "transition"


_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,160}$")
_BEAT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,120}$")


class TranscriptWord(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str
    probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("transcript word end_seconds must be greater than start_seconds")
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("transcript word text is required")
        return self


class TranscriptSegment(BaseModel):
    id: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)
    avg_logprob: Optional[float] = None
    no_speech_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_time_range(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("transcript segment end_seconds must be greater than start_seconds")
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("transcript segment text is required")
        return self


class DocumentaryTranscript(BaseModel):
    version: int = 1
    source_id: str
    source_checksum_sha256: str = ""
    language: str = ""
    language_mode: Literal["auto", "forced"] = "auto"
    requested_language: Optional[str] = None
    language_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    media_duration_seconds: Optional[float] = Field(default=None, gt=0)
    model_size: str = ""
    full_text: str = ""
    segments: list[TranscriptSegment] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not _SOURCE_ID_RE.fullmatch(value or ""):
            raise ValueError("invalid documentary source id")
        return value


class StoryEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    segment_ids: list[int] = Field(min_length=1)
    note: str = Field(default="", max_length=1000)

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not _SOURCE_ID_RE.fullmatch(value or ""):
            raise ValueError("invalid documentary source id")
        return value

    @field_validator("segment_ids")
    @classmethod
    def validate_segment_ids(cls, value: list[int]) -> list[int]:
        if any(segment_id < 0 for segment_id in value):
            raise ValueError("story evidence segment ids must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("story evidence segment ids must be unique")
        return value


class StoryBeat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default_factory=lambda: f"beat_{uuid4().hex[:12]}",
        max_length=120,
    )
    purpose: NarrativePurpose
    title: str = Field(max_length=200)
    summary: str = Field(max_length=1500)
    target_duration_seconds: float = Field(gt=0, le=180)
    narration_goal: str = Field(default="", max_length=1500)
    original_audio_priority: bool = False
    evidence: list[StoryEvidence] = Field(min_length=1)

    @field_validator("id")
    @classmethod
    def validate_beat_id(cls, value: str) -> str:
        if not _BEAT_ID_RE.fullmatch(value or ""):
            raise ValueError("invalid documentary story beat id")
        return value

    @model_validator(mode="after")
    def validate_text(self):
        self.title = self.title.strip()
        self.summary = self.summary.strip()
        self.narration_goal = self.narration_goal.strip()
        if not self.title:
            raise ValueError("story beat title is required")
        if not self.summary:
            raise ValueError("story beat summary is required")
        return self


class StoryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    title: str = Field(max_length=250)
    angle: str = Field(max_length=1500)
    hook: str = Field(max_length=1500)
    target_duration_seconds: float = Field(default=600, ge=60, le=1800)
    transcript_fingerprints: dict[str, str] = Field(default_factory=dict)
    grounding_reviewed: bool = False
    grounding_review_version: int = Field(default=0, ge=0)
    beats: list[StoryBeat] = Field(min_length=1, max_length=30)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_story_structure(self):
        self.title = self.title.strip()
        self.angle = self.angle.strip()
        self.hook = self.hook.strip()
        if not self.title:
            raise ValueError("story plan title is required")
        if not self.angle:
            raise ValueError("story plan angle is required")
        if not self.hook:
            raise ValueError("story plan hook is required")
        if self.beats[0].purpose != NarrativePurpose.hook:
            raise ValueError("first story beat must be a hook")
        beat_ids = [beat.id for beat in self.beats]
        if len(beat_ids) != len(set(beat_ids)):
            raise ValueError("story plan contains duplicate beat ids")

        planned_duration = sum(beat.target_duration_seconds for beat in self.beats)
        minimum_duration = self.target_duration_seconds * 0.65
        maximum_duration = self.target_duration_seconds * 1.35
        if planned_duration < minimum_duration or planned_duration > maximum_duration:
            raise ValueError(
                "story beat durations must stay within 35% of target duration"
            )
        return self


class ClipSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default_factory=lambda: f"clip_{uuid4().hex[:12]}",
        max_length=120,
    )
    story_beat_id: str = Field(max_length=120)
    purpose: NarrativePurpose
    source_id: str
    segment_ids: list[int] = Field(min_length=1)
    source_start_seconds: float = Field(ge=0)
    source_end_seconds: float = Field(gt=0)
    audio_mode: AudioMode = AudioMode.muted

    @field_validator("story_beat_id")
    @classmethod
    def validate_story_beat_id(cls, value: str) -> str:
        if not _BEAT_ID_RE.fullmatch(value or ""):
            raise ValueError("invalid documentary story beat id")
        return value

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not _SOURCE_ID_RE.fullmatch(value or ""):
            raise ValueError("invalid documentary source id")
        return value

    @field_validator("segment_ids")
    @classmethod
    def validate_segment_ids(cls, value: list[int]) -> list[int]:
        if any(segment_id < 0 for segment_id in value):
            raise ValueError("clip selection segment ids must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("clip selection segment ids must be unique")
        return value

    @model_validator(mode="after")
    def validate_source_range(self):
        if self.source_end_seconds <= self.source_start_seconds:
            raise ValueError(
                "clip selection source_end_seconds must be greater than "
                "source_start_seconds"
            )
        return self


class ClipPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    selector_version: Literal[1] = 1
    padding_seconds: float = Field(default=0.35, ge=0, le=5)
    max_merge_gap_seconds: float = Field(default=0.75, ge=0, le=10)
    story_plan_fingerprint: str = Field(min_length=64, max_length=64)
    transcript_fingerprints: dict[str, str] = Field(default_factory=dict)
    clips: list[ClipSelection] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("story_plan_fingerprint")
    @classmethod
    def validate_story_plan_fingerprint(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value or ""):
            raise ValueError("invalid story plan fingerprint")
        return value

    @field_validator("transcript_fingerprints")
    @classmethod
    def validate_transcript_fingerprints(cls, value: dict[str, str]) -> dict[str, str]:
        for source_id, fingerprint in value.items():
            if not _SOURCE_ID_RE.fullmatch(source_id or ""):
                raise ValueError("invalid documentary source id")
            if not re.fullmatch(r"[0-9a-f]{64}", fingerprint or ""):
                raise ValueError("invalid transcript fingerprint")
        return value


class VideoMetadata(BaseModel):
    """Technical metadata needed by transcription, clip selection, and rendering."""

    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    has_audio: bool = False
    video_codec: str = ""
    audio_codec: str = ""
    container: str = ""
    file_size_bytes: int = Field(default=0, ge=0)
    rotation_degrees: int = 0
    audio_channels: Optional[int] = Field(default=None, ge=1)
    audio_sample_rate: Optional[int] = Field(default=None, ge=1)


class SourceAsset(BaseModel):
    id: str = Field(default_factory=lambda: f"source_{uuid4().hex[:12]}")
    source_type: SourceType
    provenance: ProvenanceType = ProvenanceType.other
    title: str = ""
    source_url: str = ""
    publisher: str = ""
    publication_date: Optional[str] = None
    incident_case_id: str = ""

    original_filename: str = ""
    local_path: str = ""
    checksum_sha256: str = ""
    video_metadata: Optional[VideoMetadata] = None

    rights_status: RightsStatus = RightsStatus.unknown_review_required
    rights_note: str = ""

    youtube_video_id: str = ""
    youtube_channel: str = ""
    youtube_published_at: Optional[str] = None

    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if not _SOURCE_ID_RE.fullmatch(value or ""):
            raise ValueError("invalid documentary source id")
        return value

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_rights_status(cls, data: Any):
        """Load early Documentary manifests without treating provenance as permission.

        The first prototype incorrectly used ``official_public_source`` as a rights
        state. Preserve the provenance signal while downgrading reuse rights to
        review-required when such a manifest is loaded.
        """
        if isinstance(data, dict) and data.get("rights_status") == "official_public_source":
            migrated = dict(data)
            migrated.setdefault("provenance", ProvenanceType.official_public_source.value)
            migrated["rights_status"] = RightsStatus.unknown_review_required.value
            return migrated
        return data

    @property
    def has_local_copy(self) -> bool:
        """Whether a real local file currently exists for technical processing."""
        return bool(self.local_path) and Path(self.local_path).expanduser().is_file()

    @property
    def is_renderable(self) -> bool:
        """Technical renderability only; publication rights are checked separately."""
        return self.has_local_copy

    @property
    def rights_cleared_for_publish(self) -> bool:
        return self.rights_status in _PUBLISHABLE_RIGHTS

    @property
    def is_publishable(self) -> bool:
        return self.has_local_copy and self.rights_cleared_for_publish


class DocumentaryScene(BaseModel):
    id: str = Field(default_factory=lambda: f"scene_{uuid4().hex[:12]}")
    scene_type: SceneType
    source_id: str = ""
    source_start: Optional[float] = Field(default=None, ge=0)
    source_end: Optional[float] = Field(default=None, ge=0)
    audio_mode: AudioMode = AudioMode.narration
    narration_text: str = ""
    original_volume: float = Field(default=1.0, ge=0.0, le=2.0)
    narration_volume: float = Field(default=1.0, ge=0.0, le=2.0)
    subtitle_mode: str = "auto"
    on_screen_text: str = ""
    purpose: NarrativePurpose = NarrativePurpose.context
    story_beat_id: str = ""
    transcript_segment_ids: list[int] = Field(default_factory=list)

    @field_validator("story_beat_id")
    @classmethod
    def validate_optional_story_beat_id(cls, value: str) -> str:
        if value and not _BEAT_ID_RE.fullmatch(value):
            raise ValueError("invalid documentary story beat id")
        return value

    @field_validator("transcript_segment_ids")
    @classmethod
    def validate_transcript_segment_ids(cls, value: list[int]) -> list[int]:
        if any(segment_id < 0 for segment_id in value):
            raise ValueError("documentary scene transcript segment ids must be non-negative")
        if len(value) != len(set(value)):
            raise ValueError("documentary scene transcript segment ids must be unique")
        return value

    @model_validator(mode="after")
    def validate_source_range(self):
        source_scene_types = {
            SceneType.original_clip,
            SceneType.narration_over_source,
            SceneType.broll,
        }
        if self.scene_type in source_scene_types:
            if not self.source_id:
                raise ValueError("source-backed documentary scene requires source_id")
            if self.source_start is None or self.source_end is None:
                raise ValueError(
                    "source-backed documentary scene requires source_start and source_end"
                )
            if self.source_end <= self.source_start:
                raise ValueError("source_end must be greater than source_start")
        return self


class DocumentaryPlan(BaseModel):
    version: int = 1
    story_plan_fingerprint: str = ""
    scenes: list[DocumentaryScene] = Field(default_factory=list)

    @field_validator("story_plan_fingerprint")
    @classmethod
    def validate_story_plan_fingerprint(cls, value: str) -> str:
        if value and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid story plan fingerprint")
        return value


class DocumentaryProject(BaseModel):
    id: str
    title: str
    master_language: str = "en"
    revision: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    sources: list[SourceAsset] = Field(default_factory=list)
    plan: DocumentaryPlan = Field(default_factory=DocumentaryPlan)

    @model_validator(mode="after")
    def validate_references_and_ids(self):
        source_ids = [source.id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("documentary project contains duplicate source ids")

        scene_ids = [scene.id for scene in self.plan.scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("documentary project contains duplicate scene ids")

        known_sources = set(source_ids)
        for scene in self.plan.scenes:
            if scene.source_id and scene.source_id not in known_sources:
                raise ValueError(
                    f"documentary scene references unknown source_id: {scene.source_id}"
                )
        return self
