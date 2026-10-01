from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


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


class RightsStatus(str, Enum):
    user_owned = "user_owned"
    licensed = "licensed"
    permission_confirmed = "permission_confirmed"
    public_domain = "public_domain"
    official_public_source = "official_public_source"
    unknown_review_required = "unknown_review_required"


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
    escalation = "escalation"
    reveal = "reveal"
    payoff = "payoff"
    transition = "transition"


class SourceAsset(BaseModel):
    id: str = Field(default_factory=lambda: f"source_{uuid4().hex[:12]}")
    source_type: SourceType
    title: str = ""
    source_url: str = ""
    publisher: str = ""
    publication_date: Optional[str] = None
    incident_case_id: str = ""

    original_filename: str = ""
    local_path: str = ""
    checksum_sha256: str = ""

    rights_status: RightsStatus = RightsStatus.unknown_review_required
    rights_note: str = ""

    youtube_video_id: str = ""
    youtube_channel: str = ""
    youtube_published_at: Optional[str] = None

    created_at: datetime = Field(default_factory=utc_now)

    @property
    def is_renderable(self) -> bool:
        return bool(self.local_path)


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
    scenes: list[DocumentaryScene] = Field(default_factory=list)


class DocumentaryProject(BaseModel):
    id: str
    title: str
    master_language: str = "en"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    sources: list[SourceAsset] = Field(default_factory=list)
    plan: DocumentaryPlan = Field(default_factory=DocumentaryPlan)
