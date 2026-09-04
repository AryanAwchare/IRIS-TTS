"""
SQLAlchemy ORM models + Pydantic v2 schemas for VoiceLib.

ORM Models:
    User, Voice, Generation

Pydantic Schemas (separate from ORM):
    Auth: UserCreate, UserOut, Token, TokenData
    Voice: VoiceOut
    Generation: GenerateRequest, GenerationOut
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, func, JSON, Float, Integer
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    voices: Mapped[list["Voice"]] = relationship(
        "Voice", back_populates="owner", cascade="all, delete-orphan"
    )
    generations: Mapped[list["Generation"]] = relationship(
        "Generation", back_populates="user", cascade="all, delete-orphan"
    )
    song_covers: Mapped[list["SongCover"]] = relationship(
        "SongCover", back_populates="user", cascade="all, delete-orphan"
    )


class Voice(Base):
    __tablename__ = "voices"
    __table_args__ = (
        Index("idx_voices_owner_id", "owner_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    sample_s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    # consent_confirmed must be True at the API layer — stored here for audit trail
    consent_confirmed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Capability flags and versioned singing identity
    speech_capable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    singing_capable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    singing_identity: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Optimized neural acoustic model weights per voice (cfg_weight, lora_rank, temperature, top_p, pitch_bias)
    opt_weights: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Custom per-voice word pronunciation dictionary (word -> phonetic spelling / ARPAbet)
    pronunciation_dict: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    owner: Mapped["User"] = relationship("User", back_populates="voices")
    generations: Mapped[list["Generation"]] = relationship(
        "Generation", back_populates="voice", cascade="all, delete-orphan"
    )
    song_covers: Mapped[list["SongCover"]] = relationship(
        "SongCover", back_populates="voice", cascade="all, delete-orphan"
    )



class Generation(Base):
    __tablename__ = "generations"
    __table_args__ = (
        Index("idx_generations_user_id", "user_id"),
        Index("idx_generations_voice_id", "voice_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    voice_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("voices.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    input_text: Mapped[str] = mapped_column(Text, nullable=False)
    audio_s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    engine: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default="gpt-sovits-v3")
    emotion: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, default="neutral")
    speed: Mapped[Optional[float]] = mapped_column(nullable=True, default=1.0)
    
    # Automated multi-metric objective evaluation fields
    eval_status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)  # pending, completed, failed
    speaker_similarity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)         # ECAPA-TDNN cosine sim [0.0 - 1.0]
    word_error_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # faster-whisper WER [0.0 - 1.0+]
    prosody_f0_std: Mapped[Optional[float]] = mapped_column(Float, nullable=True)             # Pitch dynamic range (F0 standard dev in Hz)
    composite_grade: Mapped[Optional[str]] = mapped_column(String(4), nullable=True)          # Composite quality grade: "A", "B", "C", "D"
    composite_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)            # Composite quality score [0.0 - 1.0]
    eval_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)                    # Error traceback if evaluation failed
    evaluated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    voice: Mapped["Voice"] = relationship("Voice", back_populates="generations")
    user: Mapped["User"] = relationship("User", back_populates="generations")


class SongCover(Base):
    __tablename__ = "song_covers"
    __table_args__ = (
        Index("idx_song_covers_user_id", "user_id"),
        Index("idx_song_covers_voice_id", "voice_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    voice_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("voices.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="Untitled Song Cover")
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)  # pending, separating, analyzing, converting, mixing, completed, failed
    progress: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    pitch_shift: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    index_rate: Mapped[float] = mapped_column(Float, default=0.75, nullable=False)
    protect_voiceless: Mapped[float] = mapped_column(Float, default=0.33, nullable=False)
    is_preview: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), default="UPLOAD", nullable=False)  # UPLOAD, SEARCH, LIBRARY
    source_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    song_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    original_audio_s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    vocals_s3_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    instrumental_s3_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    converted_vocals_s3_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    final_mix_s3_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    preview_s3_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship("User", back_populates="song_covers")
    voice: Mapped["Voice"] = relationship("Voice", back_populates="song_covers")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

# Auth
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: uuid.UUID
    email: EmailStr
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class TokenData(BaseModel):
    user_id: str


class VoiceOut(BaseModel):
    id: uuid.UUID
    name: str
    consent_confirmed: bool
    sample_url: Optional[str] = None
    speech_capable: bool = True
    singing_capable: bool = False
    singing_identity: Optional[dict] = None
    opt_weights: Optional[dict] = None
    pronunciation_dict: Optional[dict] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class VoiceSettingsUpdate(BaseModel):
    cfg_weight: Optional[float] = Field(default=0.70, ge=0.20, le=1.0, description="Speaker & Accent Fidelity Scale")
    pitch_bias: Optional[float] = Field(default=0.0, ge=-12.0, le=12.0, description="Pitch calibration in semitones")
    speed_scale: Optional[float] = Field(default=1.0, ge=0.5, le=2.0, description="Speaking pace multiplier")
    temperature: Optional[float] = Field(default=0.7, ge=0.2, le=1.5, description="Sampling naturalness")
    top_p: Optional[float] = Field(default=0.82, ge=0.1, le=1.0, description="Nucleus sampling top-p")
    warmth_gain_db: Optional[float] = Field(default=0.0, ge=-6.0, le=8.0, description="Vocal warmth & body boost in dB")
    exaggeration: Optional[float] = Field(default=0.0, ge=0.0, le=0.5, description="Expressive exaggeration factor")
    de_robotize: Optional[bool] = Field(default=True, description="Enable anti-robotic harmonic smoothing")


# Generation
class GenerateRequest(BaseModel):
    voice_id: uuid.UUID
    text: str = Field(min_length=1, max_length=5000)
    engine: Optional[str] = Field(default="gpt-sovits-v3", description="Selected TTS engine")
    emotion: Optional[str] = Field(default="auto", description="Emotion style preset (auto, neutral, calm, happy, excited, sad, angry, fearful, disgusted)")
    user_intensity: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Manual emotional delivery intensity override (0.0 - 1.0)")
    emotions: Optional[dict[str, float]] = Field(default=None, description="8D emotion vector weights")
    rank: Optional[int] = Field(default=128, description="LoRA / Model rank")
    top_p: Optional[float] = Field(default=0.8, ge=0.0, le=1.0, description="Nucleus sampling top-p")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    speed: Optional[float] = Field(default=1.0, ge=0.2, le=3.0, description="Speed multiplier")
    pitch: Optional[float] = Field(default=0.0, ge=-12.0, le=12.0, description="Pitch shift in semitones")
    text_lang: Optional[str] = Field(default="en", description="Target text language")
    # Pocket TTS Fine-Tuning Parameters
    carrier_voice: Optional[str] = Field(default=None, description="Pocket TTS carrier voice override (auto, jean, marius, françois, alba, laura, anna)")
    morph_strength: Optional[float] = Field(default=0.85, ge=0.0, le=1.0, description="Acoustic timbre morphing intensity")
    warmth_gain_db: Optional[float] = Field(default=0.0, ge=-6.0, le=6.0, description="Low-mid vocal warmth gain in dB")
    brightness_gain_db: Optional[float] = Field(default=0.0, ge=-6.0, le=6.0, description="High-frequency presence/clarity gain in dB")


# Pocket TTS Studio Presets (carrier_voice, morph, warmth_db, brightness_db, speed)
POCKET_TTS_PRESETS = {
    "natural_conversational": {
        "carrier_voice": None, "morph_strength": 0.85,
        "warmth_gain_db": 0.0, "brightness_gain_db": 0.0, "speed": 1.0,
    },
    "studio_broadcast": {
        "carrier_voice": None, "morph_strength": 0.90,
        "warmth_gain_db": 1.5, "brightness_gain_db": 1.0, "speed": 0.95,
    },
    "crisp_narration": {
        "carrier_voice": None, "morph_strength": 0.80,
        "warmth_gain_db": -0.5, "brightness_gain_db": 2.0, "speed": 0.90,
    },
    "deep_warmth": {
        "carrier_voice": None, "morph_strength": 0.90,
        "warmth_gain_db": 3.5, "brightness_gain_db": -1.0, "speed": 0.92,
    },
}


class GenerationOut(BaseModel):
    id: uuid.UUID
    voice_id: uuid.UUID
    input_text: str
    audio_url: str
    engine: Optional[str] = "gpt-sovits-v3"
    emotion: Optional[str] = "neutral"
    speed: Optional[float] = 1.0
    eval_status: Optional[str] = "pending"
    speaker_similarity: Optional[float] = None
    word_error_rate: Optional[float] = None
    prosody_f0_std: Optional[float] = None
    composite_grade: Optional[str] = None
    composite_score: Optional[float] = None
    evaluated_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class GenerationEvalOut(BaseModel):
    generation_id: uuid.UUID
    voice_id: uuid.UUID
    eval_status: str
    speaker_similarity: Optional[float] = None
    word_error_rate: Optional[float] = None
    prosody_f0_std: Optional[float] = None
    composite_grade: Optional[str] = None
    composite_score: Optional[float] = None
    eval_error: Optional[str] = None
    evaluated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)


# Song Covers
class SongCoverCreate(BaseModel):
    voice_id: uuid.UUID
    title: Optional[str] = Field(default=None, max_length=255)
    pitch_shift: Optional[int] = Field(default=0, ge=-24, le=24, description="Semitone transposition (-24 to +24)")
    index_rate: Optional[float] = Field(default=0.75, ge=0.0, le=1.0, description="RVC feature retrieval ratio")
    protect_voiceless: Optional[float] = Field(default=0.33, ge=0.0, le=0.5, description="Voiceless consonant protection")
    preview_only: Optional[bool] = Field(default=False, description="Render fast 20-30s preview snippet")
    source_type: Optional[str] = Field(default="UPLOAD", description="Input method: UPLOAD | SEARCH | LIBRARY")
    source_url: Optional[str] = Field(default=None, description="URL for SEARCH/fetch source")
    library_song_hash: Optional[str] = Field(default=None, description="Song hash for LIBRARY personal/curated reuse")
    tos_confirmed: Optional[bool] = Field(default=True, description="Personal non-commercial use affirmation")


class SongCoverOut(BaseModel):
    id: uuid.UUID
    voice_id: uuid.UUID
    title: str
    status: str
    progress: float
    pitch_shift: int
    source_type: str = "UPLOAD"
    song_hash: Optional[str] = None
    audio_url: Optional[str] = None
    preview_url: Optional[str] = None
    instrumental_url: Optional[str] = None
    vocals_url: Optional[str] = None
    is_preview: bool
    metadata_json: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SongCoverStatusOut(BaseModel):
    id: uuid.UUID
    status: str
    progress: float
    audio_url: Optional[str] = None
    preview_url: Optional[str] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class CuratedSongOut(BaseModel):
    song_hash: str
    title: str
    artist: str
    duration: float
    genre: str
    preview_audio_url: Optional[str] = None
