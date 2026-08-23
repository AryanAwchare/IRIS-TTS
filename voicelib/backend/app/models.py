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
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, func, JSON
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    voice: Mapped["Voice"] = relationship("Voice", back_populates="generations")
    user: Mapped["User"] = relationship("User", back_populates="generations")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────

# Auth
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


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
    emotion: Optional[str] = Field(default="neutral", description="Emotion style preset")
    emotions: Optional[dict[str, float]] = Field(default=None, description="8D emotion vector weights")
    rank: Optional[int] = Field(default=128, description="LoRA / Model rank")
    top_p: Optional[float] = Field(default=0.8, ge=0.0, le=1.0, description="Nucleus sampling top-p")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    speed: Optional[float] = Field(default=1.0, ge=0.2, le=3.0, description="Speed multiplier")
    pitch: Optional[float] = Field(default=0.0, ge=-12.0, le=12.0, description="Pitch shift in semitones")
    text_lang: Optional[str] = Field(default="en", description="Target text language")


class GenerationOut(BaseModel):
    id: uuid.UUID
    voice_id: uuid.UUID
    input_text: str
    audio_url: str
    engine: Optional[str] = "gpt-sovits-v3"
    emotion: Optional[str] = "neutral"
    speed: Optional[float] = 1.0
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
