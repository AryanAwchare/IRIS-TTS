"""
VoiceLib application configuration and settings management.
Loads configuration from environment variables and .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field

    class Settings(BaseSettings):
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        # Database
        database_url: str = Field(
            default="sqlite+aiosqlite:///./voicelib_dev.db",
            alias="DATABASE_URL",
        )

        # Storage (S3 / Cloudflare R2 / MinIO / Local Disk Fallback)
        storage_endpoint_url: str = Field(default="", alias="STORAGE_ENDPOINT_URL")
        storage_access_key: str = Field(default="local", alias="STORAGE_ACCESS_KEY")
        storage_secret_key: str = Field(default="local", alias="STORAGE_SECRET_KEY")
        storage_bucket_name: str = Field(default="voicelib", alias="STORAGE_BUCKET_NAME")
        storage_region: str = Field(default="us-east-1", alias="STORAGE_REGION")
        storage_public_base_url: str = Field(default="", alias="STORAGE_PUBLIC_BASE_URL")

        # JWT Auth
        jwt_secret_key: str = Field(
            default="voicelib-localhost-secret-key-32chars",
            alias="JWT_SECRET_KEY",
        )
        jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
        access_token_expire_minutes: int = Field(default=10080, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

        # TTS & Compute
        voicelib_use_mock_tts: bool = Field(default=False, alias="VOICELIB_USE_MOCK_TTS")
        tts_engine: str = Field(default="gpt-sovits-v3", alias="TTS_ENGINE")
        colab_gpu_api_url: str = Field(default="http://localhost:8008", alias="COLAB_GPU_API_URL")
        colab_register_secret: str = Field(default="voicelib-colab-dev-secret", alias="COLAB_REGISTER_SECRET")
        voice_similarity_threshold: float = Field(default=60.0, alias="VOICE_SIMILARITY_THRESHOLD")
        
        # Audio Limits & Formats
        min_sample_duration_seconds: float = Field(default=3.0, alias="MIN_SAMPLE_DURATION_SECONDS")
        max_sample_duration_seconds: float = Field(default=300.0, alias="MAX_SAMPLE_DURATION_SECONDS")
        max_upload_size_bytes: int = Field(default=104857600, alias="MAX_UPLOAD_SIZE_BYTES")
        allowed_audio_formats: List[str] = Field(
            default=[
                "audio/wav", "audio/mpeg", "audio/mp3", "audio/x-mp3",
                "audio/mpeg3", "audio/x-mpeg-3", "audio/x-mpeg", "audio/ogg",
                "audio/flac", "audio/x-wav", "audio/wave", "audio/mp4",
                "audio/m4a", "audio/x-m4a", "audio/aac",
                "application/octet-stream", "binary/octet-stream"
            ],
            alias="ALLOWED_AUDIO_FORMATS",
        )

        # Emotion Intelligence
        emotion_detection_enabled: bool = Field(default=True, alias="EMOTION_DETECTION_ENABLED")
        emotion_blend_mode: str = Field(default="auto", alias="EMOTION_BLEND_MODE")
        emotion_model_name: str = Field(
            default="j-hartmann/emotion-english-distilroberta-base",
            alias="EMOTION_MODEL_NAME"
        )

        # Frontend
        vite_api_base_url: str = Field(default="http://localhost:8000", alias="VITE_API_BASE_URL")

except ImportError:
    # Fallback if pydantic-settings is not installed
    from pydantic import BaseModel

    class Settings(BaseModel):
        database_url: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./voicelib_dev.db")
        storage_endpoint_url: str = os.getenv("STORAGE_ENDPOINT_URL", "")
        storage_access_key: str = os.getenv("STORAGE_ACCESS_KEY", "local")
        storage_secret_key: str = os.getenv("STORAGE_SECRET_KEY", "local")
        storage_bucket_name: str = os.getenv("STORAGE_BUCKET_NAME", "voicelib")
        storage_region: str = os.getenv("STORAGE_REGION", "us-east-1")
        storage_public_base_url: str = os.getenv("STORAGE_PUBLIC_BASE_URL", "")

        jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "voicelib-localhost-secret-key-32chars")
        jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
        access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

        voicelib_use_mock_tts: bool = os.getenv("VOICELIB_USE_MOCK_TTS", "false").lower() in ("true", "1")
        tts_engine: str = os.getenv("TTS_ENGINE", "gpt-sovits-v3")
        colab_gpu_api_url: str = os.getenv("COLAB_GPU_API_URL", "http://localhost:8008")
        colab_register_secret: str = os.getenv("COLAB_REGISTER_SECRET", "voicelib-colab-dev-secret")
        voice_similarity_threshold: float = float(os.getenv("VOICE_SIMILARITY_THRESHOLD", "60.0"))

        min_sample_duration_seconds: float = float(os.getenv("MIN_SAMPLE_DURATION_SECONDS", "3.0"))
        max_sample_duration_seconds: float = float(os.getenv("MAX_SAMPLE_DURATION_SECONDS", "120.0"))
        max_upload_size_bytes: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", "20971520"))
        allowed_audio_formats: List[str] = [
            "audio/wav", "audio/mpeg", "audio/mp3", "audio/x-mp3",
            "audio/mpeg3", "audio/x-mpeg-3", "audio/x-mpeg", "audio/ogg",
            "audio/flac", "audio/x-wav", "audio/wave", "audio/mp4",
            "audio/m4a", "audio/x-m4a", "audio/aac",
            "application/octet-stream", "binary/octet-stream"
        ]
        vite_api_base_url: str = os.getenv("VITE_API_BASE_URL", "http://localhost:8000")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
