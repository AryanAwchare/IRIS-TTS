"""
Audio file validation utilities.

Validates uploaded audio files for:
    - MIME type / file extension
    - File size (must not exceed max_upload_size_bytes)
    - Duration (must not exceed max_sample_duration_seconds)
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import HTTPException, UploadFile, status
from mutagen import File as MutagenFile
from mutagen.mp3 import MP3
from mutagen.wave import WAVE

from app.config import get_settings


KNOWN_EXTENSIONS = {
    ".mp3", ".wav", ".wave", ".m4a", ".aac", ".ogg", ".flac", ".mp4", ".webm"
}


async def validate_audio_upload(file: UploadFile) -> bytes:
    """
    Read and validate an uploaded audio file.

    Returns raw bytes if valid.
    Raises HTTPException on validation failure.
    """
    s = get_settings()

    # ── 1. MIME type & extension check ───────────────────────────────────────
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    ext = Path(filename).suffix.lower()

    is_valid_type = (
        content_type in s.allowed_audio_formats
        or "audio/" in content_type
        or content_type in ["application/octet-stream", "binary/octet-stream"]
        or ext in KNOWN_EXTENSIONS
    )

    if not is_valid_type:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported audio format '{content_type or ext}'. "
                "Allowed formats: MP3, WAV, M4A, AAC, OGG, FLAC"
            ),
        )

    # ── 2. Size check ────────────────────────────────────────────────────────
    max_bytes = s.max_upload_size_bytes
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File is too large. Maximum allowed size is "
                f"{max_bytes // (1024 * 1024)} MB."
            ),
        )

    if len(data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # ── 3. Duration check via mutagen (with fallbacks) ──────────────────────
    duration: float | None = None
    
    try:
        audio_meta = MutagenFile(io.BytesIO(data))
        if audio_meta and hasattr(audio_meta, "info"):
            duration = getattr(audio_meta.info, "length", None)
    except Exception:
        pass

    if duration is None:
        try:
            mp3_meta = MP3(io.BytesIO(data))
            duration = getattr(mp3_meta.info, "length", None)
        except Exception:
            pass

    if duration is None:
        try:
            wav_meta = WAVE(io.BytesIO(data))
            duration = getattr(wav_meta.info, "length", None)
        except Exception:
            pass

    min_dur = getattr(s, "min_sample_duration_seconds", 6.0)
    max_dur = s.max_sample_duration_seconds

    if duration is not None:
        if duration < min_dur:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Audio sample is {duration:.1f} seconds long. "
                    f"Minimum required is {min_dur:.0f} seconds to extract voice characteristics and timbre. "
                    "Please record a longer, clearer speech sample."
                ),
            )
        if duration > max_dur:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Audio sample is {duration:.1f} seconds long. "
                    f"Maximum allowed duration is {max_dur:.0f} seconds. "
                    "Please trim your recording and try again."
                ),
            )

    return data
