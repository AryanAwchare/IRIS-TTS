"""
Voices router — upload, list, and delete cloned voices.

POST   /voices        — Upload a voice sample, derive voice state, persist metadata
GET    /voices        — List the current user's voices
DELETE /voices/{id}   — Remove a voice (storage + DB + TTS cache)
"""
from __future__ import annotations

import tempfile
import os
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage, tts
from app.auth import get_current_user
from app.db import get_db
from app.models import User, Voice, VoiceOut, VoiceSettingsUpdate
from app.utils.audio import validate_audio_upload
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "",
    response_model=VoiceOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a voice sample and add it to your voice library",
)
async def create_voice(
    file: UploadFile,
    name: Annotated[str, Form(min_length=1, max_length=100)],
    consent_confirmed: Annotated[bool, Form()],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> VoiceOut:
    # ── 1. Consent gate ────────────────────────────────────────────────────
    if not consent_confirmed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "You must confirm consent before cloning a voice. "
                "Only clone voices with explicit permission from the speaker."
            ),
        )

    # ── 2. Validate audio ──────────────────────────────────────────────────
    raw_audio_bytes = await validate_audio_upload(file)

    # ── 2.1 Preprocess & clean audio for voice cloning ─────────────────────
    from app.utils.audio_preprocess import preprocess_voice_sample
    audio_bytes = preprocess_voice_sample(
        raw_audio_bytes,
        target_sr=22050,
        trim_silence=True,
        denoise=True,
        normalize=True,
    )

    # ── 3. Upload cleaned sample to object storage ─────────────────────────
    sample_key = storage.upload_bytes(
        data=audio_bytes,
        content_type="audio/wav",
        prefix="voice-samples",
    )

    # ── 3.5. Extract unique per-voice acoustic parameter profile ────────────
    try:
        from app.utils.voice_profiler import extract_voice_acoustic_profile
        voice_profile = extract_voice_acoustic_profile(audio_bytes)
    except Exception as prof_err:
        logger.warning(f"Acoustic profiling note for {name}: {prof_err}")
        voice_profile = None

    # ── 4. Persist Voice row ───────────────────────────────────────────────
    voice = Voice(
        name=name.strip(),
        owner_id=current_user.id,
        sample_s3_key=sample_key,
        consent_confirmed=True,
        opt_weights=voice_profile,
    )
    db.add(voice)
    await db.commit()
    await db.refresh(voice)

    # ── 5. Derive voice state and pre-warm cache ───────────────────────────
    # Write bytes to a temp file so pocket-tts can read it
    suffix = os.path.splitext(file.filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        tts.derive_voice_state(tmp_path, str(voice.id))
    except Exception as exc:
        logger.warning(
            f"Voice state derivation failed for {voice.id}: {exc}"
        )
    finally:
        os.unlink(tmp_path)

    v_out = VoiceOut.model_validate(voice)
    v_out.sample_url = storage.generate_presigned_url(voice.sample_s3_key)
    return v_out


@router.get(
    "",
    response_model=list[VoiceOut],
    summary="List all voices in your voice library",
)
async def list_voices(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[VoiceOut]:
    result = await db.execute(
        select(Voice)
        .where(Voice.owner_id == current_user.id)
        .order_by(Voice.created_at.desc())
    )
    voices = result.scalars().all()

    # Auto-backfill unique opt_weights profile for existing voices if missing or missing detailed pitch
    dirty = False
    for v in voices:
        if not v.opt_weights or "median_f0_hz" not in v.opt_weights:
            try:
                raw_b = storage.download_bytes(v.sample_s3_key)
                from app.utils.voice_profiler import extract_voice_acoustic_profile
                profile = extract_voice_acoustic_profile(raw_b)
                if isinstance(v.opt_weights, dict):
                    v.opt_weights = {**v.opt_weights, **profile}
                else:
                    v.opt_weights = profile
                db.add(v)
                dirty = True
            except Exception as e:
                logger.warning(f"Opt_weights pitch backfill skipped for {v.name}: {e}")
    if dirty:
        await db.commit()


    output_list = []
    for v in voices:
        vo = VoiceOut.model_validate(v)
        vo.sample_url = storage.generate_presigned_url(v.sample_s3_key)
        output_list.append(vo)
    return output_list


@router.delete(
    "/{voice_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a voice from your library (storage + DB + cache)",
)
async def delete_voice(
    voice_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    # ── 1. Fetch + ownership check ─────────────────────────────────────────
    result = await db.execute(select(Voice).where(Voice.id == voice_id))
    voice = result.scalar_one_or_none()

    if voice is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found.")

    if str(voice.owner_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this voice.",
        )

    # ── 2. Delete from object storage ──────────────────────────────────────
    try:
        storage.delete_object(voice.sample_s3_key)
    except Exception:
        pass  # Log but continue — storage cleanup is best-effort

    # ── 3. Delete from DB (cascades to generations) ────────────────────────
    await db.delete(voice)
    await db.commit()

    # ── 4. Invalidate TTS cache ────────────────────────────────────────────
    tts.invalidate_voice_cache(voice_id)


@router.patch(
    "/{voice_id}/settings",
    response_model=VoiceOut,
    summary="Update individual acoustic tuning, accent lock, and de-robotization settings for a voice",
)
async def update_voice_settings(
    voice_id: str,
    payload: VoiceSettingsUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> VoiceOut:
    result = await db.execute(select(Voice).where(Voice.id == voice_id))
    voice = result.scalar_one_or_none()

    if voice is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found.")

    if str(voice.owner_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this voice.",
        )

    # Merge existing opt_weights with new updates
    current_opt = voice.opt_weights or {}
    if not isinstance(current_opt, dict):
        current_opt = {}

    updated_opt = {
        **current_opt,
        "cfg_weight": payload.cfg_weight,
        "pitch_bias": payload.pitch_bias,
        "speed_scale": payload.speed_scale,
        "temperature": payload.temperature,
        "top_p": payload.top_p,
        "warmth_gain_db": payload.warmth_gain_db,
        "exaggeration": payload.exaggeration,
        "de_robotize": payload.de_robotize,
    }

    voice.opt_weights = updated_opt
    db.add(voice)
    await db.commit()
    await db.refresh(voice)

    # Invalidate cache so fresh calibrated settings are applied immediately
    tts.invalidate_voice_cache(voice_id)
    logger.info(f"Updated settings for voice '{voice.name}' ({voice.id}): {updated_opt}")

    vo = VoiceOut.model_validate(voice)
    vo.sample_url = storage.generate_presigned_url(voice.sample_s3_key)
    return vo

