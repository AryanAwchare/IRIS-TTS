"""
Voices router — upload, list, and delete cloned voices.

POST   /voices              — Upload a voice sample, derive voice state, persist metadata
GET    /voices              — List the current user's voices
DELETE /voices/{id}         — Remove a voice (storage + DB + TTS cache)
PATCH  /voices/{id}/settings — Update acoustic tuning parameters
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
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


async def _backfill_voice_profiles(voice_ids: list[str]) -> None:
    """
    Background task: re-extract deep acoustic profiles for voices missing v2 data.
    Runs in its own DB session so it doesn't block the request that triggered it.
    """
    from app.db import AsyncSessionLocal
    from app.utils.voice_profiler import extract_voice_acoustic_profile

    async with AsyncSessionLocal() as session:
        for vid in voice_ids:
            try:
                v_uuid = uuid.UUID(vid)
                result = await session.execute(select(Voice).where(Voice.id == v_uuid))
                voice = result.scalar_one_or_none()
                if voice is None:
                    continue

                loop = asyncio.get_running_loop()
                raw_b = await loop.run_in_executor(None, storage.download_bytes, voice.sample_s3_key)
                profile = await loop.run_in_executor(None, extract_voice_acoustic_profile, raw_b)

                if isinstance(voice.opt_weights, dict):
                    voice.opt_weights = {**voice.opt_weights, **profile}
                else:
                    voice.opt_weights = profile

                session.add(voice)
                logger.info(f"Voice '{voice.name}' ({voice.id}) upgraded to Deep Acoustic DNA Profile v2")
            except Exception as e:
                logger.warning(f"Backfill skipped for voice {vid}: {e}")

        try:
            await session.commit()
        except Exception as commit_err:
            logger.warning(f"Backfill commit notice: {commit_err}")


@router.post(
    "",
    response_model=VoiceOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a voice sample or paste a YouTube URL to add it to your voice library",
)
async def create_voice(
    name: Annotated[str, Form(min_length=1, max_length=100)],
    consent_confirmed: Annotated[bool, Form()],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: Optional[UploadFile] = File(default=None),
    youtube_url: Annotated[Optional[str], Form()] = None,
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

    loop = asyncio.get_running_loop()

    # ── 2. Validate & Obtain Audio (File or YouTube URL) ────────────────────
    if youtube_url and youtube_url.strip():
        from app.services.song_fetcher import fetch_and_canonicalize_url
        try:
            raw_audio_bytes, detected_title, _ = await loop.run_in_executor(
                None, lambda: fetch_and_canonicalize_url(youtube_url.strip(), max_duration_sec=300.0)
            )
            # Try isolating vocals if the YouTube video contains background audio/music
            try:
                from app.vocal_separation import separate_vocals_and_instrumental
                sep_res = await loop.run_in_executor(
                    None, lambda: separate_vocals_and_instrumental(raw_audio_bytes)
                )
                raw_audio_bytes = sep_res.vocals_asset.to_bytes()
                logger.info(f"Isolated clean vocals from YouTube video for voice '{name}'")
            except Exception as sep_e:
                logger.warning(f"Vocal isolation skipped for YouTube URL ({sep_e})")
        except Exception as fetch_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to fetch audio from YouTube URL: {fetch_err}",
            )
    elif file is not None:
        raw_audio_bytes = await validate_audio_upload(file)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either an audio file upload or a valid youtube_url must be provided.",
        )

    # ── 2.1 Preprocess & clean audio for voice cloning ─────────────────────
    # FIX: target_sr=32000 (Chatterbox native — was 22050 causing double resampling)
    # denoise=False: Colab applies its own targeted cleaning; pre-denoising strips vocal harmonics
    loop = asyncio.get_running_loop()
    from app.utils.audio_preprocess import preprocess_voice_sample
    audio_bytes = await loop.run_in_executor(
        None,
        lambda: preprocess_voice_sample(
            raw_audio_bytes,
            target_sr=32000,
            trim_silence=True,
            denoise=False,
            normalize=True,
        ),
    )

    # ── 2.2 VAD-based reference segment selection ──────────────────────────
    # Use Silero VAD to select the cleanest speech segment for storage
    try:
        from app.preprocessing.reference_selector import select_best_segment as _vad_select

        with tempfile.NamedTemporaryFile(suffix="_upload.wav", delete=False) as tmp_vad:
            tmp_vad.write(audio_bytes)
            tmp_vad_path = tmp_vad.name

        vad_segment_path = await loop.run_in_executor(None, _vad_select, tmp_vad_path, 10.0)

        with open(vad_segment_path, "rb") as vf:
            vad_bytes = vf.read()

        for _p in (tmp_vad_path, vad_segment_path):
            try:
                if os.path.exists(_p):
                    os.unlink(_p)
            except Exception:
                pass

        # Store the VAD-selected clean segment as the reference
        sample_key = await loop.run_in_executor(
            None,
            lambda: storage.upload_bytes(vad_bytes, "audio/wav", "voice-samples"),
        )
        logger.info(f"Stored VAD-cleaned reference: {len(vad_bytes):,} bytes for '{name}'")
    except Exception as vad_err:
        logger.warning(f"VAD segment selection skipped for '{name}' ({vad_err}) — using full preprocessed audio")
        sample_key = await loop.run_in_executor(
            None,
            lambda: storage.upload_bytes(audio_bytes, "audio/wav", "voice-samples"),
        )

    # ── 3. Extract unique per-voice acoustic parameter profile ─────────────
    try:
        from app.utils.voice_profiler import extract_voice_acoustic_profile
        voice_profile = await loop.run_in_executor(None, extract_voice_acoustic_profile, audio_bytes)
    except Exception as prof_err:
        logger.warning(f"Acoustic profiling note for '{name}': {prof_err}")
        voice_profile = None

    # ── 4. Persist Voice row ───────────────────────────────────────────────
    voice = Voice(
        name=name.strip(),
        owner_id=current_user.id,
        sample_s3_key=sample_key,
        consent_confirmed=True,
        speech_capable=True,
        singing_capable=True,
        singing_identity={
            "source": "speech_derived",
            "sample_s3_key": sample_key,
            "version": 1,
        },
        opt_weights=voice_profile,
    )
    db.add(voice)
    await db.commit()
    await db.refresh(voice)

    # ── 5. Derive voice state and pre-warm cache ───────────────────────────
    suffix = os.path.splitext(file.filename or ".wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        await loop.run_in_executor(None, tts.derive_voice_state, tmp_path, str(voice.id))
    except Exception as exc:
        logger.warning(f"Voice state derivation note for {voice.id}: {exc}")
    finally:
        # FIX: swallow OSError so it doesn't mask derive_voice_state exceptions
        try:
            os.unlink(tmp_path)
        except OSError as _unlink_err:
            logger.debug(f"Could not delete temp file {tmp_path}: {_unlink_err}")

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
    background_tasks: BackgroundTasks,
) -> list[VoiceOut]:
    result = await db.execute(
        select(Voice)
        .where(Voice.owner_id == current_user.id)
        .order_by(Voice.created_at.desc())
    )
    voices = list(result.scalars().all())

    # If the user has 0 custom voices, show available shared starter voices
    if not voices:
        demo_result = await db.execute(
            select(Voice)
            .order_by(Voice.created_at.desc())
            .limit(10)
        )
        voices = list(demo_result.scalars().all())

    # FIX: backfill is now async background task — no longer blocks the GET response
    voices_needing_backfill: list[str] = []
    for v in voices:
        opt = v.opt_weights if isinstance(v.opt_weights, dict) else {}
        if not opt or opt.get("profile_version", 1) < 2 or "formants_hz" not in opt:
            voices_needing_backfill.append(str(v.id))

    if voices_needing_backfill:
        background_tasks.add_task(_backfill_voice_profiles, voices_needing_backfill)
        logger.info(f"Queued v2 profile backfill for {len(voices_needing_backfill)} voice(s)")

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
    voice_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    result = await db.execute(select(Voice).where(Voice.id == voice_id))
    voice = result.scalar_one_or_none()

    if voice is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found.")

    if str(voice.owner_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this voice.",
        )

    try:
        storage.delete_object(voice.sample_s3_key)
    except Exception:
        pass

    await db.delete(voice)
    await db.commit()

    # Invalidate cache for all engines
    for engine_id in ["gpt-sovits-v3", "pocket-tts", "zonos-expressive"]:
        try:
            tts.invalidate_cache(str(voice_id), engine_id=engine_id)
        except Exception:
            pass


@router.patch(
    "/{voice_id}/settings",
    response_model=VoiceOut,
    summary="Update acoustic tuning, accent lock, and de-robotization settings for a voice",
)
async def update_voice_settings(
    voice_id: uuid.UUID,
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

    current_opt = voice.opt_weights or {}
    if not isinstance(current_opt, dict):
        current_opt = {}

    updated_opt = {
        **current_opt,
        "cfg_weight":    payload.cfg_weight,
        "pitch_bias":    payload.pitch_bias,
        "speed_scale":   payload.speed_scale,
        "temperature":   payload.temperature,
        "top_p":         payload.top_p,
        "warmth_gain_db": payload.warmth_gain_db,
        "exaggeration":  payload.exaggeration,
        "de_robotize":   payload.de_robotize,
    }

    voice.opt_weights = updated_opt
    db.add(voice)
    await db.commit()
    await db.refresh(voice)

    # Invalidate cache for all engines so fresh settings apply immediately
    for engine_id in ["gpt-sovits-v3", "pocket-tts", "zonos-expressive"]:
        try:
            tts.invalidate_cache(voice_id, engine_id=engine_id)
        except Exception:
            pass

    logger.info(f"Updated settings for voice '{voice.name}' ({voice.id})")

    vo = VoiceOut.model_validate(voice)
    vo.sample_url = storage.generate_presigned_url(voice.sample_s3_key)
    return vo


@router.post(
    "/{voice_id}/singing-identity",
    response_model=VoiceOut,
    summary="Upload dedicated acapella singing sample to train high-fidelity singing identity",
)
async def upload_singing_identity(
    voice_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
) -> VoiceOut:
    res = await db.execute(select(Voice).where(Voice.id == voice_id))
    voice = res.scalar_one_or_none()
    if voice is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found.")

    if str(voice.owner_id) != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied for this voice.")

    audio_bytes = await validate_audio_upload(file)

    loop = asyncio.get_running_loop()
    singing_sample_key = await loop.run_in_executor(
        None, lambda: storage.upload_bytes(audio_bytes, "audio/wav", prefix="voice-singing-samples")
    )

    voice.singing_capable = True
    voice.singing_identity = {
        "source": "native_acapella",
        "sample_s3_key": singing_sample_key,
        "filename": file.filename,
        "version": 2,
    }
    db.add(voice)
    await db.commit()
    await db.refresh(voice)

    vo = VoiceOut.model_validate(voice)
    vo.sample_url = storage.generate_presigned_url(voice.sample_s3_key)
    return vo

