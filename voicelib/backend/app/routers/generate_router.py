"""
Generate router — TTS generation endpoint and history.

POST /generate          — Generate speech from text in a cloned voice
GET  /generations       — List past generations for the current user
"""
from __future__ import annotations

import logging
import tempfile
import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage, tts
from app.auth import get_current_user
from app.db import get_db
from app.models import Generation, GenerateRequest, GenerationOut, User, Voice

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/generate",
    response_model=GenerationOut,
    status_code=status.HTTP_201_CREATED,
    summary="Generate speech from text using a cloned voice",
)
async def generate_speech(
    payload: GenerateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GenerationOut:
    # ── 1. Fetch voice + ownership check ───────────────────────────────────
    result = await db.execute(select(Voice).where(Voice.id == payload.voice_id))
    voice = result.scalar_one_or_none()

    if voice is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found.")

    if str(voice.owner_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to use this voice.",
        )

    # ── 2. Get or re-derive voice state ────────────────────────────────────
    voice_id_str = str(voice.id)
    voice_state = tts.get_cached_voice_state(voice_id_str)

    if voice_state is None:
        raw_bytes = storage.download_bytes(voice.sample_s3_key)
        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        try:
            voice_state = tts.derive_voice_state(tmp_path, voice_id_str, engine_id=payload.engine)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── 3. Merge per-voice unique acoustic profile parameters ────────────────
    import asyncio
    import json
    from functools import partial

    opt = {}
    if isinstance(voice.opt_weights, dict):
        opt = voice.opt_weights
    elif isinstance(voice.opt_weights, str):
        try:
            opt = json.loads(voice.opt_weights)
        except Exception:
            opt = {}

    active_speed = payload.speed if (payload.speed and abs(payload.speed - 1.0) > 0.05) else opt.get("speed_scale", 1.0)
    active_pitch = payload.pitch if (payload.pitch and abs(payload.pitch) > 0.05) else float(opt.get("pitch_bias", 0.0))
    active_top_p = payload.top_p if (payload.top_p and abs(payload.top_p - 0.8) > 0.05) else opt.get("top_p", 0.8)
    active_temp = payload.temperature if (payload.temperature and abs(payload.temperature - 0.7) > 0.05) else opt.get("temperature", 0.7)
    active_exaggeration = float(opt.get("exaggeration", 0.00))
    active_cfg = float(opt.get("cfg_weight", 0.35))

    logger.info(
        f"Generate speech for voice '{voice.name}' ({voice.id}): active_pitch={active_pitch:+.2f} semitones "
        f"(opt pitch_bias={opt.get('pitch_bias', 0.0)}, median_f0={opt.get('median_f0_hz', 'N/A')}Hz, register='{opt.get('pitch_register', 'N/A')}')"
    )


    try:
        loop = asyncio.get_running_loop()
        gen_fn = partial(
            tts.generate_audio,
            voice_state,
            payload.text,
            engine_id=payload.engine,
            emotion=payload.emotion or "neutral",
            emotions=payload.emotions,
            speed=active_speed,
            pitch=active_pitch,
            rank=payload.rank or 128,
            top_p=active_top_p,
            temperature=active_temp,
            text_lang=payload.text_lang or "en",
            exaggeration=active_exaggeration,
            cfg_weight=active_cfg,
        )
        wav_bytes = await loop.run_in_executor(None, gen_fn)
    except Exception as exc:
        logger.error(f"TTS generation failed: {type(exc).__name__}: {exc}")
        error_msg = str(exc)
        is_offline = "offline" in error_msg.lower() or "not connected" in error_msg.lower()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE if is_offline else status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg if is_offline else "TTS generation failed. Please try again.",
        )

    # ── 3.5. Voice Enhancement Post-Processing ─────────────────────────────
    # Apply noise gate → de-robotisation → harmonic warmth → EQ → compression
    try:
        from app.utils.voice_enhance import enhance_voice_audio
        enhance_fn = partial(enhance_voice_audio, wav_bytes)
        wav_bytes = await loop.run_in_executor(None, enhance_fn)
        logger.info("Voice enhancement pipeline applied successfully.")
    except Exception as exc:
        logger.warning(f"Voice enhancement skipped (non-fatal): {exc}")

    # ── 4. Upload generated audio to object storage ────────────────────────
    audio_key = storage.upload_bytes(wav_bytes, "audio/wav", prefix="generated")

    # ── 5. Persist generation record ──────────────────────────────────────
    generation = Generation(
        voice_id=voice.id,
        user_id=current_user.id,
        input_text=payload.text,
        audio_s3_key=audio_key,
        engine=payload.engine or "gpt-sovits-v3",
        emotion=payload.emotion or "neutral",
        speed=payload.speed or 1.0,
    )
    db.add(generation)
    await db.commit()
    await db.refresh(generation)

    # ── 6. Return with fresh presigned URL ─────────────────────────────────
    audio_url = storage.generate_presigned_url(audio_key, expires_in=3600)
    return GenerationOut(
        id=generation.id,
        voice_id=generation.voice_id,
        input_text=generation.input_text,
        audio_url=audio_url,
        engine=generation.engine,
        emotion=generation.emotion,
        speed=generation.speed,
        created_at=generation.created_at,
    )


@router.get(
    "/generations",
    response_model=list[GenerationOut],
    summary="List past TTS generations for the current user",
)
async def list_generations(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    voice_id: str | None = Query(default=None, description="Filter by voice ID"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> list[GenerationOut]:
    query = (
        select(Generation)
        .where(Generation.user_id == current_user.id)
        .order_by(Generation.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if voice_id:
        query = query.where(Generation.voice_id == voice_id)

    result = await db.execute(query)
    generations = result.scalars().all()

    # Freshen presigned URLs on every list call (URLs expire after 1h)
    return [
        GenerationOut(
            id=g.id,
            voice_id=g.voice_id,
            input_text=g.input_text,
            audio_url=storage.generate_presigned_url(g.audio_s3_key, expires_in=3600),
            created_at=g.created_at,
        )
        for g in generations
    ]


@router.get(
    "/generations/{generation_id}/audio",
    summary="Direct stream for audio player playback",
)
@router.get(
    "/generate/{generation_id}/audio",
    summary="Direct stream for audio player playback",
)
async def stream_generation_audio(
    generation_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    format: str = Query("wav", description="Audio format: 'wav' or 'mp3'"),
):
    """Stream generated audio directly for HTML5 audio playback with correct Content-Type."""
    from fastapi.responses import Response
    import io

    result = await db.execute(
        select(Generation).where(Generation.id == generation_id)
    )
    generation = result.scalar_one_or_none()

    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")

    try:
        raw_bytes = storage.download_bytes(generation.audio_s3_key)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found in storage.",
        )

    if format.lower() == "mp3":
        try:
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(raw_bytes))
            buf = io.BytesIO()
            sf.write(buf, data, sr, format="MP3")
            mp3_bytes = buf.getvalue()
            return Response(content=mp3_bytes, media_type="audio/mpeg", headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"})
        except Exception:
            pass  # Fallback to WAV

    return Response(
        content=raw_bytes,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
    )


@router.get(
    "/generations/{generation_id}/download",
    summary="Download generated audio in MP3 or WAV format",
)
@router.get(
    "/generate/{generation_id}/download",
    summary="Download generated audio in MP3 or WAV format",
)
async def download_generation(
    generation_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    format: str = Query("wav", description="Audio format: 'wav' or 'mp3'"),
    token: str | None = Query(default=None, description="Optional JWT token in query"),
    current_user: Annotated[User | None, Depends(get_current_user)] = None,
):
    """Download the generated audio file with chosen format (MP3 or WAV)."""
    from fastapi.responses import Response
    import io
    from app.auth import get_user_from_token_str

    result = await db.execute(
        select(Generation).where(Generation.id == generation_id)
    )
    generation = result.scalar_one_or_none()

    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")

    # Validate authentication via Bearer user or query token
    auth_user = current_user
    if auth_user is None and token:
        auth_user = await get_user_from_token_str(token, db)

    if auth_user is not None and str(generation.user_id) != str(auth_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to download this generation.",
        )


    try:
        raw_bytes = storage.download_bytes(generation.audio_s3_key)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found in storage.",
        )

    clean_filename = f"voicelib-{generation_id[:8]}"

    if format.lower() == "mp3":
        try:
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(raw_bytes))
            buf = io.BytesIO()
            sf.write(buf, data, sr, format="MP3")
            mp3_bytes = buf.getvalue()
            return Response(
                content=mp3_bytes,
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": f'attachment; filename="{clean_filename}.mp3"',
                    "Cache-Control": "no-cache",
                },
            )
        except Exception as e:
            logger.warning(f"MP3 conversion error: {e}, falling back to WAV")

    return Response(
        content=raw_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{clean_filename}.wav"',
            "Cache-Control": "no-cache",
        },
    )


@router.get(
    "/generations/{generation_id}/similarity",
    summary="Compare generated audio to original voice sample — acoustic similarity analysis",
)
async def get_voice_similarity(
    generation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> dict:
    """
    Runs acoustic signal analysis comparing the original voice sample
    (reference) against the generated TTS audio output.

    Returns:
    - overall_score (0-100): weighted perceptual similarity
    - mfcc_similarity: spectral timbre / voice character match
    - energy_correlation: energy / dynamics / prosody match
    - centroid_match: spectral brightness register match
    - zcr_match: consonant texture match
    - ref_spectrum / gen_spectrum: 30-point frequency curves for graphing
    """
    import asyncio
    from functools import partial
    from app.utils.similarity import compute_voice_similarity

    # 1. Load generation + ownership check
    result = await db.execute(
        select(Generation).where(Generation.id == generation_id)
    )
    generation = result.scalar_one_or_none()

    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")
    if str(generation.user_id) != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")

    # 2. Load original voice row
    from app.models import Voice
    voice_result = await db.execute(
        select(Voice).where(Voice.id == generation.voice_id)
    )
    voice = voice_result.scalar_one_or_none()
    if voice is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found.")

    # 3. Download audio bytes from storage
    try:
        ref_bytes = storage.download_bytes(voice.sample_s3_key)
    except Exception as exc:
        logger.error(f"Could not load reference voice sample: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not load voice sample from storage.",
        )
    try:
        gen_bytes = storage.download_bytes(generation.audio_s3_key)
    except Exception as exc:
        logger.error(f"Could not load generated audio: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not load generated audio from storage.",
        )

    # 4. Run CPU-bound similarity analysis in threadpool
    try:
        loop = asyncio.get_running_loop()
        sim_fn = partial(compute_voice_similarity, ref_bytes, gen_bytes)
        sim = await loop.run_in_executor(None, sim_fn)
    except Exception as exc:
        logger.error(f"Similarity analysis failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Similarity analysis failed. Please try again.",
        )

    return {
        "generation_id": generation_id,
        "voice_name": voice.name,
        "overall_score": sim.overall_score,
        "accuracy_grade": sim.accuracy_grade,
        "metrics": {
            "mfcc_similarity": sim.mfcc_similarity,
            "mcd_db": sim.mcd_db,
            "mcd_match": sim.mcd_match,
            "f0_correlation": sim.f0_correlation,
            "centroid_match": sim.centroid_match,
            "zcr_match": sim.zcr_match,
            "formants_match": sim.formants_match,
            "energy_correlation": sim.f0_correlation,  # Backwards compatibility
        },
        "spectrum": {
            "ref": sim.ref_spectrum,
            "gen": sim.gen_spectrum,
        },
        "pitch_contour": {
            "ref": sim.ref_pitch_curve,
            "gen": sim.gen_pitch_curve,
        },
        "formants": {
            "ref": sim.ref_formants,
            "gen": sim.gen_formants,
        },
        "audio_info": {
            "ref_duration_s": sim.ref_duration_s,
            "gen_duration_s": sim.gen_duration_s,
            "analysis_sample_rate": sim.sample_rate,
        },
    }
