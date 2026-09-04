"""
Generate router — TTS generation endpoint and history.

POST /generate          — Generate speech from text in a cloned voice
GET  /generations       — List past generations for the current user
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from datetime import datetime, timezone
from functools import partial
import json
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage, tts
from app.auth import get_current_user
from app.db import get_db
from app.models import Generation, GenerateRequest, GenerationOut, GenerationEvalOut, User, Voice
from app.evaluation.eval_pipeline import run_async_evaluation

logger = logging.getLogger(__name__)
router = APIRouter()


async def _vad_clean_reference(raw_bytes: bytes, loop: asyncio.AbstractEventLoop) -> bytes:
    """
    Run Silero VAD-based reference segment selection in a thread pool.
    Returns the cleaned reference bytes, or the original on any error.
    """
    try:
        from app.preprocessing.reference_selector import select_best_segment as _vad_select

        with tempfile.NamedTemporaryFile(suffix="_ref_raw.wav", delete=False) as tmp_in:
            tmp_in.write(raw_bytes)
            tmp_in_path = tmp_in.name

        clean_path = await loop.run_in_executor(None, _vad_select, tmp_in_path, 10.0)

        with open(clean_path, "rb") as cf:
            clean_bytes = cf.read()

        for _p in (tmp_in_path, clean_path):
            try:
                if os.path.exists(_p):
                    os.unlink(_p)
            except Exception:
                pass

        logger.info(f"VAD pre-clean: {len(raw_bytes):,} → {len(clean_bytes):,} bytes")
        return clean_bytes
    except Exception as vad_err:
        logger.debug(f"VAD pre-clean skipped ({vad_err}) — using raw reference bytes")
        return raw_bytes


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
    response: Response,
    background_tasks: BackgroundTasks,
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
    settings_obj = None
    try:
        from app.config import get_settings
        settings_obj = get_settings()
    except Exception:
        pass

    engine_id = payload.engine or (settings_obj.tts_engine if settings_obj else "gpt-sovits-v3")
    voice_id_str = str(voice.id)

    voice_state = tts.get_cached_voice_state(voice_id_str, engine_id=engine_id)

    if voice_state is None:
        loop = asyncio.get_running_loop()
        raw_bytes = await loop.run_in_executor(None, storage.download_bytes, voice.sample_s3_key)

        # VAD-clean the reference before deriving voice state
        clean_bytes = await _vad_clean_reference(raw_bytes, loop)

        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(clean_bytes)
            tmp_path = tmp.name
        try:
            voice_state = tts.derive_voice_state(tmp_path, voice_id_str, engine_id=engine_id)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ── 3. Merge per-voice unique acoustic profile + emotion intelligence ───
    opt = {}
    if isinstance(voice.opt_weights, dict):
        opt = voice.opt_weights
    elif isinstance(voice.opt_weights, str):
        try:
            opt = json.loads(voice.opt_weights)
        except Exception:
            opt = {}

    from app.config import get_settings
    from app.utils.emotion_analyzer import compute_modulated_synthesis_parameters
    settings = get_settings()
    blend_mode = getattr(settings, "emotion_blend_mode", "auto")

    resolved_params = compute_modulated_synthesis_parameters(
        text=payload.text,
        requested_emotion=payload.emotion,
        opt_weights=opt,
        user_speed=payload.speed,
        user_pitch=payload.pitch,
        blend_mode=blend_mode,
        user_intensity=payload.user_intensity,
    )

    active_emotion     = resolved_params["resolved_emotion"]
    active_speed       = resolved_params["speed"]
    active_pitch       = resolved_params["pitch"]
    active_cfg         = resolved_params["cfg_weight"]
    active_exaggeration = resolved_params["exaggeration"]
    active_top_p       = resolved_params["top_p"]
    active_temp        = resolved_params["temperature"]
    analysis_meta      = resolved_params.get("analysis", {})

    response.headers["X-Detected-Emotion"]   = str(analysis_meta.get("detected_emotion", "neutral"))
    response.headers["X-Emotion-Intensity"]  = str(resolved_params.get("intensity", 0.0))
    response.headers["X-Resolved-CFG"]       = str(active_cfg)
    response.headers["X-Resolved-Exaggeration"] = str(active_exaggeration)

    logger.info(
        f"Generating speech for voice '{voice.name}' ({voice.id}): "
        f"emotion='{active_emotion}' cfg={active_cfg:.2f} exag={active_exaggeration:.2f} "
        f"speed={active_speed:.2f} pitch={active_pitch:+.2f}st"
    )

    try:
        loop = asyncio.get_running_loop()
        gen_fn = partial(
            tts.generate_audio,
            voice_state,
            payload.text,
            engine_id=payload.engine,
            emotion=active_emotion,
            emotions=payload.emotions,
            speed=active_speed,
            pitch=active_pitch,
            rank=payload.rank or 128,
            top_p=active_top_p,
            temperature=active_temp,
            text_lang=payload.text_lang or "en",
            exaggeration=active_exaggeration,
            cfg_weight=active_cfg,
            carrier_voice=payload.carrier_voice,
            morph_strength=payload.morph_strength,
            warmth_gain_db=payload.warmth_gain_db,
            brightness_gain_db=payload.brightness_gain_db,
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
    try:
        from app.utils.voice_enhance import enhance_voice_audio
        enhance_fn = partial(enhance_voice_audio, wav_bytes)
        wav_bytes = await loop.run_in_executor(None, enhance_fn)
        logger.info("Voice enhancement pipeline applied.")
    except Exception as exc:
        logger.warning(f"Voice enhancement skipped (non-fatal): {exc}")

    # ── 4. Upload generated audio to object storage ────────────────────────
    audio_key = await loop.run_in_executor(
        None, partial(storage.upload_bytes, wav_bytes, "audio/wav", "generated")
    )

    # ── 5. Persist generation record ──────────────────────────────────────
    gen_id: uuid.UUID
    try:
        generation = Generation(
            voice_id=voice.id,
            user_id=current_user.id,
            input_text=payload.text,
            audio_s3_key=audio_key,
            engine=payload.engine or "gpt-sovits-v3",
            emotion=active_emotion,
            speed=active_speed,
            eval_status="pending",
        )
        db.add(generation)
        await db.commit()
        await db.refresh(generation)
        gen_id = generation.id
    except Exception as db_err:
        logger.warning(f"Generation record save notice: {db_err}")
        try:
            await db.rollback()
        except Exception:
            pass
        gen_id = uuid.uuid4()

    # ── 5.5 Enqueue Async Multi-Metric Evaluation Task ──────────────────────
    background_tasks.add_task(
        run_async_evaluation,
        generation_id=gen_id,
        voice_id=voice.id,
        input_text=payload.text,
        audio_s3_key=audio_key,
        sample_s3_key=voice.sample_s3_key,
    )

    # ── 6. Return with fresh presigned URL ─────────────────────────────────
    audio_url = storage.generate_presigned_url(audio_key, expires_in=3600)
    return GenerationOut(
        id=gen_id,
        voice_id=voice.id,
        input_text=payload.text,
        audio_url=audio_url,
        engine=payload.engine or "gpt-sovits-v3",
        emotion=active_emotion,
        speed=active_speed,
        eval_status="pending",
        speaker_similarity=None,
        word_error_rate=None,
        prosody_f0_std=None,
        composite_grade=None,
        composite_score=None,
        evaluated_at=None,
        created_at=datetime.now(timezone.utc),
    )


@router.get(
    "/generations",
    response_model=list[GenerationOut],
    summary="List past TTS generations for the current user",
)
async def list_generations(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    voice_id: Optional[str] = Query(default=None, description="Filter by voice ID"),
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
        try:
            target_vid = uuid.UUID(voice_id)
            query = query.where(Generation.voice_id == target_vid)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid voice_id format — must be a valid UUID.",
            )

    result = await db.execute(query)
    generations = result.scalars().all()

    return [
        GenerationOut(
            id=g.id,
            voice_id=g.voice_id,
            input_text=g.input_text,
            audio_url=storage.generate_presigned_url(g.audio_s3_key, expires_in=3600),
            engine=g.engine,
            emotion=g.emotion,
            speed=g.speed,
            eval_status=g.eval_status or "pending",
            speaker_similarity=g.speaker_similarity,
            word_error_rate=g.word_error_rate,
            prosody_f0_std=g.prosody_f0_std,
            composite_grade=g.composite_grade,
            composite_score=g.composite_score,
            evaluated_at=g.evaluated_at,
            created_at=g.created_at,
        )
        for g in generations
    ]


@router.get(
    "/generations/{generation_id}/eval",
    response_model=GenerationEvalOut,
    summary="Fetch real-time evaluation status and metrics for a generation",
)
async def get_generation_evaluation(
    generation_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> GenerationEvalOut:
    try:
        gen_uuid = uuid.UUID(generation_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid generation UUID format.")

    result = await db.execute(select(Generation).where(Generation.id == gen_uuid))
    generation = result.scalar_one_or_none()

    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")

    if str(generation.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view evaluation for this generation.",
        )

    return GenerationEvalOut(
        generation_id=generation.id,
        voice_id=generation.voice_id,
        eval_status=generation.eval_status or "pending",
        speaker_similarity=generation.speaker_similarity,
        word_error_rate=generation.word_error_rate,
        prosody_f0_std=generation.prosody_f0_std,
        composite_grade=generation.composite_grade,
        composite_score=generation.composite_score,
        eval_error=generation.eval_error,
        evaluated_at=generation.evaluated_at,
        created_at=generation.created_at,
    )


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
    current_user: Annotated[User, Depends(get_current_user)],   # FIX: auth required
    db: Annotated[AsyncSession, Depends(get_db)],
    format: str = Query("wav", description="Audio format: 'wav' or 'mp3'"),
):
    """Stream generated audio directly for HTML5 audio playback with correct Content-Type."""
    import io as _io

    # FIX: parse UUID properly to avoid type mismatch on PostgreSQL
    try:
        gen_uuid = uuid.UUID(generation_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid generation ID format.")

    result = await db.execute(select(Generation).where(Generation.id == gen_uuid))
    generation = result.scalar_one_or_none()

    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")

    # FIX: ownership check (was completely missing)
    if str(generation.user_id) != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")

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
            data, sr = sf.read(_io.BytesIO(raw_bytes))
            buf = _io.BytesIO()
            sf.write(buf, data, sr, format="MP3")
            return Response(
                content=buf.getvalue(),
                media_type="audio/mpeg",
                headers={"Accept-Ranges": "bytes", "Cache-Control": "public, max-age=3600"},
            )
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
    token: Optional[str] = Query(default=None, description="Optional JWT token in query"),
    current_user: Annotated[Optional[User], Depends(get_current_user)] = None,
):
    """Download the generated audio file with chosen format (MP3 or WAV)."""
    import io as _io
    from app.auth import get_user_from_token_str

    # FIX: parse UUID properly to avoid type mismatch on PostgreSQL
    try:
        gen_uuid = uuid.UUID(generation_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid generation ID format.")

    result = await db.execute(select(Generation).where(Generation.id == gen_uuid))
    generation = result.scalar_one_or_none()

    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")

    # FIX: resolve auth_user from Bearer header or query token
    auth_user = current_user
    if auth_user is None and token:
        auth_user = await get_user_from_token_str(token, db)

    # FIX: auth is now mandatory — no unauthenticated downloads
    if auth_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to download audio.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if str(generation.user_id) != str(auth_user.id):
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
            data, sr = sf.read(_io.BytesIO(raw_bytes))
            buf = _io.BytesIO()
            sf.write(buf, data, sr, format="MP3")
            return Response(
                content=buf.getvalue(),
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
    against the generated TTS audio output.
    """
    from functools import partial as _partial
    from app.utils.similarity import compute_voice_similarity

    try:
        gen_uuid = uuid.UUID(generation_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid generation ID format.")

    result = await db.execute(select(Generation).where(Generation.id == gen_uuid))
    generation = result.scalar_one_or_none()

    if generation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation not found.")
    if str(generation.user_id) != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden.")

    voice_result = await db.execute(select(Voice).where(Voice.id == generation.voice_id))
    voice = voice_result.scalar_one_or_none()
    if voice is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found.")

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

    try:
        loop = asyncio.get_running_loop()
        sim_fn = _partial(compute_voice_similarity, ref_bytes, gen_bytes)
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
            # FIX: removed incorrect energy_correlation alias that was returning f0_correlation
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


# ── Engine Status & Presets Endpoints ──────────────────────────────────────

@router.get(
    "/engines/status",
    summary="Live readiness status for all TTS engines",
)
async def engine_status() -> list[dict]:
    from app.tts_engines import get_all_engine_status
    return get_all_engine_status()


@router.get(
    "/engines/presets",
    summary="Pocket TTS studio presets",
)
async def engine_presets() -> dict:
    from app.models import POCKET_TTS_PRESETS
    return {"presets": POCKET_TTS_PRESETS}
