"""
song_cover.py — Production Router for v2 Song Voice Conversion & Song Cover Pipeline.

Provides:
  - POST /song-covers/         — Ingest song via UPLOAD, SEARCH (URL), or LIBRARY (Personal/Curated)
  - GET  /song-covers/{id}/status — Poll job progress (0-100%), stage message, and preview URL
  - GET  /song-covers/{id}     — Get full song cover details & presigned audio URLs
  - GET  /song-covers/         — List past song covers for the current user
  - GET  /song-covers/curated  — List curated demo songs for instant 0ms-separation conversion
  - GET  /song-covers/library  — List previously processed songs in user's personal library for 1-click re-use
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy import distinct, select
from sqlalchemy.ext.asyncio import AsyncSession

from app import storage
from app.audio_mixer import mix_and_master_song
from app.auth import get_current_user
from app.db import AsyncSessionLocal, get_db
from app.models import (
    CuratedSongOut,
    SongCover,
    SongCoverCreate,
    SongCoverOut,
    SongCoverStatusOut,
    User,
    Voice,
)
from app.services.curated_catalog import get_curated_song_stems, get_curated_songs
from app.services.job_queue import ARTIFACTS_ROOT, job_manager
from app.services.song_fetcher import fetch_and_canonicalize_url, validate_song_url
from app.svc_engines import get_svc_engine
from app.utils.audio import validate_audio_upload
from app.utils.audio_asset import AudioAsset, load_audio_asset
from app.vocal_analysis import analyze_vocal_track
from app.vocal_separation import SeparationArtifact, separate_vocals_and_instrumental

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/song-covers", tags=["song-covers"])

MAX_SONG_DURATION_SEC = 300.0  # Strict 5-minute limit


async def _background_song_cover_worker(
    job_id: str,
    user_id: uuid.UUID,
    voice_id: uuid.UUID,
    raw_audio_bytes: bytes,
    title: str,
    pitch_shift: int,
    index_rate: float,
    protect_voiceless: float,
    preview_only: bool,
    source_type: str = "UPLOAD",
    source_url: Optional[str] = None,
    library_song_hash: Optional[str] = None,
):
    """
    Asynchronous background worker executed on the event loop:
      Step 1: Stem separation (Demucs / Colab GPU / Cached / Curated)
      Step 2: Vocal analysis (F0 pitch, voicing, median range)
      Step 3: Chunked singing voice conversion with checkpoints
      Step 4: Studio mixing & mastering
      Step 5: Storage upload and metadata logging
    """
    song_hash = library_song_hash or job_manager.get_audio_hash(raw_audio_bytes)
    song_dir = ARTIFACTS_ROOT / song_hash
    song_dir.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()

    try:
        # ── Step 1: Stem Separation ───────────────────────────────────────────
        job_manager.update_progress(job_id, "separating", 10.0, "Retrieving isolated stems (Demucs / Cached)...")
        # Check curated stems first
        separation = get_curated_song_stems(song_hash)
        if separation is None:
            separation = await loop.run_in_executor(
                None, lambda: separate_vocals_and_instrumental(raw_audio_bytes, song_hash=song_hash)
            )
        job_manager.update_progress(job_id, "separating", 30.0, "Vocal and instrumental stems isolated.")

        # ── Step 2: Vocal Analysis & Pitch Range Profiling ────────────────────
        job_manager.update_progress(job_id, "analyzing", 35.0, "Extracting F0 pitch contour and vocal register...")
        vocals_asset = separation.vocals_asset
        analysis = await loop.run_in_executor(
            None, lambda: analyze_vocal_track(vocals_asset, song_hash=song_hash)
        )

        # Apply recommended pitch transposition if user left pitch_shift at 0 and cross-register detected
        effective_pitch_shift = pitch_shift
        if pitch_shift == 0 and abs(analysis.recommended_pitch_shift) >= 10:
            logger.info(f"Applying auto-transposition recommendation: {analysis.recommended_pitch_shift:+d} semitones")
            effective_pitch_shift = analysis.recommended_pitch_shift

        # ── Step 3: Chunked Singing Voice Conversion (RVC v2) ─────────────────
        job_manager.update_progress(job_id, "converting", 40.0, f"Converting vocals to target voice (pitch_shift={effective_pitch_shift:+d}st)...")
        svc = get_svc_engine("rvc-v2")
        checkpoints_dir = song_dir / f"checkpoints_{voice_id}"

        # Fetch target voice sample for neural acoustic timbre matching
        target_voice_bytes: Optional[bytes] = None
        try:
            async with AsyncSessionLocal() as session:
                v_res = await session.execute(select(Voice).where(Voice.id == voice_id))
                voice_rec = v_res.scalar_one_or_none()
                if voice_rec and voice_rec.sample_s3_key:
                    target_voice_bytes = await loop.run_in_executor(
                        None, lambda: storage.get_bytes(voice_rec.sample_s3_key)
                    )
        except Exception as v_err:
            logger.warning(f"Could not load reference sample for voice {voice_id}: {v_err}")

        def _svc_progress(pct: float, msg: str):
            job_manager.update_progress(job_id, "converting", pct, msg)

        converted_vocals_asset = await loop.run_in_executor(
            None,
            lambda: svc.convert_full_vocals(
                vocals_asset=vocals_asset,
                voice_id=str(voice_id),
                pitch_shift=effective_pitch_shift,
                index_rate=index_rate,
                protect_voiceless=protect_voiceless,
                preview_only=preview_only,
                checkpoint_dir=checkpoints_dir,
                progress_callback=_svc_progress,
                target_voice_bytes=target_voice_bytes,
            )
        )

        # ── Step 4: Studio Mixing & Mastering ─────────────────────────────────
        job_manager.update_progress(job_id, "mixing", 85.0, "Mastering vocals and summing with instrumental...")
        instrumental_asset = separation.instrumental_asset

        if preview_only:
            # Match instrumental length to preview vocals
            preview_samples = converted_vocals_asset.num_samples
            inst_samples = instrumental_asset.samples
            inst_trimmed = inst_samples[:, :preview_samples] if inst_samples.ndim == 2 else inst_samples[:preview_samples]
            instrumental_asset = AudioAsset(samples=inst_trimmed, sample_rate=instrumental_asset.sample_rate)

        output_filename = f"cover_preview_{voice_id}.wav" if preview_only else f"cover_master_{voice_id}.wav"
        final_output_path = song_dir / output_filename

        mastered_asset = await loop.run_in_executor(
            None,
            lambda: mix_and_master_song(
                converted_vocals_asset=converted_vocals_asset,
                instrumental_asset=instrumental_asset,
                output_path=final_output_path,
                original_vocals_asset=vocals_asset,
                vocal_gain_db=0.5,
                ducking_enabled=True,
            )
        )

        # ── Step 5: Upload to Storage & Finalize ──────────────────────────────
        job_manager.update_progress(job_id, "mixing", 95.0, "Uploading final master and stems to storage...")
        final_bytes = mastered_asset.to_bytes()
        prefix = "song-previews" if preview_only else "song-covers"
        final_key = await loop.run_in_executor(
            None, lambda: storage.upload_bytes(final_bytes, "audio/wav", prefix=prefix)
        )
        conv_vocals_key = await loop.run_in_executor(
            None, lambda: storage.upload_bytes(converted_vocals_asset.to_bytes(), "audio/wav", prefix="song-vocals")
        )
        inst_key = await loop.run_in_executor(
            None, lambda: storage.upload_bytes(instrumental_asset.to_bytes(), "audio/wav", prefix="song-instrumentals")
        )
        final_url = storage.generate_presigned_url(final_key, expires_in=86400)

        # Build comprehensive engine and execution metadata
        metadata_payload = {
            "source_type": source_type,
            "source_url": source_url,
            "song_hash": song_hash,
            "svc_engine": "rvc-v2",
            "f0_model": "pyin",
            "separator": "htdemucs",
            "model_versions": {"rvc": "2.0", "demucs": "4.0"},
            "pitch_shift_applied": effective_pitch_shift,
            "source_median_f0": analysis.median_f0,
            "duration_seconds": mastered_asset.duration,
            "sample_rate": mastered_asset.sample_rate,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Sync with Database directly on current event loop
        async with AsyncSessionLocal() as session:
            stmt = select(SongCover).where(SongCover.id == uuid.UUID(job_id))
            res = await session.execute(stmt)
            cover = res.scalar_one_or_none()
            if cover:
                cover.status = "completed"
                cover.progress = 100.0
                cover.completed_at = datetime.now(timezone.utc)
                cover.metadata_json = metadata_payload
                cover.song_hash = song_hash
                cover.vocals_s3_key = conv_vocals_key
                cover.converted_vocals_s3_key = conv_vocals_key
                cover.instrumental_s3_key = inst_key
                if preview_only:
                    cover.preview_s3_key = final_key
                else:
                    cover.final_mix_s3_key = final_key
                await session.commit()

        job_manager.update_progress(
            job_id=job_id,
            status="completed",
            progress=100.0,
            message="Song cover rendered successfully!",
            audio_url=final_url if not preview_only else None,
            preview_url=final_url if preview_only else None,
        )
        logger.info(f"Song cover job {job_id} finished successfully!")

    except Exception as exc:
        logger.error(f"Song cover job {job_id} failed: {exc}", exc_info=True)
        try:
            async with AsyncSessionLocal() as session:
                stmt = select(SongCover).where(SongCover.id == uuid.UUID(job_id))
                res = await session.execute(stmt)
                cover = res.scalar_one_or_none()
                if cover:
                    cover.status = "failed"
                    cover.error_message = str(exc)
                    await session.commit()
        except Exception as db_err:
            logger.error(f"Failed to update error status in DB for job {job_id}: {db_err}")

        job_manager.update_progress(
            job_id=job_id,
            status="failed",
            progress=100.0,
            message="Conversion failed",
            error=str(exc),
        )


@router.get(
    "/curated",
    response_model=List[CuratedSongOut],
    summary="List curated demo songs ready for instant 1-click conversion with zero separation wait",
)
async def list_curated_songs() -> List[CuratedSongOut]:
    """Returns pre-processed royalty-free demo tracks with cached stems."""
    return get_curated_songs()


@router.get(
    "/library",
    response_model=List[SongCoverOut],
    summary="List user's previously separated songs with cached stems for instant re-conversion",
)
async def list_library_songs(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> List[SongCoverOut]:
    """Returns distinct songs that have already been separated for the user."""
    stmt = (
        select(SongCover)
        .where(SongCover.user_id == current_user.id)
        .where(SongCover.status == "completed")
        .order_by(SongCover.created_at.desc())
    )
    res = await db.execute(stmt)
    covers = res.scalars().all()

    seen_hashes = set()
    unique_library: List[SongCoverOut] = []

    for c in covers:
        shash = c.song_hash or str(c.id)
        if shash not in seen_hashes:
            seen_hashes.add(shash)
            audio_url = storage.generate_presigned_url(c.final_mix_s3_key) if c.final_mix_s3_key else None
            unique_library.append(
                SongCoverOut(
                    id=c.id,
                    voice_id=c.voice_id,
                    title=c.title,
                    status=c.status,
                    progress=c.progress,
                    pitch_shift=c.pitch_shift,
                    source_type=c.source_type,
                    song_hash=c.song_hash,
                    audio_url=audio_url,
                    preview_url=None,
                    vocals_url=None,
                    instrumental_url=None,
                    is_preview=c.is_preview,
                    metadata_json=c.metadata_json,
                    error_message=c.error_message,
                    created_at=c.created_at,
                    completed_at=c.completed_at,
                )
            )
    return unique_library


@router.post(
    "",
    response_model=SongCoverStatusOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a song cover or 20s preview using UPLOAD, SEARCH (URL), or LIBRARY",
)
async def create_song_cover(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    background_tasks: BackgroundTasks,
    voice_id: Annotated[uuid.UUID, Form()],
    file: Annotated[Optional[UploadFile], File()] = None,
    source_type: Annotated[str, Form()] = "UPLOAD",
    source_url: Annotated[Optional[str], Form()] = None,
    library_song_hash: Annotated[Optional[str], Form()] = None,
    title: Annotated[Optional[str], Form()] = None,
    pitch_shift: Annotated[int, Form()] = 0,
    index_rate: Annotated[float, Form()] = 0.75,
    protect_voiceless: Annotated[float, Form()] = 0.33,
    preview_only: Annotated[bool, Form()] = False,
    tos_confirmed: Annotated[bool, Form()] = True,
) -> SongCoverStatusOut:
    # ── 1. Voice Verification & Ownership ─────────────────────────────────────
    res = await db.execute(select(Voice).where(Voice.id == voice_id))
    voice = res.scalar_one_or_none()
    if voice is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Voice not found.")

    if voice.owner_id and str(voice.owner_id) != str(current_user.id):
        from sqlalchemy import func
        user_voice_count = await db.scalar(
            select(func.count(Voice.id)).where(Voice.owner_id == current_user.id)
        )
        if user_voice_count and user_voice_count > 0:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Permission denied for this voice.")

    # ── 2. Handle Entry Point (UPLOAD, SEARCH, LIBRARY) & Validate ≤ 5 min ────
    norm_source = source_type.upper().strip()
    detected_title = title

    if norm_source == "SEARCH":
        if not source_url:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="source_url is required for SEARCH source_type.")
        try:
            loop = asyncio.get_running_loop()
            raw_audio_bytes, fetched_title, duration = await loop.run_in_executor(
                None, lambda: fetch_and_canonicalize_url(source_url, max_duration_sec=MAX_SONG_DURATION_SEC)
            )
            detected_title = title or fetched_title
        except ValueError as val_err:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch song from URL: {e}")

    elif norm_source == "LIBRARY":
        if not library_song_hash:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="library_song_hash is required for LIBRARY source_type.")
        # Check curated demo stems
        cached_stem = get_curated_song_stems(library_song_hash)
        if cached_stem:
            raw_audio_bytes = cached_stem.vocals_asset.to_bytes()
            detected_title = title or f"Library Song ({library_song_hash})"
        else:
            # Check personal cache
            cached_vocals = ARTIFACTS_ROOT / library_song_hash / "vocals.wav"
            if not cached_vocals.exists():
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No cached stems found for library song hash '{library_song_hash}'.")
            with open(cached_vocals, "rb") as vf:
                raw_audio_bytes = vf.read()
            detected_title = title or f"Library Track ({library_song_hash[:8]})"

    else:
        # Default UPLOAD flow
        if file is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file upload is required for UPLOAD source_type.")
        raw_audio_bytes = await validate_audio_upload(file)
        # Check duration limit
        asset = load_audio_asset(raw_audio_bytes)
        if asset.duration > MAX_SONG_DURATION_SEC:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Song duration ({asset.duration:.1f}s) exceeds the strict 5-minute (300s) maximum limit."
            )
        detected_title = title or f"{Path(file.filename or 'Song').stem}"

    # ── 3. Persist Initial SongCover Record ────────────────────────────────────
    cover_title = f"{detected_title} ({voice.name} Cover)"
    song_job_id = uuid.uuid4()
    computed_hash = library_song_hash or job_manager.get_audio_hash(raw_audio_bytes)

    # Upload original audio to object storage
    loop = asyncio.get_running_loop()
    orig_key = await loop.run_in_executor(
        None, lambda: storage.upload_bytes(raw_audio_bytes, "audio/wav", prefix="song-originals")
    )

    cover_record = SongCover(
        id=song_job_id,
        user_id=current_user.id,
        voice_id=voice.id,
        title=cover_title,
        status="pending",
        progress=0.0,
        pitch_shift=pitch_shift,
        index_rate=index_rate,
        protect_voiceless=protect_voiceless,
        is_preview=preview_only,
        source_type=norm_source,
        source_url=source_url,
        song_hash=computed_hash,
        original_audio_s3_key=orig_key,
    )
    db.add(cover_record)
    await db.commit()
    await db.refresh(cover_record)

    # Register in in-memory tracker
    job_manager.create_job(str(song_job_id))

    # ── 4. Enqueue Asynchronous Background Processing ─────────────────────────
    background_tasks.add_task(
        _background_song_cover_worker,
        job_id=str(song_job_id),
        user_id=current_user.id,
        voice_id=voice.id,
        raw_audio_bytes=raw_audio_bytes,
        title=cover_title,
        pitch_shift=pitch_shift,
        index_rate=index_rate,
        protect_voiceless=protect_voiceless,
        preview_only=preview_only,
        source_type=norm_source,
        source_url=source_url,
        library_song_hash=computed_hash,
    )

    return SongCoverStatusOut(
        id=song_job_id,
        status="pending",
        progress=0.0,
        audio_url=None,
        preview_url=None,
        error_message=None,
    )


@router.get(
    "/{cover_id}/status",
    response_model=SongCoverStatusOut,
    summary="Poll processing status and progress for a song cover",
)
async def get_song_cover_status(
    cover_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SongCoverStatusOut:
    # 1. Check in-memory fast status tracker first
    prog = job_manager.get_progress(str(cover_id))
    if prog:
        return SongCoverStatusOut(
            id=cover_id,
            status=prog.status,
            progress=prog.progress,
            audio_url=prog.audio_url,
            preview_url=prog.preview_url,
            error_message=prog.error_message,
        )

    # 2. Fallback to Database
    res = await db.execute(select(SongCover).where(SongCover.id == cover_id))
    cover = res.scalar_one_or_none()
    if cover is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Song cover not found.")

    if str(cover.user_id) != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    audio_url = storage.generate_presigned_url(cover.final_mix_s3_key) if cover.final_mix_s3_key else None
    preview_url = storage.generate_presigned_url(cover.preview_s3_key) if cover.preview_s3_key else None

    return SongCoverStatusOut(
        id=cover.id,
        status=cover.status,
        progress=cover.progress,
        audio_url=audio_url,
        preview_url=preview_url,
        error_message=cover.error_message,
    )


@router.get(
    "/{cover_id}",
    response_model=SongCoverOut,
    summary="Get full details, metadata, and audio URLs of a song cover",
)
async def get_song_cover_detail(
    cover_id: uuid.UUID,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SongCoverOut:
    res = await db.execute(select(SongCover).where(SongCover.id == cover_id))
    cover = res.scalar_one_or_none()
    if cover is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Song cover not found.")

    if str(cover.user_id) != str(current_user.id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    audio_url = storage.generate_presigned_url(cover.final_mix_s3_key) if cover.final_mix_s3_key else None
    preview_url = storage.generate_presigned_url(cover.preview_s3_key) if cover.preview_s3_key else None
    vocals_url = storage.generate_presigned_url(cover.vocals_s3_key) if cover.vocals_s3_key else None
    instrumental_url = storage.generate_presigned_url(cover.instrumental_s3_key) if cover.instrumental_s3_key else None

    return SongCoverOut(
        id=cover.id,
        voice_id=cover.voice_id,
        title=cover.title,
        status=cover.status,
        progress=cover.progress,
        pitch_shift=cover.pitch_shift,
        source_type=cover.source_type,
        song_hash=cover.song_hash,
        audio_url=audio_url,
        preview_url=preview_url,
        vocals_url=vocals_url,
        instrumental_url=instrumental_url,
        is_preview=cover.is_preview,
        metadata_json=cover.metadata_json,
        error_message=cover.error_message,
        created_at=cover.created_at,
        completed_at=cover.completed_at,
    )


@router.get(
    "",
    response_model=List[SongCoverOut],
    summary="List all song covers created by the current user",
)
async def list_song_covers(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> List[SongCoverOut]:
    stmt = (
        select(SongCover)
        .where(SongCover.user_id == current_user.id)
        .order_by(SongCover.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    res = await db.execute(stmt)
    covers = res.scalars().all()

    out: List[SongCoverOut] = []
    for c in covers:
        audio_url = storage.generate_presigned_url(c.final_mix_s3_key) if c.final_mix_s3_key else None
        preview_url = storage.generate_presigned_url(c.preview_s3_key) if c.preview_s3_key else None
        out.append(
            SongCoverOut(
                id=c.id,
                voice_id=c.voice_id,
                title=c.title,
                status=c.status,
                progress=c.progress,
                pitch_shift=c.pitch_shift,
                source_type=c.source_type,
                song_hash=c.song_hash,
                audio_url=audio_url,
                preview_url=preview_url,
                vocals_url=None,
                instrumental_url=None,
                is_preview=c.is_preview,
                metadata_json=c.metadata_json,
                error_message=c.error_message,
                created_at=c.created_at,
                completed_at=c.completed_at,
            )
        )
    return out
