"""
Async Multi-Metric Evaluation Pipeline.

Executes objective evaluation (ECAPA-TDNN Speaker Similarity, Whisper WER, F0 Prosody)
in a background task, persisting metrics and composite grade to the database.

FIX: Added asyncio.wait_for(timeout=300) on the run_in_executor call.
     Without a timeout, a hanging ECAPA-TDNN or Whisper inference (e.g. corrupted
     audio, GPU OOM) would block a thread pool worker indefinitely.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import os
import tempfile
import uuid
from typing import Optional

from sqlalchemy import update

from app import storage
from app.db import AsyncSessionLocal
from app.models import Generation
from app.evaluation.speaker_similarity import speaker_similarity
from app.evaluation.content_accuracy import word_error_rate
from app.evaluation.prosody_metric import compute_generated_prosody
from app.evaluation.grade import compute_composite_grade

logger = logging.getLogger(__name__)


def _run_metrics_sync(ref_wav_path: str, gen_wav_path: str, input_text: str) -> dict:
    """Synchronous CPU worker to evaluate all 3 metrics + composite grade."""
    sim = speaker_similarity(ref_wav_path, gen_wav_path)
    wer = word_error_rate(input_text, gen_wav_path)
    prosody = compute_generated_prosody(gen_wav_path)
    f0_std = prosody.get("f0_std_hz", 0.0)
    grade, score = compute_composite_grade(sim, wer, f0_std)

    return {
        "speaker_similarity": round(float(sim), 4),
        "word_error_rate":    round(float(wer), 4),
        "prosody_f0_std":     round(float(f0_std), 2),
        "composite_grade":    grade,
        "composite_score":    score,
        "prosody_meta":       prosody,
    }


async def run_async_evaluation(
    generation_id: uuid.UUID | str,
    voice_id: uuid.UUID | str,
    input_text: str,
    audio_s3_key: str,
    sample_s3_key: str,
) -> None:
    """
    Background worker task dispatched immediately after audio generation.
    Downloads reference and generated audio, evaluates objective metrics,
    and updates the generation record in the database.
    """
    gen_uuid = uuid.UUID(str(generation_id))
    logger.info(f"Starting async evaluation for generation {gen_uuid}...")

    ref_tmp_path: Optional[str] = None
    gen_tmp_path: Optional[str] = None

    try:
        if not sample_s3_key or not audio_s3_key:
            raise ValueError(f"Missing S3 storage keys (sample: {sample_s3_key}, audio: {audio_s3_key})")

        ref_bytes = storage.download_bytes(sample_s3_key)
        with tempfile.NamedTemporaryFile(suffix="_ref.wav", delete=False) as ref_tmp:
            ref_tmp.write(ref_bytes)
            ref_tmp_path = ref_tmp.name

        gen_bytes = storage.download_bytes(audio_s3_key)
        with tempfile.NamedTemporaryFile(suffix="_gen.wav", delete=False) as gen_tmp:
            gen_tmp.write(gen_bytes)
            gen_tmp_path = gen_tmp.name

        loop = asyncio.get_running_loop()

        # FIX: added 300-second timeout. Without it, a hanging ECAPA-TDNN or
        # Whisper inference would block a thread pool worker indefinitely.
        results = await asyncio.wait_for(
            loop.run_in_executor(
                None, _run_metrics_sync, ref_tmp_path, gen_tmp_path, input_text
            ),
            timeout=300.0,
        )

        logger.info(
            f"Generation {gen_uuid} evaluated: "
            f"SIM={results['speaker_similarity']:.3f}, "
            f"WER={results['word_error_rate']:.3f}, "
            f"F0_std={results['prosody_f0_std']:.1f}Hz → "
            f"Grade [{results['composite_grade']}] (Score: {results['composite_score']:.3f})"
        )

        async with AsyncSessionLocal() as session:
            stmt = (
                update(Generation)
                .where(Generation.id == gen_uuid)
                .values(
                    eval_status="completed",
                    speaker_similarity=results["speaker_similarity"],
                    word_error_rate=results["word_error_rate"],
                    prosody_f0_std=results["prosody_f0_std"],
                    composite_grade=results["composite_grade"],
                    composite_score=results["composite_score"],
                    eval_error=None,
                    evaluated_at=datetime.now(timezone.utc),
                )
            )
            await session.execute(stmt)
            await session.commit()

    except asyncio.TimeoutError:
        timeout_msg = "Evaluation timed out after 300 seconds (metrics computation too slow)"
        logger.error(f"Evaluation timeout for generation {gen_uuid}: {timeout_msg}")
        try:
            async with AsyncSessionLocal() as session:
                stmt = (
                    update(Generation)
                    .where(Generation.id == gen_uuid)
                    .values(
                        eval_status="failed",
                        eval_error=timeout_msg,
                        evaluated_at=datetime.now(timezone.utc),
                    )
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as db_err:
            logger.error(f"Failed to record evaluation timeout to DB: {db_err}")

    except Exception as exc:
        logger.error(f"Async evaluation failed for generation {gen_uuid}: {exc}", exc_info=True)
        try:
            async with AsyncSessionLocal() as session:
                stmt = (
                    update(Generation)
                    .where(Generation.id == gen_uuid)
                    .values(
                        eval_status="failed",
                        eval_error=str(exc),
                        evaluated_at=datetime.now(timezone.utc),
                    )
                )
                await session.execute(stmt)
                await session.commit()
        except Exception as db_err:
            logger.error(f"Failed to record evaluation failure to DB: {db_err}")

    finally:
        for path in (ref_tmp_path, gen_tmp_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass
