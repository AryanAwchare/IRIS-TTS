"""
Content Accuracy Module using Whisper and JiWER.

Transcribes generated audio using faster-whisper on CPU and computes
the exact Word Error Rate (WER) against the intended prompt text.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_WHISPER_MODEL: Optional[object] = None
_INIT_LOCK = threading.Lock()


def _get_whisper_model() -> object:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        with _INIT_LOCK:
            if _WHISPER_MODEL is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError:
                    raise ImportError(
                        "faster-whisper package is required for content_accuracy. "
                        "Please install it using: pip install faster-whisper"
                    )
                logger.info("Loading faster-whisper (small.en) model on CPU...")
                _WHISPER_MODEL = WhisperModel(
                    "small.en",
                    device="cpu",
                    compute_type="int8",
                    download_root=os.path.join(os.path.expanduser("~"), ".cache", "whisper"),
                )
                logger.info("faster-whisper loaded and ready.")
    return _WHISPER_MODEL


def word_error_rate(intended_text: str, gen_wav_path: str) -> float:
    """
    Transcribes the generated audio and calculates Word Error Rate (WER) against intended text.

    Args:
        intended_text: The prompt string that was requested to be spoken.
        gen_wav_path: Path to the synthesized audio WAV file.

    Returns:
        Word Error Rate as a float (e.g. 0.0 = perfect match, 0.25 = 25% error rate).
    """
    if not os.path.exists(gen_wav_path):
        logger.error(f"Generated audio file not found: {gen_wav_path}")
        return 1.0

    if not intended_text or not intended_text.strip():
        return 0.0

    try:
        import jiwer
    except ImportError:
        logger.warning("jiwer package not found. Install via: pip install jiwer")
        return 0.0

    try:
        model = _get_whisper_model()
        segments, _ = model.transcribe(gen_wav_path, beam_size=5, language="en")
        transcribed_text = " ".join([s.text.strip() for s in segments]).strip()

        # Clean punctuation and casing for fair WER evaluation
        transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])

        ref_clean = transform(intended_text)
        hyp_clean = transform(transcribed_text)

        if not ref_clean:
            return 0.0 if not hyp_clean else 1.0

        wer = jiwer.wer(ref_clean, hyp_clean)
        return float(wer)

    except Exception as exc:
        logger.error(f"Failed to compute WER: {exc}")
        return 1.0
