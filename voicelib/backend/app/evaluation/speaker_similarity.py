"""
Speaker Similarity Module using SpeechBrain ECAPA-TDNN.

Computes 192-dimensional speaker embeddings from VoxCeleb-pretrained ECAPA-TDNN model
and calculates exact cosine similarity between reference and generated audio files.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# SpeechBrain version-compatible import
try:
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    try:
        from speechbrain.pretrained import EncoderClassifier
    except ImportError:
        EncoderClassifier = None

_CLASSIFIER: Optional[object] = None
_INIT_LOCK = threading.Lock()
_SAVEDIR = os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "spkrec-ecapa-voxceleb")


def _get_classifier() -> object:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        with _INIT_LOCK:
            if _CLASSIFIER is None:
                if EncoderClassifier is None:
                    raise ImportError(
                        "speechbrain package is required for speaker_similarity. "
                        "Please install it using: pip install speechbrain"
                    )
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Loading SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb) model on {device.upper()}...")
                _CLASSIFIER = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=_SAVEDIR,
                    run_opts={"device": device},
                )
                logger.info(f"SpeechBrain ECAPA-TDNN loaded and ready on {device.upper()}.")
    return _CLASSIFIER


def speaker_similarity(ref_wav_path: str, gen_wav_path: str) -> float:
    """
    Computes raw cosine similarity between 192-dimensional ECAPA-TDNN speaker embeddings.

    Args:
        ref_wav_path: Path to reference audio sample (.wav).
        gen_wav_path: Path to generated output audio sample (.wav).

    Returns:
        Raw cosine similarity float in range [0.0, 1.0].
    """
    if not os.path.exists(ref_wav_path):
        logger.error(f"Reference audio file not found: {ref_wav_path}")
        return 0.0
    if not os.path.exists(gen_wav_path):
        logger.error(f"Generated audio file not found: {gen_wav_path}")
        return 0.0

    try:
        classifier = _get_classifier()

        # Load audio signals (SpeechBrain load_audio resamples automatically to 16kHz)
        signal_ref = classifier.load_audio(ref_wav_path)
        signal_gen = classifier.load_audio(gen_wav_path)

        if signal_ref.shape[-1] == 0 or signal_gen.shape[-1] == 0:
            logger.warning("Empty audio signal encountered in speaker_similarity.")
            return 0.0

        with torch.no_grad():
            emb_ref = classifier.encode_batch(signal_ref)
            emb_gen = classifier.encode_batch(signal_gen)

        v_ref = emb_ref.squeeze()
        v_gen = emb_gen.squeeze()

        cos_sim = F.cosine_similarity(v_ref, v_gen, dim=0).item()

        # Clamp to [0.0, 1.0] for clean metric reporting
        return float(max(0.0, min(1.0, cos_sim)))

    except Exception as exc:
        logger.error(f"Failed to compute speaker similarity: {exc}")
        return 0.0
