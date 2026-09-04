"""
Composite Quality Grade Module for Voice Synthesis.

Combines multi-metric evaluation into an intuitive composite grade (A, B, C, D):
- Speaker Similarity (ECAPA-TDNN): 50% weight
- Linguistic Accuracy (1 - WER): 35% weight
- Prosodic Expressiveness (F0 dynamic range): 15% weight
"""
from __future__ import annotations

from typing import Tuple


def compute_composite_grade(
    speaker_similarity: float,
    word_error_rate: float,
    prosody_f0_std: float,
) -> Tuple[str, float]:
    """
    Calculates composite quality score [0.0 - 1.0] and assigns grade A, B, C, or D.

    Args:
        speaker_similarity: Cosine similarity [0.0, 1.0].
        word_error_rate: Word Error Rate [0.0, 1.0+].
        prosody_f0_std: Pitch standard deviation in Hz.

    Returns:
        (grade: str, composite_score: float)
    """
    sim_clamped = max(0.0, min(1.0, float(speaker_similarity if speaker_similarity is not None else 0.0)))
    wer_val = 1.0 if word_error_rate is None else float(word_error_rate)
    wer_accuracy = max(0.0, min(1.0, 1.0 - wer_val))
    # 35 Hz standard deviation is considered standard natural expressive range
    f0_score = max(0.0, min(1.0, float(prosody_f0_std if prosody_f0_std is not None else 0.0) / 35.0))

    composite_score = (0.50 * sim_clamped) + (0.35 * wer_accuracy) + (0.15 * f0_score)
    composite_score = round(composite_score, 3)

    if composite_score >= 0.82:
        grade = "A"
    elif composite_score >= 0.68:
        grade = "B"
    elif composite_score >= 0.52:
        grade = "C"
    else:
        grade = "D"

    return grade, composite_score
