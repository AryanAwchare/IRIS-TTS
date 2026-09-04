"""
Evaluation package for IRIS / VoiceLib.
Provides objective speaker similarity (ECAPA-TDNN) and content accuracy (Whisper WER).
"""
from app.evaluation.speaker_similarity import speaker_similarity
from app.evaluation.content_accuracy import word_error_rate
from app.evaluation.prosody_metric import compute_generated_prosody
from app.evaluation.grade import compute_composite_grade
from app.evaluation.eval_pipeline import run_async_evaluation

__all__ = [
    "speaker_similarity",
    "word_error_rate",
    "compute_generated_prosody",
    "compute_composite_grade",
    "run_async_evaluation",
]
