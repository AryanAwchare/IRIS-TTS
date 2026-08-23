"""
Evaluation package for IRIS / VoiceLib.
Provides objective speaker similarity (ECAPA-TDNN) and content accuracy (Whisper WER).
"""
from app.evaluation.speaker_similarity import speaker_similarity
from app.evaluation.content_accuracy import word_error_rate

__all__ = ["speaker_similarity", "word_error_rate"]
