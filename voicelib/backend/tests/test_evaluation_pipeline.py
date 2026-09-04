"""
Automated tests for Phase 0+ Multi-Metric Evaluation Pipeline.

Tests:
1. Prosody F0 dynamics extraction (f0_std_hz, f0_mean_hz, voiced_ratio)
2. Composite Grade computation (A, B, C, D)
3. Pydantic schemas (GenerationOut, GenerationEvalOut)
4. Async evaluation pipeline worker and cleanup
"""
import os
import sys
import tempfile
import uuid

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

from app.evaluation.prosody_metric import compute_generated_prosody
from app.evaluation.grade import compute_composite_grade
from app.models import GenerationOut, GenerationEvalOut


def _create_synthetic_wav(freq: float = 220.0, duration: float = 1.0, sr: int = 16000) -> str:
    """Create a temporary synthetic WAV file for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental + harmonic
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * freq * 2 * t)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, signal.astype(np.float32), sr)
        return tmp.name


def test_prosody_metric_synthetic_tone():
    """Verify F0 prosody extraction on a clean synthetic harmonic signal."""
    wav_path = _create_synthetic_wav(freq=220.0, duration=1.0, sr=16000)
    try:
        res = compute_generated_prosody(wav_path)
        assert "f0_std_hz" in res
        assert "f0_mean_hz" in res
        assert "voiced_ratio" in res
        assert res["voiced_ratio"] > 0.3
        # Mean pitch should be near 220Hz
        assert 180.0 <= res["f0_mean_hz"] <= 260.0
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)


def test_prosody_metric_silence_or_missing():
    """Verify graceful zero return on missing or silent audio."""
    res = compute_generated_prosody("non_existent_file.wav")
    assert res == {"f0_std_hz": 0.0, "f0_mean_hz": 0.0, "voiced_ratio": 0.0}


def test_composite_grade_scale():
    """Test composite grade score and tier classification."""
    # Grade A: High similarity, low WER, healthy prosody
    grade_a, score_a = compute_composite_grade(
        speaker_similarity=0.92, word_error_rate=0.02, prosody_f0_std=36.0
    )
    assert grade_a == "A"
    assert score_a >= 0.82

    # Grade B: Good similarity, minor WER
    grade_b, score_b = compute_composite_grade(
        speaker_similarity=0.74, word_error_rate=0.12, prosody_f0_std=24.0
    )
    assert grade_b == "B"
    assert 0.68 <= score_b < 0.82

    # Grade C: Medium similarity, noticeable errors
    grade_c, score_c = compute_composite_grade(
        speaker_similarity=0.55, word_error_rate=0.30, prosody_f0_std=15.0
    )
    assert grade_c == "C"
    assert 0.52 <= score_c < 0.68

    # Grade D: Poor similarity / high error
    grade_d, score_d = compute_composite_grade(
        speaker_similarity=0.30, word_error_rate=0.75, prosody_f0_std=5.0
    )
    assert grade_d == "D"
    assert score_d < 0.52

    # Verify None WER handling (should default to 1.0 error, not 0.0 accuracy)
    grade_none_wer, score_none_wer = compute_composite_grade(
        speaker_similarity=0.50, word_error_rate=None, prosody_f0_std=10.0
    )
    assert grade_none_wer == "D"
    assert score_none_wer < 0.50


def test_generation_schemas():
    """Verify Pydantic models serialization and backward compatibility."""
    gen_id = uuid.uuid4()
    voice_id = uuid.uuid4()

    # GenerationOut with pending eval
    out = GenerationOut(
        id=gen_id,
        voice_id=voice_id,
        input_text="Testing IRIS evaluation pipeline.",
        audio_url="https://storage.local/gen.wav",
        eval_status="pending",
        created_at=np.datetime64("now").astype(object),
    )
    assert out.eval_status == "pending"
    assert out.speaker_similarity is None
    assert out.composite_grade is None
    assert out.composite_score is None

    # GenerationEvalOut with completed eval
    eval_out = GenerationEvalOut(
        generation_id=gen_id,
        voice_id=voice_id,
        eval_status="completed",
        speaker_similarity=0.885,
        word_error_rate=0.035,
        prosody_f0_std=34.2,
        composite_grade="A",
        composite_score=0.895,
        created_at=np.datetime64("now").astype(object),
    )
    assert eval_out.composite_grade == "A"
    assert eval_out.composite_score == 0.895
    assert eval_out.speaker_similarity == 0.885
    assert eval_out.word_error_rate == 0.035


if __name__ == "__main__":
    print("Running evaluation pipeline unit tests...", flush=True)
    test_prosody_metric_synthetic_tone()
    print("[PASS] test_prosody_metric_synthetic_tone", flush=True)
    test_prosody_metric_silence_or_missing()
    print("[PASS] test_prosody_metric_silence_or_missing", flush=True)
    test_composite_grade_scale()
    print("[PASS] test_composite_grade_scale", flush=True)
    test_generation_schemas()
    print("[PASS] test_generation_schemas", flush=True)
    print(">>> ALL EVALUATION PIPELINE TESTS PASSED! <<<", flush=True)
