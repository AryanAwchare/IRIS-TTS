"""
Voice Enhancement & Studio Mastering Pipeline — Backend post-processing pass.

Since the Colab server already applies mastering (master_audio), this backend
pass is intentionally lightweight — it acts as a safety net rather than a
full second processing stage.

Changes vs previous version:
- HP cutoff raised from 45Hz → 80Hz (male voice safe)
- Noise gate is SKIPPED if audio is already at target level (avoids double-gating
  which caused pumping artifacts on already-processed Colab output)
- Peak limiter target changed from -0.5dBFS → -1.0dBFS (matches Colab master_audio)
"""
from __future__ import annotations

import io
import logging
import struct
import wave

import numpy as np
from scipy import signal
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)


def _wav_to_float(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes → float32 mono array + sample rate."""
    with wave.open(io.BytesIO(wav_bytes)) as wf:
        sr = wf.getframerate()
        n_ch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        s = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        s = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        s = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    if n_ch > 1:
        s = s.reshape(-1, n_ch).mean(axis=1)
    return s, sr


def _float_to_wav(audio: np.ndarray, sr: int) -> bytes:
    """Float32 mono array → WAV bytes."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    raw = pcm.tobytes()
    data_size = len(raw)
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE', b'fmt ', 16, 1,
        1, sr, sr * 2, 2, 16, b'data', data_size
    )
    return header + raw


def _smooth_noise_gate(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -46.0,
    attack_ms: float = 15.0,
    release_ms: float = 200.0,   # FIX: 200ms release (was 100ms) prevents pumping
) -> np.ndarray:
    """Smooth noise gate. Longer release prevents volume-pumping on already-gated audio."""
    threshold = 10 ** (threshold_db / 20.0)
    frame = max(1, int(sr * 0.010))
    out = audio.copy()
    atk_samples = max(1, int(sr * attack_ms / 1000))
    rel_samples = max(1, int(sr * release_ms / 1000))
    gain = 1.0

    for i in range(0, len(audio), frame):
        chunk = audio[i : i + frame]
        rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
        target = 1.0 if rms >= threshold else 0.05
        if target > gain:
            gain = min(1.0, gain + frame / atk_samples)
        else:
            gain = max(0.05, gain - frame / rel_samples)
        out[i : i + frame] = chunk * gain
    return out.astype(np.float32)


def enhance_voice_audio(wav_bytes: bytes) -> bytes:
    """
    Transparent backend voice enhancer.

    Since the Colab server already applies master_audio() (HP + noise gate +
    peak limiter), this function acts as a lightweight safety net:
    - Applies 80Hz high-pass to remove any residual sub-bass
    - Skips noise gate if audio is already at target level (avoids double-gating)
    - Applies peak limiter only if signal exceeds -1dBFS ceiling
    """
    try:
        audio, file_sr = _wav_to_float(wav_bytes)
    except Exception as e:
        logger.warning(f"Voice enhancer: could not decode WAV ({e})")
        return wav_bytes

    if len(audio) < file_sr * 0.1:
        return wav_bytes

    try:
        nyq = file_sr * 0.5

        # 1. 80Hz high-pass (FIX: was 45Hz — raised to preserve male chest resonance)
        if (80.0 / nyq) < 1.0 and len(audio) > 15:
            sos = butter(2, 80.0 / nyq, btype='highpass', output='sos')
            audio = sosfilt(sos, audio).astype(np.float32)

        # 2. Noise gate — ONLY apply if audio peak is above -1dBFS
        # (if Colab already mastered the audio, peak will be at ~-1dBFS and we skip)
        peak = float(np.max(np.abs(audio)))
        target_peak = 10 ** (-1.0 / 20.0)  # -1.0 dBFS

        if peak > target_peak * 1.05:   # 5% headroom before we intervene
            # Audio is louder than expected — apply noise gate + limit
            audio = _smooth_noise_gate(audio, file_sr, threshold_db=-46.0,
                                        attack_ms=15.0, release_ms=200.0)
            # Re-check peak after gating
            peak = float(np.max(np.abs(audio))) + 1e-9
            if peak > target_peak:
                audio = audio * (target_peak / peak)
        else:
            # Audio is already at correct level — peak limiter only as safety
            if peak > target_peak:
                audio = audio * (target_peak / (peak + 1e-9))

        audio = np.clip(audio, -0.95, 0.95)

    except Exception as e:
        logger.warning(f"Voice enhancer fallback: {e}")
        try:
            audio, _ = _wav_to_float(wav_bytes)
        except Exception:
            return wav_bytes

    return _float_to_wav(audio, file_sr)
