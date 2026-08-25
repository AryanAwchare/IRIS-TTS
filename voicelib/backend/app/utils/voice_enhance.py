"""
Voice Enhancement & Studio Mastering Pipeline.

Transparent, natural, and warm mastering:
  - Gentle sub-rumble highpass (45 Hz, order 2)
  - Smooth noise gate for silent pauses (no breath cutoff)
  - Pure, transparent peak normalization and soft ceiling (-0.5 dBFS)
  - Zero artificial presence boost or high-shelf air (prevents digital sharpness)
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
    release_ms: float = 100.0,
) -> np.ndarray:
    """Smooth noise gate to mute dead space without clipping word decays."""
    threshold = 10 ** (threshold_db / 20.0)
    frame = max(1, int(sr * 0.010))  # 10ms window
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


def enhance_voice_audio(wav_bytes: bytes, sr: int = 32000) -> bytes:
    """
    Transparent voice enhancer:
    Preserves 100% of authentic speaker vocal warmth, soft tone, and dynamics.
    Eliminates digital EQ harshness and inter-word silence buzz.
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

        # 1. Gentle Sub-Rumble Cut (45 Hz, order 2) — Soft, preserves low-end chest warmth
        if (45.0 / nyq) < 1.0 and len(audio) > 15:
            sos = butter(2, 45.0 / nyq, btype='highpass', output='sos')
            audio = sosfilt(sos, audio).astype(np.float32)

        # 2. Smooth Noise Gate (mutes pause hiss without truncating words)
        audio = _smooth_noise_gate(audio, file_sr, threshold_db=-46.0, attack_ms=15.0, release_ms=100.0)

        # 3. Transparent Peak Limiter (-0.5 dBFS ceiling)
        max_val = float(np.max(np.abs(audio))) + 1e-9
        target_peak = 10 ** (-0.5 / 20.0)  # ~0.944
        if max_val > target_peak:
            audio = audio * (target_peak / max_val)
        audio = np.clip(audio, -0.98, 0.98)

    except Exception as e:
        logger.warning(f"Voice enhancer fallback: {e}")
        audio, _ = _wav_to_float(wav_bytes)

    return _float_to_wav(audio, file_sr)
