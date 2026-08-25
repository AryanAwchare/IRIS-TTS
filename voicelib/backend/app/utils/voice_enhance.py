"""
Voice Enhancement & Studio Mastering Pipeline.

Applies a phase-coherent, high-fidelity acoustic mastering chain to TTS audio:
  Stage 1 — Sub-Bass Clean (65 Hz highpass to eliminate mic rumble / DC offset)
  Stage 2 — Intelligent Smooth Noise Gate (kills background hiss/buzz during silence)
  Stage 3 — Anti-Harshness Dynamic Sculpt (cuts piercing 3.2kHz-4kHz digital glare)
  Stage 4 — De-Esser & High-Shelf Tamer (softens sharp sibilance 's', 'sh' above 7.5kHz)
  Stage 5 — Chest Resonance & Warmth (adds rich +1.8dB body at 250Hz)
  Stage 6 — Analog Tube Warmth (smooth tanh harmonic saturation rounding harsh spikes)
  Stage 7 — Ultrasonic Cut (13.8 kHz lowpass removing high-frequency vocoder hiss)
  Stage 8 — True-Peak Mastering Limiter (-0.5 dBFS ceiling)
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


def _butter_filter(audio: np.ndarray, sr: int, cutoff, btype: str, order: int = 3) -> np.ndarray:
    """Apply butterworth filter with safe Nyquist handling."""
    nyq = sr * 0.5
    if isinstance(cutoff, (list, tuple)):
        Wn = [min(0.99, max(0.01, c / nyq)) for c in cutoff]
    else:
        Wn = min(0.99, max(0.01, cutoff / nyq))
    if len(audio) <= order * 3:
        return audio
    sos = butter(order, Wn, btype=btype, output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def _peaking_eq(audio: np.ndarray, sr: int, center_freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """Digital biquad peaking bell filter for surgical frequency sculpting."""
    if abs(gain_db) < 0.05 or len(audio) < 15:
        return audio
    nyq = sr * 0.5
    if center_freq >= nyq * 0.98:
        return audio
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * center_freq / sr
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([a0, a1, a2], dtype=np.float64) / a0
    return signal.lfilter(b, a, audio).astype(np.float32)


def _high_shelf(audio: np.ndarray, sr: int, cutoff: float, gain_db: float) -> np.ndarray:
    """Digital biquad high shelf filter for softening or boosting high-end frequencies."""
    if abs(gain_db) < 0.05 or len(audio) < 15:
        return audio
    nyq = sr * 0.5
    if cutoff >= nyq * 0.98:
        return audio
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * cutoff / sr
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / 0.707 - 1.0) + 2.0)
    cos_w0 = np.cos(w0)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([a0, a1, a2], dtype=np.float64) / a0
    return signal.lfilter(b, a, audio).astype(np.float32)


def _smooth_noise_gate(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -44.0,
    attack_ms: float = 12.0,
    release_ms: float = 90.0,
) -> np.ndarray:
    """
    Intelligent smooth noise gate:
    Mutes background hiss in speech pauses without truncating word decays or consonant tails.
    """
    threshold = 10 ** (threshold_db / 20.0)
    frame = max(1, int(sr * 0.008))  # 8ms window
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


def _warmth_saturation(audio: np.ndarray, drive: float = 0.04) -> np.ndarray:
    """
    Adds warm analog tube saturation and rounds off sharp, brittle digital spikes.
    Applies gentle even-order harmonics (warmth) and smooth tanh compression.
    """
    x = audio.copy()
    # Add subtle even harmonic
    even = (x ** 2) - np.mean(x ** 2)
    warmed = x + drive * even
    # Soft tanh curve to eliminate harsh digital transients
    return np.tanh(warmed * 1.02).astype(np.float32)


def enhance_voice_audio(wav_bytes: bytes, sr: int = 32000) -> bytes:
    """
    Master anti-harshness & de-noising voice studio pipeline.
    Sculpts the vocal timbre to sound warm, natural, intimate, and free of metallic sharpness.
    """
    try:
        audio, file_sr = _wav_to_float(wav_bytes)
    except Exception as e:
        logger.warning(f"Voice enhancer: could not decode WAV ({e})")
        return wav_bytes

    if len(audio) < file_sr * 0.1:
        return wav_bytes

    try:
        # 1. Clean sub-bass rumble & DC offset (<65Hz)
        audio = _butter_filter(audio, file_sr, 65.0, 'highpass', order=3)

        # 2. Intelligent noise gate (kills background room hiss in pauses)
        audio = _smooth_noise_gate(audio, file_sr, threshold_db=-44.0, attack_ms=10.0, release_ms=90.0)

        # 3. Add rich chest resonance & vocal body (+1.8dB @ 240Hz, Q=0.9)
        audio = _peaking_eq(audio, file_sr, center_freq=240.0, gain_db=1.8, q=0.9)

        # 4. Clean boxy/nasal midrange (-1.5dB @ 680Hz, Q=1.4)
        audio = _peaking_eq(audio, file_sr, center_freq=680.0, gain_db=-1.5, q=1.4)

        # 5. ANTI-HARSHNESS: Cut piercing digital glare (-3.0dB @ 3.4kHz, Q=1.3)
        # This directly eliminates the brittle, sharp "AI synthesizer" edge.
        audio = _peaking_eq(audio, file_sr, center_freq=3400.0, gain_db=-3.0, q=1.3)

        # 6. DE-ESSER / SIBILANCE TAMER: Smooth high-shelf reduction (-2.5dB @ 7.2kHz)
        # Softens piercing 's', 'sh', 't' consonant bursts.
        audio = _high_shelf(audio, file_sr, cutoff=7200.0, gain_db=-2.5)

        # 7. Ultrasonic lowpass filter (13.5kHz)
        # Eliminates high-frequency vocoder hiss and white noise.
        nyq = file_sr * 0.5
        if 13500.0 < nyq * 0.95:
            audio = _butter_filter(audio, file_sr, 13500.0, 'lowpass', order=2)

        # 8. Analog tube saturation (rounds sharp peaks, adds harmonic warmth)
        audio = _warmth_saturation(audio, drive=0.035)

        # 9. True-Peak Limiter (-0.5 dBFS ceiling)
        max_val = float(np.max(np.abs(audio))) + 1e-9
        target_peak = 10 ** (-0.5 / 20.0)  # ~0.944
        if max_val > target_peak:
            audio = audio * (target_peak / max_val)
        audio = np.clip(audio, -0.98, 0.98)

    except Exception as e:
        logger.warning(f"Voice enhancer fallback: {e}")
        audio, _ = _wav_to_float(wav_bytes)

    return _float_to_wav(audio, file_sr)
