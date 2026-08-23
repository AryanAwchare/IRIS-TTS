"""
Voice Enhancement Post-Processing Pipeline.

Applies a chain of DSP effects to raw Pocket TTS output to reduce
robotic artifacts and improve naturalness:

  Stage 1 — Noise gate (remove inter-word silence buzz)
  Stage 2 — De-robotisation (spectral subtraction / cepstral smoothing)
  Stage 3 — Harmonic enhancement (add subtle natural overtones)
  Stage 4 — EQ sculpting (roll-off sub-bass, boost speech clarity band)
  Stage 5 — Dynamic range compression (natural loudness leveling)
  Stage 6 — Subtle reverb / air (add room presence)
  Stage 7 — Normalisation + peak limiting

Works entirely on CPU with numpy/scipy — no GPU required.
If 'pedalboard' is installed (Spotify), uses it for higher-quality FX.
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

TARGET_SR = 22_050   # XTTS-v2 / GPT-SoVITS neural voice engine rate


# ─────────────────────────────────────────────────────────────────────────────
# WAV I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# DSP building blocks
# ─────────────────────────────────────────────────────────────────────────────

def _butter_filter(audio: np.ndarray, sr: int,
                   cutoff, btype: str, order: int = 4) -> np.ndarray:
    """Apply a butterworth filter (lowpass / highpass / bandpass)."""
    nyq = sr / 2.0
    if isinstance(cutoff, (list, tuple)):
        Wn = [c / nyq for c in cutoff]
    else:
        Wn = cutoff / nyq
    sos = butter(order, Wn, btype=btype, output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def _noise_gate(audio: np.ndarray, sr: int,
                threshold_db: float = -48.0, attack_ms: float = 5.0,
                release_ms: float = 80.0) -> np.ndarray:
    """
    Hard noise gate — mutes samples below threshold.
    Smooth open/close with attack/release envelopes.
    """
    threshold = 10 ** (threshold_db / 20.0)
    frame = int(sr * 0.01)  # 10 ms frames
    out = audio.copy()
    gate_open = False
    atk_samples = max(1, int(sr * attack_ms / 1000))
    rel_samples = max(1, int(sr * release_ms / 1000))
    gain = 0.0
    for i in range(0, len(audio), frame):
        chunk = audio[i:i + frame]
        rms = np.sqrt((chunk ** 2).mean() + 1e-12)
        target = 1.0 if rms >= threshold else 0.0
        if target > gain:
            gain = min(1.0, gain + frame / atk_samples)
        else:
            gain = max(0.0, gain - frame / rel_samples)
        out[i:i + frame] = chunk * gain
    return out


def _spectral_subtraction(audio: np.ndarray, sr: int,
                           n_fft: int = 1024, hop: int = 256,
                           noise_floor_db: float = -60.0) -> np.ndarray:
    """
    Gentle stationary-noise reduction (de-robotisation).
    Uses a conservative subtraction factor (0.8x instead of 1.5x) to avoid
    over-subtraction — the #1 cause of metallic / robotic artifacts.
    Estimates noise from the quietest 5% of frames.
    """
    _, _, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft,
                             noverlap=n_fft - hop, window='hann')
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise from quietest frames (5% — more conservative)
    frame_energy = mag.sum(axis=0)
    n_noise = max(1, int(len(frame_energy) * 0.05))
    noise_frames = np.argsort(frame_energy)[:n_noise]
    # Conservative factor of 0.8 — avoids over-subtraction metallic artifacts
    noise_profile = mag[:, noise_frames].mean(axis=1, keepdims=True) * 0.8

    # Wiener-style soft subtraction with musical noise reduction floor
    snr = mag / (noise_profile + 1e-9)
    wiener_gain = np.maximum(snr / (snr + 1.0), 0.1)  # floor at 10% gain
    mag_clean = mag * wiener_gain

    Zxx_clean = mag_clean * np.exp(1j * phase)
    _, audio_clean = signal.istft(Zxx_clean, fs=sr, nperseg=n_fft,
                                   noverlap=n_fft - hop, window='hann')

    # Trim to original length
    audio_clean = audio_clean[:len(audio)].astype(np.float32)
    return audio_clean


def _peaking_eq(audio: np.ndarray, sr: int, center_freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """Digital biquad peaking bell filter for phase-coherent vocal EQ sculpt."""
    if abs(gain_db) < 0.1:
        return audio
    nyq = sr / 2.0
    if center_freq >= nyq:
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
    """Digital biquad high shelf filter for vocal air enhancement."""
    if abs(gain_db) < 0.1:
        return audio
    nyq = sr / 2.0
    if cutoff >= nyq:
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


def _harmonic_enhancer(audio: np.ndarray, sr: int, amount: float = 0.03) -> np.ndarray:
    """
    Add subtle 2nd-order even harmonics — tube warmth without harsh odd-order distortion.
    Polynomial saturation: y = x + amount * (x^2 - mean(x^2)) - 0.08 * amount * x^3
    """
    even_harmonic = audio ** 2
    even_harmonic = even_harmonic - np.mean(even_harmonic)
    out = audio + amount * even_harmonic - 0.08 * amount * (audio ** 3)
    return out.astype(np.float32)


def _parametric_eq(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Phase-coherent 4-band studio vocal EQ:
      - High-pass at 80 Hz   (remove sub-bass mic rumble)
      - Bell cut at 250 Hz   (remove boxiness / mud)
      - Presence boost 3.2 kHz (+2.2 dB bell, Q=1.2) for speech clarity
      - High-shelf at 7.5 kHz (+1.5 dB) for vocal air
    """
    # High-pass at 80 Hz
    audio = _butter_filter(audio, sr, 80.0, 'highpass', order=3)
    # De-box cut at 250 Hz (-1.8 dB)
    audio = _peaking_eq(audio, sr, center_freq=250.0, gain_db=-1.8, q=1.0)
    # Presence clarity boost at 3.2 kHz (+2.2 dB)
    audio = _peaking_eq(audio, sr, center_freq=3200.0, gain_db=2.2, q=1.2)
    # High-shelf air boost at 7.5 kHz (+1.5 dB)
    audio = _high_shelf(audio, sr, cutoff=7500.0, gain_db=1.5)
    return audio.astype(np.float32)


def _compressor(audio: np.ndarray, sr: int,
                threshold_db: float = -18.0, ratio: float = 3.5,
                attack_ms: float = 10.0, release_ms: float = 120.0) -> np.ndarray:
    """
    Feed-forward RMS compressor — evens out loudness dynamics.
    Makes voice sound more natural and less monotone.
    """
    threshold = 10 ** (threshold_db / 20.0)
    frame = int(sr * 0.005)  # 5 ms frames
    atk = frame / max(1, int(sr * attack_ms / 1000))
    rel = frame / max(1, int(sr * release_ms / 1000))
    gain_db = 0.0
    out = audio.copy()
    for i in range(0, len(audio), frame):
        chunk = audio[i:i + frame]
        rms = np.sqrt((chunk ** 2).mean() + 1e-12)
        if rms > threshold:
            target_db = 20 * np.log10(threshold / rms) * (1 - 1 / ratio)
        else:
            target_db = 0.0
        gain_db = gain_db + (target_db - gain_db) * (atk if target_db < gain_db else rel)
        gain = 10 ** (gain_db / 20.0)
        out[i:i + frame] = chunk * gain
    return out.astype(np.float32)


def _subtle_room_reverb(audio: np.ndarray, sr: int, wet: float = 0.03) -> np.ndarray:
    """
    Applies artificial room acoustic impulse response (reverb & air).
    Adds natural studio room depth so voice doesn't sound artificially dry/anechoic.
    """
    ir_len = int(sr * 0.10)  # 100ms decay room
    t_ir = np.linspace(0, 0.10, ir_len, endpoint=False)
    decay = np.exp(-38.0 * t_ir)
    noise = np.random.normal(0, 1.0, ir_len)
    ir = noise * decay
    # High-frequency acoustic absorption damping in room IR
    nyq = sr * 0.5
    if 3500.0 / nyq < 1.0:
        b_lp, a_lp = signal.butter(2, 3500.0 / nyq, btype='lowpass')
        ir = signal.filtfilt(b_lp, a_lp, ir)
    ir = ir / (np.max(np.abs(ir)) + 1e-9)

    reverb_tail = signal.fftconvolve(audio, ir, mode='full')[:len(audio)].astype(np.float32)
    out = (1.0 - wet) * audio + wet * reverb_tail
    return out.astype(np.float32)


def _normalize(audio: np.ndarray, target_db: float = -14.0) -> np.ndarray:
    """Loudness normalize to target dBFS, then apply a soft peak limiter."""
    rms = np.sqrt((audio ** 2).mean() + 1e-12)
    target_amp = 10 ** (target_db / 20.0)
    audio = audio * (target_amp / rms)
    ceiling = 10 ** (-0.5 / 20.0)
    audio = np.tanh(audio / ceiling) * ceiling
    return audio.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def enhance_voice_audio(wav_bytes: bytes, sr: int = 32000) -> bytes:
    """
    Apply transparent voice mastering to raw TTS WAV output.
    Preserves native neural sample rate (32kHz) and speaker formants without robotic coloration.
    """
    try:
        audio, file_sr = _wav_to_float(wav_bytes)
    except Exception as e:
        logger.warning(f"Voice enhancer: could not decode WAV input — skipping. ({e})")
        return wav_bytes

    if len(audio) < file_sr * 0.1:
        return wav_bytes

    try:
        # 1. Gentle DC-offset / rumble cut (<30 Hz)
        audio = _butter_filter(audio, file_sr, 30.0, 'highpass', order=2)

        # 2. Transparent peak normalization (-0.3 dBFS ceiling)
        max_val = np.max(np.abs(audio)) + 1e-9
        if max_val > 0.01:
            target_peak = 10 ** (-0.3 / 20.0)
            if max_val > target_peak:
                audio = audio * (target_peak / max_val)
            audio = np.clip(audio, -0.99, 0.99)

    except Exception as e:
        logger.warning(f"Voice enhancer: mastering skipped — returning raw audio. ({e})")
        audio, _ = _wav_to_float(wav_bytes)

    return _float_to_wav(audio, file_sr)

