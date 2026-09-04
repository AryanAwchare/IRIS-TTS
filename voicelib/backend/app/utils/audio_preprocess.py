"""
Audio preprocessing pipeline for voice cloning.
Applies:
  1. Universal audio decoding (WAV, MP3, M4A, FLAC, OGG, etc.)
  2. Resampling to target sample rate (default 32,000 Hz — Chatterbox native)
  3. High-pass filtering (cuts sub-80Hz rumble, preserves male chest resonance)
  4. Silence trimming (removes leading/trailing dead air)
  5. RMS Loudness Normalization to -18 dBFS target

FIX: target_sr changed from 22050 → 32000 (Chatterbox native rate).
     This eliminates the double-resampling artifact that occurred when the
     Colab server resampled again from 22kHz → 32kHz, causing phase distortion
     and timbral smearing in the reference audio.

FIX: HP cutoff raised from 45Hz → 80Hz to preserve male vocal chest resonance
     (male F0 fundamentals start at ~80-130Hz — cutting at 45Hz was safe but
     80Hz is the correct value for removing true sub-bass rumble only).
"""
from __future__ import annotations

import io
import logging
import os
from typing import Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def _load_audio_to_mono_float(audio_input: bytes | str) -> Tuple[np.ndarray, int]:
    """Decode audio input bytes or file path to mono float32 array and sample rate."""
    import soundfile as sf

    if isinstance(audio_input, bytes):
        try:
            arr, sr = sf.read(io.BytesIO(audio_input), dtype="float32", always_2d=True)
            mono = arr.mean(axis=1).astype(np.float32)
            return mono, int(sr)
        except Exception:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                tmp.write(audio_input)
                tmp_p = tmp.name
            try:
                import librosa
                mono, sr = librosa.load(tmp_p, sr=None, mono=True)
                return mono.astype(np.float32), int(sr)
            finally:
                if os.path.exists(tmp_p):
                    os.unlink(tmp_p)
    else:
        try:
            arr, sr = sf.read(audio_input, dtype="float32", always_2d=True)
            mono = arr.mean(axis=1).astype(np.float32)
            return mono, int(sr)
        except Exception:
            import librosa
            mono, sr = librosa.load(audio_input, sr=None, mono=True)
            return mono.astype(np.float32), int(sr)


def preprocess_voice_sample(
    audio_source: bytes | str,
    target_sr: int = 32000,
    trim_silence: bool = True,
    denoise: bool = False,
    normalize: bool = True,
) -> bytes:
    """
    Cleans, resamples, and normalizes a voice recording sample.
    Returns 16-bit PCM WAV bytes at 32kHz suitable for Chatterbox zero-shot cloning.

    Args:
        audio_source:  Raw audio bytes or file path (any format soundfile supports).
        target_sr:     Output sample rate. Default 32000 (Chatterbox native).
                       Changed from 22050 — eliminates double-resampling in Colab.
        trim_silence:  Remove leading/trailing silence.
        denoise:       Apply spectral noise reduction. Off by default — denoising
                       at upload strips vocal harmonics Chatterbox needs for identity.
        normalize:     RMS normalize to -18 dBFS.
    """
    import soundfile as sf

    try:
        y, sr = _load_audio_to_mono_float(audio_source)
    except Exception as exc:
        logger.warning(f"Soundfile direct read failed ({exc}), attempting librosa decode...")
        try:
            import librosa
            if isinstance(audio_source, bytes):
                y, sr = librosa.load(io.BytesIO(audio_source), sr=None, mono=True)
            else:
                y, sr = librosa.load(audio_source, sr=None, mono=True)
            y = y.astype(np.float32)
        except Exception as e2:
            logger.error(f"Failed to load audio for preprocessing: {e2}")
            if isinstance(audio_source, bytes):
                return audio_source
            with open(audio_source, "rb") as f:
                return f.read()

    if len(y) == 0:
        logger.warning("Empty audio array encountered during preprocessing.")
        out_buf = io.BytesIO()
        sf.write(out_buf, np.zeros(target_sr, dtype=np.float32), target_sr, format="WAV", subtype="PCM_16")
        return out_buf.getvalue()

    # 1. Resample to target_sr (single pass — 32kHz matches Chatterbox natively)
    if sr != target_sr:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        except Exception:
            from numpy import gcd
            g = int(gcd(sr, target_sr))
            y = signal.resample_poly(y, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr

    # 2. High-pass filter at 80Hz (order 2, Butterworth)
    # FIX: raised from 45Hz → 80Hz. Male chest F0 starts at ~80Hz.
    # 45Hz was cutting into bass/baritone fundamentals; 80Hz removes only true sub-bass.
    try:
        nyq = sr * 0.5
        cutoff = 80.0 / nyq
        if cutoff < 1.0:
            b, a = signal.butter(2, cutoff, btype="highpass")
            y = signal.filtfilt(b, a, y).astype(np.float32)
    except Exception as e:
        logger.debug(f"High-pass filtering note: {e}")

    # 3. Silence Trimming
    if trim_silence and len(y) > sr * 0.5:
        try:
            import librosa
            yt, _ = librosa.effects.trim(y, top_db=35)
            if len(yt) > sr * 0.5:
                y = yt
        except Exception:
            frame_len = int(sr * 0.02)
            if len(y) > frame_len * 4:
                energy = np.array([
                    np.mean(y[i : i + frame_len] ** 2)
                    for i in range(0, len(y) - frame_len, frame_len)
                ])
                thresh = np.max(energy) * 0.005 if len(energy) > 0 else 0
                non_silent = np.where(energy > thresh)[0]
                if len(non_silent) > 0:
                    start = non_silent[0] * frame_len
                    end = min(len(y), (non_silent[-1] + 1) * frame_len)
                    if end - start > sr * 0.5:
                        y = y[start:end]

    # 4. RMS Loudness Normalization to -18 dBFS
    if normalize:
        rms = np.sqrt(np.mean(y ** 2) + 1e-9)
        target_rms = 0.125  # ~-18 dBFS
        if rms > 1e-4:
            y = y * (target_rms / rms)
        y = np.clip(y, -0.98, 0.98)

    # 5. Export to 16-bit PCM WAV bytes
    out_buf = io.BytesIO()
    sf.write(out_buf, y, sr, format="WAV", subtype="PCM_16")
    return out_buf.getvalue()
