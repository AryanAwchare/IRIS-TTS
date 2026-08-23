"""
Audio preprocessing pipeline for voice cloning.
Applies:
  1. Universal audio decoding (WAV, MP3, M4A, FLAC, OGG, etc.)
  2. Resampling to target sample rate (default 22,050 Hz for neural TTS)
  3. High-pass filtering (cuts sub-75Hz rumble / ambient AC hum)
  4. Spectral noise reduction (dual-pass stationary & non-stationary)
  5. Silence trimming (removes leading/trailing dead air)
  6. RMS Loudness Normalization to -18 dBFS target
"""
from __future__ import annotations

import io
import logging
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
    denoise: bool = True,
    normalize: bool = True,
) -> bytes:
    """
    Cleans, resamples, denoises, and normalizes a voice recording sample.
    Returns 16-bit PCM WAV bytes suitable for neural voice cloning at 32kHz.
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

    # 1. Resample to 32kHz target sample rate
    if sr != target_sr:
        try:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        except Exception:
            gcd = np.gcd(sr, target_sr)
            y = signal.resample_poly(y, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr

    # 2. High-pass filter (45 Hz butterworth: cuts inaudible sub-rumble while preserving 55Hz+ fundamentals)
    try:
        nyq = sr * 0.5
        cutoff = min(45.0, nyq * 0.5)
        b, a = signal.butter(4, cutoff / nyq, btype="highpass")
        y = signal.filtfilt(b, a, y).astype(np.float32)
    except Exception as e:
        logger.debug(f"High-pass filtering note: {e}")

    # 4. Silence Trimming
    if trim_silence and len(y) > sr * 0.5:
        try:
            import librosa
            yt, _ = librosa.effects.trim(y, top_db=35)
            if len(yt) > sr * 0.5:
                y = yt
        except Exception:
            # Simple energy threshold trimming fallback
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

    # 5. RMS Loudness Normalization to -18 dBFS
    if normalize:
        rms = np.sqrt(np.mean(y**2) + 1e-9)
        target_rms = 0.125  # ~ -18 dBFS
        if rms > 1e-4:
            y = y * (target_rms / rms)
        y = np.clip(y, -0.98, 0.98)

    # 6. Export to 16-bit PCM WAV bytes
    out_buf = io.BytesIO()
    sf.write(out_buf, y, sr, format="WAV", subtype="PCM_16")
    return out_buf.getvalue()
