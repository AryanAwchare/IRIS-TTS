"""
VoiceLib Timbre Morpher & Acoustic Profile Extractor.

Performs high-fidelity acoustic feature extraction and dynamic formant/spectral
morphing to adapt base TTS voice synthesis to match target reference speakers on CPU.
Avoids robotic/metallic artifacts by using smooth biquad filter banks and harmonic warmth.
"""
from __future__ import annotations

import io
import logging
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def extract_acoustic_profile(
    audio_source: str | bytes | np.ndarray,
    sr: int = 24000,
) -> Dict[str, Any]:
    """
    Extract a comprehensive acoustic fingerprint from reference audio:
      - F0 (Pitch) mean, median, standard deviation
      - Formants (F1, F2, F3 resonances)
      - Spectral Centroid (brightness)
      - Spectral Flatness (breathiness vs tonality)
      - Vocal Warmth (low-mid energy ratio)
    """
    profile: Dict[str, Any] = {
        "mean_f0": 150.0,
        "std_f0": 20.0,
        "f1": 600.0,
        "f2": 1700.0,
        "f3": 2500.0,
        "spectral_centroid": 1800.0,
        "spectral_tilt": 0.0,
        "warmth_gain_db": 0.0,
        "pitch_ratio": 1.0,
        "gender_hint": "neutral",
    }

    try:
        import librosa
        import soundfile as sf

        y: np.ndarray
        if isinstance(audio_source, bytes):
            y, file_sr = sf.read(io.BytesIO(audio_source))
        elif isinstance(audio_source, str):
            y, file_sr = librosa.load(audio_source, sr=sr)
        elif isinstance(audio_source, np.ndarray):
            y = audio_source
            file_sr = sr
        else:
            return profile

        # Ensure mono float32
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)

        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

        # Trim leading and trailing silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if len(y_trimmed) > sr * 0.5:
            y = y_trimmed

        # 1. Fundamental Frequency (F0) Extraction
        f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=500, sr=sr)
        valid_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])

        if len(valid_f0) > 10:
            mean_f0 = float(np.median(valid_f0))
            std_f0 = float(np.std(valid_f0))
            profile["mean_f0"] = mean_f0
            profile["std_f0"] = std_f0
            profile["gender_hint"] = "male" if mean_f0 < 145.0 else "female"
        else:
            profile["mean_f0"] = 150.0
            profile["gender_hint"] = "female"

        # 2. Spectral Centroid & Spectral Tilt (Brightness & Warmth)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        mean_cent = float(np.median(spec_cent)) if spec_cent.size > 0 else 1800.0
        profile["spectral_centroid"] = mean_cent

        # Spectral energy distribution: Low (<500Hz), Mid (500-2500Hz), High (>2500Hz)
        spec = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        low_band = np.sum(spec[freqs < 500, :]) + 1e-6
        mid_band = np.sum(spec[(freqs >= 500) & (freqs < 2500), :]) + 1e-6
        high_band = np.sum(spec[freqs >= 2500, :]) + 1e-6

        warmth_ratio = low_band / (mid_band + high_band)
        # Convert warmth to dB compensation target (-4.0dB to +4.0dB)
        warmth_db = float(np.clip((warmth_ratio - 0.25) * 6.0, -4.0, 4.0))
        profile["warmth_gain_db"] = warmth_db

        # 3. Formant Frequency Estimation via Linear Predictive Coding (LPC)
        try:
            # Pre-emphasis
            y_pre = librosa.effects.preemphasis(y)
            # LPC order rule of thumb: 2 + sr / 1000 -> 26 for 24kHz
            lpc_order = int(2 + sr / 1000)
            a = librosa.lpc(y_pre, order=min(lpc_order, 28))
            roots = np.roots(a)
            roots = [r for r in roots if np.imag(r) >= 0.01]

            formants = []
            for r in roots:
                freq = np.arctan2(np.imag(r), np.real(r)) * (sr / (2 * np.pi))
                if 200.0 < freq < (sr / 2.0 - 500.0):
                    formants.append(freq)
            formants.sort()

            if len(formants) >= 1:
                profile["f1"] = float(formants[0])
            if len(formants) >= 2:
                profile["f2"] = float(formants[1])
            if len(formants) >= 3:
                profile["f3"] = float(formants[2])
        except Exception as lpc_err:
            logger.debug(f"LPC formant estimation fallback: {lpc_err}")

        logger.info(
            f"Acoustic Profile extracted: F0={profile['mean_f0']:.1f}Hz, "
            f"Centroid={profile['spectral_centroid']:.0f}Hz, Warmth={profile['warmth_gain_db']:+.1f}dB, "
            f"F1={profile['f1']:.0f}Hz, F2={profile['f2']:.0f}Hz"
        )

    except Exception as exc:
        logger.warning(f"Failed to extract full acoustic profile ({exc}) — using defaults.")

    return profile


def morph_timbre(
    audio: np.ndarray,
    sr: int,
    target_profile: Dict[str, Any],
    base_voice_f0: float = 160.0,
    morph_strength: float = 0.85,
    warmth_override_db: float = 0.0,
    brightness_override_db: float = 0.0,
) -> np.ndarray:
    """
    Apply natural acoustic timbre and spectral morphing to synthesized audio.
    
    1. Formant matching & Vocal Tract Length scaling
    2. Dynamic parametric equalization matching target spectral envelope
    3. Warmth and harmonic presence enhancement
    4. Brightness / presence control
    5. Anti-clipping soft saturation
    """
    if len(audio) == 0:
        return audio

    try:
        from scipy import signal

        out = audio.copy().astype(np.float32)
        nyq = sr * 0.5

        # 1. Calculate Pitch / Formant Ratio
        target_f0 = target_profile.get("mean_f0", 150.0)
        pitch_ratio = target_f0 / max(base_voice_f0, 80.0)
        # Limit pitch ratio to natural bounds (0.75x to 1.35x) to prevent chipmunk/mechanical sound
        pitch_ratio = float(np.clip(pitch_ratio, 0.75, 1.35))
        pitch_shift_semitones = float(12.0 * np.log2(pitch_ratio)) * morph_strength

        # Gentle pitch shifting if needed
        if abs(pitch_shift_semitones) > 0.4:
            try:
                import librosa
                out = librosa.effects.pitch_shift(out, sr=sr, n_steps=pitch_shift_semitones)
            except Exception as ps_err:
                logger.debug(f"Pitch shift step notice: {ps_err}")

        # 2. Vocal Warmth / Low-End Presence (Peaking / Shelf Filter around 220Hz)
        # User override takes priority; otherwise use profile-extracted warmth
        warmth_db = warmth_override_db if abs(warmth_override_db) > 0.01 else (
            target_profile.get("warmth_gain_db", 0.0) * morph_strength
        )
        if abs(warmth_db) > 0.4:
            out = _apply_peaking_biquad(out, sr, f_center=220.0, gain_db=warmth_db, Q=1.0)

        # 3. Formant Resonance Matching (Mid-range F1/F2 adjustment)
        target_f2 = target_profile.get("f2", 1700.0)
        if 1000.0 < target_f2 < 3000.0 and target_f2 < nyq - 200:
            f2_gain_db = 1.5 * morph_strength
            out = _apply_peaking_biquad(out, sr, f_center=target_f2, gain_db=f2_gain_db, Q=1.8)

        # 4. Brightness / Presence Control (Peaking Filter around 4kHz)
        # User-controlled brightness override
        if abs(brightness_override_db) > 0.01:
            brightness_f = min(4000.0, nyq - 500)
            out = _apply_peaking_biquad(out, sr, f_center=brightness_f, gain_db=brightness_override_db, Q=1.2)
        else:
            # Fallback: auto brightness based on spectral centroid
            centroid = target_profile.get("spectral_centroid", 1800.0)
            if centroid > 2200.0:
                # Bright speaker -> boost presence air (+1.5 dB)
                air_f = min(6000.0, nyq - 500)
                b_high, a_high = signal.butter(1, air_f / nyq, btype='high')
                out += signal.lfilter(b_high, a_high, out) * 0.15 * morph_strength
            elif centroid < 1400.0:
                # Mellow/deep speaker -> gentle high-cut
                cut_f = min(7000.0, nyq - 500)
                b_low, a_low = signal.butter(2, cut_f / nyq, btype='low')
                out = signal.lfilter(b_low, a_low, out).astype(np.float32)

        # 5. Subtle Harmonic Warmth (Soft non-linear saturation)
        # Prevents robotic / metallic dry sound
        harmonics = np.tanh(out * 1.1) * 0.12
        out = (out * 0.88 + harmonics).astype(np.float32)

        # 6. Peak Normalization with headroom
        max_val = np.max(np.abs(out))
        if max_val > 1e-4:
            out = out / max_val * 0.92

        return out

    except Exception as exc:
        logger.warning(f"Timbre morphing encountered error ({exc}) — returning standard audio.")
        return audio


def _apply_peaking_biquad(
    audio: np.ndarray,
    sr: int,
    f_center: float,
    gain_db: float,
    Q: float = 1.0,
) -> np.ndarray:
    """Apply a 2nd-order peaking EQ biquad filter at f_center with gain_db."""
    from scipy import signal

    w0 = 2 * np.pi * f_center / sr
    alpha = np.sin(w0) / (2 * Q)
    gain_linear = 10.0 ** (gain_db / 20.0)
    A = np.sqrt(gain_linear)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return signal.lfilter(b, a, audio).astype(np.float32)
