"""
Deep Acoustic Voice Profiler — Extracts comprehensive acoustic DNA per voice.
Uses digital signal processing and speech science to analyze:
  - Fundamental frequency (F0) distribution & register classification via pYIN
  - Formant resonances (F1–F4) via Linear Predictive Coding (LPC)
  - Timbral spectral fingerprint (13 MFCC coefficients)
  - Harmonics-to-Noise Ratio (HNR)
  - Spectral tilt & spectral centroid / rolloff
  - Speaking rate (syllables/sec) & energy dynamics
  - Jitter (pitch perturbation) & Shimmer (amplitude perturbation)
  - Voiced/Unvoiced speech ratio & Zero-Crossing Rate
  - Generation parameter auto-tuning: cfg_weight, exaggeration, temp, top_p, speed_scale

FIX: Uses the actual sample rate from soundfile (not a hardcoded value).
     After audio_preprocess.py was changed to output 32kHz, the profiler
     must use the real SR from the file to compute correct pYIN, LPC, and
     spectral features.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _estimate_formants(y: np.ndarray, sr: int, n_formants: int = 4) -> List[float]:
    """
    Estimate vocal tract formant center frequencies (F1-F4 in Hz) using LPC.
    Always resamples to 16kHz for LPC analysis — optimal range for F1-F4.
    """
    try:
        import librosa
        from scipy import signal

        # Resample to 16kHz for formant analysis (captures F1-F4 in 200-3500Hz)
        if sr != 16000:
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr_proc = 16000
        else:
            y_16k = y
            sr_proc = sr

        y_pre = signal.lfilter([1, -0.97], [1], y_16k)
        lpc_order = int(2 + sr_proc / 1000)  # ~18 for 16kHz
        a = librosa.lpc(y_pre, order=lpc_order)

        roots = np.roots(a)
        roots = roots[np.imag(roots) >= 0]

        angz = np.arctan2(np.imag(roots), np.real(roots))
        freqs = angz * (sr_proc / (2 * np.pi))
        bandwidths = -0.5 * (sr_proc / (2 * np.pi)) * np.log(np.abs(roots) + 1e-9)

        valid_idx = np.where((freqs > 50) & (freqs < 5500) & (bandwidths < 700))[0]
        formants = np.sort(freqs[valid_idx])

        result: List[float] = []
        for f in formants:
            if len(result) == 0 or (f - result[-1]) > 250:
                result.append(float(round(f, 1)))
            if len(result) >= n_formants:
                break

        default_formants = [500.0, 1500.0, 2500.0, 3500.0]
        while len(result) < n_formants:
            result.append(default_formants[len(result)])

        return result[:n_formants]
    except Exception as exc:
        logger.debug(f"Formant extraction fallback: {exc}")
        return [520.0, 1480.0, 2650.0, 3800.0]


def _calculate_hnr(y: np.ndarray, sr: int) -> float:
    try:
        import librosa
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_pow = np.mean(y_harm ** 2) + 1e-12
        perc_pow = np.mean(y_perc ** 2) + 1e-12
        hnr = 10.0 * np.log10(harm_pow / perc_pow)
        return float(np.clip(hnr, 0.0, 35.0))
    except Exception:
        return 16.5


def _calculate_jitter_shimmer(valid_f0: np.ndarray, y: np.ndarray, sr: int) -> Tuple[float, float]:
    try:
        if len(valid_f0) > 4:
            periods = 1.0 / np.maximum(valid_f0, 50.0)
            diffs = np.abs(np.diff(periods))
            jitter_pct = float(np.mean(diffs) / np.mean(periods) * 100.0)
        else:
            jitter_pct = 0.85

        frame_len = int(sr * 0.02)
        if len(y) > frame_len * 4:
            amps = np.array([
                np.max(np.abs(y[i : i + frame_len]))
                for i in range(0, len(y) - frame_len, frame_len)
            ])
            amps = amps[amps > 0.01]
            if len(amps) > 4:
                amp_diffs = np.abs(np.diff(amps))
                shimmer_pct = float(np.mean(amp_diffs) / np.mean(amps) * 100.0)
            else:
                shimmer_pct = 3.2
        else:
            shimmer_pct = 3.2

        return float(np.clip(jitter_pct, 0.1, 5.0)), float(np.clip(shimmer_pct, 0.5, 12.0))
    except Exception:
        return 0.85, 3.2


def _estimate_speaking_rate(y: np.ndarray, sr: int) -> float:
    try:
        from scipy import signal
        env = np.abs(y)
        nyq = sr * 0.5
        b, a = signal.butter(2, min(20.0, nyq * 0.5) / nyq, btype='low')
        smooth_env = signal.filtfilt(b, a, env)
        min_dist = int(sr * 0.15)
        peaks, _ = signal.find_peaks(smooth_env, distance=min_dist, prominence=np.max(smooth_env) * 0.15)
        dur_sec = len(y) / sr
        if dur_sec > 1.0 and len(peaks) > 0:
            rate = len(peaks) / dur_sec
            return float(np.clip(rate, 2.0, 6.5))
        return 3.8
    except Exception:
        return 3.8


def _calculate_spectral_tilt(y: np.ndarray, sr: int) -> float:
    try:
        import librosa
        spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        mean_spec = np.mean(spec, axis=1) + 1e-9
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        idx = np.where((freqs >= 100) & (freqs <= min(7000, sr * 0.48)))[0]
        if len(idx) > 10:
            x_oct = np.log2(freqs[idx] / 100.0)
            y_db = 20.0 * np.log10(mean_spec[idx])
            slope, _ = np.polyfit(x_oct, y_db, 1)
            return float(round(slope, 2))
        return -6.0
    except Exception:
        return -6.0


def extract_voice_acoustic_profile(audio_input: bytes | str) -> dict:
    """
    Extracts deep 20+ dimensional acoustic DNA and auto-tunes TTS hyperparameters.

    FIX: Uses the actual sample rate read from the file (soundfile), not a hardcoded
         assumption. After audio_preprocess.py was changed to output 32kHz, this
         function must work at whatever SR the stored file actually has.
    """
    import soundfile as sf

    # 1. Universal audio decoding — uses ACTUAL SR from file header
    y: np.ndarray
    sr: int
    if isinstance(audio_input, bytes):
        try:
            arr, sr = sf.read(io.BytesIO(audio_input), dtype="float32", always_2d=True)
            y = arr.mean(axis=1).astype(np.float32)
        except Exception:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
                tmp.write(audio_input)
                tmp_p = tmp.name
            try:
                import librosa
                y, sr = librosa.load(tmp_p, sr=None, mono=True)
                y = y.astype(np.float32)
            finally:
                if os.path.exists(tmp_p):
                    os.unlink(tmp_p)
    else:
        try:
            arr, sr = sf.read(audio_input, dtype="float32", always_2d=True)
            y = arr.mean(axis=1).astype(np.float32)
        except Exception:
            import librosa
            y, sr = librosa.load(audio_input, sr=None, mono=True)
            y = y.astype(np.float32)

    # sr is now the REAL sample rate from the file (e.g. 32000 after upload preprocessing)

    # 2. Fundamental Frequency (F0) Analysis via pYIN — uses real sr
    mean_f0 = 160.0
    median_f0 = 160.0
    std_f0 = 22.0
    f0_min = 85.0
    f0_max = 320.0
    voiced_ratio = 0.65
    valid_f0 = np.array([])

    try:
        import librosa
        f0_vals, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),   # ~65 Hz
            fmax=librosa.note_to_hz('C6'),   # ~1046 Hz
            sr=sr,                            # FIX: use actual SR, not hardcoded
            frame_length=2048,
            hop_length=512,
        )
        valid_f0 = f0_vals[~np.isnan(f0_vals)]
        if len(valid_f0) > 10:
            mean_f0 = float(np.mean(valid_f0))
            median_f0 = float(np.median(valid_f0))
            std_f0 = float(np.std(valid_f0))
            f0_min = float(np.percentile(valid_f0, 5))
            f0_max = float(np.percentile(valid_f0, 95))
            voiced_ratio = float(len(valid_f0) / max(1, len(f0_vals)))
    except Exception as exc:
        logger.debug(f"pYIN F0 extraction notice: {exc}")

    # 3. Categorize Vocal Register
    if median_f0 < 130.0:
        pitch_register = "Bass / Baritone"
        base_cfg = 0.60
        temperature = 0.68
        top_p = 0.82
    elif median_f0 < 175.0:
        pitch_register = "Tenor / Mid Male"
        base_cfg = 0.57
        temperature = 0.70
        top_p = 0.84
    elif median_f0 < 240.0:
        pitch_register = "Alto / Mezzo-Soprano"
        base_cfg = 0.55
        temperature = 0.72
        top_p = 0.85
    else:
        pitch_register = "High Soprano"
        base_cfg = 0.53
        temperature = 0.74
        top_p = 0.88

    # 4. Formants F1-F4 (LPC internally resamples to 16kHz — correct behavior)
    formants = _estimate_formants(y, sr, n_formants=4)

    # 5. MFCC 13 Timbral Coefficients
    mfcc_mean: List[float] = []
    try:
        import librosa
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = [float(round(m, 3)) for m in np.mean(mfccs, axis=1)]
    except Exception:
        mfcc_mean = [0.0] * 13

    # 6. Acoustic Physics Metrics
    hnr_db = _calculate_hnr(y, sr)
    jitter_pct, shimmer_pct = _calculate_jitter_shimmer(valid_f0, y, sr)
    speaking_rate = _estimate_speaking_rate(y, sr)
    spectral_tilt = _calculate_spectral_tilt(y, sr)

    # 7. Spectral Centroid, Rolloff, ZCR
    mean_sc = 1850.0
    mean_ro = 3600.0
    zcr_mean = 0.05
    energy_std = 0.03
    try:
        import librosa
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        if sc.size > 0:
            mean_sc = float(np.mean(sc))
        ro = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        if ro.size > 0:
            mean_ro = float(np.mean(ro))
        zcr = librosa.feature.zero_crossing_rate(y)
        if zcr.size > 0:
            zcr_mean = float(np.mean(zcr))
        rms = librosa.feature.rms(y=y)
        if rms.size > 0:
            energy_std = float(np.std(rms))
    except Exception:
        pass

    # 8. Auto-tune generation hyperparameters from acoustic profile
    # FIX: cfg caps lowered to match recalibrated emotion_analyzer.py (max 0.65)
    cfg_weight = base_cfg
    if hnr_db > 20.0:
        cfg_weight = min(0.65, cfg_weight + 0.04)
    elif hnr_db < 12.0:
        cfg_weight = max(0.45, cfg_weight - 0.04)

    # FIX: exaggeration minimum raised to 0.10 (was 0.04/0.06/0.08)
    # Natural speech requires at least 0.10 exag even for flat/neutral speakers
    if std_f0 < 15.0:
        auto_exaggeration = 0.14   # Flat speaker — needs intonation lift
    elif std_f0 > 35.0:
        auto_exaggeration = 0.10   # Dynamic speaker — keep natural, don't over-exaggerate
    else:
        auto_exaggeration = 0.12   # Normal range

    if speaking_rate > 4.5:
        auto_speed = 1.05
    elif speaking_rate < 3.0:
        auto_speed = 0.95
    else:
        auto_speed = 1.00

    profile: Dict[str, Any] = {
        "profile_version": 2,
        "mean_f0_hz": round(mean_f0, 1),
        "median_f0_hz": round(median_f0, 1),
        "std_f0_hz": round(std_f0, 1),
        "f0_min_hz": round(f0_min, 1),
        "f0_max_hz": round(f0_max, 1),
        "pitch_register": pitch_register,
        "pitch_bias": 0.0,
        "voiced_ratio": round(voiced_ratio, 3),
        "formants_hz": formants,
        "mfcc_mean": mfcc_mean,
        "spectral_centroid_hz": round(mean_sc, 1),
        "spectral_rolloff_hz": round(mean_ro, 1),
        "spectral_tilt_db_oct": round(spectral_tilt, 2),
        "zcr_mean": round(zcr_mean, 4),
        "hnr_db": round(hnr_db, 2),
        "jitter_percent": round(jitter_pct, 2),
        "shimmer_percent": round(shimmer_pct, 2),
        "speaking_rate_syl_s": round(speaking_rate, 2),
        "energy_std": round(energy_std, 4),
        "cfg_weight": round(cfg_weight, 2),
        "exaggeration": round(auto_exaggeration, 2),
        "temperature": round(temperature, 2),
        "top_p": round(top_p, 2),
        "speed_scale": round(auto_speed, 2),
    }

    logger.info(
        f"Acoustic DNA Profile v2: F0 Median={median_f0:.1f}Hz ({pitch_register}), "
        f"SR={sr}Hz, Formants={formants}, HNR={hnr_db:.1f}dB, "
        f"Auto-Tuned: cfg={cfg_weight:.2f}, exag={auto_exaggeration:.2f}, speed={auto_speed:.2f}"
    )
    return profile
