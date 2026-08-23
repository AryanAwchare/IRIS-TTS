"""
Voice Similarity Analysis — Advanced Acoustic Signal Processing & Feature Extraction.

Computes perceptual and acoustic similarity between a reference voice sample and a
generated TTS output using:
  1. MFCC Spectral Cosine Similarity (Timbre & vocal tract envelope match)
  2. Mel-Cepstral Distance (MCD in dB) via Dynamic Time Warping (DTW)
  3. Fundamental Pitch (F0) Contour Tracking & Dynamic Pearson Correlation
  4. LPC-based Vocal Formant Estimation (F1, F2, F3 resonance mapping)
  5. Spectral Centroid Correlation (Vocal brightness & register match)
  6. Zero-Crossing Rate (ZCR) Correlation (Consonant texture match)
  7. 40-Point Log-Mel Filterbank Spectrum Curves (High-resolution graphing)
  8. 30-Point Pitch (F0) Trajectory Curves (Intonation & prosody graphing)

Supports any format readable by soundfile: WAV, MP3, OGG, FLAC, AIFF, etc.
"""
from __future__ import annotations

import io
import logging
from typing import NamedTuple, Tuple, List

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)

TARGET_SR = 16_000


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

class SimilarityResult(NamedTuple):
    overall_score: float          # 0–100 weighted vocal match accuracy
    accuracy_grade: str           # e.g., "A+ Studio Grade", "A Near-Identical", "B+ High Match"
    mfcc_similarity: float        # 0–100 spectral timbre match
    mcd_db: float                 # Mel-Cepstral Distance in dB (lower = better)
    mcd_match: float              # 0–100 normalized MCD match score
    f0_correlation: float         # 0–100 pitch & intonation correlation
    centroid_match: float         # 0–100 spectral brightness match
    zcr_match: float              # 0–100 consonant texture match
    formants_match: float         # 0–100 vocal tract resonance match
    ref_spectrum: list[float]     # 40-point frequency curve (original)
    gen_spectrum: list[float]     # 40-point frequency curve (generated)
    ref_pitch_curve: list[float]  # 30-point normalized pitch contour (original)
    gen_pitch_curve: list[float]  # 30-point normalized pitch contour (generated)
    ref_formants: list[float]     # [F1, F2, F3] in Hz
    gen_formants: list[float]     # [F1, F2, F3] in Hz
    ref_duration_s: float
    gen_duration_s: float
    sample_rate: int


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading & preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _load_audio_bytes(data: bytes) -> tuple[np.ndarray, int]:
    """Decode any audio format → (float32 mono array, sample_rate)."""
    import soundfile as sf
    try:
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        mono = arr.mean(axis=1)
        return mono, int(sr)
    except Exception as e:
        raise RuntimeError(f"Could not decode audio (soundfile): {e}") from e


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample to target sample rate."""
    if orig_sr == target_sr:
        return audio
    gcd = np.gcd(orig_sr, target_sr)
    return signal.resample_poly(audio, target_sr // gcd, orig_sr // gcd).astype(np.float32)


def _preemphasis(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])


def _framing(audio: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    n_frames = 1 + (len(audio) - frame_len) // hop_len
    if n_frames <= 0:
        return np.zeros((1, frame_len), dtype=np.float32)
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, frame_len),
        strides=(audio.strides[0] * hop_len, audio.strides[0]),
    )
    return frames.copy()


# ─────────────────────────────────────────────────────────────────────────────
# MFCC & Mel-Cepstral Distance (MCD) with Dynamic Time Warping (DTW)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_mfcc(audio: np.ndarray, sr: int,
                  n_mfcc: int = 16, n_fft: int = 512,
                  hop_len: int = 256, n_mels: int = 40) -> np.ndarray:
    """Compute MFCC matrix (n_mfcc × T)."""
    audio_pre = _preemphasis(audio)
    _, _, Zxx = signal.stft(audio_pre, fs=sr, nperseg=n_fft,
                             noverlap=n_fft - hop_len, window="hann", padded=True)
    mag = np.abs(Zxx)

    n_freqs = n_fft // 2 + 1
    fmin, fmax = 80.0, min(sr / 2.0, 8000.0)
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_indices = np.floor((n_fft + 1) * hz_points / sr).astype(int).clip(0, n_freqs - 1)

    filterbank = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        lo, center, hi = bin_indices[m - 1], bin_indices[m], bin_indices[m + 1]
        for k in range(lo, center):
            if center != lo:
                filterbank[m - 1, k] = (k - lo) / (center - lo)
        for k in range(center, hi):
            if hi != center:
                filterbank[m - 1, k] = (hi - k) / (hi - center)

    mel_spec = filterbank @ mag
    log_mel = np.log(mel_spec + 1e-9)

    dct_matrix = np.cos(
        np.pi * np.arange(n_mfcc)[:, None] *
        (2 * np.arange(n_mels)[None, :] + 1) / (2 * n_mels)
    ) * np.sqrt(2.0 / n_mels)
    dct_matrix[0] *= 1.0 / np.sqrt(2)
    return (dct_matrix @ log_mel).astype(np.float32)


def _compute_mcd_dtw(mfcc_ref: np.ndarray, mfcc_gen: np.ndarray) -> Tuple[float, float]:
    """
    Computes Mel-Cepstral Distance (MCD in dB) using Dynamic Time Warping (DTW) alignment.
    Returns (mcd_db, normalized_mcd_match_percentage).
    """
    # Exclude 0th energy coefficient, transpose to (T, D)
    ref = mfcc_ref[1:].T
    gen = mfcc_gen[1:].T

    n_ref, n_gen = len(ref), len(gen)
    if n_ref == 0 or n_gen == 0:
        return 6.0, 75.0

    # Downsample frames if audio is very long to ensure instant DTW computation
    max_frames = 400
    if n_ref > max_frames:
        idx = np.linspace(0, n_ref - 1, max_frames).astype(int)
        ref = ref[idx]
        n_ref = max_frames
    if n_gen > max_frames:
        idx = np.linspace(0, n_gen - 1, max_frames).astype(int)
        gen = gen[idx]
        n_gen = max_frames

    # Pairwise Euclidean distance matrix (Euclidean distance between MFCC vectors)
    diff = ref[:, np.newaxis, :] - gen[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Constant factor for dB conversion: (10 * sqrt(2)) / ln(10) ≈ 6.14185
    factor = (10.0 * np.sqrt(2.0)) / np.log(10.0)
    dist_matrix_db = dist_matrix * factor

    # DTW dynamic programming
    cost = np.full((n_ref + 1, n_gen + 1), np.inf, dtype=np.float32)
    cost[0, 0] = 0.0

    for i in range(1, n_ref + 1):
        for j in range(1, n_gen + 1):
            d = dist_matrix_db[i - 1, j - 1]
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    # Normalized path cost
    path_len = n_ref + n_gen
    mcd_db = float(cost[n_ref, n_gen] / max(1, path_len))
    mcd_db = round(np.clip(mcd_db, 1.0, 15.0), 2)

    # Convert to 0–100 match score (2.5dB = 100%, 10.5dB = 0%)
    # Cloned voices typically range between 3.5dB (studio) and 7.5dB
    mcd_match = float(np.clip(100.0 - (mcd_db - 2.5) * 12.5, 10.0, 99.0))
    return mcd_db, round(mcd_match, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Pitch (F0) Autocorrelation & Contour Tracking
# ─────────────────────────────────────────────────────────────────────────────

def _extract_f0_contour(audio: np.ndarray, sr: int, n_points: int = 30) -> Tuple[np.ndarray, list[float]]:
    """
    Extracts fundamental frequency (F0) trajectory across time.
    Returns (raw_f0_array, normalized_30_point_curve).
    """
    frame_len = int(sr * 0.030)  # 30 ms window
    hop_len = int(sr * 0.015)    # 15 ms step
    frames = _framing(audio, frame_len, hop_len)

    min_f0, max_f0 = 75.0, 500.0
    min_lag = max(1, int(sr / max_f0))
    max_lag = min(frame_len - 1, int(sr / min_f0))

    f0_list = []
    for frame in frames:
        # Window & center
        w = frame * np.hanning(len(frame))
        w = w - np.mean(w)
        if np.std(w) < 1e-4:
            f0_list.append(0.0)
            continue

        # Autocorrelation
        r = np.correlate(w, w, mode='full')
        r = r[len(w) - 1:]
        r0 = r[0] if r[0] > 0 else 1.0

        if max_lag < len(r):
            search_region = r[min_lag:max_lag + 1] / r0
            best_lag_rel = np.argmax(search_region)
            peak_val = search_region[best_lag_rel]
            best_lag = min_lag + best_lag_rel

            # Voicing threshold
            if peak_val > 0.30 and best_lag > 0:
                f0_list.append(float(sr / best_lag))
            else:
                f0_list.append(0.0)
        else:
            f0_list.append(0.0)

    f0_arr = np.array(f0_list, dtype=np.float32)

    # Create 30-point normalized curve for graphing
    if len(f0_arr) == 0 or np.all(f0_arr == 0):
        return f0_arr, [0.0] * n_points

    # Resample / interpolate contour to n_points
    x_old = np.linspace(0, 1, len(f0_arr))
    x_new = np.linspace(0, 1, n_points)
    interp_f0 = np.interp(x_new, x_old, f0_arr)

    # Normalize to 0–100 scale (with min 80Hz max 400Hz reference)
    norm_curve = np.clip((interp_f0 - 75.0) / (400.0 - 75.0) * 100.0, 0.0, 100.0)
    return f0_arr, [round(float(v), 1) for v in norm_curve]


def _correlate_f0(f0_ref: np.ndarray, f0_gen: np.ndarray) -> float:
    """Computes dynamic pitch correlation between voiced speech frames."""
    min_len = min(len(f0_ref), len(f0_gen))
    if min_len < 4:
        return 80.0

    r = f0_ref[:min_len]
    g = f0_gen[:min_len]

    # Voiced frame mask (where at least one has pitch activity)
    mask = (r > 0) | (g > 0)
    if np.sum(mask) < 4:
        return 80.0

    r_v, g_v = r[mask], g[mask]
    if np.std(r_v) < 1e-3 or np.std(g_v) < 1e-3:
        # Monotone/stable pitch match
        mean_diff = abs(np.mean(r_v) - np.mean(g_v))
        return float(np.clip(100.0 - mean_diff * 0.25, 60.0, 95.0))

    corr = np.corrcoef(r_v, g_v)[0, 1]
    if np.isnan(corr):
        return 75.0

    # Scale -1..1 to 0..100
    return float(np.clip((corr + 1.0) / 2.0 * 100.0, 0.0, 100.0))


# ─────────────────────────────────────────────────────────────────────────────
# LPC Formant Frequency Estimation (Vocal Tract Resonance)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_formants(audio: np.ndarray, sr: int) -> List[float]:
    """
    Estimates the first 3 vocal formants (F1, F2, F3) in Hz using LPC polynomial roots.
    Includes energy-gating and pole-radius constraints to prevent numerical instability.
    """
    if len(audio) < sr * 0.15:
        return [500.0, 1500.0, 2500.0]

    # Energy gate check
    energy = float(np.mean(audio ** 2))
    if energy < 1e-6:
        return [500.0, 1500.0, 2500.0]

    # Pre-emphasis + middle high-energy segment
    audio_pre = _preemphasis(audio)
    mid_start = len(audio_pre) // 3
    mid_end = mid_start + int(sr * 0.5)
    sample = audio_pre[mid_start:mid_end] if mid_end <= len(audio_pre) else audio_pre

    # Window
    w = sample * np.hamming(len(sample))
    if np.std(w) < 1e-5:
        return [500.0, 1500.0, 2500.0]

    # Autocorrelation for Levinson-Durbin
    lpc_order = int(2 + sr / 1000)  # ~18 for 16kHz
    r = np.correlate(w, w, mode='full')
    r = r[len(w) - 1: len(w) - 1 + lpc_order + 1]

    if r[0] < 1e-6:
        return [500.0, 1500.0, 2500.0]

    # Safe Levinson-Durbin recursion
    a = np.zeros(lpc_order + 1)
    a[0] = 1.0
    e = float(r[0])
    for i in range(1, lpc_order + 1):
        if abs(e) < 1e-12:
            break
        gamma = -np.dot(a[:i], r[i:0:-1]) / e
        if abs(gamma) >= 1.0:
            break
        a[1:i + 1] += gamma * a[i - 1::-1]
        e = max(1e-12, e * (1.0 - gamma ** 2))

    # Roots of LPC polynomial
    try:
        roots = np.roots(a)
        roots = roots[np.iscomplex(roots)]
        roots = roots[np.imag(roots) > 0]
    except Exception:
        return [500.0, 1500.0, 2500.0]

    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sr / (2 * np.pi))
    bandwidths = -0.5 * (sr / (2 * np.pi)) * np.log(np.clip(np.abs(roots), 1e-9, 0.999))

    # Filter realistic formant ranges with pole radius constraints
    valid = []
    for f, bw, r_val in zip(freqs, bandwidths, np.abs(roots)):
        if 200.0 <= f <= 4000.0 and bw < 450.0 and r_val > 0.65:
            valid.append(float(f))

    valid.sort()
    while len(valid) < 3:
        valid.append(500.0 * (len(valid) + 1))
    return [round(valid[0], 1), round(valid[1], 1), round(valid[2], 1)]


def _formants_similarity(f_ref: List[float], f_gen: List[float]) -> float:
    """Computes percentage resonance match across first 3 formants."""
    diffs = [abs(r - g) / (r + 1e-5) for r, g in zip(f_ref[:3], f_gen[:3])]
    avg_rel_diff = np.mean(diffs)
    return float(np.clip(100.0 - avg_rel_diff * 100.0, 50.0, 99.0))


# ─────────────────────────────────────────────────────────────────────────────
# Spectral Centroid, Energy, ZCR & Log-Mel Spectrum Curve
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_centroid(audio: np.ndarray, sr: int,
                        n_fft: int = 512, hop_len: int = 256) -> np.ndarray:
    _, _, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft,
                             noverlap=n_fft - hop_len, window="hann")
    mag = np.abs(Zxx)
    freqs = np.linspace(0, sr / 2, mag.shape[0])
    return (freqs[:, None] * mag).sum(axis=0) / (mag.sum(axis=0) + 1e-9)


def _energy_envelope(audio: np.ndarray, frame_len: int = 512, hop_len: int = 256) -> np.ndarray:
    frames = _framing(audio, frame_len, hop_len)
    return np.sqrt((frames ** 2).mean(axis=1))


def _zcr(audio: np.ndarray, frame_len: int = 512, hop_len: int = 256) -> np.ndarray:
    frames = _framing(audio, frame_len, hop_len)
    if frames.shape[1] < 2:
        return np.zeros(frames.shape[0])
    return (np.diff(np.sign(frames), axis=1) != 0).sum(axis=1).astype(float) / frame_len


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    min_t = min(a.shape[1], b.shape[1])
    if min_t == 0:
        return 0.5
    a, b = a[:, :min_t], b[:, :min_t]
    dots = np.einsum("mt,mt->t", a, b)
    norms = (np.linalg.norm(a, axis=0) + 1e-9) * (np.linalg.norm(b, axis=0) + 1e-9)
    return float(np.clip((dots / norms).mean(), 0.0, 1.0))


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    min_t = min(len(a), len(b))
    a, b = a[:min_t], b[:min_t]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.5
    r = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(r):
        return 0.5
    return float(np.clip((r + 1.0) / 2.0, 0.0, 1.0))


def _log_mel_curve(audio: np.ndarray, sr: int, n_points: int = 40) -> list[float]:
    """Return a normalized 40-point mel-scale magnitude curve for high-res graphing."""
    n_fft = 1024
    _, _, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, window="hann")
    mag_mean = np.abs(Zxx).mean(axis=1)

    fmin, fmax = 80.0, min(sr / 2.0, 8000.0)
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_bins = np.linspace(mel_min, mel_max, n_points + 1)
    hz_bins = 700 * (10 ** (mel_bins / 2595) - 1)
    freq_axis = np.linspace(0, sr / 2, len(mag_mean))

    curve = []
    for i in range(n_points):
        mask = (freq_axis >= hz_bins[i]) & (freq_axis < hz_bins[i + 1])
        val = float(mag_mean[mask].mean()) if mask.any() else 0.0
        curve.append(max(val, 0.0))

    curve_log = [float(np.log1p(v)) for v in curve]
    max_v = max(curve_log) if max(curve_log) > 0 else 1.0
    return [round(v / max_v * 100.0, 2) for v in curve_log]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_voice_similarity(ref_audio_bytes: bytes, gen_audio_bytes: bytes) -> SimilarityResult:
    """
    Runs multi-dimensional acoustic feature extraction and comparative analysis
    between original voice sample (reference) and TTS generated audio.
    """
    # 1. Load & resample
    ref_raw, ref_sr = _load_audio_bytes(ref_audio_bytes)
    gen_raw, gen_sr = _load_audio_bytes(gen_audio_bytes)

    ref = _resample(ref_raw, ref_sr, TARGET_SR)
    gen = _resample(gen_raw, gen_sr, TARGET_SR)

    # Clip to max 30s to keep real-time performance
    max_samples = TARGET_SR * 30
    ref = ref[:max_samples]
    gen = gen[:max_samples]

    ref_dur = len(ref) / TARGET_SR
    gen_dur = len(gen) / TARGET_SR

    # 2. MFCC & Timbre match
    mfcc_ref = _compute_mfcc(ref, TARGET_SR, n_mfcc=16)
    mfcc_gen = _compute_mfcc(gen, TARGET_SR, n_mfcc=16)
    mfcc_sim = _cosine_sim(mfcc_ref, mfcc_gen) * 100.0

    # 3. Dynamic Time Warping (DTW) Mel-Cepstral Distance (MCD)
    mcd_db, mcd_match = _compute_mcd_dtw(mfcc_ref, mfcc_gen)

    # 4. Pitch (F0) Tracking & Dynamic Prosody Correlation
    f0_ref_arr, ref_pitch_curve = _extract_f0_contour(ref, TARGET_SR, n_points=30)
    f0_gen_arr, gen_pitch_curve = _extract_f0_contour(gen, TARGET_SR, n_points=30)
    f0_corr = _correlate_f0(f0_ref_arr, f0_gen_arr)

    # 5. Formant Resonance
    ref_formants = _estimate_formants(ref, TARGET_SR)
    gen_formants = _estimate_formants(gen, TARGET_SR)
    formants_match = _formants_similarity(ref_formants, gen_formants)

    # 6. Spectral Centroid & Consonant ZCR
    centroid_match = _safe_pearson(_spectral_centroid(ref, TARGET_SR),
                                   _spectral_centroid(gen, TARGET_SR)) * 100.0
    zcr_match = _safe_pearson(_zcr(ref), _zcr(gen)) * 100.0

    # 7. Unified Weighted Vocal Match Accuracy Formula
    # Weights: Timbre (35%), MCD Distance Match (25%), Pitch Correlation (20%), Centroid (10%), ZCR (10%)
    overall = (
        mfcc_sim       * 0.35 +
        mcd_match      * 0.25 +
        f0_corr        * 0.20 +
        centroid_match * 0.10 +
        zcr_match      * 0.10
    )
    overall_score = round(float(np.clip(overall, 10.0, 99.5)), 1)

    # Qualitative Grade
    if overall_score >= 90.0:
        accuracy_grade = "A+ Studio Clone"
    elif overall_score >= 82.0:
        accuracy_grade = "A Near-Identical"
    elif overall_score >= 70.0:
        accuracy_grade = "B+ High Match"
    elif overall_score >= 55.0:
        accuracy_grade = "B Moderate Match"
    else:
        accuracy_grade = "C Partial Match"

    return SimilarityResult(
        overall_score=overall_score,
        accuracy_grade=accuracy_grade,
        mfcc_similarity=round(mfcc_sim, 1),
        mcd_db=mcd_db,
        mcd_match=mcd_match,
        f0_correlation=round(f0_corr, 1),
        centroid_match=round(centroid_match, 1),
        zcr_match=round(zcr_match, 1),
        formants_match=round(formants_match, 1),
        ref_spectrum=_log_mel_curve(ref, TARGET_SR, n_points=40),
        gen_spectrum=_log_mel_curve(gen, TARGET_SR, n_points=40),
        ref_pitch_curve=ref_pitch_curve,
        gen_pitch_curve=gen_pitch_curve,
        ref_formants=ref_formants,
        gen_formants=gen_formants,
        ref_duration_s=round(ref_dur, 2),
        gen_duration_s=round(gen_dur, 2),
        sample_rate=TARGET_SR,
    )

