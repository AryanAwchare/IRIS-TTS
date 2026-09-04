"""
Prosody Metric Module for Generated Speech Evaluation.

Computes fundamental frequency (F0) dynamics on synthesized audio:
- f0_std_hz: standard deviation of voiced pitch frames (measure of expressive pitch variation)
- f0_mean_hz: mean pitch across voiced frames
- voiced_ratio: fraction of audio containing voiced speech
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_generated_prosody(gen_wav_path: str) -> Dict[str, Any]:
    """
    Extracts fundamental frequency (F0) pitch dynamics from generated audio.

    Args:
        gen_wav_path: Path to synthesized WAV file.

    Returns:
        dict containing:
            - f0_std_hz (float): pitch standard deviation in Hz (0.0 if unvoiced)
            - f0_mean_hz (float): mean pitch in Hz
            - voiced_ratio (float): voiced frames fraction [0.0, 1.0]
    """
    if not os.path.exists(gen_wav_path):
        logger.error(f"Generated audio file not found for prosody evaluation: {gen_wav_path}")
        return {"f0_std_hz": 0.0, "f0_mean_hz": 0.0, "voiced_ratio": 0.0}

    try:
        import soundfile as sf
        audio, sr = sf.read(gen_wav_path, dtype="float32", always_2d=True)
        y = audio.mean(axis=1)  # Convert to mono
    except Exception as read_err:
        logger.warning(f"Soundfile read failed in compute_generated_prosody ({read_err})")
        return {"f0_std_hz": 0.0, "f0_mean_hz": 0.0, "voiced_ratio": 0.0}

    if len(y) == 0:
        return {"f0_std_hz": 0.0, "f0_mean_hz": 0.0, "voiced_ratio": 0.0}

    # Resample to 16kHz for pitch estimation if needed
    target_sr = 16000
    if sr != target_sr:
        try:
            import librosa
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        except Exception:
            y_16k = y
    else:
        y_16k = y

    # Extract F0 with librosa.pyin (focused on human vocal speech range 75Hz - 500Hz)
    try:
        import librosa

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y_16k,
            fmin=75.0,
            fmax=500.0,
            sr=target_sr,
            frame_length=1024,
            hop_length=512,
        )

        if f0 is not None and voiced_flag is not None:
            valid_f0 = f0[voiced_flag & (~np.isnan(f0))]
            if len(valid_f0) > 0:
                f0_std = float(np.std(valid_f0))
                f0_mean = float(np.mean(valid_f0))
                voiced_ratio = float(len(valid_f0) / max(len(f0), 1))
                return {
                    "f0_std_hz": round(f0_std, 2),
                    "f0_mean_hz": round(f0_mean, 2),
                    "voiced_ratio": round(voiced_ratio, 3),
                }

    except Exception as pyin_err:
        logger.debug(f"pYIN prosody extraction fallback ({pyin_err})")

    # Fast Autocorrelation Fallback
    try:
        frame_size = int(target_sr * 0.03)  # 30ms frames
        hop_size = int(target_sr * 0.01)    # 10ms hop
        f0_estimates = []

        for i in range(0, len(y_16k) - frame_size, hop_size):
            frame = y_16k[i : i + frame_size]
            if np.max(np.abs(frame)) < 0.01:
                continue  # Silence
            corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
            # Search lag between 50Hz and 500Hz
            min_lag = int(target_sr / 500)
            max_lag = int(target_sr / 50)
            if max_lag < len(corr):
                peak_lag = min_lag + np.argmax(corr[min_lag:max_lag])
                if corr[peak_lag] > 0.3 * corr[0]:
                    pitch = target_sr / peak_lag
                    f0_estimates.append(pitch)

        if len(f0_estimates) > 0:
            arr = np.array(f0_estimates)
            return {
                "f0_std_hz": round(float(np.std(arr)), 2),
                "f0_mean_hz": round(float(np.mean(arr)), 2),
                "voiced_ratio": round(float(len(f0_estimates) / max(1, len(y_16k) / hop_size)), 3),
            }

    except Exception as fallback_err:
        logger.warning(f"Autocorrelation fallback failed in compute_generated_prosody: {fallback_err}")

    return {"f0_std_hz": 0.0, "f0_mean_hz": 0.0, "voiced_ratio": 0.0}
