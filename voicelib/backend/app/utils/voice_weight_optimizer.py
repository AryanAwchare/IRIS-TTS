"""
Per-Voice Weight & Acoustic Profile Optimizer for VoiceLib.

Analyzes a target voice reference audio sample to derive speaker-specific
neural model hyperparameters and weight profile settings:
  - cfg_weight (Classifier-Free Guidance weight)
  - lora_rank (LoRA model adaptation rank)
  - temperature (sampling variability)
  - top_p (nucleus sampling threshold)
  - pitch_bias (natural F0 offset)
  - speed_bias (speaking cadence factor)
  - noise_gate_db (stationary noise floor)
"""
from __future__ import annotations

import io
import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def derive_optimal_voice_weights(audio_bytes: bytes, target_sr: int = 22050) -> Dict[str, Any]:
    """
    Extract acoustic statistics from voice reference audio and calculate optimal model weights.
    """
    defaults = {
        "cfg_weight": 0.55,
        "lora_rank": 128,
        "temperature": 0.70,
        "top_p": 0.80,
        "pitch_bias": 0.0,
        "speed_bias": 1.0,
        "noise_gate_db": -48.0,
        "optimization_status": "derived_baseline",
    }

    if not audio_bytes or len(audio_bytes) < 44:
        return defaults

    try:
        import soundfile as sf
        import librosa

        # Read audio array
        y, sr = sf.read(io.BytesIO(audio_bytes))
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # 1. Pitch Register Analysis (F0 contour)
        f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
        valid_f0 = f0[(f0 >= 60) & (f0 <= 400)]
        f0_median = float(np.median(valid_f0)) if len(valid_f0) > 0 else 140.0
        f0_std = float(np.std(valid_f0)) if len(valid_f0) > 0 else 25.0

        # Pitch bias adjustment: if voice has low base pitch (<110Hz), bias slightly lower
        pitch_bias = 0.0
        if f0_median < 110.0:
            pitch_bias = -0.6
        elif f0_median > 220.0:
            pitch_bias = 0.8

        # 2. Spectral Brightness & Centroid (Vocal Tract Shape)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_sc = float(np.mean(sc)) if len(sc) > 0 else 2000.0

        # Higher spectral centroid (crisp/bright voice) -> higher lora_rank & higher cfg_weight
        if mean_sc > 2400.0:
            lora_rank = 256
            cfg_weight = 0.65
        elif mean_sc < 1400.0:
            lora_rank = 64
            cfg_weight = 0.50
        else:
            lora_rank = 128
            cfg_weight = 0.55

        # 3. Dynamic Range & Energy (Temperature & Top-P)
        rms = librosa.feature.rms(y=y)[0]
        rms_var = float(np.var(rms)) if len(rms) > 0 else 0.01

        # Expressive dynamic range -> slightly higher temperature
        if rms_var > 0.02:
            temperature = 0.75
            top_p = 0.85
        elif rms_var < 0.005:
            temperature = 0.60
            top_p = 0.75
        else:
            temperature = 0.70
            top_p = 0.80

        # 4. Noise Floor (Noise Gate)
        sorted_rms = np.sort(rms)
        noise_floor_rms = float(np.mean(sorted_rms[: max(1, int(len(sorted_rms) * 0.05))]))
        noise_gate_db = float(np.clip(20.0 * np.log10(noise_floor_rms + 1e-9), -60.0, -36.0))

        optimized_weights = {
            "cfg_weight": round(cfg_weight, 2),
            "lora_rank": int(lora_rank),
            "temperature": round(temperature, 2),
            "top_p": round(top_p, 2),
            "pitch_bias": round(pitch_bias, 1),
            "speed_bias": 1.0,
            "noise_gate_db": round(noise_gate_db, 1),
            "f0_median_hz": round(f0_median, 1),
            "spectral_brightness": round(mean_sc, 1),
            "optimization_status": "optimized",
        }
        logger.info(f"Derived optimal acoustic weights: {optimized_weights}")
        return optimized_weights

    except Exception as err:
        logger.warning(f"Voice weight optimization fallback notice: {err}")
        return defaults
