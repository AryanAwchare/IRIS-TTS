"""
Reference Audio Quality Selector and VAD Pre-filter.

Applies Silero VAD to detect speech regions and select the highest-SNR
reference segment for zero-shot voice cloning prompt intake.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)


_VAD_MODEL = None
_VAD_UTILS = None
_VAD_LOCK = threading.Lock()


def _get_vad_model():
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is None:
        with _VAD_LOCK:
            if _VAD_MODEL is None:
                try:
                    import torch
                    _VAD_MODEL, _VAD_UTILS = torch.hub.load(
                        repo_or_dir="snakers4/silero-vad",
                        model="silero_vad",
                        onnx=True,
                        trust_repo=True,
                    )
                    logger.info("Silero VAD ONNX model loaded successfully.")
                except Exception as e:
                    logger.warning(f"Could not load Silero VAD from torch.hub ({e}). Using energy-based VAD fallback.")
                    _VAD_MODEL = "fallback"
    return _VAD_MODEL, _VAD_UTILS


def _energy_vad_trim(audio: np.ndarray, sr: int, top_db: float = 30.0) -> np.ndarray:
    """Fallback energy-based trimming if neural VAD is unavailable."""
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed if len(trimmed) > sr * 0.5 else audio
    except Exception:
        return audio


def select_best_segment(audio_path: str, target_duration: float = 8.0) -> str:
    """
    Selects the cleanest, highest-SNR speech window from an intake audio sample.

    Args:
        audio_path: Path to reference audio file (.wav, .mp3, etc.).
        target_duration: Desired segment length in seconds (default: 8.0s).

    Returns:
        Path to a temporary WAV file containing the selected clean speech segment.
    """
    if not os.path.exists(audio_path):
        return audio_path

    try:
        import soundfile as sf
    except ImportError:
        logger.warning("soundfile not installed, skipping segment selection.")
        return audio_path

    try:
        y, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        y = y.mean(axis=1)  # Mono
    except Exception as read_err:
        logger.warning(f"Audio read failed in select_best_segment ({read_err}). Returning original.")
        return audio_path

    total_duration = len(y) / sr
    target_sr = 16000

    # Resample to 16kHz for VAD processing if needed
    if sr != target_sr:
        try:
            import librosa
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        except Exception:
            y_16k = y.copy()
    else:
        y_16k = y.copy()

    # 1. Run Silero VAD
    vad_model, vad_utils = _get_vad_model()
    speech_audio = y

    if vad_model != "fallback" and vad_utils is not None:
        try:
            import torch
            get_speech_timestamps = vad_utils[0]
            wav_tensor = torch.from_numpy(y_16k)
            speech_timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=16000)

            if speech_timestamps:
                # Concatenate speech portions
                speech_chunks = []
                for ts in speech_timestamps:
                    start_orig = int((ts["start"] / target_sr) * sr)
                    end_orig = int((ts["end"] / target_sr) * sr)
                    speech_chunks.append(y[start_orig:end_orig])
                if speech_chunks:
                    speech_audio = np.concatenate(speech_chunks)
        except Exception as vad_err:
            logger.debug(f"Silero VAD timestamp extraction notice: {vad_err}")
            speech_audio = _energy_vad_trim(y, sr)
    else:
        speech_audio = _energy_vad_trim(y, sr)

    speech_duration = len(speech_audio) / sr

    # 2. If short (<12s), return trimmed speech as-is
    if speech_duration < 12.0 or speech_duration <= target_duration:
        out_f = tempfile.NamedTemporaryFile(suffix="_vad.wav", delete=False)
        out_path = out_f.name
        out_f.close()
        sf.write(out_path, speech_audio, sr, format="WAV", subtype="PCM_16")
        return out_path

    # 3. If long (>=12s), score candidate windows by simple SNR
    window_samples = int(target_duration * sr)
    step_samples = int(1.0 * sr)  # 1-second slide
    best_window = speech_audio[:window_samples]
    best_snr = -float("inf")

    for start in range(0, len(speech_audio) - window_samples + 1, step_samples):
        candidate = speech_audio[start : start + window_samples]
        
        # Signal power vs approximate noise floor power
        sig_power = np.mean(candidate ** 2)
        # Approximate noise floor from lowest 10% energy frames
        frame_size = int(0.025 * sr)
        hop = int(0.010 * sr)
        frames = [candidate[i:i+frame_size] for i in range(0, len(candidate) - frame_size, hop)]
        if frames:
            frame_powers = [np.mean(f**2) for f in frames]
            noise_power = np.percentile(frame_powers, 10) + 1e-9
            snr = 10.0 * np.log10((sig_power + 1e-9) / noise_power)
        else:
            snr = sig_power

        if snr > best_snr:
            best_snr = snr
            best_window = candidate

    out_f = tempfile.NamedTemporaryFile(suffix="_vad_snr.wav", delete=False)
    out_path = out_f.name
    out_f.close()
    sf.write(out_path, best_window, sr, format="WAV", subtype="PCM_16")
    return out_path

