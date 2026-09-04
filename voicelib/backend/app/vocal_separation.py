"""
vocal_separation.py — Demucs-powered Stem Isolation & Separation Artifact Manager.

Provides:
  - SeparationArtifact: persistent cache container for vocals.wav and instrumental.wav
  - Demucs v4 (Hybrid Transformer) integration
  - Remote Colab GPU offload via /separate_stems
  - Offline center-channel extraction fallback for testing without GPU
  - Caching by song SHA-256 hash: duplicate conversions never re-separate audio!
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf

from app.config import get_settings
from app.services.job_queue import ARTIFACTS_ROOT, SongCoverJobManager
from app.utils.audio_asset import AudioAsset, load_audio_asset

logger = logging.getLogger(__name__)


@dataclass
class SeparationArtifact:
    """Persistent container for isolated stems."""
    song_hash: str
    vocals_path: Path
    instrumental_path: Path
    sample_rate: int
    duration: float

    @property
    def vocals_asset(self) -> AudioAsset:
        return load_audio_asset(self.vocals_path)

    @property
    def instrumental_asset(self) -> AudioAsset:
        return load_audio_asset(self.instrumental_path)


def _compute_hash(data: bytes) -> str:
    return hashlib.sha256(data[:1024 * 1024]).hexdigest()[:16]


def separate_vocals_and_instrumental(
    source: Union[str, Path, bytes],
    song_hash: Optional[str] = None,
    shifts: int = 2,
) -> SeparationArtifact:
    """
    Isolates vocals and instrumental stems from a song mix.
    Checks artifact cache first to prevent redundant computation.
    """
    if isinstance(source, (bytes, bytearray)):
        raw_bytes = bytes(source)
        computed_hash = song_hash or _compute_hash(raw_bytes)
        source_path = None
    else:
        source_path = Path(source)
        with open(source_path, "rb") as f:
            raw_bytes = f.read()
        computed_hash = song_hash or _compute_hash(raw_bytes)

    song_dir = ARTIFACTS_ROOT / computed_hash
    song_dir.mkdir(parents=True, exist_ok=True)
    vocals_file = song_dir / "vocals.wav"
    instrumental_file = song_dir / "instrumental.wav"

    # ── 1. Check Artifact Cache ───────────────────────────────────────────────
    if vocals_file.exists() and instrumental_file.exists():
        logger.info(f"Using cached separation artifacts for song '{computed_hash}'")
        v_info = sf.info(str(vocals_file))
        return SeparationArtifact(
            song_hash=computed_hash,
            vocals_path=vocals_file,
            instrumental_path=instrumental_file,
            sample_rate=v_info.samplerate,
            duration=v_info.duration,
        )

    # ── 2. Attempt Colab GPU Separation (if live tunnel active) ───────────────
    from app.tts_engines.gptsovits_engine import get_live_colab_url
    settings = get_settings()
    colab_url = get_live_colab_url() or getattr(settings, "colab_gpu_api_url", "")

    if colab_url and colab_url.startswith("http"):
        try:
            logger.info(f"Offloading stem separation to Colab GPU at {colab_url}...")
            # Multi-part request to Colab /separate_stems
            boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
            body_parts = [
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"song.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode("utf-8"),
                raw_bytes,
                f"\r\n--{boundary}--\r\n".encode("utf-8"),
            ]
            full_body = b"".join(body_parts)
            req = urllib.request.Request(
                f"{colab_url.rstrip('/')}/separate_stems",
                data=full_body,
                headers={
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                    "User-Agent": "VoiceLib-Backend",
                    "ngrok-skip-browser-warning": "true",
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120.0) as res:
                if res.status == 200:
                    resp_data = json.loads(res.read().decode("utf-8"))
                    import base64
                    v_bytes = base64.b64decode(resp_data["vocals_base64"])
                    i_bytes = base64.b64decode(resp_data["instrumental_base64"])
                    with open(vocals_file, "wb") as vf:
                        vf.write(v_bytes)
                    with open(instrumental_file, "wb") as inf:
                        inf.write(i_bytes)
                    logger.info("Demucs GPU stem separation completed via Colab!")
                    v_info = sf.info(str(vocals_file))
                    return SeparationArtifact(
                        song_hash=computed_hash,
                        vocals_path=vocals_file,
                        instrumental_path=instrumental_file,
                        sample_rate=v_info.samplerate,
                        duration=v_info.duration,
                    )
        except Exception as colab_err:
            logger.warning(f"Colab GPU separation skipped ({colab_err}) — attempting local execution")

    # ── 3. Attempt Local Demucs (if installed) ────────────────────────────────
    try:
        import torch
        import demucs.separate
        logger.info("Running Demucs local separation...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            tmp_in.write(raw_bytes)
            tmp_in_path = tmp_in.name

        out_tmp_dir = tempfile.mkdtemp()
        try:
            demucs.separate.main([
                "-n", "htdemucs",
                "--two-stems", "vocals",
                "--shifts", str(shifts),
                "-o", out_tmp_dir,
                tmp_in_path,
            ])
            # Find output files
            model_out = Path(out_tmp_dir) / "htdemucs" / Path(tmp_in_path).stem
            shutil.copy2(model_out / "vocals.wav", vocals_file)
            shutil.copy2(model_out / "no_vocals.wav", instrumental_file)
            v_info = sf.info(str(vocals_file))
            return SeparationArtifact(
                song_hash=computed_hash,
                vocals_path=vocals_file,
                instrumental_path=instrumental_file,
                sample_rate=v_info.samplerate,
                duration=v_info.duration,
            )
        finally:
            shutil.rmtree(out_tmp_dir, ignore_errors=True)
            if os.path.exists(tmp_in_path):
                os.unlink(tmp_in_path)
    except Exception as demucs_local_err:
        logger.warning(f"Local Demucs not available ({demucs_local_err}). Using DSP center-channel separation fallback.")

    # ── 4. High-Fidelity DSP Center-Channel Phase Extraction Fallback ─────────
    # Allows local environments to develop, test, and process stems even without CUDA Demucs!
    asset = load_audio_asset(raw_bytes)
    if asset.channels == 1:
        # Mono mix: use vocal harmonic filtering (300Hz-3.4kHz pass vs reject)
        from scipy import signal
        b_voc, a_voc = signal.butter(4, [250, 4000], btype="bandpass", fs=asset.sample_rate)
        vocals_data = signal.filtfilt(b_voc, a_voc, asset.samples)
        inst_data = asset.samples - vocals_data * 0.7
    else:
        # Stereo mix: Mid/Side (M/S) Center Channel Separation
        left = asset.samples[0]
        right = asset.samples[1]
        mid = 0.5 * (left + right)   # Lead vocal is panned center
        side = 0.5 * (left - right)  # Instruments/panned stereo backing

        # High-pass center mid to avoid bass/kick bleed
        from scipy import signal
        b_hp, a_hp = signal.butter(4, 200, btype="highpass", fs=asset.sample_rate)
        vocals_mono = signal.filtfilt(b_hp, a_hp, mid)

        # Reconstruct stereo instrumental without the center vocal
        inst_left = left - 0.85 * vocals_mono
        inst_right = right - 0.85 * vocals_mono
        vocals_data = np.stack([vocals_mono, vocals_mono])
        inst_data = np.stack([inst_left, inst_right])

    sf.write(str(vocals_file), vocals_data.T, asset.sample_rate, subtype="PCM_16")
    sf.write(str(instrumental_file), inst_data.T, asset.sample_rate, subtype="PCM_16")

    return SeparationArtifact(
        song_hash=computed_hash,
        vocals_path=vocals_file,
        instrumental_path=instrumental_file,
        sample_rate=asset.sample_rate,
        duration=asset.duration,
    )
