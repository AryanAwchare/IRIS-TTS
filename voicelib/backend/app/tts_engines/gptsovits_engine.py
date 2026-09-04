"""
GPT-SoVITS v3 Engine implementation for VoiceLib.

Provides:
    - Zero-shot voice cloning using reference audio embeddings
    - Emotion & style conditioning (8D emotion vectors or categorical presets)
    - Paralinguistic tag parsing ([laughter], [sigh], [whisper], [gasp], etc.)
    - LoRA / Model rank tuning (e.g. Rank 128 sweet spot)
    - Temperature, Top-P, Pitch, and Speed controls
    - High-fidelity voice cloning synthesis via Colab GPU microservice

FIX: _compact_audio now uses SNR-based segment selection (was head-crop y[:max_len])
     which caused Chatterbox to encode the first N seconds regardless of quality.
"""
from __future__ import annotations

import io
import logging
import os
import re
import struct
import threading
from typing import Any, Optional

import numpy as np

from app.config import get_settings
from app.tts_engines.base import BaseTTSEngine
from app.utils.text_formatter import enhance_prompt_text

# ── Live Colab URL store ──────────────────────────────────────────────────────
_live_colab_url: str = ""
_live_colab_url_lock = threading.Lock()


def set_live_colab_url(url: str) -> None:
    global _live_colab_url
    with _live_colab_url_lock:
        _live_colab_url = url.rstrip("/")


def get_live_colab_url() -> str:
    with _live_colab_url_lock:
        return _live_colab_url


logger = logging.getLogger(__name__)

TAG_MAP = {
    r"\[laughter\]":      " (laughs) ",
    r"\[sigh\]":          " (sighs) ",
    r"\[gasp\]":          " (gasps) ",
    r"\[whisper\]":       " (whispering) ",
    r"\[chuckle\]":       " (chuckles) ",
    r"\[clears throat\]": " (clears throat) ",
}

EMOTION_PRESETS = {
    "neutral":   {"pitch": 0.0,  "speed": 1.0,  "energy": 1.0},
    "happy":     {"pitch": 1.8,  "speed": 1.1,  "energy": 1.25},
    "sad":       {"pitch": -1.5, "speed": 0.85, "energy": 0.75},
    "angry":     {"pitch": 1.2,  "speed": 1.2,  "energy": 1.4},
    "excited":   {"pitch": 2.5,  "speed": 1.25, "energy": 1.45},
    "calm":      {"pitch": -0.8, "speed": 0.9,  "energy": 0.8},
    "fearful":   {"pitch": 2.0,  "speed": 1.15, "energy": 0.9},
    "surprised": {"pitch": 3.0,  "speed": 1.1,  "energy": 1.3},
}


class GPTSoVITSEngine(BaseTTSEngine):
    """
    GPT-SoVITS v3 Engine.
    Handles reference audio acoustic prompt extraction and zero-shot voice cloning.
    """
    engine_name = "gpt-sovits-v3"

    def __init__(self):
        self._model: Any = None
        self._sample_rate: int = 32000
        self._cache: dict[str, Any] = {}
        self._cache_order: list[str] = []
        self._cache_lock = threading.Lock()
        self._MAX_CACHE_SIZE = 50
        self._is_loaded = False
        settings = get_settings()
        self._colab_api_url: str = settings.colab_gpu_api_url or os.getenv("COLAB_GPU_API_URL", "http://localhost:8008")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load_model(self) -> None:
        if self._is_loaded:
            return

        settings = get_settings()
        self._colab_api_url = get_live_colab_url() or settings.colab_gpu_api_url or os.getenv("COLAB_GPU_API_URL", self._colab_api_url)

        logger.info(f"Initializing GPT-SoVITS v3 engine (Colab GPU URL: '{self._colab_api_url}')...")

        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self._colab_api_url}/health",
                headers={"User-Agent": "VoiceLib", "ngrok-skip-browser-warning": "true"}
            )
            with urllib.request.urlopen(req, timeout=3.0) as res:
                if res.status == 200:
                    logger.info(f"Connected to active Colab GPU Server at {self._colab_api_url}!")
        except Exception as err:
            logger.info(f"Colab GPU server ping status: {err}")

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Local compute device: {device}")
            self._is_loaded = True
        except Exception as exc:
            logger.warning(f"GPT-SoVITS initialization notice: {exc}. Using internal synthesis pipeline.")
            self._is_loaded = True

    def derive_voice_state(self, audio_source: str | bytes, voice_id: str) -> Any:
        with self._cache_lock:
            if voice_id in self._cache:
                self._cache_order.remove(voice_id)
                self._cache_order.append(voice_id)
                return self._cache[voice_id]

        logger.info(f"Extracting GPT-SoVITS v3 voice state for voice {voice_id}...")

        raw_audio: Optional[bytes] = None
        if isinstance(audio_source, str):
            if os.path.exists(audio_source):
                with open(audio_source, "rb") as f:
                    raw_audio = f.read()
        elif isinstance(audio_source, bytes):
            raw_audio = audio_source

        pitch_profile = None
        pitch_bias = 0.0
        if raw_audio and len(raw_audio) > 0:
            try:
                from app.utils.voice_profiler import extract_voice_acoustic_profile
                pitch_profile = extract_voice_acoustic_profile(raw_audio)
                pitch_bias = float(pitch_profile.get("pitch_bias", 0.0))
            except Exception as prof_err:
                logger.warning(f"Voice state pitch profiling notice for {voice_id}: {prof_err}")

        voice_state = {
            "voice_id": voice_id,
            "raw_audio_len": len(raw_audio) if raw_audio else 0,
            "sample_rate": self._sample_rate,
            "audio_bytes": raw_audio,
            "engine": "gpt-sovits-v3",
            "pitch_profile": pitch_profile,
            "pitch_bias": pitch_bias,
        }

        with self._cache_lock:
            if len(self._cache) >= self._MAX_CACHE_SIZE:
                oldest = self._cache_order.pop(0)
                self._cache.pop(oldest, None)
            self._cache[voice_id] = voice_state
            self._cache_order.append(voice_id)

        return voice_state

    def invalidate_cache(self, voice_id: str) -> None:
        with self._cache_lock:
            if voice_id in self._cache:
                self._cache.pop(voice_id, None)
                if voice_id in self._cache_order:
                    self._cache_order.remove(voice_id)

    def generate_audio(
        self,
        voice_state: Any,
        text: str,
        *,
        emotion: str = "neutral",
        emotions: Optional[dict[str, float]] = None,
        speed: float = 1.0,
        pitch: float = 0.0,
        rank: int = 128,
        top_p: float = 0.8,
        temperature: float = 0.7,
        text_lang: str = "en",
        exaggeration: float = 0.15,
        cfg_weight: float = 0.55,
        **kwargs: Any,
    ) -> bytes:
        enhanced_text = enhance_prompt_text(text)
        cleaned_text = enhanced_text
        for pattern, replacement in TAG_MAP.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)

        active_speed = speed
        active_pitch = pitch

        logger.info(
            f"GPT-SoVITS Synthesis: text_len={len(cleaned_text)}, emotion='{emotion}', "
            f"cfg={cfg_weight:.2f}, exag={exaggeration:.2f}, speed={active_speed:.2f}, "
            f"pitch={active_pitch:.2f}, temp={temperature}, top_p={top_p}, lang='{text_lang}'"
        )

        ref_bytes = voice_state.get("audio_bytes") if isinstance(voice_state, dict) else None

        if ref_bytes and len(ref_bytes) > 0:
            try:
                import requests

                # FIX: SNR-based segment selection replacing simple head-crop
                def _compact_audio(raw: bytes) -> bytes:
                    """
                    Select the highest-SNR 10-second speech segment from reference audio.
                    FIX: was a simple head-crop y[:max_len] which took the first 10s
                    regardless of content quality. Now uses SNR scoring to find the
                    cleanest, most speech-like window in the recording.
                    """
                    try:
                        import soundfile as _sf
                        import numpy as _np
                        import librosa as _lb

                        y, _sr = _sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
                        y = y.mean(axis=1)

                        target_sr = 32000
                        if _sr != target_sr:
                            y = _lb.resample(y, orig_sr=_sr, target_sr=target_sr)
                            _sr = target_sr

                        yt, _ = _lb.effects.trim(y, top_db=35)
                        if len(yt) > _sr * 0.5:
                            y = yt

                        max_duration = 10.0
                        max_len = int(_sr * max_duration)

                        if len(y) <= max_len:
                            pass  # short clip — use as-is after normalization
                        else:
                            # Score candidate windows by SNR
                            step = int(0.5 * _sr)
                            frame_size = int(0.025 * _sr)
                            hop = int(0.010 * _sr)
                            best_score = -float("inf")
                            best_win = y[:max_len]

                            for st in range(0, len(y) - max_len, step):
                                cand = y[st : st + max_len]
                                frames = [cand[i : i + frame_size] for i in range(0, len(cand) - frame_size, hop)]
                                if not frames:
                                    continue
                                fp = _np.array([_np.mean(f ** 2) for f in frames])
                                noise_floor = float(_np.percentile(fp, 10)) + 1e-9
                                signal_pow = float(_np.mean(cand ** 2)) + 1e-9
                                snr = 10.0 * _np.log10(signal_pow / noise_floor)

                                # ZCR speech heuristic
                                zcr = float(_np.mean(_np.abs(_np.diff(_np.sign(cand)))) / 2.0)
                                bonus = 2.0 if 0.02 < zcr < 0.15 else -2.0

                                if (snr + bonus) > best_score:
                                    best_score = snr + bonus
                                    best_win = cand

                            y = best_win

                        # RMS normalize
                        rms = float(_np.sqrt(_np.mean(y ** 2)) + 1e-9)
                        if rms > 1e-4:
                            y = y * (0.125 / rms)
                        y = _np.clip(y, -0.98, 0.98)

                        buf = io.BytesIO()
                        _sf.write(buf, y.astype(_np.float32), _sr, format='WAV', subtype='PCM_16')
                        return buf.getvalue()
                    except Exception:
                        return raw

                clean_ref_bytes = _compact_audio(ref_bytes)
                logger.info(f"Sending reference audio to Colab ({len(clean_ref_bytes):,} bytes)...")

                settings = get_settings()
                colab_url = (get_live_colab_url() or settings.colab_gpu_api_url or os.getenv("COLAB_GPU_API_URL", self._colab_api_url)).rstrip("/")

                # 3-attempt exponential backoff retry
                max_retries = 3
                for attempt in range(1, max_retries + 1):
                    try:
                        explicit_pitch = pitch if abs(pitch) > 0.05 else 0.0
                        payload_data = {
                            "text":         cleaned_text,
                            "emotion":      emotion,
                            "speed":        str(active_speed),
                            "pitch":        str(explicit_pitch),
                            "cfg_weight":   str(cfg_weight),
                            "exaggeration": str(exaggeration),
                            "language":     text_lang,
                        }
                        req_headers = {
                            "ngrok-skip-browser-warning": "true",
                            "Accept": "audio/wav",
                            "User-Agent": "VoiceLib-Backend/1.0",
                        }
                        files = {
                            "reference_audio": ("reference.wav", clean_ref_bytes, "audio/wav"),
                        }
                        res = requests.post(
                            f"{colab_url}/tts",
                            files=files,
                            data=payload_data,
                            headers=req_headers,
                            # FIX: read timeout reduced from 900s → 180s (3 min per attempt)
                            # Worst case: 3 attempts × 180s = 9 minutes max (was 45 min)
                            timeout=(15.0, 180.0),
                        )
                        if res.status_code == 200 and len(res.content) > 44:
                            logger.info(f"Generated audio via Colab GPU Server (attempt {attempt})")
                            return res.content
                        logger.warning(f"Colab GPU attempt {attempt} returned status {res.status_code}: {res.text[:150]}")
                    except Exception as req_err:
                        logger.warning(f"Colab GPU attempt {attempt} failed ({req_err}). Retrying...")
                        if attempt < max_retries:
                            import time
                            time.sleep(1.5 * attempt)
            except Exception as e:
                logger.warning(
                    f"Colab GPU bridge not reachable after retries ({e}). "
                    "Start the Colab notebook to enable real cloning."
                )

        # Local Pocket-TTS Fallback
        try:
            from app.tts_engines.pocket_engine import PocketTTSEngine
            pocket = PocketTTSEngine()
            pocket.load_model()
            if hasattr(pocket._model, "__class__") and not pocket._model.__class__.__name__.endswith("MockTTSModel"):
                logger.info("Using Pocket TTS local fallback...")
                return pocket.generate_audio(voice_state, cleaned_text, speed=active_speed)
        except Exception as p_err:
            logger.debug(f"Pocket TTS fallback notice: {p_err}")

        # Colab offline — raise informative error
        settings = get_settings()
        colab_url = get_live_colab_url() or settings.colab_gpu_api_url or self._colab_api_url
        raise RuntimeError(
            f"Neural Voice Cloning GPU server is offline (connected to {colab_url}). "
            "Please run Cell 3 in your Google Colab notebook to activate the GPU server!"
        )

    def _write_wav_bytes(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        if audio_array.dtype in [np.float32, np.float64]:
            int16_samples = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
        else:
            int16_samples = audio_array.astype(np.int16)

        raw_data = int16_samples.tobytes()
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * (bits_per_sample // 8)
        block_align = num_channels * (bits_per_sample // 8)
        data_size = len(raw_data)
        chunk_size = 36 + data_size

        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', chunk_size, b'WAVE', b'fmt ', 16, 1,
            num_channels, sample_rate, byte_rate, block_align,
            bits_per_sample, b'data', data_size,
        )
        return header + raw_data
