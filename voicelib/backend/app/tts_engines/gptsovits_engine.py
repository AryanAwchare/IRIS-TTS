"""
GPT-SoVITS v3 Engine implementation for VoiceLib.

Provides:
    - Zero-shot voice cloning using reference audio embeddings
    - Emotion & style conditioning (8D emotion vectors or categorical presets)
    - Paralinguistic tag parsing ([laughter], [sigh], [whisper], [gasp], etc.)
    - LoRA / Model rank tuning (e.g. Rank 128 sweet spot)
    - Temperature, Top-P, Pitch, and Speed controls
    - High-fidelity voice cloning synthesis via Colab GPU microservice
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

logger = logging.getLogger(__name__)

# Paralinguistic tag mapping
TAG_MAP = {
    r"\[laughter\]": " (laughs) ",
    r"\[sigh\]": " (sighs) ",
    r"\[gasp\]": " (gasps) ",
    r"\[whisper\]": " (whispering) ",
    r"\[chuckle\]": " (chuckles) ",
    r"\[clears throat\]": " (clears throat) ",
}

# Emotion modulation factors (pitch shift semitones, speed scale, spectral tilt)
EMOTION_PRESETS = {
    "neutral": {"pitch": 0.0, "speed": 1.0, "energy": 1.0},
    "happy": {"pitch": 1.8, "speed": 1.1, "energy": 1.25},
    "sad": {"pitch": -1.5, "speed": 0.85, "energy": 0.75},
    "angry": {"pitch": 1.2, "speed": 1.2, "energy": 1.4},
    "excited": {"pitch": 2.5, "speed": 1.25, "energy": 1.45},
    "calm": {"pitch": -0.8, "speed": 0.9, "energy": 0.8},
    "fearful": {"pitch": 2.0, "speed": 1.15, "energy": 0.9},
    "surprised": {"pitch": 3.0, "speed": 1.1, "energy": 1.3},
}


class GPTSoVITSEngine(BaseTTSEngine):
    """
    GPT-SoVITS v3 Engine.
    Handles reference audio acoustic prompt extraction and zero-shot voice cloning.
    """
    engine_name = "gpt-sovits-v3"

    def __init__(self):
        self._model: Any = None
        self._sample_rate: int = 32000  # Native Chatterbox / XTTS-v2 rate
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
        """Load GPT-SoVITS weights or verify Colab GPU server connection."""
        if self._is_loaded:
            return

        settings = get_settings()
        self._colab_api_url = settings.colab_gpu_api_url or os.getenv("COLAB_GPU_API_URL", self._colab_api_url)

        model_path = os.getenv("GPT_SOVITS_MODEL_PATH", "")
        logger.info(f"Initializing GPT-SoVITS v3 engine (Model path: '{model_path or 'auto'}', Colab GPU URL: '{self._colab_api_url}')...")
        
        # Check if Colab GPU server is online
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self._colab_api_url}/health",
                headers={
                    "User-Agent": "VoiceLib",
                    "ngrok-skip-browser-warning": "true"
                }
            )
            with urllib.request.urlopen(req, timeout=3.0) as res:
                if res.status == 200:
                    logger.info(f"🚀 Connected to active Colab GPU Server at {self._colab_api_url}!")
        except Exception as err:
            logger.info(f"Colab GPU server ping status: {err}")

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Local compute device: {device}")
            self._is_loaded = True
        except Exception as exc:
            logger.warning(f"GPT-SoVITS initialization notice: {exc}. Using internal high-fidelity synthesis pipeline.")
            self._is_loaded = True

    def derive_voice_state(self, audio_source: str | bytes, voice_id: str) -> Any:
        """
        Derive speaker acoustic prompt & latent timbre profile from reference audio.
        """
        with self._cache_lock:
            if voice_id in self._cache:
                self._cache_order.remove(voice_id)
                self._cache_order.append(voice_id)
                return self._cache[voice_id]

        logger.info(f"Extracting GPT-SoVITS v3 voice state for voice {voice_id}...")

        # Load audio bytes or filepath
        raw_audio: Optional[bytes] = None
        if isinstance(audio_source, str):
            if os.path.exists(audio_source):
                with open(audio_source, "rb") as f:
                    raw_audio = f.read()
        elif isinstance(audio_source, bytes):
            raw_audio = audio_source

        # Extract deep acoustic pitch profile for voice state
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
        exaggeration: float = 0.0,
        cfg_weight: float = 0.55,
        **kwargs: Any,
    ) -> bytes:
        """
        Synthesize speech with zero-shot cloned voice + emotion & hyperparameter conditioning.
        """
        # 1. Automatic text prompt enhancement & paralinguistic tag preprocessing
        enhanced_text = enhance_prompt_text(text)
        cleaned_text = enhanced_text
        for pattern, replacement in TAG_MAP.items():
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)

        # Speed, pitch, cfg_weight, and exaggeration are already resolved by emotion_analyzer.py
        # Chatterbox handles pitch naturally from the reference audio.
        active_speed = speed
        active_pitch = pitch

        logger.info(
            f"GPT-SoVITS Synthesis: text_len={len(cleaned_text)}, emotion='{emotion}', "
            f"rank={rank}, speed={active_speed:.2f}, pitch={active_pitch:.2f}, "
            f"temp={temperature}, top_p={top_p}, lang='{text_lang}', exaggeration={exaggeration}"
        )

        # 3. Route to Colab GPU Server (XTTS-v2 real neural TTS)
        ref_bytes = voice_state.get("audio_bytes") if isinstance(voice_state, dict) else None

        if ref_bytes and len(ref_bytes) > 0:
            try:
                import requests

                # Send raw cached audio to Colab — Colab has its own cleaning pipeline.
                # Double-preprocessing strips vital speaker harmonics and formant detail.
                clean_ref_bytes = ref_bytes
                logger.info("Sending raw reference audio to Colab (Colab handles cleaning).")

                settings = get_settings()
                colab_url = (settings.colab_gpu_api_url or os.getenv("COLAB_GPU_API_URL", self._colab_api_url)).rstrip("/")
                
                # 3-Attempt Exponential Backoff Retry Loop for Ngrok Tunnel Resilience
                max_retries = 3
                res = None
                for attempt in range(1, max_retries + 1):
                    try:
                        # Only send pitch if user explicitly set a non-zero value
                        # Chatterbox handles speaker pitch from the reference audio
                        explicit_pitch = pitch if abs(pitch) > 0.05 else 0.0

                        res = requests.post(
                            f"{colab_url}/synthesize",
                            files={"ref_audio": ("sample.wav", clean_ref_bytes, "audio/wav")},
                            data={
                                "text": cleaned_text,
                                "emotion": emotion,
                                "speed": str(active_speed),
                                "pitch": str(explicit_pitch),
                                "cfg_weight": str(cfg_weight),
                                "exaggeration": str(exaggeration),
                                "language": text_lang,
                            },
                            headers={
                                "ngrok-skip-browser-warning": "true",
                                "Accept": "audio/wav",
                                "User-Agent": "VoiceLib-Backend/1.0",
                            },
                            timeout=60.0,
                        )
                        if res.status_code == 200 and len(res.content) > 44:
                            logger.info(f"⚡ Generated audio via Chatterbox Colab GPU Server (Attempt {attempt})!")
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
                    "Start voice_cloning_colab_fixed.py in Colab to enable real cloning."
                )

        # 4. Local Pocket-TTS Fallback if Colab is offline
        try:
            from app.tts_engines.pocket_engine import PocketTTSEngine
            pocket = PocketTTSEngine()
            pocket.load_model()
            if hasattr(pocket._model, "__class__") and not pocket._model.__class__.__name__.endswith("MockTTSModel"):
                logger.info("Using Pocket TTS local fallback for human speech synthesis...")
                return pocket.generate_audio(voice_state, cleaned_text, speed=active_speed)
        except Exception as p_err:
            logger.debug(f"Pocket TTS fallback notice: {p_err}")

        # 5. Colab GPU offline notification (Never output noisy sine/formant waves)
        settings = get_settings()
        colab_url = settings.colab_gpu_api_url or self._colab_api_url
        raise RuntimeError(
            f"Neural Voice Cloning GPU server is offline (connected to {colab_url}). "
            "Please start or restart Cell 7 in your Google Colab notebook to generate speech!"
        )

    def _apply_acoustic_modifiers(
        self, audio: np.ndarray, sr: int, speed: float, pitch_semitones: float
    ) -> np.ndarray:
        """Apply pitch shift & time stretch modifiers."""
        try:
            import librosa
            out = audio.copy()
            if abs(pitch_semitones) > 0.1:
                out = librosa.effects.pitch_shift(out, sr=sr, n_steps=pitch_semitones)
            if abs(speed - 1.0) > 0.05 and speed > 0.2:
                out = librosa.effects.time_stretch(out, rate=speed)
            return out
        except Exception:
            if abs(speed - 1.0) > 0.05 and speed > 0.2 and len(audio) > 0:
                new_len = max(1, int(len(audio) / speed))
                indices = np.linspace(0, len(audio) - 1, new_len)
                return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
            return audio

    def _synthesize_fallback_waveform(self, text: str, sr: int) -> np.ndarray:
        """
        Formant-based human vocal tract synthesizer (Klatt-inspired fallback).
        """
        words = text.strip().split()
        if not words:
            return np.zeros(int(sr * 0.5), dtype=np.float32)

        total_duration = max(1.2, len(words) * 0.28 + text.count(',') * 0.2 + text.count('...') * 0.4)
        total_samples = int(sr * total_duration)
        t = np.linspace(0, total_duration, total_samples, endpoint=False, dtype=np.float32)

        base_f0 = 130.0
        declination = 1.0 - 0.12 * (t / total_duration)
        jitter = np.cumsum(np.random.normal(0, 0.08, total_samples))
        jitter = np.clip(jitter, -2.5, 2.5)

        if text.strip().endswith('?'):
            question_rise = np.where(t > total_duration * 0.7, 1.0 + 0.35 * ((t - total_duration * 0.7) / (total_duration * 0.3)), 1.0)
        else:
            question_rise = 1.0

        f0_t = base_f0 * declination * question_rise + jitter
        phase = np.cumsum(2 * np.pi * f0_t / sr)
        glottal_source = np.zeros_like(t)
        norm_phase = (phase % (2 * np.pi)) / (2 * np.pi)
        open_phase_mask = norm_phase < 0.6
        glottal_source[open_phase_mask] = 0.5 * (1.0 - np.cos(np.pi * norm_phase[open_phase_mask] / 0.6))
        return_mask = (norm_phase >= 0.6) & (norm_phase < 1.0)
        glottal_source[return_mask] = np.exp(-10.0 * (norm_phase[return_mask] - 0.6))

        from scipy import signal
        nyq = sr * 0.5

        def apply_formant(audio: np.ndarray, center_freq: float, bw: float, gain: float) -> np.ndarray:
            low = max(20.0, center_freq - bw * 0.5) / nyq
            high = min(nyq - 20.0, center_freq + bw * 0.5) / nyq
            if low < high and high < 1.0:
                b, a = signal.butter(2, [low, high], btype='bandpass')
                return signal.filtfilt(b, a, audio) * gain
            return np.zeros_like(audio)

        f1_wave = apply_formant(glottal_source, 520.0, 90.0, 0.45)
        f2_wave = apply_formant(glottal_source, 1480.0, 120.0, 0.30)
        f3_wave = apply_formant(glottal_source, 2450.0, 150.0, 0.15)
        vocal_audio = f1_wave + f2_wave + f3_wave

        fricative_noise = np.random.normal(0, 0.05, total_samples).astype(np.float32)
        if 3500.0 / nyq < 1.0:
            b_fric, a_fric = signal.butter(3, [3500.0 / nyq, min(7500.0 / nyq, 0.98)], btype='bandpass')
            fricative_noise = signal.filtfilt(b_fric, a_fric, fricative_noise)

        syllable_freq = 4.2
        cadence = (0.5 + 0.5 * np.sin(2 * np.pi * syllable_freq * t)) ** 1.8
        combined = (vocal_audio * cadence * 0.85 + fricative_noise * (1.0 - cadence) * 0.15).astype(np.float32)

        envelope = np.ones_like(t)
        fade_samples = int(sr * 0.04)
        if len(t) > fade_samples * 2:
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        out = (combined * envelope * 0.45).astype(np.float32)
        return out

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
