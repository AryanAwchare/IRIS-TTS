"""
Pocket TTS engine wrapper implementation with Dynamic Acoustic Morphing.

Provides high-fidelity voice cloning on CPU by combining Pocket TTS's natural
human prosody with dynamic formant resonance shifting, vocal warmth enhancement,
and multi-speaker acoustic profiling.
"""
from __future__ import annotations

import logging
import os
import struct
import threading
from typing import Any, Dict, Optional

import numpy as np

from app.tts_engines.base import BaseTTSEngine
from app.utils.timbre_morpher import extract_acoustic_profile, morph_timbre

logger = logging.getLogger(__name__)

# Catalog voice baseline acoustic profiles
CATALOG_VOICES = {
    "jean": {"f0": 105.0, "centroid": 1400.0, "gender": "male"},
    "marius": {"f0": 130.0, "centroid": 1600.0, "gender": "male"},
    "françois": {"f0": 120.0, "centroid": 1500.0, "gender": "male"},
    "alba": {"f0": 175.0, "centroid": 1950.0, "gender": "female"},
    "laura": {"f0": 185.0, "centroid": 2100.0, "gender": "female"},
    "anna": {"f0": 215.0, "centroid": 2300.0, "gender": "female"},
}


class _MockTTSModel:
    sample_rate = 24000

    def get_state_for_audio_prompt(self, audio_source: Any) -> dict:
        logger.warning("MockTTSModel: returning dummy voice state")
        return {"mock": True}

    def generate_audio(self, voice_state: Any, text: str) -> Any:
        logger.warning(f"MockTTSModel: generating dummy audio for text: {text[:40]!r}")
        t = np.linspace(0, 1.0, self.sample_rate, dtype=np.float32)
        sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        class _Tensor:
            def __init__(self, arr):
                self._arr = arr
            def numpy(self):
                return self._arr

        return _Tensor(sine)


class PocketTTSEngine(BaseTTSEngine):
    engine_name = "pocket-tts"

    def __init__(self):
        self._model: Any = None
        self._sample_rate: int = 24000
        self._cache: dict[str, Any] = {}
        self._cache_order: list[str] = []
        self._cache_lock = threading.Lock()
        self._MAX_CACHE_SIZE = 50

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def load_model(self) -> None:
        use_mock = os.getenv("VOICELIB_USE_MOCK_TTS", "false").lower() == "true"
        if use_mock:
            logger.warning("VOICELIB_USE_MOCK_TTS=true — using MockTTSModel")
            self._model = _MockTTSModel()
            self._sample_rate = self._model.sample_rate
            return

        try:
            from pocket_tts import TTSModel
            logger.info("Loading Pocket TTS model...")
            self._model = TTSModel.load_model()
            self._sample_rate = getattr(self._model, "sample_rate", 24000)
            logger.info(f"Pocket TTS loaded. Sample rate: {self._sample_rate} Hz")
        except Exception as exc:
            logger.warning(f"Pocket TTS model load failed ({exc}). Falling back to MockTTSModel.")
            self._model = _MockTTSModel()
            self._sample_rate = self._model.sample_rate

    def derive_voice_state(self, audio_source: str | bytes, voice_id: str) -> Any:
        if self._model is None:
            self.load_model()

        with self._cache_lock:
            if voice_id in self._cache:
                self._cache_order.remove(voice_id)
                self._cache_order.append(voice_id)
                return self._cache[voice_id]

        # 1. Extract detailed acoustic fingerprint from reference audio
        profile = extract_acoustic_profile(audio_source, sr=self._sample_rate)
        ref_f0 = profile.get("mean_f0", 150.0)
        ref_centroid = profile.get("spectral_centroid", 1800.0)

        # 2. Find closest catalog base carrier voice using Euclidean acoustic distance
        best_voice = "alba"
        best_score = float("inf")
        base_f0 = 175.0

        for name, meta in CATALOG_VOICES.items():
            # Normalized distance in F0 and spectral centroid
            f0_dist = ((ref_f0 - meta["f0"]) / 60.0) ** 2
            cent_dist = ((ref_centroid - meta["centroid"]) / 600.0) ** 2
            gender_penalty = 0.0 if profile.get("gender_hint") == meta["gender"] else 1.5
            score = f0_dist + cent_dist + gender_penalty

            if score < best_score:
                best_score = score
                best_voice = name
                base_f0 = meta["f0"]

        logger.info(
            f"Pocket TTS: reference acoustic profile (F0={ref_f0:.1f}Hz, Centroid={ref_centroid:.0f}Hz) "
            f"mapped to optimal base carrier '{best_voice}' (base F0={base_f0:.1f}Hz)"
        )

        # 3. Load base prompt state for carrier voice
        carrier_prompt_state = None
        try:
            carrier_prompt_state = self._model.get_state_for_audio_prompt(best_voice)
        except Exception as exc:
            logger.warning(f"Failed to get carrier prompt state for '{best_voice}' ({exc}) — trying default 'alba'")
            try:
                carrier_prompt_state = self._model.get_state_for_audio_prompt("alba")
                best_voice = "alba"
                base_f0 = 175.0
            except Exception:
                carrier_prompt_state = "alba"

        voice_state = {
            "voice_id": voice_id,
            "carrier_voice": best_voice,
            "carrier_state": carrier_prompt_state,
            "base_f0": base_f0,
            "acoustic_profile": profile,
            "engine": "pocket-tts-morph",
        }

        with self._cache_lock:
            if len(self._cache) >= self._MAX_CACHE_SIZE:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]
            self._cache[voice_id] = voice_state
            self._cache_order.append(voice_id)

        return voice_state

    def invalidate_cache(self, voice_id: str) -> None:
        with self._cache_lock:
            if voice_id in self._cache:
                del self._cache[voice_id]
                self._cache_order.remove(voice_id)

    def generate_audio(self, voice_state: Any, text: str, **kwargs: Any) -> bytes:
        if self._model is None:
            self.load_model()

        carrier_state = voice_state
        acoustic_profile: Optional[Dict[str, Any]] = None
        base_f0 = 160.0
        voice_state_dict: Optional[Dict[str, Any]] = None

        if isinstance(voice_state, dict):
            voice_state_dict = voice_state
            if "carrier_state" in voice_state and "acoustic_profile" in voice_state:
                carrier_state = voice_state["carrier_state"]
                acoustic_profile = voice_state["acoustic_profile"]
                base_f0 = voice_state.get("base_f0", 160.0)
            elif "audio_bytes" in voice_state and voice_state["audio_bytes"]:
                derived = self.derive_voice_state(voice_state["audio_bytes"], voice_state.get("voice_id", "default"))
                carrier_state = derived["carrier_state"]
                acoustic_profile = derived["acoustic_profile"]
                base_f0 = derived.get("base_f0", 160.0)
            elif "voice_id" in voice_state:
                derived = self.derive_voice_state("alba", voice_state["voice_id"])
                carrier_state = derived["carrier_state"]
                acoustic_profile = derived["acoustic_profile"]
                base_f0 = derived.get("base_f0", 160.0)
        elif isinstance(voice_state, str):
            derived = self.derive_voice_state(voice_state, "default")
            carrier_state = derived["carrier_state"]
            acoustic_profile = derived["acoustic_profile"]
            base_f0 = derived.get("base_f0", 160.0)

        # ── Carrier voice override from frontend ──────────────────────────
        carrier_voice_override = kwargs.get("carrier_voice")
        if carrier_voice_override and carrier_voice_override != "auto":
            override_key = carrier_voice_override.lower().strip()
            if override_key in CATALOG_VOICES:
                try:
                    carrier_state = self._model.get_state_for_audio_prompt(override_key)
                    base_f0 = CATALOG_VOICES[override_key]["f0"]
                    logger.info(f"Pocket TTS: carrier voice overridden to '{override_key}' (F0={base_f0:.1f}Hz)")
                except Exception as ov_err:
                    logger.warning(f"Carrier override '{override_key}' failed ({ov_err}), keeping auto-selected")
            else:
                logger.warning(f"Unknown carrier voice '{override_key}', keeping auto-selected")

        # ── Fine-tuning parameters from frontend ──────────────────────────
        morph_strength = kwargs.get("morph_strength", 0.85)
        warmth_gain_db = kwargs.get("warmth_gain_db", 0.0)
        brightness_gain_db = kwargs.get("brightness_gain_db", 0.0)

        # 1. Synthesize base human speech via Pocket TTS
        audio_tensor = self._model.generate_audio(carrier_state, text)
        if hasattr(audio_tensor, "detach"):
            audio_array = audio_tensor.detach().cpu().numpy().astype(np.float32)
        elif hasattr(audio_tensor, "numpy"):
            audio_array = audio_tensor.numpy().astype(np.float32)
        else:
            audio_array = np.asarray(audio_tensor, dtype=np.float32)

        audio_array = audio_array.flatten()

        # 2. Apply Dynamic Acoustic Timbre Morphing matching user reference sample
        if acoustic_profile is not None:
            try:
                audio_array = morph_timbre(
                    audio_array,
                    sr=self._sample_rate,
                    target_profile=acoustic_profile,
                    base_voice_f0=base_f0,
                    morph_strength=morph_strength,
                    warmth_override_db=warmth_gain_db,
                    brightness_override_db=brightness_gain_db,
                )
            except Exception as morph_err:
                logger.warning(f"Acoustic timbre morphing fallback ({morph_err})")

        # 3. Speed adjustments if requested
        speed = kwargs.get("speed", 1.0)
        if abs(speed - 1.0) > 0.05 and speed > 0.2:
            new_len = int(len(audio_array) / speed)
            indices = np.linspace(0, len(audio_array) - 1, new_len)
            audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array).astype(np.float32)

        return self._write_wav_bytes(audio_array, self._sample_rate)

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
