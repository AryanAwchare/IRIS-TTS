"""
VoiceLib TTS Router & Cache Manager.

Delegates synthesis and voice state derivation to the configured TTS engine
(e.g., GPT-SoVITS v3, Zonos Expressive, or Pocket TTS).
"""
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Any, Optional

from app.config import get_settings
from app.tts_engines import get_engine

logger = logging.getLogger(__name__)


class _LRUCache:
    """Thread-safe LRU cache with move-on-access eviction."""

    def __init__(self, max_size: int = 50):
        self._max_size = max_size
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def remove(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# Single-layer LRU cache mapping voice_id -> voice_state
_cache = _LRUCache(max_size=50)


def load_model() -> None:
    """Pre-warm configured default TTS engine at startup."""
    settings = get_settings()
    engine_id = settings.tts_engine
    logger.info(f"Loading active TTS engine: '{engine_id}'...")
    engine = get_engine(engine_id)
    engine.load_model()
    logger.info(f"TTS Engine '{engine_id}' ready. Sample rate: {engine.sample_rate} Hz")


def get_sample_rate(engine_id: Optional[str] = None) -> int:
    settings = get_settings()
    eid = engine_id or settings.tts_engine
    return get_engine(eid).sample_rate


def get_cached_voice_state(voice_id: str) -> Any | None:
    """Retrieve cached voice state without re-derivation."""
    return _cache.get(voice_id)


def derive_voice_state(
    audio_source: str | bytes,
    voice_id: str,
    engine_id: Optional[str] = None,
) -> Any:
    """Derive voice acoustic embedding / state for the given voice_id."""
    settings = get_settings()
    eid = engine_id or settings.tts_engine
    engine = get_engine(eid)
    state = engine.derive_voice_state(audio_source, voice_id)
    _cache.put(voice_id, state)
    return state


def invalidate_cache(voice_id: str, engine_id: Optional[str] = None) -> None:
    """Invalidate voice state in memory."""
    _cache.remove(voice_id)
    settings = get_settings()
    eid = engine_id or settings.tts_engine
    get_engine(eid).invalidate_cache(voice_id)


def generate_audio(
    voice_state: Any,
    text: str,
    *,
    engine_id: Optional[str] = None,
    emotion: str = "neutral",
    emotions: Optional[dict[str, float]] = None,
    speed: float = 1.0,
    pitch: float = 0.0,
    rank: int = 128,
    top_p: float = 0.8,
    temperature: float = 0.7,
    text_lang: str = "en",
    **kwargs: Any,
) -> bytes:
    """Synthesize speech using selected TTS engine with emotional & hyperparameter conditioning."""
    settings = get_settings()
    eid = engine_id or settings.tts_engine
    engine = get_engine(eid)

    return engine.generate_audio(
        voice_state,
        text,
        emotion=emotion,
        emotions=emotions,
        speed=speed,
        pitch=pitch,
        rank=rank,
        top_p=top_p,
        temperature=temperature,
        text_lang=text_lang,
        **kwargs,
    )


def get_engine_info(engine_id: Optional[str] = None) -> dict[str, Any]:
    """Return runtime metadata and capabilities for the active TTS engine."""
    settings = get_settings()
    eid = engine_id or settings.tts_engine
    engine = get_engine(eid)

    return {
        "engine": getattr(engine, "engine_name", eid),
        "sample_rate": engine.sample_rate,
        "emotion_support": True,
        "supported_emotions": [
            "neutral", "happy", "sad", "angry", "excited", "calm", "fearful", "surprised"
        ],
        "supported_languages": ["en", "zh", "ja", "ko"],
        "min_sample_duration_seconds": settings.min_sample_duration_seconds,
        "max_sample_duration_seconds": settings.max_sample_duration_seconds,
        "speed_range": [0.5, 2.0],
        "mock_mode": settings.voicelib_use_mock_tts,
    }
