"""
Abstract Base Class for all TTS engines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTTSEngine(ABC):
    @abstractmethod
    def load_model(self) -> None:
        """Initialize and load model into memory at startup."""
        pass

    @abstractmethod
    def derive_voice_state(self, audio_source: str | bytes, voice_id: str) -> Any:
        """Derive opaque voice state/embeddings from audio reference."""
        pass

    @abstractmethod
    def generate_audio(self, voice_state: Any, text: str, **kwargs: Any) -> bytes:
        """Synthesize text using derived voice state into raw WAV bytes."""
        pass

    @abstractmethod
    def invalidate_cache(self, voice_id: str) -> None:
        """Evict voice_id from engine memory cache."""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output audio sample rate in Hz."""
        pass
