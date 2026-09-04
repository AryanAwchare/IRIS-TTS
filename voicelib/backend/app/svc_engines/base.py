"""
Base interface for Singing Voice Conversion (SVC / RVC) engines.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from app.utils.audio_asset import AudioAsset, AudioChunk


class BaseVoiceConversionEngine(ABC):
    """Abstract interface for Singing Voice Conversion engines."""

    engine_name: str = "base-svc"

    @abstractmethod
    def convert_vocal_chunk(
        self,
        chunk: AudioChunk,
        voice_id: str,
        pitch_shift: int = 0,
        index_rate: float = 0.75,
        protect_voiceless: float = 0.33,
        target_voice_bytes: Optional[bytes] = None,
    ) -> np.ndarray:
        """Convert a single audio chunk to the target vocal timbre."""
        pass

    @abstractmethod
    def convert_full_vocals(
        self,
        vocals_asset: AudioAsset,
        voice_id: str,
        pitch_shift: int = 0,
        index_rate: float = 0.75,
        protect_voiceless: float = 0.33,
        preview_only: bool = False,
        checkpoint_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        target_voice_bytes: Optional[bytes] = None,
    ) -> AudioAsset:
        """Process full vocal track with chunking, checkpointing, and crossfade stitching."""
        pass
