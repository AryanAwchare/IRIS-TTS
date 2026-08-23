"""
TTS Engine abstraction package.
"""
from __future__ import annotations

from app.tts_engines.base import BaseTTSEngine
from app.tts_engines.pocket_engine import PocketTTSEngine
from app.tts_engines.gptsovits_engine import GPTSoVITSEngine

_engines: dict[str, BaseTTSEngine] = {}


def get_engine(engine_id: str = "gpt-sovits-v3") -> BaseTTSEngine:
    """
    Return singleton engine instance for engine_id:
    - 'gpt-sovits-v3' / 'gpt-sovits'
    - 'zonos-expressive' / 'zonos'
    - 'pocket-tts' / 'pocket'
    """
    normalized_id = (engine_id or "gpt-sovits-v3").lower().strip()
    
    if normalized_id not in _engines:
        if "pocket" in normalized_id:
            _engines[normalized_id] = PocketTTSEngine()
        else:
            # GPT-SoVITS / Zonos Expressive use the advanced emotion & rank voice cloning pipeline
            _engines[normalized_id] = GPTSoVITSEngine()
            
    return _engines[normalized_id]
