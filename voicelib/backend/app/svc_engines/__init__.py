"""
Singing Voice Conversion engine registry.
"""
from __future__ import annotations

from typing import Dict, Type
from app.svc_engines.base import BaseVoiceConversionEngine

_ENGINES: Dict[str, Type[BaseVoiceConversionEngine]] = {}


def register_svc_engine(name: str):
    def decorator(cls: Type[BaseVoiceConversionEngine]):
        _ENGINES[name.lower()] = cls
        return cls
    return decorator


def get_svc_engine(name: str = "rvc-v2") -> BaseVoiceConversionEngine:
    name_norm = name.lower()
    if name_norm not in _ENGINES:
        # Import rvc_engine to populate registry
        from app.svc_engines.rvc_engine import RVCEngine
        _ENGINES["rvc-v2"] = RVCEngine
    cls = _ENGINES.get(name_norm, _ENGINES.get("rvc-v2"))
    return cls()
