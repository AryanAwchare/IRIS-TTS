"""
TTS Engine abstraction package.

Manages engine singletons, aliases, and real-time status for the model
switcher UI.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from app.tts_engines.base import BaseTTSEngine
from app.tts_engines.pocket_engine import PocketTTSEngine
from app.tts_engines.gptsovits_engine import GPTSoVITSEngine

logger = logging.getLogger(__name__)

_engines: dict[str, BaseTTSEngine] = {}

# Canonical engine IDs
ENGINE_POCKET = "pocket-tts"
ENGINE_NEURAL = "gpt-sovits-v3"
ENGINE_ZONOS = "zonos-expressive"

# Alias map → canonical ID
_ALIASES: dict[str, str] = {
    "pocket": ENGINE_POCKET,
    "pocket-tts": ENGINE_POCKET,
    "gpt-sovits": ENGINE_NEURAL,
    "gpt-sovits-v3": ENGINE_NEURAL,
    "neural": ENGINE_NEURAL,
    "zonos": ENGINE_ZONOS,
    "zonos-expressive": ENGINE_ZONOS,
}

# Which engines are enabled (env var fallback: all)
_ENABLED_RAW = os.getenv("ENABLED_ENGINES", "pocket-tts,gpt-sovits-v3,zonos-expressive")
ENABLED_ENGINES: set[str] = {_ALIASES.get(e.strip(), e.strip()) for e in _ENABLED_RAW.split(",")}


def _canonicalize(engine_id: str) -> str:
    normalized = (engine_id or ENGINE_NEURAL).lower().strip()
    return _ALIASES.get(normalized, normalized)


def get_engine(engine_id: str = "gpt-sovits-v3") -> BaseTTSEngine:
    """
    Return singleton engine instance for engine_id.
    """
    canonical = _canonicalize(engine_id)

    if canonical not in _engines:
        if canonical == ENGINE_POCKET:
            _engines[canonical] = PocketTTSEngine()
        else:
            # GPT-SoVITS / Zonos Expressive use the Colab GPU pipeline
            _engines[canonical] = GPTSoVITSEngine()

    return _engines[canonical]


# ── Engine catalog metadata (for frontend UI cards) ────────────────────────

ENGINE_CATALOG: list[dict[str, Any]] = [
    {
        "id": ENGINE_POCKET,
        "name": "Pocket TTS",
        "description": "CPU-local voice cloning with acoustic morphing",
        "compute": "cpu",
        "supports_fine_tuning": True,
    },
    {
        "id": ENGINE_NEURAL,
        "name": "Neural Voice Cloning",
        "description": "GPU-accelerated GPT-SoVITS v3 cloning via Colab",
        "compute": "gpu_colab",
        "supports_fine_tuning": False,
    },
    {
        "id": ENGINE_ZONOS,
        "name": "Zonos Expressive",
        "description": "GPU-accelerated expressive TTS with 8D emotion vectors",
        "compute": "gpu_colab",
        "supports_fine_tuning": False,
    },
]


def get_all_engine_status() -> list[dict[str, Any]]:
    """
    Return live readiness status for every known engine.
    Used by GET /engines/status so the frontend can grey out offline engines.
    """
    statuses = []
    for meta in ENGINE_CATALOG:
        eid = meta["id"]
        enabled = eid in ENABLED_ENGINES
        ready = False

        if not enabled:
            status_label = "disabled"
        elif eid == ENGINE_POCKET:
            # Pocket TTS is always available locally — "ready" if the model singleton exists
            try:
                eng = get_engine(eid)
                ready = eng._model is not None
                status_label = "ready" if ready else "not_loaded"
            except Exception:
                status_label = "error"
        else:
            # GPU engines — check Colab connectivity
            try:
                from app.tts_engines.gptsovits_engine import get_live_colab_url
                from app.config import get_settings
                live_url = get_live_colab_url()
                url = live_url or get_settings().colab_gpu_api_url
                ready = bool(url and url != "http://localhost:8008")
                status_label = "ready" if ready else "offline"
            except Exception:
                status_label = "offline"

        statuses.append({
            **meta,
            "enabled": enabled,
            "ready": ready,
            "status": status_label,
        })

    return statuses
