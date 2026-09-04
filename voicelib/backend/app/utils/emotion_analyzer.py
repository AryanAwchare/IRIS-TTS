"""
Emotion & Sentiment Analyzer — Deep NLP Text Emotion Intelligence.
Uses transformer-based classification (e.g. DistilRoBERTa / BERT) with instant fallback
to rule-based lexicon scoring for real-time inference (<80ms).

Provides:
  - Canonical 6-Emotion taxonomy mapping (neutral, calm, happy, excited, sad, angry)
  - Continuous emotional intensity scoring (0.0 – 1.0)
  - Intelligent blending modes (auto, blend, user_only)
  - Mathematical hyperparameter scaling (CFG weight, Exaggeration, Speed, Pitch)
  - LRU caching by text hash (proper LRU eviction — no thundering herd on full cache)
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Recalibrated 8-Emotion Canonical Table ─────────────────────────────────────
CANONICAL_EMOTIONS: Dict[str, Dict[str, float]] = {
    "neutral":   {"base_cfg": 0.65, "min_exag": 0.04, "max_exag": 0.10, "base_speed": 1.00, "pitch": 0.0},
    "calm":      {"base_cfg": 0.68, "min_exag": 0.02, "max_exag": 0.08, "base_speed": 0.94, "pitch": -0.5},
    "happy":     {"base_cfg": 0.54, "min_exag": 0.14, "max_exag": 0.28, "base_speed": 1.06, "pitch": 1.0},
    "excited":   {"base_cfg": 0.46, "min_exag": 0.25, "max_exag": 0.40, "base_speed": 1.15, "pitch": 1.8},
    "sad":       {"base_cfg": 0.60, "min_exag": 0.06, "max_exag": 0.16, "base_speed": 0.88, "pitch": -1.0},
    "angry":     {"base_cfg": 0.50, "min_exag": 0.22, "max_exag": 0.36, "base_speed": 1.12, "pitch": 1.0},
    "fearful":   {"base_cfg": 0.52, "min_exag": 0.18, "max_exag": 0.32, "base_speed": 1.10, "pitch": 1.4},
    "disgusted": {"base_cfg": 0.58, "min_exag": 0.10, "max_exag": 0.20, "base_speed": 0.95, "pitch": -0.3},
}

# Transformer Label to Canonical Emotion Mapping (Full 7 -> 8 mapping, preserving fear & disgust)
MODEL_LABEL_MAP: Dict[str, str] = {
    "joy":        "happy",
    "happiness":  "happy",
    "love":       "happy",
    "surprise":   "excited",
    "excitement": "excited",
    "sadness":    "sad",
    "sorrow":     "sad",
    "anger":      "angry",
    "disgust":    "disgusted",
    "fear":       "fearful",
    "neutral":    "neutral",
    "calm":       "calm",
}


@dataclass
class EmotionProfile:
    """Strongly-typed synthesis emotion parameters with dict-like backward compatibility."""
    emotion: str
    intensity: float
    confidence: float
    cfg_weight: float
    exaggeration: float
    speed: float
    pitch: float
    temperature: float = 0.70
    top_p: float = 0.85
    analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolved_emotion": self.emotion,
            "intensity": round(self.intensity, 3),
            "confidence": round(self.confidence, 3),
            "cfg_weight": round(self.cfg_weight, 2),
            "exaggeration": round(self.exaggeration, 2),
            "speed": round(self.speed, 2),
            "pitch": round(self.pitch, 2),
            "temperature": round(self.temperature, 2),
            "top_p": round(self.top_p, 2),
            "analysis": self.analysis,
        }

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()


INTENSITY_MODIFIERS: Dict[str, List[str]] = {
    "boosters": ["very", "extremely", "so", "incredibly", "absolutely", "totally", "deeply", "unbelievably"],
    "dampeners": ["a bit", "slightly", "kind of", "somewhat", "a little"],
}


def _lexical_intensity_adjustment(text: str) -> float:
    lower = text.lower()
    adj = 0.0
    for w in INTENSITY_MODIFIERS["boosters"]:
        if w in lower:
            adj += 0.08
    for w in INTENSITY_MODIFIERS["dampeners"]:
        if w in lower:
            adj -= 0.10
    letters = [c for c in text if c.isalpha()]
    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > 0.4:
            adj += 0.15
    return adj


@dataclass
class EmotionAnalysisResult:
    emotion: str
    intensity: float
    confidence: float
    raw_label: str
    raw_score: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    suggested_tags: List[str] = field(default_factory=list)
    is_mixed: bool = False


_TRANSFORMER_PIPELINE: Any = None
_TRANSFORMER_LOAD_ATTEMPTED: bool = False
_PIPELINE_LOCK = threading.Lock()

# FIX: proper LRU cache using OrderedDict — replaces _CACHE.clear() which
# caused thundering herd when cache was full (all 300 entries evicted at once).
_CACHE: OrderedDict[str, EmotionAnalysisResult] = OrderedDict()
_CACHE_LOCK = threading.Lock()
_MAX_CACHE_SIZE = 300


def _lru_cache_put(key: str, value: EmotionAnalysisResult) -> None:
    """Insert into LRU cache, evicting the oldest entry if at capacity."""
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
        else:
            if len(_CACHE) >= _MAX_CACHE_SIZE:
                _CACHE.popitem(last=False)   # evict LRU (oldest first)
            _CACHE[key] = value
            _CACHE.move_to_end(key)


def _lru_cache_get(key: str) -> Optional[EmotionAnalysisResult]:
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
        return None


def _get_transformer_pipeline():
    global _TRANSFORMER_PIPELINE, _TRANSFORMER_LOAD_ATTEMPTED
    if _TRANSFORMER_PIPELINE is not None:
        return _TRANSFORMER_PIPELINE

    with _PIPELINE_LOCK:
        if _TRANSFORMER_LOAD_ATTEMPTED:
            return _TRANSFORMER_PIPELINE
        _TRANSFORMER_LOAD_ATTEMPTED = True

        try:
            from transformers import pipeline
            from app.config import get_settings
            settings = get_settings()
            model_name = getattr(settings, "emotion_model_name", "j-hartmann/emotion-english-distilroberta-base")
            logger.info(f"Initializing Emotion NLP model '{model_name}' on CPU...")
            _TRANSFORMER_PIPELINE = pipeline(
                "text-classification",
                model=model_name,
                top_k=None,
                device=-1,
            )
            logger.info(f"Emotion NLP model '{model_name}' ready.")
        except Exception as exc:
            logger.info(f"Emotion transformer notice: {exc}. Using lexicon engine.")
            _TRANSFORMER_PIPELINE = None
    return _TRANSFORMER_PIPELINE


def analyze_text_sentiment_and_emotion(text: str) -> EmotionAnalysisResult:
    """
    Analyzes prompt text to extract canonical emotion, intensity, and expressiveness cues.
    """
    if not text or not text.strip():
        return EmotionAnalysisResult(
            emotion="neutral",
            intensity=0.0,
            confidence=1.0,
            raw_label="neutral",
            raw_score=1.0,
            all_scores={"neutral": 1.0},
        )

    clean_text = text.strip()
    text_hash = hashlib.md5(clean_text.lower().encode("utf-8")).hexdigest()

    cached = _lru_cache_get(text_hash)
    if cached is not None:
        return cached

    pipe = _get_transformer_pipeline()
    if pipe is not None:
        try:
            outputs = pipe(clean_text[:512])
            if outputs and isinstance(outputs[0], list):
                raw_scores = {item["label"].lower(): float(item["score"]) for item in outputs[0]}
                top_label = max(raw_scores, key=raw_scores.get)
                top_score = raw_scores[top_label]

                canonical_emotion = MODEL_LABEL_MAP.get(top_label, "neutral")

                sorted_scores = sorted(raw_scores.values(), reverse=True)
                is_mixed = len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1] < 0.15) and (sorted_scores[0] < 0.60)

                # Decouple delivery intensity from classifier confidence
                base_intensity = 0.45 + _lexical_intensity_adjustment(clean_text)
                if "!" in clean_text:
                    base_intensity += 0.10 * min(3, clean_text.count("!"))
                if is_mixed:
                    base_intensity *= 0.8
                if top_score > 0.85:
                    base_intensity += 0.10
                base_intensity = float(np.clip(base_intensity, 0.05, 1.0))

                suggested_tags = []
                if canonical_emotion in ["happy", "excited"] and any(w in clean_text.lower() for w in ("haha", "lol", "laugh")):
                    suggested_tags.append("[laughter]")
                elif canonical_emotion == "sad" and any(w in clean_text.lower() for w in ("unfortunately", "sigh")):
                    suggested_tags.append("[sigh]")
                elif canonical_emotion == "excited" and any(w in clean_text.lower() for w in ("wow", "omg")):
                    suggested_tags.append("[gasp]")

                result = EmotionAnalysisResult(
                    emotion=canonical_emotion,
                    intensity=round(base_intensity, 3),
                    confidence=round(top_score, 3),
                    raw_label=top_label,
                    raw_score=round(top_score, 3),
                    all_scores=raw_scores,
                    suggested_tags=suggested_tags,
                    is_mixed=is_mixed,
                )
                _lru_cache_put(text_hash, result)
                return result
        except Exception as err:
            logger.debug(f"Transformer pipeline failed: {err}. Falling back to rule engine.")

    # Rule-based fallback (<2ms)
    from app.utils.emotion_detector import analyze_text_emotion
    fallback_res = analyze_text_emotion(clean_text)
    emo = fallback_res.get("emotion", "neutral")
    if emo not in CANONICAL_EMOTIONS:
        emo = "neutral"

    lex_adj = _lexical_intensity_adjustment(clean_text)
    raw_intensity = float(fallback_res.get("intensity", 0.0))
    if raw_intensity > 0.0:
        base_intensity = float(np.clip(raw_intensity + lex_adj, 0.10, 1.0))
    else:
        base_intensity = 0.0

    result = EmotionAnalysisResult(
        emotion=emo,
        intensity=round(base_intensity, 3),
        confidence=float(fallback_res.get("confidence", 0.7)),
        raw_label=emo,
        raw_score=round(base_intensity, 3),
        suggested_tags=fallback_res.get("suggested_tags", []),
        is_mixed=False,
    )
    _lru_cache_put(text_hash, result)
    return result


def compute_modulated_synthesis_parameters(
    text: str,
    requested_emotion: Optional[str] = "auto",
    opt_weights: Optional[Dict[str, Any]] = None,
    user_speed: Optional[float] = None,
    user_pitch: Optional[float] = None,
    blend_mode: str = "auto",
    user_intensity: Optional[float] = None,
) -> EmotionProfile:
    """
    Computes production-ready generation hyperparameters:
    - Automatically resolves emotional intent from text
    - Merges with user preference (auto / blend / user_only)
    - Allows direct user_intensity manual override
    - Scales CFG weight & Exaggeration using speaker acoustic DNA + emotion formulas
    """
    opt = opt_weights or {}
    analysis = analyze_text_sentiment_and_emotion(text)

    # 1. Determine Effective Emotion & Intensity
    req_norm = (requested_emotion or "auto").lower().strip()

    manual_intensity = None
    if user_intensity is not None:
        manual_intensity = float(np.clip(user_intensity, 0.0, 1.0))

    if req_norm in ["auto", "none", ""]:
        resolved_emotion = analysis.emotion
        effective_intensity = manual_intensity if manual_intensity is not None else analysis.intensity
    elif blend_mode == "blend" and req_norm != analysis.emotion and analysis.intensity > 0.40:
        resolved_emotion = req_norm
        effective_intensity = manual_intensity if manual_intensity is not None else max(0.20, analysis.intensity * 0.75)
    elif blend_mode == "user_only" or (req_norm in CANONICAL_EMOTIONS and req_norm != "neutral"):
        resolved_emotion = req_norm if req_norm in CANONICAL_EMOTIONS else "neutral"
        effective_intensity = manual_intensity if manual_intensity is not None else max(0.25, analysis.intensity)
    else:
        # User explicitly requested neutral
        resolved_emotion = "neutral"
        effective_intensity = manual_intensity if manual_intensity is not None else min(0.12, analysis.intensity)

    table = CANONICAL_EMOTIONS.get(resolved_emotion, CANONICAL_EMOTIONS["neutral"])

    # 2. Exaggeration Formula: min_exag + (max_exag - min_exag) * intensity
    min_ex = table["min_exag"]
    max_ex = table["max_exag"]
    profile_exag_base = float(opt.get("exaggeration", 0.05))

    if resolved_emotion == "neutral":
        final_exaggeration = float(np.clip(profile_exag_base, min_ex, max_ex))
    else:
        final_exaggeration = float(np.clip(
            min_ex + (max_ex - min_ex) * effective_intensity,
            0.00,
            0.60
        ))

    # 3. CFG Weight Formula: base_cfg * (1.05 - 0.20 * intensity)
    profile_cfg_base = float(opt.get("cfg_weight", table["base_cfg"]))
    cfg_factor = 1.05 - (0.20 * effective_intensity)
    final_cfg = float(np.clip(profile_cfg_base * cfg_factor, 0.35, 0.75))

    # 4. Speed & Pitch
    profile_speed_base = float(opt.get("speed_scale", 1.00))
    if user_speed and abs(user_speed - 1.0) > 0.05:
        final_speed = user_speed
    else:
        final_speed = float(np.clip(profile_speed_base * table["base_speed"], 0.75, 1.35))

    if user_pitch and abs(user_pitch) > 0.05:
        final_pitch = user_pitch
    else:
        final_pitch = 0.0

    temperature = float(opt.get("temperature", 0.70))
    top_p = float(opt.get("top_p", 0.85))

    if resolved_emotion in ["excited", "happy"]:
        top_p = min(0.92, top_p + 0.04)
        temperature = min(0.78, temperature + 0.03)

    return EmotionProfile(
        emotion=resolved_emotion,
        intensity=round(effective_intensity, 3),
        confidence=round(analysis.confidence, 3),
        cfg_weight=round(final_cfg, 2),
        exaggeration=round(final_exaggeration, 2),
        speed=round(final_speed, 2),
        pitch=round(final_pitch, 2),
        temperature=round(temperature, 2),
        top_p=round(top_p, 2),
        analysis={
            "detected_emotion": analysis.emotion,
            "confidence": analysis.confidence,
            "raw_label": analysis.raw_label,
            "is_mixed": analysis.is_mixed,
            "suggested_tags": analysis.suggested_tags,
        },
    )
