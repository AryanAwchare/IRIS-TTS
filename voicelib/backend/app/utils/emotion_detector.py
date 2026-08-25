"""
Text Sentiment & Emotion Intelligence Engine for VoiceLib.
Provides:
  - Fast, CPU-friendly NLP emotion & sentiment analysis (<5ms)
  - Emotion intensity scoring (0.0 – 1.0)
  - Blending of text-detected emotion with speaker acoustic DNA
  - Dynamic parameter resolution (CFG weight, Exaggeration, Speed, Pitch)
  - Contextual paralinguistic tag insertion ([laughter], [sigh], [gasp], [chuckle])
  - Thread-safe LRU caching
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Emotion Lexicon with valence, arousal, and dominant emotion mapping
EMOTION_LEXICON: Dict[str, Dict[str, Any]] = {
    # Happy / Joy / Warmth
    "happy": {"emotion": "happy", "valence": 0.8, "arousal": 0.6, "intensity": 0.8},
    "joy": {"emotion": "happy", "valence": 0.9, "arousal": 0.7, "intensity": 0.9},
    "glad": {"emotion": "happy", "valence": 0.7, "arousal": 0.4, "intensity": 0.6},
    "great": {"emotion": "happy", "valence": 0.8, "arousal": 0.6, "intensity": 0.7},
    "wonderful": {"emotion": "happy", "valence": 0.9, "arousal": 0.7, "intensity": 0.85},
    "delighted": {"emotion": "happy", "valence": 0.9, "arousal": 0.6, "intensity": 0.85},
    "love": {"emotion": "happy", "valence": 0.9, "arousal": 0.5, "intensity": 0.8},
    "laugh": {"emotion": "happy", "valence": 0.8, "arousal": 0.8, "intensity": 0.9, "tag": "[laughter]"},
    "haha": {"emotion": "happy", "valence": 0.8, "arousal": 0.8, "intensity": 0.9, "tag": "[laughter]"},
    "lol": {"emotion": "happy", "valence": 0.7, "arousal": 0.7, "intensity": 0.8, "tag": "[chuckle]"},
    "smile": {"emotion": "happy", "valence": 0.7, "arousal": 0.4, "intensity": 0.6},
    "congratulations": {"emotion": "happy", "valence": 0.8, "arousal": 0.7, "intensity": 0.8},

    # Excited / Thrilled
    "excited": {"emotion": "excited", "valence": 0.8, "arousal": 0.9, "intensity": 0.9},
    "amazing": {"emotion": "excited", "valence": 0.9, "arousal": 0.8, "intensity": 0.85},
    "awesome": {"emotion": "excited", "valence": 0.8, "arousal": 0.8, "intensity": 0.8},
    "incredible": {"emotion": "excited", "valence": 0.9, "arousal": 0.85, "intensity": 0.9},
    "unbelievable": {"emotion": "excited", "valence": 0.7, "arousal": 0.9, "intensity": 0.85},
    "wow": {"emotion": "excited", "valence": 0.8, "arousal": 0.9, "intensity": 0.85, "tag": "[gasp]"},
    "hurray": {"emotion": "excited", "valence": 0.9, "arousal": 0.9, "intensity": 0.95},
    "fantastic": {"emotion": "excited", "valence": 0.9, "arousal": 0.8, "intensity": 0.85},

    # Sad / Sorrow / Melancholy
    "sad": {"emotion": "sad", "valence": -0.8, "arousal": -0.5, "intensity": 0.8},
    "sorrow": {"emotion": "sad", "valence": -0.9, "arousal": -0.4, "intensity": 0.85},
    "unfortunate": {"emotion": "sad", "valence": -0.6, "arousal": -0.3, "intensity": 0.6},
    "unfortunately": {"emotion": "sad", "valence": -0.6, "arousal": -0.3, "intensity": 0.6, "tag": "[sigh]"},
    "depressed": {"emotion": "sad", "valence": -0.9, "arousal": -0.7, "intensity": 0.9},
    "grief": {"emotion": "sad", "valence": -0.9, "arousal": -0.6, "intensity": 0.9},
    "crying": {"emotion": "sad", "valence": -0.9, "arousal": -0.3, "intensity": 0.85},
    "lonely": {"emotion": "sad", "valence": -0.7, "arousal": -0.5, "intensity": 0.75},
    "heartbroken": {"emotion": "sad", "valence": -0.9, "arousal": -0.6, "intensity": 0.95},
    "sigh": {"emotion": "sad", "valence": -0.5, "arousal": -0.4, "intensity": 0.6, "tag": "[sigh]"},
    "alas": {"emotion": "sad", "valence": -0.7, "arousal": -0.4, "intensity": 0.75, "tag": "[sigh]"},

    # Angry / Frustrated
    "angry": {"emotion": "angry", "valence": -0.8, "arousal": 0.8, "intensity": 0.85},
    "furious": {"emotion": "angry", "valence": -0.9, "arousal": 0.95, "intensity": 0.95},
    "mad": {"emotion": "angry", "valence": -0.7, "arousal": 0.7, "intensity": 0.75},
    "frustrated": {"emotion": "angry", "valence": -0.7, "arousal": 0.6, "intensity": 0.75},
    "outraged": {"emotion": "angry", "valence": -0.9, "arousal": 0.9, "intensity": 0.9},
    "hate": {"emotion": "angry", "valence": -0.9, "arousal": 0.7, "intensity": 0.85},
    "annoying": {"emotion": "angry", "valence": -0.6, "arousal": 0.5, "intensity": 0.65},
    "terrible": {"emotion": "angry", "valence": -0.8, "arousal": 0.6, "intensity": 0.75},
    "horrible": {"emotion": "angry", "valence": -0.8, "arousal": 0.6, "intensity": 0.75},

    # Calm / Serene / Peaceful
    "calm": {"emotion": "calm", "valence": 0.5, "arousal": -0.7, "intensity": 0.7},
    "peaceful": {"emotion": "calm", "valence": 0.7, "arousal": -0.8, "intensity": 0.8},
    "relax": {"emotion": "calm", "valence": 0.6, "arousal": -0.7, "intensity": 0.7},
    "quiet": {"emotion": "calm", "valence": 0.4, "arousal": -0.8, "intensity": 0.6},
    "gentle": {"emotion": "calm", "valence": 0.6, "arousal": -0.6, "intensity": 0.65},
    "softly": {"emotion": "calm", "valence": 0.4, "arousal": -0.7, "intensity": 0.6, "tag": "[whisper]"},
    "whisper": {"emotion": "calm", "valence": 0.3, "arousal": -0.8, "intensity": 0.7, "tag": "[whisper]"},

    # Fearful / Anxious
    "scared": {"emotion": "fearful", "valence": -0.7, "arousal": 0.8, "intensity": 0.8},
    "afraid": {"emotion": "fearful", "valence": -0.7, "arousal": 0.7, "intensity": 0.75},
    "terrified": {"emotion": "fearful", "valence": -0.9, "arousal": 0.9, "intensity": 0.95},
    "anxious": {"emotion": "fearful", "valence": -0.6, "arousal": 0.6, "intensity": 0.7},
    "nervous": {"emotion": "fearful", "valence": -0.5, "arousal": 0.6, "intensity": 0.65},
    "panic": {"emotion": "fearful", "valence": -0.8, "arousal": 0.9, "intensity": 0.9},

    # Surprised / Shocked
    "surprised": {"emotion": "surprised", "valence": 0.4, "arousal": 0.85, "intensity": 0.8},
    "shocked": {"emotion": "surprised", "valence": -0.5, "arousal": 0.9, "intensity": 0.85},
    "astonished": {"emotion": "surprised", "valence": 0.6, "arousal": 0.85, "intensity": 0.85},
    "whoa": {"emotion": "surprised", "valence": 0.5, "arousal": 0.8, "intensity": 0.8, "tag": "[gasp]"},
    "oh my god": {"emotion": "surprised", "valence": 0.3, "arousal": 0.9, "intensity": 0.9, "tag": "[gasp]"},
}

# Base emotion hyperparameters (From IRIS Research Paper Table I & Chatterbox best practices)
BASE_EMOTION_TABLE: Dict[str, Dict[str, float]] = {
    "neutral":   {"cfg": 0.65, "exag": 0.05, "speed": 1.00, "pitch": 0.0},
    "calm":      {"cfg": 0.70, "exag": 0.02, "speed": 0.92, "pitch": -0.8},
    "happy":     {"cfg": 0.58, "exag": 0.15, "speed": 1.08, "pitch": 1.5},
    "excited":   {"cfg": 0.52, "exag": 0.28, "speed": 1.18, "pitch": 2.2},
    "sad":       {"cfg": 0.65, "exag": 0.08, "speed": 0.88, "pitch": -1.4},
    "angry":     {"cfg": 0.55, "exag": 0.22, "speed": 1.12, "pitch": 1.2},
    "fearful":   {"cfg": 0.58, "exag": 0.18, "speed": 1.10, "pitch": 1.8},
    "surprised": {"cfg": 0.52, "exag": 0.24, "speed": 1.08, "pitch": 2.5},
}

_EMOTION_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()
_MAX_CACHE = 200


def analyze_text_emotion(text: str) -> Dict[str, Any]:
    """
    Analyzes sentiment, arousal, and dominant emotion from prompt text.
    Returns emotion classification, intensity score, and suggested paralinguistic tags.
    """
    if not text or not text.strip():
        return {
            "emotion": "neutral",
            "intensity": 0.0,
            "confidence": 0.5,
            "suggested_tags": [],
            "valence": 0.0,
            "arousal": 0.0,
        }

    # Check cache
    text_hash = hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()
    with _CACHE_LOCK:
        if text_hash in _EMOTION_CACHE:
            return _EMOTION_CACHE[text_hash]

    words = re.findall(r"\b[\w'-]+\b", text.lower())
    scores: Dict[str, float] = {e: 0.0 for e in BASE_EMOTION_TABLE.keys()}
    scores["neutral"] = 0.2  # baseline prior
    valence_sum = 0.0
    arousal_sum = 0.0
    match_count = 0
    suggested_tags: List[str] = []

    # Check multi-word triggers first
    lower_text = text.lower()
    for phrase, meta in EMOTION_LEXICON.items():
        if " " in phrase and phrase in lower_text:
            emo = meta["emotion"]
            scores[emo] += meta["intensity"] * 2.0
            valence_sum += meta["valence"]
            arousal_sum += meta["arousal"]
            match_count += 1
            if "tag" in meta and meta["tag"] not in suggested_tags:
                suggested_tags.append(meta["tag"])

    # Single-word lexicon scanning
    for w in words:
        if w in EMOTION_LEXICON:
            meta = EMOTION_LEXICON[w]
            emo = meta["emotion"]
            scores[emo] += meta["intensity"]
            valence_sum += meta["valence"]
            arousal_sum += meta["arousal"]
            match_count += 1
            if "tag" in meta and meta["tag"] not in suggested_tags:
                suggested_tags.append(meta["tag"])

    # Exclamation & question mark punctuation energy cues
    excl_count = text.count("!")
    q_count = text.count("?")
    if excl_count >= 2:
        scores["excited"] += 0.4 * excl_count
        arousal_sum += 0.3 * excl_count
    elif excl_count == 1:
        scores["excited"] += 0.25
        arousal_sum += 0.15

    if q_count >= 2:
        scores["surprised"] += 0.35
    elif q_count == 1 and ("what" in words or "how" in words or "why" in words):
        scores["surprised"] += 0.2

    # Capitalization energy cues (ALL-CAPS words)
    caps = [w for w in re.findall(r"\b[A-Z]{2,}\b", text) if w not in ["AI", "API", "CPU", "GPU", "UI", "TTS"]]
    if caps:
        scores["excited"] += 0.3 * len(caps)
        scores["angry"] += 0.2 * len(caps)

    # Determine dominant emotion
    dominant_emotion = max(scores, key=scores.get)
    max_score = scores[dominant_emotion]

    if match_count == 0 and excl_count == 0 and q_count == 0:
        dominant_emotion = "neutral"
        intensity = 0.0
        confidence = 0.8
    else:
        # Intensity scaled proportionally with keyword intensity and match density
        avg_score = max_score / max(1.0, float(match_count))
        intensity = float(np.clip(avg_score * (1.0 + 0.1 * min(3, match_count - 1)), 0.25, 0.95))
        confidence = float(np.clip(max_score / (sum(scores.values()) + 1e-6), 0.45, 0.95))

    avg_valence = float(np.clip(valence_sum / max(1, match_count), -1.0, 1.0))
    avg_arousal = float(np.clip(arousal_sum / max(1, match_count), -1.0, 1.0))

    result = {
        "emotion": dominant_emotion,
        "intensity": round(intensity, 3),
        "confidence": round(confidence, 3),
        "suggested_tags": suggested_tags,
        "valence": round(avg_valence, 3),
        "arousal": round(avg_arousal, 3),
    }

    with _CACHE_LOCK:
        if len(_EMOTION_CACHE) >= _MAX_CACHE:
            _EMOTION_CACHE.clear()
        _EMOTION_CACHE[text_hash] = result

    return result


def resolve_synthesis_parameters(
    text: str,
    requested_emotion: Optional[str] = None,
    opt_weights: Optional[Dict[str, Any]] = None,
    user_speed: Optional[float] = None,
    user_pitch: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Intelligently blends text sentiment analysis with speaker acoustic DNA profile.
    Produces optimal generation hyperparameters.
    """
    opt = opt_weights or {}
    text_analysis = analyze_text_emotion(text)

    # 1. Resolve active emotion label
    if requested_emotion and requested_emotion.lower() not in ["auto", "none", ""]:
        resolved_emotion = requested_emotion.lower().strip()
    else:
        resolved_emotion = text_analysis["emotion"]

    base = BASE_EMOTION_TABLE.get(resolved_emotion, BASE_EMOTION_TABLE["neutral"])

    # 2. CFG Weight Modulation
    # Profile base CFG (e.g. 0.60) adjusted by emotion delta
    profile_cfg = float(opt.get("cfg_weight", 0.60))
    emotion_cfg_offset = base["cfg"] - BASE_EMOTION_TABLE["neutral"]["cfg"]
    final_cfg = float(np.clip(profile_cfg + emotion_cfg_offset, 0.45, 0.75))

    # 3. Exaggeration Modulation
    # Profile base exaggeration (e.g. 0.05) scaled by emotion intensity
    profile_exag = float(opt.get("exaggeration", 0.05))
    emotion_exag = base["exag"]
    intensity = text_analysis["intensity"]
    if resolved_emotion == "neutral":
        final_exag = profile_exag
    else:
        final_exag = float(np.clip(emotion_exag * max(0.5, intensity), 0.02, 0.35))

    # 4. Speed Modulation
    profile_speed = float(opt.get("speed_scale", 1.00))
    if user_speed and abs(user_speed - 1.0) > 0.05:
        final_speed = user_speed
    else:
        emotion_speed_factor = base["speed"]
        final_speed = float(np.clip(profile_speed * emotion_speed_factor, 0.75, 1.35))

    # 5. Pitch Modulation (Explicit user pitch only or emotion prosody cue)
    if user_pitch and abs(user_pitch) > 0.05:
        final_pitch = user_pitch
    else:
        final_pitch = 0.0  # Chatterbox handles pitch from reference audio

    resolved = {
        "resolved_emotion": resolved_emotion,
        "cfg_weight": round(final_cfg, 2),
        "exaggeration": round(final_exag, 2),
        "speed": round(final_speed, 2),
        "pitch": round(final_pitch, 2),
        "temperature": float(opt.get("temperature", 0.70)),
        "top_p": float(opt.get("top_p", 0.85)),
        "text_sentiment": text_analysis,
    }

    logger.info(
        f"🎭 Emotion Intelligence Resolved: emotion='{resolved_emotion}' (intensity={text_analysis['intensity']}), "
        f"cfg={resolved['cfg_weight']}, exag={resolved['exaggeration']}, speed={resolved['speed']}"
    )
    return resolved
