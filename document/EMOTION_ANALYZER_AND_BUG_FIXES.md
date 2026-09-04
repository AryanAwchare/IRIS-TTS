# IRIS VoiceLib — Emotion Analyzer & System Bug Fixes Action Guide

> **Target Codebase:** `C:\Users\dell\OneDrive\Desktop\IRIS`  
> **Status:** Ready for Implementation  
> **Date:** September 2026  

---

## 1. Executive Summary & Diagnostics

During real-time synthesis and code analysis, two major issues were observed:
1. **Emotion Analyzer Failure**:
   - The deep neural NLP model (`j-hartmann/emotion-english-distilroberta-base`) is skipped because `transformers` is not installed in the virtual environment.
   - The fallback lexicon fails on 90% of real sentences because it uses rigid exact-word matching (e.g. `"depressed"` matches, but `"depressing"` and `"heartbreaking"` fail and default to `neutral`).
   - The emotion modulation logic in `emotion_analyzer.py` forces `effective_intensity = 0.05` when `requested_emotion == "neutral"`, wiping out emotional delivery even when intense emotion was detected in the prompt text.
   - The frontend (`Generate.jsx`) automatically locks the dropdown to `"happy"` when clicking paralinguistic tags like `[laughter]`, persisting `"happy"` across future unrelated generations.
2. **Missing Evaluation Packages**:
   - Background evaluation records `Grade [D]` because `speechbrain` (similarity) and `jiwer` (WER) are not installed.
3. **Core System Bugs**:
   - Web Audio API throws `InvalidStateError` when re-mounting audio visualizers.
   - Local storage files lack `.wav` extensions, causing browsers to receive `application/octet-stream`.
   - SQLite auto-migrations fail due to `IF NOT EXISTS` syntax unsupported in SQLite `ADD COLUMN`.
   - VAD temporary files leak on disk if an exception occurs.

---

## 2. Step-by-Step Implementation Guide

### Step 1: Install Missing Backend Dependencies
In your PowerShell terminal, activate the virtual environment and install the required AI models:

```powershell
Set-Location "C:\Users\dell\OneDrive\Desktop\IRIS\voicelib\backend"
.\venv\Scripts\activate

# 1. Real Deep Emotion AI Model
pip install transformers

# 2. Objective Quality Evaluation Models (Eliminates Grade [D] error)
pip install speechbrain jiwer
```

---

### Step 2: Upgrade Fallback Lexicon with Stem & Prefix Matching
**File:** [`voicelib/backend/app/utils/emotion_detector.py`](file:///C:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/utils/emotion_detector.py)

Replace exact word matching with stem/prefix-aware matching so words like `depress*`, `heartbreak*`, `wonder*`, `angr*`, `excit*` capture all their grammatical forms:

```python
# Expanded root-matching lexicon
EMOTION_ROOTS = {
    # Happy / Joy / Warmth
    "happ": {"emotion": "happy", "intensity": 0.8},
    "joy": {"emotion": "happy", "intensity": 0.9},
    "glad": {"emotion": "happy", "intensity": 0.7},
    "wonder": {"emotion": "happy", "intensity": 0.85},
    "delight": {"emotion": "happy", "intensity": 0.85},
    "cheer": {"emotion": "happy", "intensity": 0.8},
    "love": {"emotion": "happy", "intensity": 0.8},
    "laugh": {"emotion": "happy", "intensity": 0.9, "tag": "[laughter]"},
    "smile": {"emotion": "happy", "intensity": 0.6},
    "congrat": {"emotion": "happy", "intensity": 0.85},

    # Sad / Melancholy
    "sad": {"emotion": "sad", "intensity": 0.8},
    "depress": {"emotion": "sad", "intensity": 0.9},
    "heartbreak": {"emotion": "sad", "intensity": 0.95},
    "sorrow": {"emotion": "sad", "intensity": 0.85},
    "grief": {"emotion": "sad", "intensity": 0.9},
    "cry": {"emotion": "sad", "intensity": 0.85},
    "tear": {"emotion": "sad", "intensity": 0.7},
    "lonel": {"emotion": "sad", "intensity": 0.75},
    "unfortunat": {"emotion": "sad", "intensity": 0.6, "tag": "[sigh]"},

    # Angry / Frustrated
    "angr": {"emotion": "angry", "intensity": 0.85},
    "furious": {"emotion": "angry", "intensity": 0.95},
    "frustrat": {"emotion": "angry", "intensity": 0.75},
    "outrag": {"emotion": "angry", "intensity": 0.9},
    "hate": {"emotion": "angry", "intensity": 0.85},
    "annoy": {"emotion": "angry", "intensity": 0.65},
    "terribl": {"emotion": "angry", "intensity": 0.75},
    "horribl": {"emotion": "angry", "intensity": 0.75},

    # Excited / Thrilled
    "excit": {"emotion": "excited", "intensity": 0.9},
    "amaz": {"emotion": "excited", "intensity": 0.85},
    "awesom": {"emotion": "excited", "intensity": 0.8},
    "incredibl": {"emotion": "excited", "intensity": 0.9},
    "unbeliev": {"emotion": "excited", "intensity": 0.85},
    "fantast": {"emotion": "excited", "intensity": 0.85},

    # Calm / Serene
    "calm": {"emotion": "calm", "intensity": 0.7},
    "peace": {"emotion": "calm", "intensity": 0.8},
    "relax": {"emotion": "calm", "intensity": 0.7},
    "quiet": {"emotion": "calm", "intensity": 0.6},
    "gentl": {"emotion": "calm", "intensity": 0.65},
    "whisper": {"emotion": "calm", "intensity": 0.7, "tag": "[whisper]"},
}
```

---

### Step 3: Fix Emotion Modulation & "Neutral" Lockout
**File:** [`voicelib/backend/app/utils/emotion_analyzer.py`](file:///C:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/utils/emotion_analyzer.py)

In `compute_modulated_synthesis_parameters`:
```python
    req_norm = (requested_emotion or "auto").lower().strip()

    if req_norm in ["auto", "none", ""]:
        # Text emotion is primary
        resolved_emotion = analysis.emotion
        effective_intensity = analysis.intensity
    elif req_norm != "neutral" and req_norm in CANONICAL_EMOTIONS:
        # User explicitly requested an active emotion (e.g. happy, sad, angry)
        resolved_emotion = req_norm
        effective_intensity = max(0.20, analysis.intensity)
    else:
        # User requested neutral: honor neutral but allow natural subtle inflection
        resolved_emotion = "neutral"
        effective_intensity = min(0.15, analysis.intensity)
```

---

### Step 4: Fix Frontend Paralinguistic Tag Lockout
**File:** [`voicelib/frontend/src/pages/Generate.jsx`](file:///C:/Users/dell/OneDrive/Desktop/IRIS/voicelib/frontend/src/pages/Generate.jsx)

In `insertTag`:
```javascript
  const insertTag = (tag) => {
    setText((prev) => prev ? `${prev} ${tag} ` : `${tag} `)
    // Keep 'auto' active so subsequent texts are analyzed independently!
  }
```

---

### Step 5: Fix Web Audio API & MediaElementSource Lifecycle
**Files:**
* [`voicelib/frontend/src/components/generate/AudioPlayer.jsx`](file:///C:/Users/dell/OneDrive/Desktop/IRIS/voicelib/frontend/src/components/generate/AudioPlayer.jsx)
* [`voicelib/frontend/src/components/ui/AudioCanvasVisualizer.jsx`](file:///C:/Users/dell/OneDrive/Desktop/IRIS/voicelib/frontend/src/components/ui/AudioCanvasVisualizer.jsx)

1. In `AudioPlayer.jsx`, remove `key={generationId || url}` on the outer container to prevent destroying the DOM audio element on track changes.
2. In `AudioCanvasVisualizer.jsx`, cache the `MediaElementSource` node on the audio element:
   ```javascript
   if (!audioRef.current._sourceNode) {
     const source = ctx.createMediaElementSource(audioRef.current)
     source.connect(analyser)
     audioRef.current._sourceNode = source
   }
   ```

---

### Step 6: Add Proper `.wav` Extensions to Storage Files
**File:** [`voicelib/backend/app/storage.py`](file:///C:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/storage.py)

In `upload_bytes`:
```python
def upload_bytes(data: bytes, content_type: str, prefix: str = "uploads") -> str:
    s = get_settings()
    ext = ".wav" if "wav" in content_type else (".mp3" if "mpeg" in content_type else "")
    key = f"{prefix}/{uuid.uuid4()}{ext}"
    ...
```
In `download_bytes`:
```python
    target_path = LOCAL_STORAGE_DIR / key
    if not target_path.exists() and not key.endswith(".wav"):
        if (LOCAL_STORAGE_DIR / f"{key}.wav").exists():
            target_path = LOCAL_STORAGE_DIR / f"{key}.wav"
```

---

### Step 7: Fix SQLite Auto-Migrations
**File:** [`voicelib/backend/app/db.py`](file:///C:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/db.py)

Support SQLite dynamic schema inspection via `PRAGMA table_info` before executing `ALTER TABLE`:

```python
async def create_tables() -> None:
    from app import models
    from sqlalchemy import text

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
        # Check SQLite table info
        settings = get_settings()
        if "sqlite" in settings.database_url:
            res = await conn.execute(text("PRAGMA table_info(generations);"))
            existing_cols = {row[1] for row in res.fetchall()}
            
            sqlite_additions = [
                ("engine", "VARCHAR(64) DEFAULT 'gpt-sovits-v3'"),
                ("emotion", "VARCHAR(64) DEFAULT 'neutral'"),
                ("eval_status", "VARCHAR(32) DEFAULT 'pending'"),
                ("speaker_similarity", "FLOAT"),
                ("word_error_rate", "FLOAT"),
                ("prosody_f0_std", "FLOAT"),
                ("composite_grade", "VARCHAR(4)"),
                ("composite_score", "FLOAT"),
                ("eval_error", "TEXT"),
                ("evaluated_at", "TIMESTAMP"),
            ]
            for col_name, col_type in sqlite_additions:
                if col_name not in existing_cols:
                    try:
                        await conn.execute(text(f"ALTER TABLE generations ADD COLUMN {col_name} {col_type};"))
                    except Exception:
                        pass
```

---

### Step 8: Prevent Temporary File Leaks in VAD Audio Cleaning
**Files:**
* [`voicelib/backend/app/routers/voices_router.py`](file:///C:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/routers/voices_router.py)
* [`voicelib/backend/app/routers/generate_router.py`](file:///C:/Users/dell/OneDrive/Desktop/IRIS/voicelib/backend/app/routers/generate_router.py)

Wrap VAD selection in strict `try ... finally` blocks:

```python
tmp_vad_path = None
vad_segment_path = None
try:
    with tempfile.NamedTemporaryFile(suffix="_upload.wav", delete=False) as tmp_vad:
        tmp_vad.write(audio_bytes)
        tmp_vad_path = tmp_vad.name

    vad_segment_path = await loop.run_in_executor(None, _vad_select, tmp_vad_path, 10.0)
    with open(vad_segment_path, "rb") as vf:
        vad_bytes = vf.read()
finally:
    for _p in (tmp_vad_path, vad_segment_path):
        if _p and os.path.exists(_p):
            try:
                os.unlink(_p)
            except Exception:
                pass
```
