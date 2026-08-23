# OmniVoice (646+ Languages) & Cross-Lingual Voice Translation Roadmap Plan

> **Document Status:** 📋 Planned / Post-Bugfix Roadmap  
> **Target Engine:** `k2-fsa/OmniVoice` (Apache 2.0)  
> **Target Capabilities:** 646+ Language Speech Synthesis, Zero-Shot Voice Cloning, and Cross-Lingual Voice Translation (e.g. English text → Spanish/Hindi cloned speech).  

---

## 1. Executive Overview

**OmniVoice** is an open-source, non-autoregressive diffusion TTS model developed by the `k2-fsa` team (creators of next-gen Kaldi and Sherpa). It supports **646+ world languages** and allows zero-shot voice cloning across any supported language from a short 3–10s reference audio sample.

This document preserves the architectural blueprint and implementation steps to integrate OmniVoice into **IRIS (VoiceLib)** once core audio library calibration and voice stabilization are finalized.

---

## 2. Core Features & Use Cases

### A. Fixing Language Script Mismatches (Hindi, Spanish, French, Japanese, Arabic)
* **Problem:** English-only tokenizers (e.g. standard Pocket-TTS or English Whisper/VITS) fail or produce gibberish noises when given Devanagari (`नमस्ते`), Cyrillic, Arabic, or CJK characters.
* **Solution:** OmniVoice natively supports 646+ languages with proper phonetic tokenization, producing fluent native pronunciation for Hindi, Spanish, Mandarin, French, German, etc.

### B. Cross-Lingual Voice Translation Pipeline
* **Flow:**
  1. User types in their native language (e.g. English: *"Hello, welcome to my AI voice studio!"*).
  2. The system translates the text to the selected target language (e.g. Spanish: *"¡Hola, bienvenidos a mi estudio de voz!"* or Hindi: *"नमस्ते, मेरे एआई वॉयस स्टूडियो में आपका स्वागत है!"*).
  3. OmniVoice takes the translated text + the user's reference voice sample (`.wav`).
  4. **Output:** The user's cloned voice speaking fluent Spanish or Hindi with native accent and intonation.

---

## 3. Architecture & System Flow

```
                      User Prompt & Reference Voice (.wav)
                                      │
                   ┌──────────────────▼──────────────────┐
                   │   Language Detector & Auto-Router   │
                   │   - Detects input script & language │
                   │   - Validates engine compatibility   │
                   └──────────────────┬──────────────────┘
                                      │
                 Is Auto-Translate ON or Language Mismatch?
                                ├── YES ──► Neural Translator (deep-translator / NMT)
                                │           (e.g., "Hello" -> "नमस्ते" / "¡Hola!")
                                └── NO ───► Keep Original Text
                                      │
                   ┌──────────────────▼──────────────────┐
                   │       Engine Selector & Router      │
                   ├──────────────────┬──────────────────┤
                   │                  │                  │
        ┌──────────▼─────────┐ ┌──────▼─────────┐ ┌──────▼─────────┐
        │ 1. Pocket TTS      │ │ 2. GPT-SoVITS  │ │ 3. OmniVoice   │
        │ - Fast English CPU │ │ - Expressive   │ │ - 646+ Langs   │
        │ - Timbre Morpher   │ │   Colab/GPU    │ │ - Cross-Lingual│
        └──────────┬─────────┘ └──────┬─────────┘ └──────┬─────────┘
                   │                  │                  │
                   └──────────────────┼──────────────────┘
                                      │
                    High-Fidelity Cloned Audio Stream (.wav)
```

---

## 4. Planned Implementation Components

### 4.1 Backend Engine & Translation Modules
1. **`app/utils/translator.py`**:
   - Lightweight text translation wrapper using `deep-translator` (Google Translate / LibreTranslate backend) with local in-memory caching.
   - Converts source text into ISO language codes (`hi`, `es`, `fr`, `de`, `ja`, `zh`, `ar`, `pt`, `ru`, etc.).
2. **`app/tts_engines/omnivoice_engine.py`**:
   - `OmniVoiceEngine` class inheriting from `BaseTTSEngine`.
   - Connects to the Colab GPU server `/synthesize_omnivoice` endpoint and handles zero-shot prompt injection.
3. **`app/models.py` & `app/routers/generate_router.py`**:
   - Add `target_lang: Optional[str] = "en"` and `auto_translate: Optional[bool] = False` to `GenerateRequest`.
   - Route multilingual text dynamically to prevent gibberish outputs.

### 4.2 Colab GPU Server Integration (`voice_cloning_colab_fixed.py`)
```python
# Colab setup
!pip install omnivoice

from omnivoice import OmniVoice

omni_model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0")

@colab_app.post("/synthesize_omnivoice")
async def synthesize_omnivoice(
    ref_audio: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form("en"),
):
    ...
```

### 4.3 Frontend Studio UI (`Generate.jsx`)
* **Target Language Dropdown:** (English, Hindi, Spanish, French, German, Japanese, Mandarin, Arabic, Portuguese, etc.).
* **"Auto-Translate Prompt" Toggle Switch:** Automatically translates prompt text before synthesis.
* **Engine Tab Selector:** Seamless switching between `Neural Voice (GPT-SoVITS)`, `OmniVoice 646 (Multilingual)`, and `Standard Fast (Pocket CPU)`.

---

## 5. Execution Checklist (When Ready)

- [ ] Install `deep-translator` in backend virtual environment.
- [ ] Add `OmniVoice` server handler in `voice_cloning_colab_fixed.py`.
- [ ] Scaffold `app/tts_engines/omnivoice_engine.py` and `app/utils/translator.py`.
- [ ] Add Language Selector dropdown and Auto-Translate toggle in `Generate.jsx`.
- [ ] Test cross-lingual synthesis for English → Hindi, English → Spanish, and English → Japanese.
