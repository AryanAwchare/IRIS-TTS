# Project Memory & Context Log — IRIS / VoiceLib

> **Project Name:** IRIS (VoiceLib)  
> **Status:** Active / v1.0 Released  
> **Last Updated:** August 22, 2026  
> **Purpose:** Keeps AI assistants synchronized on active codebase state, architecture, completed tasks, and upcoming work.  

---

## 1. Quick Context Summary

* **What is IRIS?** A full-stack AI Voice-Cloning and Text-to-Speech (TTS) web application. Users register, upload short voice audio samples (3-30s), extract voice embeddings via Kyutai Pocket-TTS, and generate custom speech audio.
* **Root Location:** `c:\Users\dell\OneDrive\Desktop\IRIS\`
* **Source App Location:** `c:\Users\dell\OneDrive\Desktop\IRIS\voicelib\`
* **Backend Tech:** FastAPI 0.111, Python 3.11, PyTorch CPU (`pocket-tts`), SQLAlchemy 2.0 Async, PostgreSQL 15, MinIO / S3 Storage (`boto3`), JWT Auth (`bcrypt` + `python-jose`), `mutagen` audio validation.
* **Frontend Tech:** React 18, Vite, Tailwind CSS v3 (Obsidian Ethereal Glass theme), Zustand (persisted state), Axios with JWT interceptor, custom canvas waveform player.

---

## 2. Implemented Features & Component Status Matrix

| Component / Feature | Path | Status | Verification / Notes |
|---|---|---|---|
| **Document Suite** | `document/*.md` | ✅ Complete | PRD, Architecture, Rules, Phases, Design, Memory created. |
| **FastAPI Core App** | `voicelib/backend/app/main.py` | ✅ Complete | CORS middleware, lifespan events, exception handlers configured. |
| **Config Loader** | `voicelib/backend/app/config.py` | ✅ Complete | Pydantic BaseSettings for DB, S3, JWT, and upload limits. |
| **Async Database ORM** | `voicelib/backend/app/models.py` | ✅ Complete | `User`, `Voice`, `Generation` ORM schemas defined. |
| **Auth System & JWT** | `voicelib/backend/app/auth.py` | ✅ Complete | Register, login, current user dependency with bcrypt hashing. |
| **TTS Engine Suite** | `voicelib/backend/app/tts.py` | ✅ Complete | GPT-SoVITS v3 / Pocket-TTS with Dynamic Timbre & Formant Morphing on CPU. |
| **Acoustic Timbre Morpher** | `voicelib/backend/app/utils/timbre_morpher.py` | ✅ Complete | Dynamic F0, LPC Formant (F1-F3) matching, spectral tilt, and harmonic warmth. |
| **S3 Storage Client** | `voicelib/backend/app/storage.py` | ✅ Complete | Presigned URL generation, file upload, object deletion wrapper. |
| **Audio Validation** | `voicelib/backend/app/utils/audio.py` | ✅ Complete | `mutagen` header verification for duration and MIME format. |
| **Frontend Auth SPA** | `voicelib/frontend/src/pages/Login.jsx` | ✅ Complete | Login & Register forms connected to Zustand store & local storage. |
| **Voice Library & Studio** | `voicelib/frontend/src/pages/VoiceLibrary.jsx` | ✅ Complete | Voice cards grid, drag-drop uploader, per-voice Tuning & Accent Calibration modal (`VoiceTuneModal.jsx`). |
| **Speech Studio UI** | `voicelib/frontend/src/pages/Generate.jsx` | ✅ Complete | Voice selector, auto-loaded calibration profiles, script editor canvas, waveform player, history. |
| **Audio Waveform Player**| `voicelib/frontend/src/components/generate/AudioPlayer.jsx` | ✅ Complete | Custom canvas audio spectrum visualizer, scrubber, WAV downloader. |
| **Docker Composition** | `voicelib/docker-compose.yml` | ✅ Complete | Postgres, MinIO, Backend, Frontend multi-container config. |
| **Supabase Setup** | `voicelib/SUPABASE_SETUP_GUIDE.md` | ✅ Complete | Migration instructions for cloud PostgreSQL database hosting. |
| **Song Cover Studio** | `voicelib/frontend/src/pages/SongCover.jsx` | 🔒 Stub / v2 | Coming soon teaser UI for Phase 7 AI Song Cover pipeline. |

---

## 3. Environment Variables Reference (`.env`)

```env
# Database Credentials
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/voicelib

# S3 / MinIO Object Storage
STORAGE_ENDPOINT_URL=http://localhost:9000
STORAGE_ACCESS_KEY=minioadmin
STORAGE_SECRET_KEY=minioadmin
STORAGE_BUCKET_NAME=voicelib
STORAGE_REGION=us-east-1

# JWT Auth Configuration
JWT_SECRET_KEY=super-secret-key-change-this-in-production-min-32-chars
ACCESS_TOKEN_EXPIRE_MINUTES=10080

# Audio & Synthesis Constraints
MAX_UPLOAD_SIZE_BYTES=20971520
MAX_SAMPLE_DURATION_SECONDS=30
VOICELIB_USE_MOCK_TTS=false

# Frontend API URL
VITE_API_BASE_URL=http://localhost:8000
```

---

## 4. Known Issues & Tech Debt Log

1. **CPU Inference Latency:** `pocket-tts` running on standard CPU hosts takes ~2-3 seconds for 10-word prompts. GPU hardware acceleration (CUDA/ROCm) recommended for production.
2. **Presigned URL Expiry Handling:** Long-lived audio links in generation history need token auto-refresh handling when presigned S3 URLs expire.
3. **Song Cover Pipeline:** Backend router (`routers/song_cover.py`) is currently a stub awaiting Demucs integration in Phase 7.

---

## 5. Next Action Items for AI Assistants

1. **Test Suite Implementation:** Write `pytest` unit tests for `/auth/*`, `/voices/*`, and `/generate/*` routers under `voicelib/backend/tests/`.
2. **Rate Limiting:** Add slowapi rate limiting middleware to FastAPI `/generate` endpoint to protect server CPU from overload.
3. **Phase 7 Demucs Pipeline:** Scaffold vocal separation service for v2 Song Cover feature.
