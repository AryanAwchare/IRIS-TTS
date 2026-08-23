# Project Implementation Phases — IRIS / VoiceLib

> **Project Name:** IRIS (VoiceLib)  
> **Status:** Phase 1 - 5 Completed (v1.0 Ready) | Phase 6 In Progress | Phase 7 Planned  
> **Last Updated:** August 2026  

---

## Roadmap Overview

```
 ┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
 │    Phase 1     │ ──► │    Phase 2     │ ──► │    Phase 3     │ ──► │    Phase 4     │
 │ Scaffold & DB  │     │ JWT Auth & UI  │     │ Voice Uploads  │     │ Pocket-TTS Engine
 └────────────────┘     └────────────────┘     └────────────────┘     └────────────────┘
                                                                               │
 ┌────────────────┐     ┌────────────────┐     ┌────────────────┐              │
 │    Phase 7     │ ◄── │    Phase 6     │ ◄── │    Phase 5     │ ◄────────────┘
 │ AI Song Covers │     │ Hardening & E2E│     │ Studio & Player│
 └────────────────┘     └────────────────┘     └────────────────┘
```

---

## Detailed Phase Breakdown

### ✅ Phase 1: Environment Scaffolding, Core DB & Storage Layer
**Goal:** Establish backend foundation, PostgreSQL schema, S3 storage client, and Docker Compose development environment.

* **Key Deliverables:**
  * FastAPI base application setup in `backend/app/main.py`.
  * Environment config parser using Pydantic `BaseSettings` (`app/config.py`).
  * Async SQLAlchemy 2.0 database engine & base session dependency (`app/db.py`).
  * Initial ORM database models (`User`, `Voice`, `Generation`) in `app/models.py`.
  * S3 / MinIO storage wrapper (`app/storage.py`) supporting uploads, presigned URLs, and deletes.
  * `docker-compose.yml` orchestrating PostgreSQL, MinIO, Backend, and Frontend containers.

---

### ✅ Phase 2: Authentication System & User Management
**Goal:** Implement secure user registration, JWT token authentication, and frontend session persistence.

* **Key Deliverables:**
  * Password hashing utilities using `bcrypt` and JWT encoder/decoder (`app/auth.py`).
  * Authentication API routes (`/auth/register`, `/auth/login`, `/auth/me`).
  * React frontend auth pages (`Login.jsx`, `Register.jsx`).
  * Zustand auth store (`useAuthStore.js`) with JWT token local storage persistence.
  * Protected route guard component (`ProtectedRoute.jsx`) for SPA client routing.

---

### ✅ Phase 3: Voice Sample Upload & Profile Pipeline
**Goal:** Enable users to upload, validate, store, and manage custom audio voice profiles.

* **Key Deliverables:**
  * Audio file validator (`app/utils/audio.py`) verifying file MIME headers, duration (3-30s), and size (<=20MB).
  * Voice management endpoints (`POST /voices`, `GET /voices`, `DELETE /voices/{id}`).
  * Frontend Drag-and-drop file uploader (`Dropzone.jsx`) with real-time format validation.
  * Voice creation modal (`AddVoiceModal.jsx`) requiring explicit legal consent verification.
  * Voice library grid (`VoiceLibrary.jsx`) rendering interactive cards with inline sample playback and deletion cascades.

---

### ✅ Phase 4: Neural TTS Inference Engine Integration
**Goal:** Integrate Kyutai Pocket-TTS for zero-shot speech synthesis with thread-safe LRU caching and dev fallback.

* **Key Deliverables:**
  * `pocket-tts` wrapper (`app/tts.py`) managing model loading and execution.
  * Thread-safe LRU in-memory cache supporting up to 50 active voice embeddings.
  * Synthetic fallback engine (`MockTTSModel`) generating clean test tone WAVs when `VOICELIB_USE_MOCK_TTS=true`.
  * Speech synthesis endpoint (`POST /generate`) and history query endpoint (`GET /generations`).

---

### ✅ Phase 5: Synthesis Studio & Audio Waveform Player UI
**Goal:** Create a high-end visual speech generator canvas with custom waveform playback and history management.

* **Key Deliverables:**
  * Interactive Speech Generator interface (`Generate.jsx`) with dynamic voice selector and prompt editor.
  * Custom audio player component (`AudioPlayer.jsx`) featuring audio scrubbing, dynamic canvas waveforms, duration display, and WAV file download.
  * Real-time generation history list with voice metadata badges and playback controls.
  * Ethereal Glass design system application (`index.css` & `tailwind.config.js`).

---

### 🔄 Phase 6: Production Hardening, Cloud Deployment & Testing
**Goal:** Prepare application for production deployment, performance optimization, and end-to-end testing.

* **Tasks & Tasks Remaining:**
  * [x] Supabase integration guide & schema migration setup (`SUPABASE_SETUP_GUIDE.md`).
  * [ ] Automated backend unit & integration tests with `pytest` for auth, voices, and generate routes.
  * [ ] Presigned URL caching and CDN edge distribution strategy.
  * [ ] Rate-limiting middleware for `/generate` endpoints.
  * [ ] Docker build optimization & multi-stage production Dockerfiles.

---

### ⏳ Phase 7 (v2 Roadmap): AI Song Cover & Vocal Conversion Engine
**Goal:** Expand IRIS into an AI Song Cover workstation.

* **Planned Features:**
  * Audio vocal separation using Demucs / UVR5 models.
  * Vocal timbre & pitch conversion leveraging existing `Voice` table embeddings.
  * Audio mixer combining converted vocals with backing instrumental tracks.
  * Frontend Song Cover studio (`SongCover.jsx` transition from stub UI to full functional canvas).
