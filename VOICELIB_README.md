# VoiceLib — Project Memory

> **Location:** `c:\Users\dell\OneDrive\Desktop\IRIS\voicelib\`
> **Status:** ✅ v1 Built — TTS Voice Cloning
> **Last updated:** 2026-08-13

---

## What Is This?

VoiceLib is a full-stack AI voice-cloning TTS app.
Users upload short audio samples → derive a reusable "voice" → generate speech in that voice.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Frontend | React 18 + Vite + Tailwind CSS v3 |
| State | Zustand (persisted JWT to localStorage) |
| HTTP | Axios with JWT interceptor + error normalization |
| Backend | FastAPI 0.111 (Python 3.11, async-first) |
| TTS Engine | `pocket-tts` (Kyutai Labs, CPU-only) + MockTTSModel fallback |
| ORM | SQLAlchemy 2.0 async + Alembic |
| Database | PostgreSQL 15 |
| Storage | S3-compatible via boto3 (MinIO local / AWS S3 / R2) |
| Auth | JWT HS256 (python-jose) + bcrypt password hashing |
| Dev infra | Docker Compose (postgres + minio + backend + frontend) |

---

## Complete File Tree

```
voicelib/
├── VOICELIB_README.md          ← this file (at IRIS root)
├── README.md                   ← project docs
├── docker-compose.yml
├── .env.example
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py             # FastAPI app + lifespan
│       ├── config.py           # Pydantic BaseSettings
│       ├── db.py               # Async SQLAlchemy engine + sessions
│       ├── models.py           # ORM models (User, Voice, Generation) + Pydantic schemas
│       ├── auth.py             # JWT + bcrypt + get_current_user dependency
│       ├── tts.py              # Pocket TTS wrapper + LRU cache (50 voices, thread-safe)
│       ├── storage.py          # S3-compatible boto3 wrapper (upload/download/presign/delete)
│       ├── utils/
│       │   └── audio.py        # mutagen-based upload validation (MIME + size + duration)
│       └── routers/
│           ├── auth_router.py  # POST /auth/register, /auth/login, GET /auth/me
│           ├── voices_router.py# POST/GET/DELETE /voices
│           ├── generate_router.py # POST /generate, GET /generations
│           └── song_cover.py   # ← STUB ONLY — v2 entry point
│
└── frontend/
    ├── Dockerfile
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx             # BrowserRouter + all routes
        ├── index.css           # Tailwind + Ethereal Glass design system
        ├── api/
        │   ├── client.js       # Axios instance (JWT interceptor, error normalization)
        │   ├── auth.js
        │   ├── voices.js
        │   └── generate.js
        ├── store/
        │   ├── useAuthStore.js # Zustand (persisted to localStorage)
        │   └── useVoiceStore.js
        ├── pages/
        │   ├── Login.jsx
        │   ├── Register.jsx
        │   ├── VoiceLibrary.jsx    # Skeleton loading, empty state, stagger anims
        │   ├── Generate.jsx        # Voice selector, text area, history panel
        │   └── SongCover.jsx       # 🔒 Coming Soon — UI only
        └── components/
            ├── layout/
            │   ├── Navbar.jsx      # Floating glass pill, mobile overlay
            │   └── ProtectedRoute.jsx
            ├── ui/
            │   ├── Spinner.jsx
            │   ├── ErrorBanner.jsx
            │   ├── Modal.jsx       # Backdrop blur, ESC close, scroll lock
            │   └── Dropzone.jsx    # Drag-drop + client-side size/duration validation
            ├── voices/
            │   ├── VoiceCard.jsx   # Double-Bezel, inline delete confirm
            │   └── AddVoiceModal.jsx # State machine: idle→uploading→success
            └── generate/
                └── AudioPlayer.jsx # Waveform bars, scrub bar, download button
```

---

## Applied Skills

### high-end-visual-design (soft-skill)
- **Vibe:** Ethereal Glass — OLED black `#050505`, radial mesh gradient orbs
- **Layout:** Asymmetrical grid with staggered animation delays
- **Cards:** Double-Bezel (Doppelrand) — outer shell + inner core with `backdrop-blur`
- **Motion:** Custom `cubic-bezier(0.32,0.72,0,1)` spring transitions, no `ease-in-out`
- **Nav:** Floating glass pill detached from top with hamburger overlay
- **Typography:** Plus Jakarta Sans variable font
- **Entry anims:** `translate-y-16 blur-md opacity-0` → `translate-y-0 blur-0 opacity-100`

### fastapi-pro
- Async-first with `asyncpg` and SQLAlchemy 2.0 async sessions
- Pydantic V2 schemas separate from ORM models
- Lifespan events for DB init, bucket creation, TTS model loading
- `Annotated` dependency injection everywhere
- Structured logging with `logging.getLogger`
- Proper HTTP status codes (201, 204, 400, 403, 404, 413, 415, 422, 500)

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| TTS model loaded once at startup | Avoid 30-60s cold start per request |
| LRU cache (max 50 voices, thread-safe) | Prevent unbounded memory growth; thread-safe for uvicorn threadpool |
| Cache miss re-derives from storage | Server restarts don't lose voice state — only adds latency once |
| `storage.py` isolated from TTS | Future `vocal_conversion.py` can import storage without touching TTS |
| `song_cover.py` stub | Clear v2 entry point; v1 router is no-op (`router` with no routes) |
| Voice schema has no `tts_*` prefixes | Voice is generic — works for TTS and future song cover feature |
| MockTTSModel fallback | Full dev+testing workflow without pocket-tts or PyTorch installed |
| `consent_confirmed` stored in DB | Audit trail; rejected at API level if False |

---

## Quick Start & Reminders

> 📌 **Supabase Setup Saved**: Complete step-by-step Supabase connection instructions for database & file storage are saved at [`voicelib/SUPABASE_SETUP_GUIDE.md`](file:///c:/Users/dell/OneDrive/Desktop/IRIS/voicelib/SUPABASE_SETUP_GUIDE.md).

```bash
cd c:\Users\dell\OneDrive\Desktop\IRIS\voicelib
# Follow SUPABASE_SETUP_GUIDE.md to update .env
```

**Dev without Docker (TTS mock):**
```bash
# Backend
cd backend
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
set VOICELIB_USE_MOCK_TTS=true
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

---

## v2 Entry Points

- **Backend:** [`backend/app/routers/song_cover.py`](./voicelib/backend/app/routers/song_cover.py) — detailed implementation notes inside
- **Frontend:** [`frontend/src/pages/SongCover.jsx`](./voicelib/frontend/src/pages/SongCover.jsx) — Coming Soon UI

The `Voice` table, `storage.py`, and `tts.py` require **no changes** for v2.
