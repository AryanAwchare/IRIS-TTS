# System Architecture Document — IRIS / VoiceLib

> **Project Name:** IRIS (VoiceLib)  
> **Architecture Version:** 1.0.0  
> **Last Updated:** August 2026  

---

## 1. High-Level Architecture Overview

IRIS is structured as a decoupled multi-tier micro-architecture comprising an single-page React client, an asynchronous FastAPI application server, PyTorch neural inference engines, PostgreSQL metadata storage, and S3-compatible object storage.

```
                           ┌────────────────────────────────────────┐
                           │          Client Tier (Vite SPA)        │
                           │   React 18 + Zustand + Tailwind CSS    │
                           └───────────────────┬────────────────────┘
                                               │
                                       HTTP / HTTPS (JWT)
                                               │
                           ┌───────────────────▼────────────────────┐
                           │       Application Tier (FastAPI)       │
                           │   Async Routers, Auth & Controllers    │
                           └────────┬──────────┬───────────┬────────┘
                                    │          │           │
           ┌────────────────────────┘          │           └────────────────────────┐
           │                                   │                                    │
┌──────────▼──────────┐             ┌──────────▼──────────┐              ┌──────────▼──────────┐
│  Database (Postgres)│             │  Neural Engine      │              │ Object Storage (S3) │
│ Users, Voices, Gens │             │ Kyutai Pocket-TTS   │              │ Raw Audio & Output  │
│ SQLAlchemy 2.0 Async│             │ Thread-safe Cache   │              │ MinIO / AWS / R2    │
└─────────────────────┘             └─────────────────────┘              └─────────────────────┘
```

---

## 2. Technical Stack Matrix

| Tier | Component | Technology | Rationale |
|---|---|---|---|
| **Frontend UI** | Framework | React 18 + Vite | Lightning-fast HMR, component modularity, optimal bundle size. |
| **Frontend Styling** | Styling Engine | Tailwind CSS v3 | Utility-first obsidian glassmorphism design system. |
| **State & HTTP** | State & Network | Zustand + Axios | Zero-boilerplate global state persistence + JWT token interceptors. |
| **Backend API** | Web Framework | FastAPI 0.111 (Python 3.11) | Native `asyncio` support, auto OpenAPI documentation, Pydantic V2. |
| **AI/TTS Engine** | Neural Synthesis | `pocket-tts` (Kyutai) | High-quality CPU-friendly zero-shot TTS engine. |
| **TTS Cache** | Memory Cache | Custom LRU Cache | In-memory caching for 50 loaded voice models to eliminate disk cold-starts. |
| **Database** | RDBMS | PostgreSQL 15 | Relational integrity, JSONB support, async driver via `asyncpg`. |
| **ORM / Migration** | Database Tooling | SQLAlchemy 2.0 Async + Alembic | Modern Python async ORM with type safe query building. |
| **File Storage** | Object Store | S3 API (boto3) / MinIO | Scalable audio file hosting with presigned URL capabilities. |
| **DevOps** | Containerization | Docker & Docker Compose | Multi-container environment orchestration for reproducible deployments. |

---

## 3. Project Directory & File Tree

```
c:\Users\dell\OneDrive\Desktop\IRIS\
├── document/                       # Formal Project Documentation
│   ├── PRD.md                      # Project Requirements Document
│   ├── Architecture.md             # System Architecture & Flow (this file)
│   ├── Rules.md                    # AI Coding Boundaries & Standards
│   ├── Phases.md                   # Implementation Phases & Timeline
│   ├── Design.md                   # Design System & UI Specifications
│   └── Memory.md                   # Active Context & Development Progress
│
├── VOICELIB_README.md              # Technical summary & file index
│
└── voicelib/                       # Source Code Root
    ├── docker-compose.yml          # Local infra orchestration (Postgres, MinIO, Backend, Frontend)
    ├── .env.example                # Template for environment credentials
    ├── test_app.html               # Standalone frontend verification page
    │
    ├── backend/                    # FastAPI Server Application
    │   ├── Dockerfile              # Python 3.11 container manifest
    │   ├── requirements.txt        # Python dependency manifest
    │   └── app/
    │       ├── __init__.py
    │       ├── main.py             # Application lifespan, middleware & CORS configuration
    │       ├── config.py           # Pydantic BaseSettings environment loader
    │       ├── db.py               # Async SQLAlchemy engine & session maker
    │       ├── models.py           # ORM Database Models & Pydantic validation schemas
    │       ├── auth.py             # JWT generation, verification & bcrypt password utilities
    │       ├── tts.py              # Pocket-TTS wrapper with LRU cache & Mock TTS fallback
    │       ├── storage.py          # S3-compatible boto3 client wrapper
    │       ├── utils/
    │       │   └── audio.py        # Mutagen audio validation (MIME, duration, size)
    │       └── routers/
    │           ├── auth_router.py  # Endpoints: /auth/register, /auth/login, /auth/me
    │           ├── voices_router.py# Endpoints: POST /voices, GET /voices, DELETE /voices/{id}
    │           ├── generate_router.py # Endpoints: POST /generate, GET /generations
    │           └── song_cover.py   # Stub endpoint for v2 AI Song Cover engine
    │
    └── frontend/                   # React Single-Page Application
        ├── Dockerfile              # Node.js build & Nginx preview container
        ├── package.json            # Node dependencies
        ├── vite.config.js          # Vite build & proxy settings
        ├── tailwind.config.js      # Custom theme, colors & glassmorphism utilities
        ├── postcss.config.js       # PostCSS plugins
        ├── index.html              # HTML entry point with font preloads
        └── src/
            ├── main.jsx            # React root mount
            ├── App.jsx             # Router definition & protected route boundaries
            ├── index.css           # Tailwind directives & global theme CSS
            ├── api/
            │   ├── client.js       # Axios client with JWT interceptor & error normalization
            │   ├── auth.js         # Auth API calls
            │   ├── voices.js       # Voice management API calls
            │   └── generate.js     # Speech generation API calls
            ├── store/
            │   ├── useAuthStore.js # Zustand store for Auth state & user persistence
            │   └── useVoiceStore.js# Zustand store for Voice profiles & selection
            ├── pages/
            │   ├── Login.jsx       # User authentication page
            │   ├── Register.jsx    # New account registration page
            │   ├── VoiceLibrary.jsx# Voice profile grid, sample playback & uploader modal
            │   ├── Generate.jsx    # Synthesis studio, text canvas & generation history
            │   └── SongCover.jsx   # Feature teaser page for v2 AI Song Covers
            └── components/
                ├── layout/
                │   ├── Navbar.jsx  # Floating glass navigation bar
                │   └── ProtectedRoute.jsx # Auth route wrapper
                ├── ui/
                │   ├── Spinner.jsx # Loading spinners
                │   ├── ErrorBanner.jsx # Global error alerts
                │   ├── Modal.jsx   # Reusable accessible glass modal dialog
                │   └── Dropzone.jsx# Interactive drag-drop audio uploader
                ├── voices/
                │   ├── VoiceCard.jsx # Voice sample card with inline delete confirmation
                │   └── AddVoiceModal.jsx # Voice creation wizard modal
                └── generate/
                    └── AudioPlayer.jsx # Customized audio player with waveform visualizer
```

---

## 4. Database Schema & Data Models

```
 ┌──────────────────────────────────┐        ┌──────────────────────────────────┐
 │              users               │        │              voices              │
 ├──────────────────────────────────┤        ├──────────────────────────────────┤
 │ id (UUID / Int, PK)              │1      *│ id (UUID / Int, PK)              │
 │ email (VARCHAR, Unique, Indexed) ├───────►│ user_id (FK -> users.id)        │
 │ hashed_password (VARCHAR)        │        │ name (VARCHAR)                   │
 │ created_at (TIMESTAMP WITH TZ)   │        │ sample_s3_key (VARCHAR)          │
 └──────────────────────────────────┘        │ consent_confirmed (BOOLEAN)      │
                                             │ created_at (TIMESTAMP WITH TZ)   │
                                             └────────────────┬─────────────────┘
                                                              │ 1
                                                              │
                                                              │ *
                                             ┌────────────────▼─────────────────┐
                                             │           generations            │
                                             ├──────────────────────────────────┤
                                             │ id (UUID / Int, PK)              │
                                             │ user_id (FK -> users.id)         │
                                             │ voice_id (FK -> voices.id)       │
                                             │ text_prompt (TEXT)               │
                                             │ output_s3_key (VARCHAR)          │
                                             │ duration_seconds (FLOAT)         │
                                             │ created_at (TIMESTAMP WITH TZ)   │
                                             └──────────────────────────────────┘
```

---

## 5. End-to-End API Sequence Flows

### 5.1 Voice Upload & Feature Embedding Flow

```
User App (Client)               FastAPI Backend             mutagen / S3             PostgreSQL
       │                               │                         │                        │
       ├─── POST /voices ─────────────►│                         │                        │
       │    (multipart file, name,     │                         │                        │
       │     consent_confirmed)        ├─── Validate Format ────►│                        │
       │                               │    & Sample Duration    │                        │
       │                               │◄── Validated ───────────┘                        │
       │                               │                                                  │
       │                               ├─── Upload Audio WAV ────────────────────────────►│ (S3 Bucket)
       │                               │◄── S3 Key Returned ──────────────────────────────┘
       │                               │                                                  │
       │                               ├─── INSERT INTO voices (user_id, s3_key, name) ───►│
       │                               │◄── Voice Record Created ─────────────────────────┘
       │◄── 201 Created (Voice Object)─┤
```

### 5.2 Speech Generation Flow

```
User App (Client)               FastAPI Backend          LRU Cache / PocketTTS       S3 / Database
       │                               │                         │                        │
       ├─── POST /generate ───────────►│                         │                        │
       │    ({ voice_id, text })       ├─── Fetch Voice S3 Key ──┼───────────────────────►│
       │                               │◄── S3 Key & Sample File ┼────────────────────────┘
       │                               │                         │                        │
       │                               ├─── Get/Load Embedding ─►│                        │
       │                               │◄── Model Embedding Ready│                        │
       │                               │                         │                        │
       │                               ├─── Execute Inference ──►│ (Synthesize Speech)    │
       │                               │◄── Audio Stream / WAV ──┘                        │
       │                               │                                                  │
       │                               ├─── Store Output WAV ────────────────────────────►│ (S3 Storage)
       │                               ├─── Record Generation Row ───────────────────────►│ (Postgres DB)
       │◄── 200 OK (Audio URL + Meta)──┤
```
