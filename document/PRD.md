# Product Requirements Document (PRD) — IRIS / VoiceLib

> **Project Name:** IRIS (VoiceLib)  
> **Status:** Active / v1.0 Released  
> **Last Updated:** August 2026  
> **Document Owner:** Lead Product Architect & AI Engineering Team  

---

## 1. Executive Summary & Vision

**IRIS (VoiceLib)** is a full-stack, state-of-the-art AI-powered Voice Cloning and Text-to-Speech (TTS) web application. Designed to deliver low-latency, zero-shot and few-shot voice replication, IRIS enables users to clone custom voices from short audio samples (3–30 seconds) and generate hyper-realistic speech audio on demand.

The long-term vision for IRIS extends beyond basic speech synthesis into an all-in-one AI Audio Workstation, incorporating AI Song Cover creation (v2), vocal timbre conversion, multi-lingual audio synthesis, and automated voice style transformation.

---

## 2. Target Audience & User Personas

| Persona | Primary Goal | Key Requirements |
|---|---|---|
| **Content Creators & Podcasters** | Generate consistent, studio-quality voiceovers without re-recording audio scripts. | High audio fidelity, quick rendering, batch text generation, history management. |
| **Game Developers & Animators** | Create distinct character voices dynamically for game NPCs and animated dialogue. | Custom voice library management, tag-based categorization, zero-shot cloning. |
| **Audiobooks & Educational Authors** | Synthesize long-form narration using personal or brand-approved voices. | Stable text handling, continuous audio generation, downloadable formats. |
| **Accessibility Developers** | Provide customized personalized voices for assistive speech technology users. | Low latency synthesis, clear audio playback, intuitive UI with screen-reader support. |

---

## 3. Scope & Key Features

### 3.1 Phase 1 (v1.0 — Core Voice Cloning & TTS)

#### Feature 1: Authentication & Account Management
* **JWT-Based Authentication:** Secure user registration and login endpoints utilizing HS256 tokens and bcrypt password hashing.
* **Persistent Session:** Client-side token persistence using Zustand and local storage with auto-refresh / logout handling.
* **User Profile & Authorization:** Protected routes ensuring private ownership of custom voices and generated audio tracks.

#### Feature 2: Voice Sample Upload & Profile Management
* **Drag-and-Drop Audio Upload:** Interactive client-side dropzone supporting `.wav`, `.mp3`, `.ogg`, and `.m4a` file formats.
* **Client & Server Audio Validation:** Strict verification of audio size (max 20MB), duration (3 to 30 seconds), and MIME type.
* **Consent & Legal Safeguards:** Mandatory checkbox requiring explicit user confirmation of voice ownership rights prior to upload.
* **Voice Library Management:** CRUD interface displaying active voice cards, sample playback, creation timestamps, and deletion cascades.

#### Feature 3: Text-to-Speech Generation Engine
* **Voice Selector & Text Canvas:** Interface for picking active voice profiles and writing or pasting synthesis text prompts.
* **Kyutai Pocket-TTS Inference:** Real-time speech synthesis driven by Kyutai's Pocket-TTS engine (CPU optimized) with LRU model caching (50-voice capacity).
* **Mock TTS Fallback:** Automatic fallback engine generating clean 1-second reference tones when running in resource-constrained or dev mode (`VOICELIB_USE_MOCK_TTS=true`).
* **Generation History & Waveform Player:** Interactive history panel featuring inline custom waveform visualizers, audio scrubbing, speed controls, and one-click `.wav` downloads.

#### Feature 4: Cloud & Local Storage Integration
* **S3-Compatible Storage Adapter:** Seamless abstraction layer supporting MinIO (local development), AWS S3, and Cloudflare R2 for storing raw samples and generated outputs.
* **Presigned Audio URLs:** Secure direct delivery of generated audio streams via presigned URLs or CDN endpoints.

---

### 3.2 Phase 2 (v2.0 — AI Song Cover & Vocal Conversion)

* **Vocal & Instrumental Separation:** Integration of Demucs / UVR5 for splitting input songs into isolated vocal and backing tracks.
* **Vocal Pitch & Timbre Transfer:** Converting source song vocals into a selected user voice profile while maintaining pitch dynamics.
* **Audio Mixer & Export:** Re-combining converted vocals with backing tracks and exporting final mixed cover songs.

---

## 4. Non-Functional Requirements (NFRs)

### 4.1 Performance & Latency
* **Synthesis Speed:** Sub-3-second time-to-first-byte (TTFB) for standard text prompts under 100 characters on CPU inference.
* **Frontend Responsiveness:** SPA page transitions under 100ms; smooth 60fps audio waveform rendering.
* **LRU Model Cache Efficiency:** Cache up to 50 active voice embeddings in server memory to minimize disk I/O and cold-start latency.

### 4.2 Security & Compliance
* **Password Hashing:** Passwords must be hashed using `bcrypt` with work factor >= 12.
* **Audio Ownership Policy:** User consent logs stored with voice metadata for audit compliance.
* **API Rate Limiting:** Throttling synthesis requests to prevent server exhaustion and DDoS attacks.
* **CORS & Data Protection:** Strict origin controls and sanitized multipart form data handling.

### 4.3 Reliability & Maintainability
* **Graceful Degradation:** Fallback to mock synthesis engine when external neural dependencies or models are unavailable.
* **Structured Error Handling:** Standardized API error format (`{ "detail": string, "code": string }`).
* **Database Isolation:** Async SQLAlchemy 2.0 with PostgreSQL transactions ensuring strict ACID guarantees.

---

## 5. User Journey & Product Flow

```
[ Visitor ] 
    │
    ├──> Register / Login (JWT Issued)
    │
[ Dashboard / Voice Library ]
    │
    ├──> Upload Voice Sample (3-30s audio + Consent Check) ──> mutagen Validation ──> MinIO/S3 & PostgreSQL
    │
[ Speech Generator Canvas ]
    │
    ├──> Select Voice Profile + Enter Text Script
    │
    ├──> Trigger Generate (POST /generate)
    │        └─> FastAPI ──> Pocket-TTS Model ──> Audio WAV Generated
    │
[ History & Player ]
    └─> Play Waveform / Download WAV / Stream Presigned URL
```

---

## 6. Success Metrics & Key Performance Indicators (KPIs)

1. **Voice Creation Success Rate:** > 98% successful voice embedding extractions from valid audio uploads.
2. **TTS Generation Latency:** Average generation time < 2.5x real-time audio length on CPU.
3. **User Retention & Engagement:** Average user creates 3+ custom voices and generates 15+ speech clips per session.
4. **Zero Crashing Standard:** 0 unhandled 500 server crashes in synthesis worker threads.
