# VoiceLib

> AI-powered voice cloning text-to-speech — powered by Kyutai's Pocket TTS

## Quick Start (Docker Compose)

```bash
git clone <repo>
cd voicelib
cp .env.example .env          # Review and update JWT_SECRET_KEY
docker-compose up
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000/docs |
| MinIO Console | http://localhost:9001 |

Default MinIO credentials: `minioadmin / minioadmin`

---

## Architecture

```
                  ┌─────────────┐
                  │  React SPA  │ :5173
                  │  + Zustand  │
                  └──────┬──────┘
                         │ Axios (JWT Bearer)
                  ┌──────▼──────┐
                  │  FastAPI    │ :8000
                  │  ├── auth  │
                  │  ├── voices│
                  │  └── gen.  │
                  └──┬──────┬──┘
                     │      │
              ┌──────▼──┐  ┌▼────────────┐
              │Postgres │  │  MinIO/S3   │
              │ :5432   │  │  :9000      │
              └─────────┘  └─────────────┘
                  +
              [TTS Model in memory]
              pocket-tts (CPU)
              LRU cache (50 voices)
```

---

## Environment Variables

| Variable | Description | Required | Default |
|---|---|---|---|
| `DATABASE_URL` | Async PostgreSQL URL | ✅ | `postgresql+asyncpg://...` |
| `STORAGE_ENDPOINT_URL` | S3 endpoint (MinIO/R2, leave blank for AWS) | — | — |
| `STORAGE_ACCESS_KEY` | S3 access key | ✅ | `minioadmin` |
| `STORAGE_SECRET_KEY` | S3 secret key | ✅ | `minioadmin` |
| `STORAGE_BUCKET_NAME` | S3 bucket name | ✅ | `voicelib` |
| `STORAGE_REGION` | S3 region | — | `us-east-1` |
| `STORAGE_PUBLIC_BASE_URL` | CDN prefix (optional) | — | — |
| `JWT_SECRET_KEY` | HS256 signing secret (min 32 chars) | ✅ | ⚠️ change this |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT expiry | — | `10080` (7d) |
| `MAX_UPLOAD_SIZE_BYTES` | Max audio upload size | — | `20971520` (20MB) |
| `MAX_SAMPLE_DURATION_SECONDS` | Max voice sample duration | — | `30` |
| `VOICELIB_USE_MOCK_TTS` | Use mock TTS (dev without pocket-tts) | — | `false` |
| `VITE_API_BASE_URL` | Frontend API base URL | — | `http://localhost:8000` |

---

## API Reference

All endpoints (except `/auth/*` and `/health`) require `Authorization: Bearer <token>`.

### Auth
| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Register — `{email, password}` → `{access_token, user}` |
| POST | `/auth/login` | Login — `{email, password}` → `{access_token, user}` |
| GET | `/auth/me` | Current user |

### Voices
| Method | Path | Description |
|---|---|---|
| POST | `/voices` | Upload voice sample (multipart: `file`, `name`, `consent_confirmed`) |
| GET | `/voices` | List your voices |
| DELETE | `/voices/{id}` | Delete a voice (cascades to generations) |

### Generate
| Method | Path | Description |
|---|---|---|
| POST | `/generate` | Generate speech — `{voice_id, text}` → `{audio_url, ...}` |
| GET | `/generations` | List history — query: `?voice_id=&limit=&offset=` |

---

## Pocket TTS Setup (CPU)

```bash
# Install PyTorch CPU build first
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
# Then install pocket-tts
pip install pocket-tts
```

For development without pocket-tts, set `VOICELIB_USE_MOCK_TTS=true` in `.env`.
The mock generates a 1-second sine-wave WAV so the full API flow can be tested.

---

## Production Deployment

### Storage (no code changes)
- **AWS S3:** Remove `STORAGE_ENDPOINT_URL`, set real `STORAGE_ACCESS_KEY`/`STORAGE_SECRET_KEY`
- **Cloudflare R2:** Set `STORAGE_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com`
- **MinIO (prod):** Keep endpoint, change credentials

### Database
Replace `DATABASE_URL` with RDS, Supabase, or Neon connection string.

### Server
Run behind Nginx/Caddy with TLS. Example Nginx upstream:
```nginx
location /api/ {
  proxy_pass http://backend:8000/;
  proxy_set_header Host $host;
}
```

---

## v2 Roadmap — Song Covers

The Song Cover pipeline will be added as a separate build. It will:
1. Accept a song file or URL
2. Separate vocals from instrumental (Demucs)
3. Convert vocals to a cloned voice (reusing `Voice` table as-is)
4. Apply optional genre transformation
5. Output the full mixed song

**Entry point for v2:** [`backend/app/routers/song_cover.py`](./backend/app/routers/song_cover.py)

The `Voice` table and `storage.py`/`tts.py` modules require no changes for v2.

---

## Project Structure

```
voicelib/
├── backend/
│   └── app/
│       ├── main.py           # FastAPI app + lifespan
│       ├── config.py         # Pydantic BaseSettings
│       ├── db.py             # SQLAlchemy async engine
│       ├── models.py         # ORM + Pydantic schemas
│       ├── auth.py           # JWT + bcrypt
│       ├── tts.py            # Pocket TTS + LRU cache
│       ├── storage.py        # S3-compatible client
│       ├── utils/audio.py    # Upload validation
│       └── routers/
│           ├── auth_router.py
│           ├── voices_router.py
│           ├── generate_router.py
│           └── song_cover.py  ← v2 stub
├── frontend/
│   └── src/
│       ├── api/              # Axios wrappers
│       ├── store/            # Zustand stores
│       ├── components/       # UI + voice + generate components
│       └── pages/            # VoiceLibrary, Generate, SongCover, Login, Register
├── docker-compose.yml
└── .env.example
```
