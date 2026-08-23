"""
VoiceLib FastAPI application factory.

Startup sequence:
    1. Create DB tables (dev mode — use Alembic for prod)
    2. Ensure S3/MinIO bucket exists
    3. Load Pocket TTS model (blocks until ready)
    4. Register all routers
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db, storage, tts
from app.routers import auth_router, generate_router, song_cover, voices_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown hooks."""
    # ── Startup ──────────────────────────────────────────────────────────
    logger.info("VoiceLib starting up...")

    # 1. Create DB tables (dev convenience; use Alembic in prod)
    await db.create_tables()
    logger.info("Database tables ready.")

    # 2. Ensure object storage bucket exists (idempotent)
    storage.ensure_bucket_exists()

    # 3. Load TTS model once — blocks until model is in memory
    tts.load_model()

    logger.info("VoiceLib ready. All systems go. 🎙️")
    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    logger.info("VoiceLib shutting down.")
    await db.engine.dispose()


app = FastAPI(
    title="VoiceLib API",
    description=(
        "AI voice-cloning text-to-speech API powered by Kyutai's Pocket TTS. "
        "Upload voice samples, build a personal voice library, and generate speech."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files (Local Disk Storage Fallback) ───────────────────────────────
from fastapi.staticfiles import StaticFiles
from pathlib import Path
local_dir = Path("./local_storage_data")
local_dir.mkdir(parents=True, exist_ok=True)
app.mount("/storage_files", StaticFiles(directory=local_dir), name="storage_files")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(voices_router.router, prefix="/voices", tags=["voices"])
app.include_router(generate_router.router, tags=["generate"])
app.include_router(song_cover.router)   # Stub — no-op in v1


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Health check endpoint for load balancers and monitoring."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/model-info", tags=["health"])
async def model_info() -> dict:
    """Returns active TTS model metadata, capabilities, emotion support, and sample limits."""
    return tts.get_engine_info()


@app.get("/colab-status", tags=["health"])
async def colab_status() -> dict:
    """Checks whether the external Colab GPU synthesis server is online."""
    import urllib.request
    import json
    from app.config import get_settings

    settings = get_settings()
    url = settings.colab_gpu_api_url.rstrip("/")

    try:
        req = urllib.request.Request(
            f"{url}/health",
            headers={
                "User-Agent": "VoiceLib",
                "ngrok-skip-browser-warning": "true",
            },
        )
        with urllib.request.urlopen(req, timeout=5.0) as res:
            if res.status == 200:
                body = json.loads(res.read().decode("utf-8"))
                return {
                    "online": True,
                    "url": url,
                    "model": body.get("model", "chatterbox-tts"),
                    "gpu": body.get("gpu", "CUDA GPU"),
                    "cuda": body.get("cuda", True),
                    "sample_rate": body.get("sample_rate", 32000),
                }
    except Exception as e:
        logger.debug(f"Colab GPU server ping failed: {e}")

    return {
        "online": False,
        "url": url,
        "model": None,
        "gpu": None,
        "cuda": False,
        "message": "Colab GPU server not running locally on port 8008. Run Step 7 in Colab notebook to activate GPU.",
    }
