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
from app.config import get_settings
from app.routers import auth_router, generate_router, song_cover, voices_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown hooks."""
    logger.info("VoiceLib starting up...")

    await db.create_tables()
    logger.info("Database tables ready.")

    settings = get_settings()
    if not getattr(settings, "debug", True) and settings.jwt_secret_key == "voicelib-localhost-secret-key-32chars":
        logger.warning(
            "CRITICAL SECURITY WARNING: Production mode running with default JWT_SECRET_KEY! "
            "Configure a strong JWT_SECRET_KEY in production."
        )

    storage.ensure_bucket_exists()

    tts.load_model()

    async def _warmup_eval_models():
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            from app.evaluation.speaker_similarity import _get_classifier
            from app.evaluation.content_accuracy import _get_whisper_model
            from app.utils.emotion_analyzer import _get_classifier as _get_emotion_classifier
            await loop.run_in_executor(None, _get_classifier)
            await loop.run_in_executor(None, _get_whisper_model)
            await loop.run_in_executor(None, _get_emotion_classifier)
            logger.info("Evaluation & Emotion models (ECAPA-TDNN, Whisper, DistilRoBERTa) pre-warmed successfully.")
        except Exception as warm_err:
            logger.debug(f"Models pre-warm notice: {warm_err}")

    import asyncio
    asyncio.create_task(_warmup_eval_models())

    logger.info("VoiceLib ready. All systems go.")
    yield

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
# FIX: removed allow_origin_regex=r"https?://.*" which matched ALL origins,
# defeating the purpose of the explicit allow_origins list.
# Add your production domain to allow_origins when deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    origin = request.headers.get("origin", "http://localhost:5173")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
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
app.include_router(song_cover.router)


@app.get("/health", tags=["health"])
async def health_check() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/model-info", tags=["health"])
async def model_info() -> dict:
    return tts.get_engine_info()


@app.get("/colab-status", tags=["health"])
async def colab_status() -> dict:
    import urllib.request
    import json
    from app.config import get_settings
    from app.tts_engines.gptsovits_engine import get_live_colab_url

    settings = get_settings()
    live_url = get_live_colab_url()
    url = (live_url or settings.colab_gpu_api_url).rstrip("/")

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
                    "model": body.get("model", body.get("engine", "chatterbox-tts")),
                    "gpu": body.get("gpu", "CUDA GPU"),
                    "cuda": body.get("cuda", True),
                    "sample_rate": body.get("sample_rate", 32000),
                    "auto_registered": bool(live_url),
                }
    except Exception as e:
        logger.debug(f"Colab GPU server ping failed: {e}")

    return {
        "online": False,
        "url": url,
        "model": None,
        "gpu": None,
        "cuda": False,
        "auto_registered": bool(live_url),
        "message": "Colab GPU server not reachable. Run Cell 3 in your Colab notebook to activate.",
    }


@app.post("/colab-register", tags=["health"])
async def colab_register(payload: dict) -> dict:
    from fastapi import HTTPException
    from app.config import get_settings
    from app.tts_engines.gptsovits_engine import set_live_colab_url

    settings = get_settings()
    provided_secret = payload.get("secret", "")
    url = (payload.get("url") or "").strip()

    if provided_secret != settings.colab_register_secret:
        raise HTTPException(status_code=403, detail="Invalid registration secret.")

    if not url or (not url.startswith("https://") and not url.startswith("http://")):
        raise HTTPException(status_code=400, detail="Payload must contain a valid 'url'.")

    set_live_colab_url(url)
    logger.info(f"Colab GPU URL auto-registered: {url}")

    return {"status": "registered", "url": url, "message": "Colab GPU server URL updated successfully."}
