"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           VoiceLib — Chatterbox TTS  GPU Microservice  (Cell 3)             ║
║           Paste this entire file into ONE Colab cell and run it.            ║
╚══════════════════════════════════════════════════════════════════════════════╝

What this does, in order:
  1. Installs / fixes all dependencies (torch, torchaudio, chatterbox, fastapi…)
  2. Applies torchvision NMS shim so transformers imports safely
  3. Loads ChatterboxTTS onto the GPU
  4. Starts a FastAPI server on port 8008 (POST /tts  +  GET /health)
  5. Opens an ngrok HTTPS tunnel
  6. Auto-registers the public URL with your local VoiceLib backend
     → No more copy-pasting URLs into .env each session!

Configuration  (edit the three lines in the CONFIG block below):
  BACKEND_URL            — your local backend URL (default: http://localhost:8000)
  COLAB_REGISTER_SECRET  — must match COLAB_REGISTER_SECRET in your backend .env
  NGROK_AUTHTOKEN        — your ngrok auth token
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 0 — CONFIG  (edit these three values)
# ═══════════════════════════════════════════════════════════════════════════════
import os

# URL of your running VoiceLib backend.
# • Local dev  → "http://localhost:8000"
# • If your backend is also tunneled (e.g. Render / Railway) → paste that URL
BACKEND_URL: str = os.getenv("VOICELIB_BACKEND_URL", "http://localhost:8000").rstrip("/")

# Must match COLAB_REGISTER_SECRET in voicelib/backend/.env
COLAB_REGISTER_SECRET: str = os.getenv("COLAB_REGISTER_SECRET", "voicelib-colab-dev-secret")

# Your ngrok authtoken  (https://dashboard.ngrok.com/get-started/your-authtoken)
NGROK_AUTHTOKEN: str = os.getenv("NGROK_AUTHTOKEN", "3I5bScJL7R0haCWXJ3FmBedIO5l_5aTLhGmF9vqmvEepVsERq")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 1 — DEPENDENCY INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════
import subprocess, sys

def _pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", *args], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

print("📦 Installing / verifying dependencies…")

import torch as _torch_check
_tv = _torch_check.__version__
_base = _tv.split("+")[0]
_cu   = _tv.split("+")[1] if "+" in _tv else "cu124"
_whl  = f"https://download.pytorch.org/whl/{_cu}"

# Re-install torchaudio to exactly match torch version
_pip("uninstall", "-y", "torchaudio")
_pip("install", "-q", "--no-cache-dir", f"torchaudio=={_base}", "--index-url", _whl)

# Core server + audio + ML stack
_pip("install", "-q", "--no-cache-dir",
     "fastapi", "uvicorn[standard]", "pyngrok", "nest-asyncio",
     "python-multipart", "requests",
     "librosa", "soundfile", "scipy", "noisereduce",
     "jiwer", "speechbrain", "faster-whisper",
     "transformers==4.47.1", "accelerate", "safetensors")

# Chatterbox runtime deps listed explicitly (avoids --no-deps skipping perth)
_pip("install", "-q", "--no-cache-dir",
     "perth",                   # pitch estimation
     "omegaconf",               # config loading
     "conformer",               # conformer encoder
     "einops",                  # tensor reshaping
     "rotary-embedding-torch",  # positional embeddings
     "vocos",                   # neural vocoder
     "s3tokenizer",             # speech tokenizer
     "diffusers>=0.21.0",       # flow-matching decoder
)

# Chatterbox itself - no-deps so it cannot downgrade torch/transformers
_pip("install", "-q", "--no-cache-dir", "--no-deps",
     "git+https://github.com/resemble-ai/chatterbox.git")

print(f"Dependencies ready  (torch {_tv})")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 2 — IMPORTS & INITIAL SETUP
# ═══════════════════════════════════════════════════════════════════════════════
import io, re, time, threading, traceback, types, importlib.util
from typing import Optional, Union

import torch
print(f"✅ torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Kill stale process on port 8008 (Linux / Colab)
if os.name == "posix":
    os.system("fuser -k 8008/tcp 2>/dev/null || true")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 3 — TORCHVISION NMS SHIM
#     Prevents transformers / LlamaModel import crash when torchvision C++ ops
#     are broken or version-mismatched (common on free Colab T4).
# ═══════════════════════════════════════════════════════════════════════════════
def _register_nms_shim() -> bool:
    """Register a pure-Python NMS op so torchvision.ops works without C++ ext."""
    # 1. Try native torchvision first
    try:
        import torchvision.ops as _tvops
        _tvops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.0]), 0.5)
        print("✅ torchvision ops native OK")
        return True
    except Exception:
        pass

    # 2. torch.library.custom_op (torch ≥ 2.1)
    try:
        @torch.library.custom_op("torchvision::nms", mutates_args=())
        def _nms(boxes: torch.Tensor, scores: torch.Tensor,
                 iou_threshold: float) -> torch.Tensor:
            return _py_nms(boxes, scores, iou_threshold)

        print("✅ torchvision::nms shimmed via torch.library.custom_op")
        return True
    except Exception:
        pass

    # 3. Module-level mock fallback
    def _py_nms(boxes: torch.Tensor, scores: torch.Tensor,
                iou: float) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort(descending=True)
        keep: list[int] = []
        while order.numel() > 0:
            i = int(order[0]); keep.append(i)
            if order.numel() == 1: break
            rest = order[1:]
            inter = ((x2[rest].clamp(max=float(x2[i])) - x1[rest].clamp(min=float(x1[i]))).clamp(0) *
                     (y2[rest].clamp(max=float(y2[i])) - y1[rest].clamp(min=float(y1[i]))).clamp(0))
            order = rest[(inter / (areas[i] + areas[rest] - inter + 1e-6)) <= iou]
        return torch.tensor(keep, dtype=torch.long)

    try:
        if "torchvision" not in sys.modules:
            sys.modules["torchvision"] = types.ModuleType("torchvision")
        tv = sys.modules["torchvision"]
        ops = types.ModuleType("torchvision.ops")
        ops.nms = _py_nms
        ops.box_iou = lambda a, b: torch.zeros(a.shape[0], b.shape[0])
        ops.batched_nms = lambda boxes, scores, idxs, thresh: _py_nms(boxes, scores, thresh)
        tv.ops = ops
        sys.modules["torchvision.ops"] = ops
        print("✅ torchvision.ops shimmed via module mock")
        return True
    except Exception as _e:
        print(f"⚠️  NMS shim notice: {_e}")
        return False

_register_nms_shim()


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 4 — TORCHAUDIO SAFE LOAD (mock if broken)
# ═══════════════════════════════════════════════════════════════════════════════
def _make_torchaudio_mock() -> types.ModuleType:
    ta = types.ModuleType("torchaudio")
    ta.__version__ = "0.0.0+mock"
    ta.__file__ = "<mock>"
    ta.__path__ = []
    ta.__package__ = "torchaudio"
    spec = importlib.util.spec_from_loader("torchaudio", loader=None)
    spec.submodule_search_locations = []  # type: ignore[union-attr]
    ta.__spec__ = spec
    for sub in ["_torchaudio", "backend", "transforms", "functional",
                "compliance", "io", "pipelines", "datasets", "utils"]:
        m = types.ModuleType(f"torchaudio.{sub}")
        m.__spec__ = importlib.util.spec_from_loader(f"torchaudio.{sub}", loader=None)
        m.__package__ = "torchaudio"
        sys.modules[f"torchaudio.{sub}"] = m
        setattr(ta, sub, m)
    sys.modules["torchaudio"] = ta
    return ta

try:
    import torchaudio
    _ = torchaudio.__version__
    print(f"✅ torchaudio {torchaudio.__version__} OK")
except Exception as _e:
    print(f"⚠️  torchaudio issue ({type(_e).__name__}) — applying mock…")
    for _k in list(sys.modules):
        if "torchaudio" in _k:
            del sys.modules[_k]
    torchaudio = _make_torchaudio_mock()
    print("✅ torchaudio mocked with valid __spec__")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 5 — AUDIO LIBRARIES
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
import nest_asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from pyngrok import ngrok

nest_asyncio.apply()
print("✅ Audio libraries loaded")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 6 — AUDIO PROCESSING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def clean_audio(audio: Union[bytes, str], target_sr: int = 32000):
    """
    Light-touch reference audio cleaner.
    - Converts to mono float32 at target_sr
    - Gentle 50 Hz high-pass (removes DC rumble, preserves chest warmth)
    - Silence trimming  (-40 dBFS)
    - RMS normalisation to -18 dBFS
    Keeps vocal harmonics intact — Chatterbox needs them for speaker identity.
    """
    if isinstance(audio, bytes):
        y, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    else:
        y, sr = librosa.load(audio, sr=None, mono=True)
        y = y.astype(np.float32)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 50 Hz high-pass
    b, a = signal.butter(2, 50.0 / (sr * 0.5), btype="highpass")
    y = signal.filtfilt(b, a, y).astype(np.float32)

    # Silence trim
    yt, _ = librosa.effects.trim(y, top_db=40)
    if len(yt) > sr * 0.5:
        y = yt

    # RMS normalise to -18 dBFS
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    if rms > 1e-4:
        y = y * (0.125 / rms)
    y = np.clip(y, -0.98, 0.98)

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), y, sr


def select_best_segment(audio_bytes: bytes, target_duration: float = 10.0) -> str:
    """
    Pick the highest-energy continuous segment of ~10 s from reference audio.
    Returns path to a temp WAV file — caller must delete it.
    Short clips (< target_duration) are returned as-is.
    """
    import tempfile
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    y = y.mean(axis=1)

    yt, _ = librosa.effects.trim(y, top_db=32)
    if len(yt) > sr * 0.5:
        y = yt

    win = int(target_duration * sr)
    if len(y) > win + int(sr * 2):
        step = int(0.25 * sr)
        best_pow, best_win = -float("inf"), y[:win]
        for st in range(0, len(y) - win, step):
            cand = y[st: st + win]
            p = float(np.mean(cand ** 2))
            if p > best_pow:
                best_pow, best_win = p, cand
        y = best_win

    tmp = tempfile.NamedTemporaryFile(suffix="_seg.wav", delete=False)
    sf.write(tmp.name, y, sr, format="WAV", subtype="PCM_16")
    tmp.close()
    return tmp.name


def master_audio(audio: np.ndarray, sr: int = 32000) -> np.ndarray:
    """
    Transparent studio mastering:
    - 45 Hz high-pass (sub-rumble removal)
    - Smooth noise gate (-46 dBFS) — clean pauses, no word truncation
    - Peak limiter (-0.5 dBFS) — no hard clipping / digital distortion
    """
    a = audio.copy().astype(np.float32)
    nyq = sr * 0.5

    # Sub-rumble high-pass
    if (45.0 / nyq) < 1.0 and len(a) > 15:
        bh, ah = signal.butter(2, 45.0 / nyq, btype="highpass")
        a = signal.filtfilt(bh, ah, a).astype(np.float32)

    # Smooth noise gate
    thr = 10 ** (-46.0 / 20.0)
    frame = max(1, int(sr * 0.01))
    atk   = max(1, int(sr * 0.015))
    rel   = max(1, int(sr * 0.10))
    g = 1.0
    for i in range(0, len(a), frame):
        ch = a[i: i + frame]
        rms = float(np.sqrt(np.mean(ch ** 2)) + 1e-12)
        tgt = 1.0 if rms >= thr else 0.05
        g = min(1.0, g + frame / atk) if tgt > g else max(0.05, g - frame / rel)
        a[i: i + frame] = ch * g

    # Peak limiter
    mv = float(np.max(np.abs(a))) + 1e-9
    tp = 10 ** (-0.5 / 20.0)
    if mv > tp:
        a = a * (tp / mv)

    return np.clip(a, -0.98, 0.98).astype(np.float32)


def pitch_preserving_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch without changing pitch.  rate > 1 → faster."""
    if abs(rate - 1.0) < 0.01 or len(y) == 0:
        return y
    try:
        return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
    except Exception:
        idx = np.linspace(0, len(y) - 1, max(1, int(len(y) / rate)))
        return np.interp(idx, np.arange(len(y)), y).astype(np.float32)


def split_sentences(text, target_chars=120, max_chars=220):
    """Chunk text for synthesis. Handles up to 5000 chars. target_chars=120 = 4-8s per T4 pass."""
    raw = re.split(r"(?<=[.!?;])\s+|\n+", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    merged, cur = [], ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= target_chars:
            cur = cur + " " + s
        else:
            merged.append(cur); cur = s
    if cur: merged.append(cur)
    final = []
    for chunk in merged:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            parts = re.split(r"(?<=,)\s+", chunk)
            sub, c2 = [], ""
            for p in parts:
                if not c2: c2 = p
                elif len(c2) + 1 + len(p) <= max_chars: c2 = c2 + " " + p
                else: sub.append(c2); c2 = p
            if c2: sub.append(c2)
            final.extend(sub if sub else [chunk])
    return final or [text]


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 7 — EMOTION PRESETS
# ═══════════════════════════════════════════════════════════════════════════════
EMOTION_PRESETS: dict[str, tuple[float, float]] = {
    # (exaggeration, cfg_weight)
    "neutral":   (0.04, 0.62),
    "calm":      (0.01, 0.68),
    "happy":     (0.20, 0.54),
    "excited":   (0.35, 0.46),
    "sad":       (0.10, 0.62),
    "angry":     (0.28, 0.50),
    "fearful":   (0.18, 0.58),
    "surprised": (0.30, 0.50),
}

TAG_REPLACEMENTS = {
    r"\[laughter\]":      " (laughs) ",
    r"\[sigh\]":          " ... (sighs) ... ",
    r"\[gasp\]":          " ... (gasps) ... ",
    r"\[whisper\]":       " (whispering) ",
    r"\[chuckle\]":       " (chuckles) ",
    r"\[clears throat\]": " (clears throat) ",
}


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 8 — LOAD CHATTERBOX MODEL (lazy, cached)
# ═══════════════════════════════════════════════════════════════════════════════
_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        print("⏳ Loading ChatterboxTTS onto GPU…")
        from chatterbox.tts import ChatterboxTTS
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = ChatterboxTTS.from_pretrained(device=dev)
        sr = getattr(_MODEL, "sr", 32000)
        print(f"🚀 ChatterboxTTS ready on {dev.upper()}!  sample_rate={sr} Hz")
    return _MODEL


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 9 — FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="VoiceLib Chatterbox GPU Server")
_GPU_LOCK = __import__("asyncio").Lock()


@app.get("/")
@app.get("/health")
def health():
    """Health probe used by the VoiceLib backend /colab-status endpoint."""
    return {
        "status": "healthy",
        "engine": "chatterbox-tts",
        "model": "chatterbox-tts",
        "cuda":  torch.cuda.is_available(),
        "gpu":   torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "sample_rate": getattr(_MODEL, "sr", 32000) if _MODEL else 32000,
        "emotion_presets": list(EMOTION_PRESETS.keys()),
    }


@app.post("/tts")
async def tts_endpoint(
    text:            str            = Form(...),
    reference_audio: UploadFile     = File(...),
    emotion:         str            = Form("neutral"),
    cfg_weight:      Optional[float]= Form(None),
    exaggeration:    Optional[float]= Form(None),
    speed:           float          = Form(1.0),
    pitch:           float          = Form(0.0),
    language:        str            = Form("en"),
):
    """
    Zero-shot voice cloning endpoint.

    Form fields
    -----------
    text             : Script to synthesise
    reference_audio  : WAV/MP3 voice sample (the speaker to clone)
    emotion          : neutral | calm | happy | excited | sad | angry | fearful | surprised
    cfg_weight       : override classifier-free guidance weight  (0.20 – 0.90)
    exaggeration     : override emotion exaggeration factor      (0.00 – 0.50)
    speed            : speaking pace multiplier                  (0.5 – 2.0)
    pitch            : pitch shift in semitones                  (-6 … +6)
    language         : target language code (informational only for Chatterbox)
    """
    t0 = time.perf_counter()

    async with _GPU_LOCK:
        model = get_model()
        if model is None:
            return Response(
                content=b"Chatterbox model not loaded.",
                status_code=503,
                media_type="text/plain",
            )

        try:
            gen_sr = getattr(model, "sr", 32000)

            # ── Reference audio: clean → best segment selection ──────────────
            raw_bytes = await reference_audio.read()
            cleaned_bytes, _, _ = clean_audio(raw_bytes, target_sr=gen_sr)
            ref_path = select_best_segment(cleaned_bytes, target_duration=10.0)

            # ── Text: clean tags + normalise ─────────────────────────────────
            clean_text = text.strip() or "Hello."
            for pat, rep in TAG_REPLACEMENTS.items():
                clean_text = re.sub(pat, rep, clean_text, flags=re.IGNORECASE)

            # ── Emotion parameters ────────────────────────────────────────────
            ne = (emotion or "neutral").lower().strip()
            exag_def, cfg_def = EMOTION_PRESETS.get(ne, EMOTION_PRESETS["neutral"])
            active_cfg  = float(np.clip(cfg_weight  if cfg_weight  is not None else cfg_def,  0.20, 0.90))
            active_exag = float(np.clip(exaggeration if exaggeration is not None else exag_def, 0.00, 0.50))

            # ── Neural synthesis ─────────────────────────────────────────────
            try:
                if len(clean_text) > 80 or re.search(r"[,;]", clean_text):
                    # Multi-sentence path — stitch with natural pauses
                    sentences = split_sentences(clean_text, min_chars=35)
                    chunks: list[np.ndarray] = []
                    fade = int(gen_sr * 0.005)

                    for i, sent in enumerate(sentences):
                        s = sent.strip()
                        if not s.endswith((".", "!", "?", ";", ",")):
                            s += "."
                        with torch.inference_mode():
                            wav_t = model.generate(
                                s,
                                audio_prompt_path=ref_path,
                                cfg_weight=active_cfg,
                                exaggeration=active_exag,
                            )
                        chunk = wav_t.squeeze().detach().cpu().numpy().astype(np.float32)

                        # Cross-fade edges to avoid clicks at joins
                        if len(chunk) > fade * 2:
                            fi = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, fade)))
                            fo = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0, 1, fade)))
                            chunk[:fade] *= fi
                            chunk[-fade:] *= fo

                        chunks.append(chunk)

                        # Natural inter-sentence pause
                        if i < total - 1:
                            pause_ms = 220.0 if s.endswith((".", "!")) else 150.0
                            chunks.append(np.zeros(int(gen_sr * pause_ms / 1000), dtype=np.float32))

                        # Flush GPU VRAM every 10 chunks ? prevents OOM on 5000-char texts
                        if (i + 1) % 10 == 0:
                            if torch.cuda.is_available(): torch.cuda.empty_cache()
                            elapsed = time.perf_counter() - chunk_t0
                            eta = (elapsed / (i + 1)) * (total - i - 1)
                            print(f"  chunk {i+1}/{total} | {elapsed:.0f}s | ~{eta:.0f}s left")

                gen_np = np.concatenate(chunks).astype(np.float32)
                print(f"All {total} chunks done in {time.perf_counter()-chunk_t0:.1f}s")
            finally:
                # Always clean up temp reference file
                try:
                    if ref_path and os.path.exists(ref_path):
                        os.unlink(ref_path)
                except Exception:
                    pass

            # ── Post-processing ───────────────────────────────────────────────
            # Silence-trim generated audio
            if len(gen_np) > gen_sr * 0.5:
                yt, _ = librosa.effects.trim(gen_np, top_db=38)
                if len(yt) > gen_sr * 0.3:
                    gen_np = yt

            # Optional pitch shift (only when explicitly requested)
            if abs(pitch) > 0.05:
                try:
                    gen_np = librosa.effects.pitch_shift(
                        gen_np, sr=gen_sr, n_steps=pitch
                    ).astype(np.float32)
                except Exception:
                    pass  # Non-fatal — skip pitch shift on error

            # Optional time-stretch for speed
            if abs(speed - 1.0) > 0.03 and 0.5 <= speed <= 2.0:
                gen_np = pitch_preserving_stretch(gen_np, rate=speed)

            # Studio mastering (noise gate + peak limiter)
            gen_np = master_audio(gen_np, sr=gen_sr)

            # Free GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ── Encode to WAV ─────────────────────────────────────────────────
            buf = io.BytesIO()
            sf.write(buf, gen_np, gen_sr, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()

            latency = time.perf_counter() - t0
            print(f"⚡ Synthesised | emotion={ne} | speed={speed:.2f}x | "
                  f"pitch={pitch:+.1f}st | {len(clean_text)} chars | {latency:.2f}s latency")

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "X-Model":                "chatterbox-tts",
                    "X-Sample-Rate":          str(gen_sr),
                    "X-Latency-Seconds":      f"{latency:.2f}",
                    "X-Resolved-Emotion":     ne,
                    "X-Resolved-CFG":         f"{active_cfg:.2f}",
                    "X-Resolved-Exaggeration":f"{active_exag:.2f}",
                    "ngrok-skip-browser-warning": "true",
                },
            )

        except Exception as exc:
            tb = traceback.format_exc()
            print(f"❌ Synthesis error: {exc}\n{tb}")
            return Response(
                content=f"Synthesis error: {exc}\n{tb}".encode(),
                status_code=500,
                media_type="text/plain",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 10 — START SERVER + NGROK TUNNEL
# ═══════════════════════════════════════════════════════════════════════════════

def _run_server() -> None:
    """Run uvicorn in a daemon thread."""
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="warning")


print("\n🚀 Starting uvicorn on port 8008…")
threading.Thread(target=_run_server, daemon=True).start()
time.sleep(2.0)  # Give uvicorn time to bind

# Pre-load model onto GPU before accepting requests
get_model()

# ── ngrok tunnel ──────────────────────────────────────────────────────────────
try:
    ngrok.kill()
except Exception:
    pass

ngrok.set_auth_token(NGROK_AUTHTOKEN)
tunnel = ngrok.connect(8008)
PUBLIC_URL: str = tunnel.public_url

print("\n" + "=" * 70)
print("🌐 VoiceLib Chatterbox GPU Server is LIVE!")
print(f"   Public URL  : {PUBLIC_URL}")
print(f"   Health check: {PUBLIC_URL}/health")
print(f"   Synthesis   : POST {PUBLIC_URL}/tts")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 11 — AUTO-REGISTER URL WITH VOICELIB BACKEND
#     Sends this ngrok URL to the backend so it updates COLAB_GPU_API_URL
#     in memory — no manual .env editing required.
# ═══════════════════════════════════════════════════════════════════════════════
import requests as _requests

_MAX_REGISTER_ATTEMPTS = 3
_registered = False

for _attempt in range(1, _MAX_REGISTER_ATTEMPTS + 1):
    try:
        _resp = _requests.post(
            f"{BACKEND_URL}/colab-register",
            json={"url": PUBLIC_URL, "secret": COLAB_REGISTER_SECRET},
            timeout=8.0,
            headers={"Content-Type": "application/json"},
        )
        if _resp.status_code == 200:
            print(f"\n✅ Backend auto-registered!  ({BACKEND_URL})")
            print(f"   Response: {_resp.json().get('message', 'OK')}")
            _registered = True
            break
        else:
            print(f"⚠️  Register attempt {_attempt}: HTTP {_resp.status_code} — {_resp.text[:120]}")
    except Exception as _re:
        print(f"⚠️  Register attempt {_attempt}: {_re}")
    time.sleep(2.0 * _attempt)

if not _registered:
    print(f"\n⚠️  Could not auto-register with backend at {BACKEND_URL}.")
    print(f"   If your backend is running elsewhere, set VOICELIB_BACKEND_URL at the top of this script.")
    print(f"   Or manually add to your backend .env:\n   COLAB_GPU_API_URL={PUBLIC_URL}")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 12 — KEEP ALIVE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n⚡ Server is live and listening for requests.  (Keep this cell running)")
print("   Stop it with Runtime → Interrupt execution  or  Ctrl+C\n")

try:
    while True:
        time.sleep(1.0)
except KeyboardInterrupt:
    print("\nServer stopped.")
