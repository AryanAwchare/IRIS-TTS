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
  NGROK_AUTHTOKEN        — your ngrok auth token (set via Colab Secrets or env var)
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 0 — CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
import os

BACKEND_URL: str = os.getenv("VOICELIB_BACKEND_URL", "http://localhost:8000").rstrip("/")
COLAB_REGISTER_SECRET: str = os.getenv("COLAB_REGISTER_SECRET", "voicelib-colab-dev-secret")

# SECURITY: Never hardcode your token here. Set it via Colab Secrets (🔑 icon)
# or as an environment variable: NGROK_AUTHTOKEN=your_token_here
NGROK_AUTHTOKEN: str = os.getenv("NGROK_AUTHTOKEN", "").strip()
if not NGROK_AUTHTOKEN:
    try:
        from google.colab import userdata as _colab_userdata
        NGROK_AUTHTOKEN = (_colab_userdata.get("NGROK_AUTHTOKEN") or "").strip()
    except Exception:
        pass
if not NGROK_AUTHTOKEN:
    raise ValueError(
        "\n❌  NGROK_AUTHTOKEN is not set!\n"
        "    1. Get your free token at: https://dashboard.ngrok.com/get-started/your-authtoken\n"
        "    2. In Colab, click the 🔑 Secrets icon (left sidebar) and add:\n"
        "       Name: NGROK_AUTHTOKEN   Value: <your_token>\n"
        "    3. Re-run this cell.\n"
    )


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

_pip("uninstall", "-y", "torchaudio")
_pip("install", "-q", "--no-cache-dir", f"torchaudio=={_base}", "--index-url", _whl)

_pip("install", "-q", "--no-cache-dir",
     "fastapi", "uvicorn[standard]", "pyngrok", "nest-asyncio",
     "python-multipart", "requests",
     "librosa", "soundfile", "scipy", "noisereduce",
     "jiwer", "speechbrain", "faster-whisper",
     "transformers==4.47.1", "accelerate", "safetensors")

_pip("install", "-q", "--no-cache-dir",
     "perth", "omegaconf", "conformer", "einops",
     "rotary-embedding-torch", "vocos", "s3tokenizer", "diffusers>=0.21.0")

_pip("install", "-q", "--no-cache-dir", "--no-deps",
     "git+https://github.com/resemble-ai/chatterbox.git")

print(f"✅ Dependencies ready  (torch {_tv})")


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

if os.name == "posix":
    os.system("fuser -k 8008/tcp 2>/dev/null || true")


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 3 — TORCHVISION NMS SHIM
# ═══════════════════════════════════════════════════════════════════════════════
def _register_nms_shim() -> bool:
    try:
        import torchvision.ops as _tvops
        _tvops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.0]), 0.5)
        print("✅ torchvision ops native OK")
        return True
    except Exception:
        pass

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
        @torch.library.custom_op("torchvision::nms", mutates_args=())
        def _nms(boxes: torch.Tensor, scores: torch.Tensor,
                 iou_threshold: float) -> torch.Tensor:
            return _py_nms(boxes, scores, iou_threshold)
        print("✅ torchvision::nms shimmed via torch.library.custom_op")
        return True
    except Exception:
        pass

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
# ██  STEP 4 — TORCHAUDIO SAFE LOAD
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
    - Converts to mono float32 at target_sr (no resample if already at 32kHz)
    - 80 Hz high-pass (preserves male chest resonance, removes only true sub-bass)
    - Silence trimming (-40 dBFS)
    - RMS normalisation to -18 dBFS
    Keeps vocal harmonics intact — Chatterbox needs them for speaker identity.
    """
    if isinstance(audio, bytes):
        y, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    else:
        y, sr = librosa.load(audio, sr=None, mono=True)
        y = y.astype(np.float32)

    # Only resample if genuinely needed — avoids double-resample artifacts
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 80Hz high-pass (was 50Hz — raised to preserve male chest F0 80-130Hz)
    b, a = signal.butter(2, 80.0 / (sr * 0.5), btype="highpass")
    y = signal.filtfilt(b, a, y).astype(np.float32)

    yt, _ = librosa.effects.trim(y, top_db=40)
    if len(yt) > sr * 0.5:
        y = yt

    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-9)
    if rms > 1e-4:
        y = y * (0.125 / rms)
    y = np.clip(y, -0.98, 0.98)

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), y, sr


def select_best_segment(audio_bytes: bytes, target_duration: float = 10.0) -> str:
    """
    Pick the highest-SNR continuous speech segment from reference audio.
    Uses signal-to-noise ratio + zero-crossing rate heuristic to prefer
    clean speech over loud-but-noisy sections.
    Returns path to a temp WAV file — caller must delete it.
    """
    import tempfile
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    y = y.mean(axis=1)

    yt, _ = librosa.effects.trim(y, top_db=32)
    if len(yt) > sr * 0.5:
        y = yt

    win = int(target_duration * sr)

    if len(y) <= win + int(sr * 2):
        # Short clip — use as-is
        tmp = tempfile.NamedTemporaryFile(suffix="_seg.wav", delete=False)
        sf.write(tmp.name, y, sr, format="WAV", subtype="PCM_16")
        tmp.close()
        return tmp.name

    # Score windows by SNR (signal power / estimated noise floor)
    step = int(0.5 * sr)
    frame_size = int(0.025 * sr)
    hop = int(0.010 * sr)
    best_score = -float("inf")
    best_win = y[:win]

    for st in range(0, len(y) - win, step):
        cand = y[st : st + win]
        frames = [cand[i : i + frame_size] for i in range(0, len(cand) - frame_size, hop)]
        if not frames:
            continue
        frame_powers = np.array([np.mean(f ** 2) for f in frames])
        noise_floor = float(np.percentile(frame_powers, 10)) + 1e-9
        signal_power = float(np.mean(cand ** 2)) + 1e-9
        snr = 10.0 * np.log10(signal_power / noise_floor)

        # Prefer ZCR typical of speech (0.02–0.15) over music/noise
        zcr = float(np.mean(np.abs(np.diff(np.sign(cand)))) / 2.0)
        speech_bonus = 2.0 if 0.02 < zcr < 0.15 else -2.0

        score = snr + speech_bonus
        if score > best_score:
            best_score = score
            best_win = cand

    tmp = tempfile.NamedTemporaryFile(suffix="_seg.wav", delete=False)
    sf.write(tmp.name, best_win, sr, format="WAV", subtype="PCM_16")
    tmp.close()
    return tmp.name


def build_reference_prompt(audio_bytes: bytes, target_duration: float = 20.0) -> str:
    """
    Build a high-quality reference prompt by selecting and concatenating the two
    highest-SNR non-overlapping speech segments (up to target_duration total).

    Benefits over a single 10-second window:
    - More complete speaker embedding (broader phoneme coverage)
    - Averages out single-segment recording artifacts
    - Better for expressive speakers whose voice varies across the recording

    Returns path to a temp WAV file — caller must delete it.
    """
    import tempfile

    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    y = y.mean(axis=1)

    yt, _ = librosa.effects.trim(y, top_db=32)
    if len(yt) > sr * 0.5:
        y = yt

    total_dur = len(y) / sr
    segment_dur = target_duration / 2.0  # Two segments of half the target duration

    if total_dur <= segment_dur + 2.0:
        # Too short for two segments — return as-is
        tmp = tempfile.NamedTemporaryFile(suffix="_ref.wav", delete=False)
        sf.write(tmp.name, y, sr, format="WAV", subtype="PCM_16")
        tmp.close()
        return tmp.name

    win = int(segment_dur * sr)
    step = int(0.5 * sr)
    frame_size = int(0.025 * sr)
    hop = int(0.010 * sr)

    # Score all candidate windows
    scored: list[tuple[float, int, np.ndarray]] = []
    for st in range(0, len(y) - win, step):
        cand = y[st : st + win]
        frames = [cand[i : i + frame_size] for i in range(0, len(cand) - frame_size, hop)]
        if not frames:
            continue
        fp = np.array([np.mean(f ** 2) for f in frames])
        noise_floor = float(np.percentile(fp, 10)) + 1e-9
        snr = 10.0 * np.log10(float(np.mean(cand ** 2) + 1e-9) / noise_floor)
        zcr = float(np.mean(np.abs(np.diff(np.sign(cand)))) / 2.0)
        bonus = 2.0 if 0.02 < zcr < 0.15 else -2.0
        scored.append((snr + bonus, st, cand))

    if not scored:
        tmp = tempfile.NamedTemporaryFile(suffix="_ref.wav", delete=False)
        sf.write(tmp.name, y[:win], sr, format="WAV", subtype="PCM_16")
        tmp.close()
        return tmp.name

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_start, best_seg = scored[0]

    # Find second-best non-overlapping segment
    second_seg = None
    for score, start, seg in scored[1:]:
        if abs(start - best_start) >= win:  # Must not overlap
            second_seg = seg
            break

    # Stitch with a 200ms silence bridge
    silence_bridge = np.zeros(int(sr * 0.2), dtype=np.float32)
    if second_seg is not None:
        reference = np.concatenate([best_seg, silence_bridge, second_seg])
    else:
        reference = best_seg

    rms = float(np.sqrt(np.mean(reference ** 2)) + 1e-9)
    if rms > 1e-4:
        reference = reference * (0.125 / rms)
    reference = np.clip(reference, -0.98, 0.98).astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix="_ref_multi.wav", delete=False)
    sf.write(tmp.name, reference, sr, format="WAV", subtype="PCM_16")
    tmp.close()
    return tmp.name


def master_audio(audio: np.ndarray, sr: int = 32000) -> np.ndarray:
    """
    Transparent studio mastering:
    - 80 Hz high-pass (male-voice safe — was 45 Hz, which cut chest resonance)
    - Smooth noise gate (-46 dBFS) with 200ms release to prevent pumping
    - Peak limiter (-1.0 dBFS) for clean headroom
    """
    a = audio.copy().astype(np.float32)
    nyq = sr * 0.5

    # 80Hz high-pass — preserves male chest F0 while removing sub-bass rumble
    if (80.0 / nyq) < 1.0 and len(a) > 15:
        bh, ah = signal.butter(2, 80.0 / nyq, btype="highpass")
        a = signal.filtfilt(bh, ah, a).astype(np.float32)

    # Smooth noise gate with longer release (200ms) to prevent pumping
    thr = 10 ** (-46.0 / 20.0)
    frame = max(1, int(sr * 0.01))
    atk   = max(1, int(sr * 0.015))
    rel   = max(1, int(sr * 0.20))   # 200ms release (was 100ms)
    g = 1.0
    for i in range(0, len(a), frame):
        ch = a[i : i + frame]
        rms = float(np.sqrt(np.mean(ch ** 2)) + 1e-12)
        tgt = 1.0 if rms >= thr else 0.05
        g = min(1.0, g + frame / atk) if tgt > g else max(0.05, g - frame / rel)
        a[i : i + frame] = ch * g

    # Peak limiter at -1.0 dBFS (slightly more headroom than old -0.5 dBFS)
    mv = float(np.max(np.abs(a))) + 1e-9
    tp = 10 ** (-1.0 / 20.0)
    if mv > tp:
        a = a * (tp / mv)

    return np.clip(a, -0.95, 0.95).astype(np.float32)


def pitch_preserving_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch without changing pitch. rate > 1 → faster."""
    if abs(rate - 1.0) < 0.01 or len(y) == 0:
        return y
    try:
        return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
    except Exception:
        idx = np.linspace(0, len(y) - 1, max(1, int(len(y) / rate)))
        return np.interp(idx, np.arange(len(y)), y).astype(np.float32)


def split_sentences(text: str, target_chars: int = 180, max_chars: int = 280) -> list[str]:
    """
    Split text into synthesis chunks optimised for Chatterbox prosodic coherence.

    Key improvements over previous version:
    - target_chars raised to 180 (was 120) — longer chunks = better prosodic continuity
    - max_chars raised to 280 (was 220) — allows complete compound sentences
    - Minimum chunk size enforced (60 chars) — prevents single-clause isolation
    - Smart re-merge: short trailing chunks merged back into the previous chunk
    """
    MIN_CHUNK = 60

    raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    if not sentences:
        return [text]

    # Merge short sentences up to target_chars
    merged: list[str] = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= target_chars:
            cur = cur + " " + s
        else:
            merged.append(cur)
            cur = s
    if cur:
        merged.append(cur)

    # Split oversized chunks at clause boundaries
    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            parts = re.split(r"(?<=[,;])\s+", chunk)
            sub_cur = ""
            for p in parts:
                if not sub_cur:
                    sub_cur = p
                elif len(sub_cur) + 1 + len(p) <= max_chars:
                    sub_cur = sub_cur + " " + p
                else:
                    final.append(sub_cur)
                    sub_cur = p
            if sub_cur:
                final.append(sub_cur)

    # Re-merge short trailing chunks to avoid orphaned 2-3 word fragments
    result: list[str] = []
    for chunk in final:
        if result and len(chunk) < MIN_CHUNK and len(result[-1]) + 1 + len(chunk) <= max_chars:
            result[-1] = result[-1] + " " + chunk
        else:
            result.append(chunk)

    return result or [text]


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 7 — EMOTION PRESETS  (recalibrated for natural Chatterbox output)
#
#  Previous values had neutral exag=0.04 (too flat/robotic) and cfg=0.62–0.68
#  (too high — over-constrains prosody). New values based on Chatterbox community
#  testing and IRIS paper Table I revision.
#
#  Key insight: Chatterbox needs exag >= 0.10 even for "neutral" to sound
#  like natural speech. cfg > 0.65 produces robotic, clipped prosody.
# ═══════════════════════════════════════════════════════════════════════════════
EMOTION_PRESETS: dict[str, tuple[float, float]] = {
    # (exaggeration, cfg_weight)
    "neutral":   (0.15, 0.55),   # was (0.04, 0.62) — raised exag, lowered cfg
    "calm":      (0.08, 0.60),   # was (0.01, 0.68) — raised exag, lowered cfg
    "happy":     (0.22, 0.50),   # was (0.20, 0.54) — slight adjustments
    "excited":   (0.38, 0.44),   # was (0.35, 0.46) — slight adjustments
    "sad":       (0.12, 0.58),   # was (0.10, 0.62) — raised exag, lowered cfg
    "angry":     (0.30, 0.46),   # was (0.28, 0.50) — slight adjustments
    "fearful":   (0.22, 0.52),   # was (0.18, 0.58) — raised exag, lowered cfg
    "surprised": (0.32, 0.46),   # was (0.30, 0.50) — slight adjustments
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
        "status":          "healthy",
        "engine":          "chatterbox-tts",
        "model":           "chatterbox-tts",
        "cuda":            torch.cuda.is_available(),
        "gpu":             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch":           torch.__version__,
        "sample_rate":     getattr(_MODEL, "sr", 32000) if _MODEL else 32000,
        "emotion_presets": list(EMOTION_PRESETS.keys()),
    }


@app.post("/tts")
async def tts_endpoint(
    text:            str             = Form(...),
    reference_audio: UploadFile      = File(...),
    emotion:         str             = Form("neutral"),
    cfg_weight:      Optional[float] = Form(None),
    exaggeration:    Optional[float] = Form(None),
    speed:           float           = Form(1.0),
    pitch:           float           = Form(0.0),
    language:        str             = Form("en"),
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

        ref_path: Optional[str] = None

        try:
            gen_sr = getattr(model, "sr", 32000)

            # ── Reference audio: clean → multi-segment reference build ────────
            raw_bytes = await reference_audio.read()
            cleaned_bytes, _, _ = clean_audio(raw_bytes, target_sr=gen_sr)
            # Use multi-segment reference (up to 20s from 2 best SNR windows)
            ref_path = build_reference_prompt(cleaned_bytes, target_duration=20.0)

            # ── Text: clean tags + normalise ──────────────────────────────────
            clean_text = text.strip() or "Hello."
            for pat, rep in TAG_REPLACEMENTS.items():
                clean_text = re.sub(pat, rep, clean_text, flags=re.IGNORECASE)

            # ── Emotion parameters ─────────────────────────────────────────────
            ne = (emotion or "neutral").lower().strip()
            exag_def, cfg_def = EMOTION_PRESETS.get(ne, EMOTION_PRESETS["neutral"])
            active_cfg  = float(np.clip(cfg_weight  if cfg_weight  is not None else cfg_def,  0.20, 0.90))
            active_exag = float(np.clip(exaggeration if exaggeration is not None else exag_def, 0.00, 0.50))

            # ── Neural synthesis ───────────────────────────────────────────────
            sentences = split_sentences(clean_text)
            total = len(sentences)
            print(f"Synthesising {len(clean_text)} chars in {total} chunk(s) | "
                  f"emotion={ne} cfg={active_cfg:.2f} exag={active_exag:.2f}")
            chunks: list[np.ndarray] = []
            fade = int(gen_sr * 0.012)   # 12ms cross-fade (was 5ms)
            chunk_t0 = time.perf_counter()

            for i, sent in enumerate(sentences):
                s = sent.strip()
                if not s.endswith((".", "!", "?", ";", ",")):
                    s += "."

                with torch.inference_mode():   # saves 25-35% VRAM vs no context
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
                    chunk[:fade]  *= fi
                    chunk[-fade:] *= fo

                chunks.append(chunk)

                # Natural inter-sentence pause (longer = more natural breathing room)
                if i < total - 1:
                    pause_ms = 280.0 if s.endswith((".", "!")) else 180.0
                    chunks.append(np.zeros(int(gen_sr * pause_ms / 1000), dtype=np.float32))

                # Flush GPU VRAM every 8 chunks — prevents OOM on 5000-char texts
                if (i + 1) % 8 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    elapsed = time.perf_counter() - chunk_t0
                    eta = (elapsed / (i + 1)) * (total - i - 1)
                    print(f"  chunk {i+1}/{total} | {elapsed:.0f}s elapsed | ~{eta:.0f}s left")

            gen_np = np.concatenate(chunks).astype(np.float32)
            print(f"✅ All {total} chunks done in {time.perf_counter() - chunk_t0:.1f}s")

            # ── Post-processing ────────────────────────────────────────────────
            if len(gen_np) > gen_sr * 0.5:
                yt, _ = librosa.effects.trim(gen_np, top_db=38)
                if len(yt) > gen_sr * 0.3:
                    gen_np = yt

            if abs(pitch) > 0.05:
                try:
                    gen_np = librosa.effects.pitch_shift(
                        gen_np, sr=gen_sr, n_steps=pitch
                    ).astype(np.float32)
                except Exception:
                    pass

            if abs(speed - 1.0) > 0.03 and 0.5 <= speed <= 2.0:
                gen_np = pitch_preserving_stretch(gen_np, rate=speed)

            gen_np = master_audio(gen_np, sr=gen_sr)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            buf = io.BytesIO()
            sf.write(buf, gen_np, gen_sr, format="WAV", subtype="PCM_16")
            wav_bytes = buf.getvalue()

            latency = time.perf_counter() - t0
            print(f"⚡ Done | emotion={ne} | speed={speed:.2f}x | "
                  f"pitch={pitch:+.1f}st | {len(clean_text)} chars | {latency:.2f}s")

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={
                    "X-Model":                 "chatterbox-tts",
                    "X-Sample-Rate":           str(gen_sr),
                    "X-Latency-Seconds":       f"{latency:.2f}",
                    "X-Resolved-Emotion":      ne,
                    "X-Resolved-CFG":          f"{active_cfg:.2f}",
                    "X-Resolved-Exaggeration": f"{active_exag:.2f}",
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
        finally:
            # Always clean up temp reference file(s)
            if ref_path:
                try:
                    if os.path.exists(ref_path):
                        os.unlink(ref_path)
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════════════════════
# ██  STEP 10 — START SERVER + NGROK TUNNEL
# ═══════════════════════════════════════════════════════════════════════════════

def _run_server() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="warning")


print("\n🚀 Starting uvicorn on port 8008…")
threading.Thread(target=_run_server, daemon=True).start()
time.sleep(2.0)

get_model()

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
    print(f"   Set VOICELIB_BACKEND_URL env var, or manually add to .env:")
    print(f"   COLAB_GPU_API_URL={PUBLIC_URL}")


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
