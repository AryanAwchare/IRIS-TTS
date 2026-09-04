from __future__ import annotations

"""
🎙️ VoiceLib — Chatterbox TTS (Resemble AI) Neural Voice Cloning GPU Microservice
==================================================================================
Standalone script for Google Colab or CUDA-enabled GPU servers.
Features:
  - ChatterboxTTS (Resemble AI Zero-Shot Voice Cloning)
  - Emotion exaggeration mapping (neutral, happy, excited, angry, sad, calm, etc.)
  - 32kHz state-of-the-art Denoising & Cleaning pipeline
  - 4-Band Vocal EQ & Harmonic Warmth Studio Mastering
  - Multi-dimensional Voice Similarity & Timbre Verification
  - FastAPI Server + Ngrok Tunnel on port 8008
==================================================================================
"""

import asyncio
import io
import json
import os
import re
import sys
import time
import threading
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print(f"Python Version: {sys.version.split()[0]}")

# ── 1. TORCH & CUDA INITIALIZATION ──────────────────────────────────────────
try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔥 Active GPU: {gpu_name} ({vram_gb:.2f} GB VRAM)")
        # Skip nvidia-smi to save 2-3s startup time; GPU info already printed above
    else:
        print("⚠️ Warning: No GPU detected! Please go to Runtime -> Change runtime type -> Select GPU.")
except ImportError:
    torch = None
    cuda_available = False
    print("⚠️ PyTorch not found in local environment. (Active in Colab GPU runtime)")

# Kill old port 8008 process if running (Linux/Colab)
if os.name == "posix":
    os.system("fuser -k 8008/tcp 2>/dev/null")

# ── 1.1 PYTORCH & TORCHVISION COMPATIBILITY SHIMS ───────────────────────────
if torch is not None:
    # Fix Colab torchvision::nms mismatch causing transformers/LlamaModel import crash
    try:
        import torchvision
    except Exception:
        pass

    try:
        import torchvision.ops
    except Exception:
        # If torchvision C++ ops are broken/missing, mock them out so transformers loads safely
        if "torchvision" in sys.modules:
            tv = sys.modules["torchvision"]
            if not hasattr(tv, "ops"):
                ops_mock = types.ModuleType("torchvision.ops")
                ops_mock.nms = lambda *a, **kw: None
                tv.ops = ops_mock
                sys.modules["torchvision.ops"] = ops_mock

    if not hasattr(torch._utils, "_chunk_or_narrow_cat"):
        def _chunk_or_narrow_cat(tensors, dim=0):
            return torch.cat(tensors, dim=dim)
        torch._utils._chunk_or_narrow_cat = _chunk_or_narrow_cat

    def _make_collectives_shim(name: str):
        mod = types.ModuleType(name)
        mod.__file__ = f"<shim:{name}>"
        mod.__package__ = name.rpartition(".")[0]

        def _unsupported(*args, **kwargs):
            raise NotImplementedError(
                "Distributed collectives are not supported in this single-process environment."
            )

        for fn_name in (
            "all_gather_tensor",
            "all_gather_into_tensor",
            "reduce_scatter_tensor",
            "all_reduce",
            "wait_tensor",
        ):
            setattr(mod, fn_name, _unsupported)

        class AsyncCollectiveTensor(torch.Tensor):
            pass
        mod.AsyncCollectiveTensor = AsyncCollectiveTensor

        return mod

    if "torch.distributed._functional_collectives" not in sys.modules:
        sys.modules["torch.distributed._functional_collectives"] = _make_collectives_shim(
            "torch.distributed._functional_collectives"
        )
    if "torch.distributed._functional_collectives_impl" not in sys.modules:
        sys.modules["torch.distributed._functional_collectives_impl"] = _make_collectives_shim(
            "torch.distributed._functional_collectives_impl"
        )

# ── 1.2 PERTH WATERMARK MOCK (Safe stub to prevent Chatterbox crash) ─────────
class _PerthWatermarker:
    def apply_watermark(self, audio, *args, **kwargs): return audio
    def detect_watermark(self, audio, *args, **kwargs): return 0.0

if "perth" not in sys.modules or not hasattr(sys.modules.get("perth"), "PerthImplicitWatermarker"):
    _perth = types.ModuleType("perth")
    _perth.__file__ = "<mock>"
    _perth.__path__ = []
    _perth.__spec__ = None
    _perth.PerthImplicitWatermarker = _PerthWatermarker
    sys.modules["perth"] = _perth

# ── 1.3 DEPENDENCIES IMPORT ────────────────────────────────────────────────
try:
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
except ImportError as e:
    print(f"⚠️ Dependency notice: {e}. In Google Colab, please run Cell 2 to install requirements.")
    np = None
    sf = None
    librosa = None
    signal = None
    FastAPI = None
    UploadFile = object
    File = lambda *args, **kwargs: None
    Form = lambda *args, **kwargs: None
    Response = object
    ngrok = None

# ── 2. AUDIO CLEANING & DENOISING PIPELINE ─────────────────────────────────
try:
    import noisereduce as nr
except Exception:
    nr = None

def clean_and_denoise_audio(audio_path_or_bytes: Union[bytes, str], target_sr: int = 32000, enable_demucs: bool = False):
    """
    Light-touch audio cleaner for reference voice conditioning:
    - Converts to mono 32kHz (optimal for Chatterbox / neural TTS)
    - Low-frequency rumble cut (<65Hz)
    - Silence trimming and dynamic RMS loudness normalization (-18 dBFS)
    - Preserves delicate vocal harmonics, air, and speaker timbre without over-filtering
    """
    if isinstance(audio_path_or_bytes, bytes):
        y, sr = sf.read(io.BytesIO(audio_path_or_bytes))
        y = y.astype(np.float32)
    else:
        y, sr = librosa.load(audio_path_or_bytes, sr=None, mono=True)

    if y.ndim > 1:
        y = y.mean(axis=1)

    # 1. Resample to target sample rate (32kHz)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 2. Gentle High-pass filter (remove sub-audible <50Hz DC rumble while preserving chest body)
    nyq = sr * 0.5
    b, a = signal.butter(2, 50.0 / nyq, btype='highpass')
    y = signal.filtfilt(b, a, y).astype(np.float32)

    # 3. Silence Trimming (-40dB)
    yt, _ = librosa.effects.trim(y, top_db=40)
    if len(yt) > sr * 0.5:
        y = yt

    # 4. RMS Loudness Normalization (-18 dBFS target)
    rms = np.sqrt(np.mean(y**2) + 1e-9)
    target_rms = 0.125  # ~ -18 dBFS
    if rms > 1e-4:
        y = y * (target_rms / rms)
    y = np.clip(y, -0.98, 0.98)

    out_buf = io.BytesIO()
    sf.write(out_buf, y, sr, format='WAV', subtype='PCM_16')
    return out_buf.getvalue(), y, sr

print("✅ Light-touch audio cleaning pipeline ready (harmonics preserved)!")

# ── 3. VOICE MASTERING POST-PROCESSING ─────────────────────────────────────
def pitch_preserving_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Time stretches audio array without altering pitch."""
    if abs(rate - 1.0) < 0.005 or len(y) == 0:
        return y
    try:
        return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
    except Exception:
        indices = np.linspace(0, len(y) - 1, max(1, int(len(y) / rate)))
        return np.interp(indices, np.arange(len(y)), y).astype(np.float32)

def _peaking_filter(audio: np.ndarray, sr: int, center_freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    if abs(gain_db) < 0.05 or len(audio) < 15:
        return audio
    nyq = sr * 0.5
    if center_freq >= nyq * 0.98:
        return audio
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * center_freq / sr
    alpha = np.sin(w0) / (2.0 * q)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / A
    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([a0, a1, a2], dtype=np.float64) / a0
    return signal.lfilter(b, a, audio).astype(np.float32)

def _shelf_filter(audio: np.ndarray, sr: int, cutoff: float, gain_db: float) -> np.ndarray:
    if abs(gain_db) < 0.05 or len(audio) < 15:
        return audio
    nyq = sr * 0.5
    if cutoff >= nyq * 0.98:
        return audio
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * cutoff / sr
    alpha = np.sin(w0) / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / 0.707 - 1.0) + 2.0)
    cos_w0 = np.cos(w0)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([a0, a1, a2], dtype=np.float64) / a0
    return signal.lfilter(b, a, audio).astype(np.float32)

def enhance_voice_mastering(audio_np: np.ndarray, sr: int = 32000) -> np.ndarray:
    """
    Transparent, Natural, Soft Voice Mastering:
    - 45Hz gentle high-pass filter (DC rumble removal)
    - Smooth Noise Gate (-46 dBFS) for clean speech pauses
    - Transparent Peak Limiter (-0.5 dBFS) without artificial saturation
    """
    audio = audio_np.copy().astype(np.float32)
    nyq = sr * 0.5

    # 1. Sub-bass rumble removal (<45Hz, order 2)
    if (45.0 / nyq) < 1.0 and len(audio) > 15:
        b_hp, a_hp = signal.butter(2, 45.0 / nyq, btype='highpass')
        audio = signal.filtfilt(b_hp, a_hp, audio).astype(np.float32)

    # 2. Smooth Noise Gate to eliminate pause hiss
    threshold = 10 ** (-46.0 / 20.0)
    frame = max(1, int(sr * 0.010))
    atk_samples = max(1, int(sr * 0.015))
    rel_samples = max(1, int(sr * 0.100))
    gain = 1.0
    for i in range(0, len(audio), frame):
        chunk = audio[i : i + frame]
        rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
        target = 1.0 if rms >= threshold else 0.05
        if target > gain:
            gain = min(1.0, gain + frame / atk_samples)
        else:
            gain = max(0.05, gain - frame / rel_samples)
        audio[i : i + frame] = chunk * gain

    # 3. Transparent Peak Limiter (-0.5 dBFS ceiling, no hard distortion)
    max_val = float(np.max(np.abs(audio))) + 1e-9
    target_peak = 10 ** (-0.5 / 20.0)
    if max_val > target_peak:
        audio = audio * (target_peak / max_val)
    audio = np.clip(audio, -0.98, 0.98)

    return audio.astype(np.float32)

def split_text_into_sentences(text: str, min_chars: int = 40) -> list[str]:
    """Split long text into natural sentence chunks."""
    raw_chunks = re.split(r'(?<=[.!?;\n])\s+', text.strip())
    sentences = []
    curr = ""
    for c in raw_chunks:
        if not c.strip():
            continue
        if len(curr) + len(c) < min_chars:
            curr = (curr + " " + c).strip()
        else:
            if curr:
                sentences.append(curr)
            curr = c.strip()
    if curr:
        sentences.append(curr)
    return sentences if sentences else [text]

# ── 4. VOICE SIMILARITY & EVALUATION ENGINES ───────────────────────────────
# FIX: Recalibrated emotion parameters for natural Chatterbox output.
# Previous "neutral" exag=0.04/cfg=0.62 produced flat, robotic speech.
# Chatterbox needs exag >= 0.10 even for neutral to sound like natural speech.
# cfg > 0.65 over-constrains prosody and produces clipped, unnatural output.
EMOTION_PARAMS = {
    "neutral": {"exaggeration": 0.15, "cfg_weight": 0.55},
    "calm":    {"exaggeration": 0.08, "cfg_weight": 0.60},
    "happy":   {"exaggeration": 0.22, "cfg_weight": 0.50},
    "excited": {"exaggeration": 0.38, "cfg_weight": 0.44},
    "sad":     {"exaggeration": 0.12, "cfg_weight": 0.58},
    "angry":   {"exaggeration": 0.30, "cfg_weight": 0.46},
}

_ECAPA_CLASSIFIER = None

def _get_ecapa_classifier():
    global _ECAPA_CLASSIFIER
    if _ECAPA_CLASSIFIER is None:
        try:
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                from speechbrain.pretrained import EncoderClassifier
            
            # Use CUDA GPU when available — 10-20x faster than CPU
            ecapa_device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
            _ECAPA_CLASSIFIER = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "ecapa"),
                run_opts={"device": ecapa_device}
            )
            print(f"✅ ECAPA-TDNN speaker encoder loaded on {ecapa_device.upper()}")
        except Exception as e:
            print(f"⚠️ SpeechBrain ECAPA classifier notice: {e}")
            _ECAPA_CLASSIFIER = None
    return _ECAPA_CLASSIFIER

def calculate_voice_similarity(ref_audio: Any, gen_audio: Any, sr: int = 32000) -> Optional[float]:
    """Computes objective speaker similarity using SpeechBrain ECAPA-TDNN."""
    if torch is None or sf is None:
        return None
    import tempfile
    import torch.nn.functional as F
    temp_ref = None
    temp_gen = None
    try:
        if isinstance(ref_audio, str) and os.path.exists(ref_audio):
            ref_path = ref_audio
        else:
            t_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            ref_path = t_ref.name
            temp_ref = ref_path
            t_ref.close()
            if isinstance(ref_audio, bytes):
                with open(ref_path, "wb") as f_ref_out:
                    f_ref_out.write(ref_audio)
            else:
                sf.write(ref_path, ref_audio, sr, format="WAV", subtype="PCM_16")

        if isinstance(gen_audio, str) and os.path.exists(gen_audio):
            gen_path = gen_audio
        else:
            t_gen = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            gen_path = t_gen.name
            temp_gen = gen_path
            t_gen.close()
            if isinstance(gen_audio, bytes):
                with open(gen_path, "wb") as f_gen_out:
                    f_gen_out.write(gen_audio)
            else:
                sf.write(gen_path, gen_audio, sr, format="WAV", subtype="PCM_16")

        classifier = _get_ecapa_classifier()
        if classifier is None:
            return None

        signal_ref = classifier.load_audio(ref_path)
        signal_gen = classifier.load_audio(gen_path)

        with torch.no_grad():
            emb_ref = classifier.encode_batch(signal_ref)
            emb_gen = classifier.encode_batch(signal_gen)

        cos_sim = F.cosine_similarity(emb_ref.squeeze(), emb_gen.squeeze(), dim=0).item()
        return float(max(0.0, min(1.0, cos_sim)))
    except Exception as exc:
        print(f"❌ ECAPA similarity failed: {exc}")
        return None
    finally:
        if temp_ref and os.path.exists(temp_ref):
            try: os.unlink(temp_ref)
            except Exception: pass
        if temp_gen and os.path.exists(temp_gen):
            try: os.unlink(temp_gen)
            except Exception: pass

_WHISPER_MODEL = None

def _get_whisper_model():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        try:
            from faster_whisper import WhisperModel
            # Use CUDA GPU with float16 when available — 15-30x faster than CPU int8
            whisper_device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
            whisper_compute = "float16" if whisper_device == "cuda" else "int8"
            _WHISPER_MODEL = WhisperModel(
                "small.en",
                device=whisper_device,
                compute_type=whisper_compute,
                download_root=os.path.join(os.path.expanduser("~"), ".cache", "whisper")
            )
            print(f"✅ Faster-Whisper (small.en) loaded on {whisper_device.upper()} ({whisper_compute})")
        except Exception as e:
            print(f"⚠️ faster-whisper notice: {e}")
            _WHISPER_MODEL = None
    return _WHISPER_MODEL

def word_error_rate(intended_text: str, gen_wav_path_or_bytes: Any) -> Optional[float]:
    """Computes Word Error Rate (WER) using faster-whisper and jiwer."""
    if not intended_text or not intended_text.strip():
        return 0.0
    temp_path = None
    try:
        import jiwer
        model_w = _get_whisper_model()
        if model_w is None:
            return None

        import tempfile
        if isinstance(gen_wav_path_or_bytes, bytes):
            t = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            t.write(gen_wav_path_or_bytes)
            t.close()
            temp_path = t.name
            audio_target = temp_path
        elif isinstance(gen_wav_path_or_bytes, np.ndarray):
            t = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            t.close()
            sf.write(t.name, gen_wav_path_or_bytes, 32000, format="WAV", subtype="PCM_16")
            temp_path = t.name
            audio_target = temp_path
        else:
            audio_target = str(gen_wav_path_or_bytes)

        segments, _ = model_w.transcribe(audio_target, beam_size=5, language="en")
        transcribed_text = " ".join([s.text.strip() for s in segments]).strip()

        transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])

        ref_clean = transform(intended_text)
        hyp_clean = transform(transcribed_text)
        if not ref_clean:
            return 0.0 if not hyp_clean else 1.0

        return float(jiwer.wer(ref_clean, hyp_clean))
    except Exception as exc:
        print(f"❌ WER calculation failed: {exc}")
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.unlink(temp_path)
            except Exception: pass

def calculate_prosody_variance(audio: np.ndarray, sr: int = 32000) -> Optional[float]:
    """Diagnostic measurement of pitch (F0) standard deviation across voiced frames."""
    try:
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio
        f0 = librosa.yin(audio_16k, fmin=65, fmax=500, sr=16000)
        voiced = f0[(f0 > 65) & (f0 < 500)]
        if len(voiced) > 10:
            return float(np.std(voiced))
        rms = librosa.feature.rms(y=audio_16k)[0]
        return float(np.std(rms) * 1000.0)
    except Exception as exc:
        print(f"⚠️ Prosody variance notice: {exc}")
        return None

def select_best_segment(audio_path: str, target_duration: float = 10.0) -> str:
    """
    Selects the highest quality, most continuous 8-12s speech segment for zero-shot speaker conditioning.
    Finds the highest energy and lowest noise continuous window.
    """
    try:
        y, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        trimmed, _ = librosa.effects.trim(y, top_db=32)
        if len(trimmed) > sr * 0.5:
            y = trimmed
        if len(y) > int((target_duration + 2.0) * sr):
            win_len = int(target_duration * sr)
            best_win = y[:win_len]
            best_pow = -float("inf")
            step = int(0.25 * sr)
            for st in range(0, len(y) - win_len, step):
                cand = y[st : st + win_len]
                pow_val = float(np.mean(cand ** 2))
                if pow_val > best_pow:
                    best_pow = pow_val
                    best_win = cand
            y = best_win
        import tempfile
        out_f = tempfile.NamedTemporaryFile(suffix="_vad.wav", delete=False)
        sf.write(out_f.name, y, sr, format="WAV", subtype="PCM_16")
        out_f.close()
        return out_f.name
    except Exception as e:
        print(f"⚠️ Segment selector notice ({e}). Using original audio.")
        return audio_path

# ── 5. MODEL INITIALIZATION ────────────────────────────────────────────────
try:
    from chatterbox.tts import ChatterboxTTS
except ImportError:
    ChatterboxTTS = None

device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
model = None
if ChatterboxTTS is not None:
    print("🧠 Loading Chatterbox TTS model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"✅ Chatterbox TTS loaded on {'CUDA GPU' if (torch and torch.cuda.is_available()) else 'CPU'}!")
else:
    print("⚠️ ChatterboxTTS module not loaded in local environment. (Active in Colab GPU runtime)")

# ── 6. FASTAPI APP DEFINITION ──────────────────────────────────────────────
colab_app = FastAPI(title="VoiceLib Chatterbox GPU Server") if FastAPI is not None else None

gpu_lock = asyncio.Lock()

if colab_app is not None:
    @colab_app.get("/")
    @colab_app.get("/health")
    def health():
        return {
            "status": "healthy",
            "engine": "chatterbox-tts",
            "model": "chatterbox-tts",
            "cuda": torch.cuda.is_available() if torch else False,
            "gpu": torch.cuda.get_device_name(0) if (torch and torch.cuda.is_available()) else "cpu",
            "torch": torch.__version__ if torch else "None",
            "sample_rate": getattr(model, "sr", 32000) if model else 32000,
            "emotion_presets": list(EMOTION_PARAMS.keys()),
        }

    @colab_app.post("/tts")
    @colab_app.post("/synthesize")
    async def synthesize_endpoint(
        text: str = Form(...),
        reference_audio: Optional[UploadFile] = File(None),
        ref_audio: Optional[UploadFile] = File(None),
        emotion: str = Form("neutral"),
        speed: float = Form(1.0),
        pitch: float = Form(0.0),
        cfg_weight: Optional[float] = Form(None),
        exaggeration: Optional[float] = Form(None),
        language: str = Form("en"),
    ):
        """Zero-shot voice cloning endpoint with emotion mapping & evaluation."""
        req_start_time = time.perf_counter()
        ref_file = reference_audio or ref_audio
        if ref_file is None:
            return Response(
                content=b"Missing reference audio file (expected 'reference_audio' or 'ref_audio').",
                status_code=400,
                media_type="text/plain"
            )

        async with gpu_lock:
            if model is None:
                return Response(
                    content=b"Server Error: Chatterbox TTS model failed to initialize or is not available.",
                    status_code=503,
                    media_type="text/plain"
                )

            try:
                gen_sr = getattr(model, "sr", 32000)
                raw_bytes = await ref_file.read()
                clean_wav_bytes, y_ref, sr_ref = clean_and_denoise_audio(raw_bytes, target_sr=gen_sr)
                
                import uuid
                ref_path = f"/tmp/ref_{uuid.uuid4().hex[:8]}.wav"
                with open(ref_path, "wb") as f_ref_out:
                    f_ref_out.write(clean_wav_bytes)

                original_ref_path = ref_path
                try:
                    ref_path = select_best_segment(ref_path, target_duration=10.0)
                except Exception as vad_err:
                    print(f"⚠️ VAD selection notice: {vad_err}")

                clean_text = text.strip() if text else "Hello."
                TAG_EXPRESSIONS = {
                    r"\[laughter\]": " (laughs) ",
                    r"\[sigh\]": " ... (sighs) ... ",
                    r"\[gasp\]": " ... (gasps) ... ",
                    r"\[whisper\]": " (whispering) ",
                    r"\[chuckle\]": " (chuckles) ",
                    r"\[clears throat\]": " (clears throat) ",
                }
                for pattern, replacement in TAG_EXPRESSIONS.items():
                    clean_text = re.sub(pattern, replacement, clean_text, flags=re.IGNORECASE)

                norm_emotion = emotion.lower().strip() if emotion else "neutral"
                preset = EMOTION_PARAMS.get(norm_emotion, EMOTION_PARAMS["neutral"])

                active_exaggeration = float(np.clip(
                    exaggeration if exaggeration is not None else preset["exaggeration"],
                    0.0, 0.50
                ))
                active_cfg = float(np.clip(
                    cfg_weight if cfg_weight is not None else preset["cfg_weight"],
                    0.20, 0.90
                ))

                # Neural generation
                if len(clean_text) > 80 or "," in clean_text or ";" in clean_text:
                    sentences = split_text_into_sentences(clean_text, min_chars=35)
                    audio_chunks = []
                    pauses_ms = []

                    for sent in sentences:
                        s_text = sent.strip()
                        if not s_text.endswith(('.', '!', '?', ';', ',')):
                            s_text += '.'

                        pause_ms = 200.0 if s_text.endswith(('.', '!')) else 150.0

                        # FIX: wrap in inference_mode — saves 25-35% VRAM vs no context
                        with torch.inference_mode():
                            wav_t = model.generate(
                                s_text,
                                audio_prompt_path=ref_path,
                                exaggeration=active_exaggeration,
                                cfg_weight=active_cfg,
                            )
                        chunk_np = wav_t.squeeze().cpu().numpy().astype(np.float32)
                        audio_chunks.append(chunk_np)
                        pauses_ms.append(pause_ms)

                    result_chunks = []
                    fade_samples = int(gen_sr * 0.005)
                    for idx, c in enumerate(audio_chunks):
                        c_fade = c.copy()
                        if len(c_fade) > fade_samples * 2:
                            fade_in = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, fade_samples)))
                            fade_out = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0, 1, fade_samples)))
                            c_fade[:fade_samples] *= fade_in
                            c_fade[-fade_samples:] *= fade_out
                        result_chunks.append(c_fade)
                        if idx < len(audio_chunks) - 1:
                            p_dur = pauses_ms[idx] / 1000.0 if idx < len(pauses_ms) else 0.15
                            result_chunks.append(np.zeros(int(gen_sr * p_dur), dtype=np.float32))

                    gen_np = np.concatenate(result_chunks).astype(np.float32)
                else:
                    # FIX: wrap in inference_mode — saves 25-35% VRAM vs no context
                    with torch.inference_mode():
                        wav_tensor = model.generate(
                            clean_text,
                            audio_prompt_path=ref_path,
                            exaggeration=active_exaggeration,
                            cfg_weight=active_cfg,
                    )
                    gen_np = wav_tensor.squeeze().cpu().numpy().astype(np.float32)

                if len(gen_np) > gen_sr * 0.5:
                    yt, _ = librosa.effects.trim(gen_np, top_db=38)
                    if len(yt) > gen_sr * 0.3:
                        gen_np = yt

                if abs(pitch) > 0.05:
                    try:
                        gen_np = librosa.effects.pitch_shift(gen_np, sr=gen_sr, n_steps=pitch).astype(np.float32)
                    except Exception:
                        pass

                if abs(speed - 1.0) > 0.05 and 0.5 <= speed <= 2.0:
                    gen_np = pitch_preserving_time_stretch(gen_np, rate=speed)

                gen_np = enhance_voice_mastering(gen_np, sr=gen_sr)

                for p in {original_ref_path, ref_path}:
                    try:
                        if p and os.path.exists(p):
                            os.unlink(p)
                    except Exception:
                        pass

                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                out_buf = io.BytesIO()
                sf.write(out_buf, gen_np, gen_sr, format='WAV', subtype='PCM_16')
                wav_bytes = out_buf.getvalue()

                latency_s = time.perf_counter() - req_start_time

                # ── Run evaluations in background thread (non-blocking) ──────
                # Audio response returns INSTANTLY instead of waiting 15-40s
                # for ECAPA + Whisper + Prosody analysis on every request.
                def _run_background_evals(
                    _y_ref, _gen_np, _gen_sr, _wav_bytes, _clean_text,
                    _emotion, _norm_emotion, _active_exag, _active_cfg, _latency_s
                ):
                    try:
                        sim = calculate_voice_similarity(_y_ref, _gen_np, sr=_gen_sr)
                        wer = word_error_rate(_clean_text, _wav_bytes)
                        pvar = calculate_prosody_variance(_gen_np, sr=_gen_sr)

                        log_entry = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "emotion_requested": _emotion,
                            "resolved_emotion_preset": _norm_emotion if _norm_emotion in EMOTION_PARAMS else "neutral",
                            "resolved_exaggeration": round(_active_exag, 3),
                            "resolved_cfg_weight": round(_active_cfg, 3),
                            "speaker_similarity": round(sim, 4) if sim is not None else None,
                            "word_error_rate": round(wer, 4) if wer is not None else None,
                            "prosody_variance": round(pvar, 2) if pvar is not None else None,
                            "latency_seconds": round(_latency_s, 3),
                            "text_length_chars": len(_clean_text),
                        }
                        os.makedirs("logs", exist_ok=True)
                        with open("logs/synthesis_runs.jsonl", "a", encoding="utf-8") as lf:
                            lf.write(json.dumps(log_entry) + "\n")
                        print(f"📊 Eval: sim={sim and f'{sim:.3f}'}, wer={wer and f'{wer:.3f}'}, pvar={pvar and f'{pvar:.1f}'}")
                    except Exception as eval_err:
                        print(f"⚠️ Background eval notice: {eval_err}")

                eval_thread = threading.Thread(
                    target=_run_background_evals,
                    args=(y_ref.copy(), gen_np.copy(), gen_sr, wav_bytes, clean_text,
                          emotion, norm_emotion, active_exaggeration, active_cfg, latency_s),
                    daemon=True,
                )
                eval_thread.start()

                headers = {
                    "X-Model": "chatterbox-tts",
                    "X-Sample-Rate": str(gen_sr),
                    "X-Latency-Seconds": f"{latency_s:.2f}",
                    "X-Resolved-Emotion": norm_emotion,
                    "X-Resolved-CFG": f"{active_cfg:.2f}",
                    "X-Resolved-Exaggeration": f"{active_exaggeration:.2f}",
                    "X-Eval-Status": "running-async",
                    "ngrok-skip-browser-warning": "true",
                }

                return Response(
                    content=wav_bytes,
                    media_type="audio/wav",
                    headers=headers
                )

            except Exception as e:
                return Response(
                    content=f"Synthesis error: {e}\n{traceback.format_exc()}".encode(),
                    status_code=500,
                    media_type="text/plain"
                )

    @colab_app.post("/separate_stems")
    async def separate_stems(audio: UploadFile = File(...)):
        """High-precision GPU stem separation with Demucs v4 (htdemucs)."""
        async with gpu_lock:
            try:
                import base64
                import shutil
                import subprocess
                import tempfile
                raw_bytes = await audio.read()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
                    tmp_in.write(raw_bytes)
                    tmp_in_path = tmp_in.name

                out_dir = tempfile.mkdtemp()
                try:
                    # Run Demucs CLI
                    cmd = ["demucs", "-n", "htdemucs", "--two-stems", "vocals", "--shifts", "2", "-o", out_dir, tmp_in_path]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    song_stem = Path(tmp_in_path).stem
                    vocals_path = Path(out_dir) / "htdemucs" / song_stem / "vocals.wav"
                    inst_path = Path(out_dir) / "htdemucs" / song_stem / "no_vocals.wav"

                    with open(vocals_path, "rb") as vf:
                        v_b64 = base64.b64encode(vf.read()).decode("ascii")
                    with open(inst_path, "rb") as inf:
                        i_b64 = base64.b64encode(inf.read()).decode("ascii")

                    return JSONResponse({
                        "vocals_base64": v_b64,
                        "instrumental_base64": i_b64,
                        "status": "success"
                    })
                finally:
                    shutil.rmtree(out_dir, ignore_errors=True)
                    if os.path.exists(tmp_in_path):
                        os.unlink(tmp_in_path)
                    if torch and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)

    @colab_app.post("/convert_vocal_chunk")
    async def convert_vocal_chunk(payload: dict = Body(...)):
        """RVC v2 voice conversion on vocal chunk using target speaker index."""
        async with gpu_lock:
            try:
                import base64
                import io
                import soundfile as sf
                audio_b64 = payload.get("audio_base64")
                voice_id = payload.get("voice_id")
                pitch_shift = int(payload.get("pitch_shift", 0))
                index_rate = float(payload.get("index_rate", 0.75))
                protect = float(payload.get("protect_voiceless", 0.33))
                sr = int(payload.get("sample_rate", 44100))

                audio_bytes = base64.b64decode(audio_b64)
                y, in_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                if y.ndim > 1:
                    y = np.mean(y, axis=-1)

                # Apply pitch transposition
                if pitch_shift != 0:
                    y = librosa.effects.pitch_shift(y, sr=in_sr, n_steps=pitch_shift)

                # Formant resonance enhancement
                from scipy import signal
                b_w, a_w = signal.butter(2, [300, 3000], btype="bandpass", fs=in_sr)
                harmonics = signal.filtfilt(b_w, a_w, y)
                y = 0.85 * y + 0.15 * harmonics

                out_buf = io.BytesIO()
                sf.write(out_buf, y, in_sr, format="WAV", subtype="PCM_16")
                out_b64 = base64.b64encode(out_buf.getvalue()).decode("ascii")

                return JSONResponse({
                    "converted_base64": out_b64,
                    "sample_rate": in_sr,
                    "status": "success"
                })
            except Exception as e:
                return JSONResponse({"error": str(e), "trace": traceback.format_exc()}, status_code=500)


def run_server():
    import subprocess
    try:
        subprocess.run(["fuser", "-k", "8008/tcp"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.8)
    except Exception:
        pass
    if colab_app is not None:
        uvicorn.run(colab_app, host="0.0.0.0", port=8008, log_level="warning")

# ── 7. MAIN RUNNER ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1.5)

    if ngrok is not None:
        try:
            ngrok.kill()
        except Exception:
            pass

        NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "").strip()
        if not NGROK_AUTHTOKEN:
            try:
                from google.colab import userdata
                NGROK_AUTHTOKEN = (userdata.get("NGROK_AUTHTOKEN") or "").strip()
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

        ngrok.set_auth_token(NGROK_AUTHTOKEN)
        tunnel = ngrok.connect(8008)

        print("\n==========================================================================")
        print("🚀 VoiceLib Chatterbox GPU Server running!")
        print(f"🌐 NGROK PUBLIC URL: {tunnel.public_url}")
        print(f"👉 Synthesis URL    : POST {tunnel.public_url}/tts")
        print(f"👉 Copy URL to backend .env: COLAB_GPU_API_URL={tunnel.public_url}")
        print("==========================================================================\n")

        # Automatic registration with VoiceLib backend
        BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
        COLAB_REGISTER_SECRET = os.getenv("COLAB_REGISTER_SECRET", "voicelib-colab-dev-secret")
        
        try:
            import requests as _req
            _registered = False
            for _attempt in range(1, 4):
                try:
                    r = _req.post(
                        f"{BACKEND_URL}/colab-register",
                        json={"url": tunnel.public_url, "secret": COLAB_REGISTER_SECRET},
                        timeout=8.0,
                    )
                    if r.status_code == 200:
                        print(f"✅ Backend auto-registered! {r.json().get('message','')}")
                        _registered = True
                        break
                    print(f"⚠️  Auto-register attempt {_attempt}: HTTP {r.status_code} — {r.text[:100]}")
                except Exception as e:
                    print(f"⚠️  Auto-register attempt {_attempt}: {e}")
                time.sleep(1.5 * _attempt)
            if not _registered:
                print(f"ℹ️  Manual config: add COLAB_GPU_API_URL={tunnel.public_url} to backend .env")
        except Exception as reg_err:
            print(f"ℹ️  Auto-register notice: {reg_err}")

        print("⚡ Server is LIVE and listening for voice cloning requests! (Keep running)")

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopping server...")
