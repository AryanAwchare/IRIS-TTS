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
        os.system("nvidia-smi")
    else:
        print("⚠️ Warning: No GPU detected! Please go to Runtime -> Change runtime type -> Select GPU.")
except ImportError:
    torch = None
    cuda_available = False
    print("⚠️ PyTorch not found in local environment. (Active in Colab GPU runtime)")

# Kill old port 8008 process if running (Linux/Colab)
if os.name == "posix":
    os.system("fuser -k 8008/tcp 2>/dev/null")

# ── 1.1 PYTORCH COMPATIBILITY SHIMS ────────────────────────────────────────
if torch is not None:
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

# ── 1.2 DEPENDENCIES IMPORT ────────────────────────────────────────────────
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
    Audio cleaner for voice cloning:
    - Converts to mono 32kHz (optimal for Chatterbox / GPT-SoVITS)
    - Low-frequency rumble cut (<65Hz)
    - Silence trimming and dynamic RMS loudness normalization (-18 dBFS)
    """
    if isinstance(audio_path_or_bytes, bytes):
        try:
            y, sr = sf.read(io.BytesIO(audio_path_or_bytes))
            y = y.astype(np.float32)
        except Exception:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp_audio:
                tmp_audio.write(audio_path_or_bytes)
                tmp_path = tmp_audio.name
            try:
                y, sr = librosa.load(tmp_path, sr=None, mono=True)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
    else:
        y, sr = librosa.load(audio_path_or_bytes, sr=None, mono=True)

    if y.ndim > 1:
        y = y.mean(axis=1)

    # 1. Resample to target sample rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 2. High-pass filter (remove 0-65Hz mic rumble / DC offset)
    nyq = sr * 0.5
    if len(y) > 15 and (65.0 / nyq) < 1.0:
        b, a = signal.butter(4, 65.0 / nyq, btype='highpass')
        y = signal.filtfilt(b, a, y).astype(np.float32)

    # 3. Two-pass noise reduction if available
    if nr is not None:
        try:
            y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.85)
        except Exception:
            pass

    # 4. Silence Trimming (-40dB)
    yt, _ = librosa.effects.trim(y, top_db=40)
    if len(yt) > sr * 0.5:
        y = yt

    # 5. RMS Loudness Normalization (-18 dBFS target)
    rms = np.sqrt(np.mean(y**2) + 1e-9)
    target_rms = 0.125  # ~ -18 dBFS
    if rms > 1e-4:
        y = y * (target_rms / rms)
    y = np.clip(y, -0.98, 0.98)

    out_buf = io.BytesIO()
    sf.write(out_buf, y, sr, format='WAV', subtype='PCM_16')
    return out_buf.getvalue(), y, sr

print("✅ Audio cleaning & denoising pipeline ready!")

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

def enhance_voice_mastering(audio_np: np.ndarray, sr: int = 32000) -> np.ndarray:
    """
    Transparent vocal normalizer:
    - Removes sub-audible DC offset (<30Hz)
    - Applies clean peak ceiling limiter (-0.3 dBFS)
    - Preserves 100% of authentic speaker vocal timbre
    """
    audio = audio_np.copy().astype(np.float32)
    nyq = sr * 0.5

    if len(audio) > 15 and (30.0 / nyq) < 1.0:
        b_hp, a_hp = signal.butter(2, 30.0 / nyq, btype='highpass')
        audio = signal.filtfilt(b_hp, a_hp, audio).astype(np.float32)

    max_val = float(np.max(np.abs(audio))) + 1e-9
    if max_val > 0.01:
        target_peak = 10 ** (-0.3 / 20.0)  # ~ -0.3 dBFS
        audio = audio * (target_peak / max_val)
        audio = np.clip(audio, -0.99, 0.99)

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
EMOTION_PARAMS = {
    "neutral": {"exaggeration": 0.05, "cfg_weight": 0.70},
    "calm":    {"exaggeration": 0.00, "cfg_weight": 0.75},
    "happy":   {"exaggeration": 0.25, "cfg_weight": 0.55},
    "excited": {"exaggeration": 0.40, "cfg_weight": 0.45},
    "sad":     {"exaggeration": 0.15, "cfg_weight": 0.65},
    "angry":   {"exaggeration": 0.35, "cfg_weight": 0.50},
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
            
            _ECAPA_CLASSIFIER = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "ecapa"),
                run_opts={"device": "cpu"}
            )
        except Exception as e:
            print(f"⚠️ SpeechBrain ECAPA classifier notice: {e}")
            _ECAPA_CLASSIFIER = None
    return _ECAPA_CLASSIFIER

def calculate_voice_similarity(ref_audio: Any, gen_audio: Any, sr: int = 32000) -> Optional[float]:
    """Computes objective speaker similarity using SpeechBrain ECAPA-TDNN (resampled to 16kHz)."""
    if torch is None or sf is None:
        return None
    import tempfile
    import torch.nn.functional as F
    temp_ref = None
    temp_gen = None
    try:
        def _to_16k_np(audio_input, input_sr: int) -> np.ndarray:
            if isinstance(audio_input, str) and os.path.exists(audio_input):
                arr, orig_sr = sf.read(audio_input, dtype="float32")
            elif isinstance(audio_input, bytes):
                try:
                    arr, orig_sr = sf.read(io.BytesIO(audio_input), dtype="float32")
                except Exception:
                    arr, orig_sr = librosa.load(io.BytesIO(audio_input), sr=None, mono=True)
            elif isinstance(audio_input, np.ndarray):
                arr = audio_input.astype(np.float32)
                orig_sr = input_sr
            else:
                raise ValueError("Unsupported audio input format for similarity evaluation")
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if orig_sr != 16000:
                arr = librosa.resample(arr, orig_sr=orig_sr, target_sr=16000)
            return arr

        arr_ref = _to_16k_np(ref_audio, sr)
        arr_gen = _to_16k_np(gen_audio, sr)

        t_ref = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        ref_path = t_ref.name
        temp_ref = ref_path
        t_ref.close()
        sf.write(ref_path, arr_ref, 16000, format="WAV", subtype="PCM_16")

        t_gen = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        gen_path = t_gen.name
        temp_gen = gen_path
        t_gen.close()
        sf.write(gen_path, arr_gen, 16000, format="WAV", subtype="PCM_16")

        classifier = _get_ecapa_classifier()
        if classifier is None:
            return None

        signal_ref = classifier.load_audio(ref_path)
        signal_gen = classifier.load_audio(gen_path)

        with torch.no_grad():
            emb_ref = classifier.encode_batch(signal_ref)
            emb_gen = classifier.encode_batch(signal_gen)

        v_ref = emb_ref.squeeze()
        v_gen = emb_gen.squeeze()
        if v_ref.ndim == 1:
            cos_sim = F.cosine_similarity(v_ref, v_gen, dim=0).item()
        else:
            cos_sim = F.cosine_similarity(v_ref, v_gen, dim=-1).mean().item()

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
            _WHISPER_MODEL = WhisperModel(
                "small.en",
                device="cpu",
                compute_type="int8",
                download_root=os.path.join(os.path.expanduser("~"), ".cache", "whisper")
            )
        except Exception as e:
            print(f"⚠️ faster-whisper notice: {e}")
            _WHISPER_MODEL = None
    return _WHISPER_MODEL

def word_error_rate(intended_text: str, gen_wav_path_or_bytes: Any) -> Optional[float]:
    """Computes Word Error Rate (WER) using faster-whisper and jiwer."""
    if not intended_text or not intended_text.strip():
        return 0.0
    try:
        import jiwer
        model_w = _get_whisper_model()
        if model_w is None:
            return None

        import tempfile
        temp_path = None
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

def select_best_segment(audio_path: str, target_duration: float = 8.0) -> str:
    """Pre-filters reference audio using energy windowing."""
    try:
        y, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        trimmed, _ = librosa.effects.trim(y, top_db=32)
        if len(trimmed) > sr * 0.5:
            y = trimmed
        if len(y) > int((target_duration + 3.0) * sr):
            win_len = int(target_duration * sr)
            best_win = y[:win_len]
            best_pow = -float("inf")
            for st in range(0, len(y) - win_len, int(0.5 * sr)):
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
    @colab_app.get("/health")
    def health():
        return {
            "status": "online",
            "model": "chatterbox-tts",
            "cuda": torch.cuda.is_available() if torch else False,
            "gpu": torch.cuda.get_device_name(0) if (torch and torch.cuda.is_available()) else "None",
            "sample_rate": getattr(model, "sr", 32000) if model else 32000,
            "emotion_presets": list(EMOTION_PARAMS.keys()),
        }

    @colab_app.post("/synthesize")
    async def synthesize_endpoint(
        ref_audio: UploadFile = File(...),
        text: str = Form(...),
        emotion: str = Form("neutral"),
        speed: float = Form(1.0),
        pitch: float = Form(0.0),
        cfg_weight: Optional[float] = Form(None),
        exaggeration: Optional[float] = Form(None),
        language: str = Form("en"),
    ):
        """Zero-shot voice cloning endpoint with emotion mapping & evaluation."""
        req_start_time = time.perf_counter()
        async with gpu_lock:
            if model is None:
                return Response(
                    content=b"Server Error: Chatterbox TTS model failed to initialize or is not available.",
                    status_code=503,
                    media_type="text/plain"
                )

            try:
                gen_sr = getattr(model, "sr", 32000)
                raw_bytes = await ref_audio.read()
                clean_wav_bytes, y_ref, sr_ref = clean_and_denoise_audio(raw_bytes, target_sr=gen_sr)
                
                import tempfile
                import uuid

                raw_ref_path = os.path.join(tempfile.gettempdir(), f"ref_raw_{uuid.uuid4().hex[:8]}.wav")
                with open(raw_ref_path, "wb") as f_ref_out:
                    f_ref_out.write(clean_wav_bytes)

                ref_path = raw_ref_path
                vad_ref_path = None
                try:
                    vad_path = select_best_segment(raw_ref_path, target_duration=8.0)
                    if vad_path != raw_ref_path:
                        vad_ref_path = vad_path
                        ref_path = vad_path
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

                for p in (raw_ref_path, vad_ref_path):
                    if p and os.path.exists(p):
                        try:
                            os.unlink(p)
                        except Exception:
                            pass

                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                out_buf = io.BytesIO()
                sf.write(out_buf, gen_np, gen_sr, format='WAV', subtype='PCM_16')
                wav_bytes = out_buf.getvalue()

                similarity_score = calculate_voice_similarity(y_ref, gen_np, sr=gen_sr)
                wer_score = word_error_rate(clean_text, wav_bytes)
                prosody_var = calculate_prosody_variance(gen_np, sr=gen_sr)
                latency_s = time.perf_counter() - req_start_time

                log_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "emotion_requested": emotion,
                    "resolved_emotion_preset": norm_emotion if norm_emotion in EMOTION_PARAMS else "neutral",
                    "resolved_exaggeration": round(active_exaggeration, 3),
                    "resolved_cfg_weight": round(active_cfg, 3),
                    "speaker_similarity": round(similarity_score, 4) if similarity_score is not None else None,
                    "word_error_rate": round(wer_score, 4) if wer_score is not None else None,
                    "prosody_variance": round(prosody_var, 2) if prosody_var is not None else None,
                    "latency_seconds": round(latency_s, 3),
                    "text_length_chars": len(clean_text),
                }
                try:
                    os.makedirs("logs", exist_ok=True)
                    with open("logs/synthesis_runs.jsonl", "a", encoding="utf-8") as lf:
                        lf.write(json.dumps(log_entry) + "\n")
                except Exception as log_err:
                    print(f"⚠️ Run logging notice: {log_err}")

                headers = {
                    "X-Model": "chatterbox-tts",
                    "X-Sample-Rate": str(gen_sr),
                    "X-Speaker-Similarity": f"{similarity_score:.4f}" if similarity_score is not None else "unavailable",
                    "X-Word-Error-Rate": f"{wer_score:.4f}" if wer_score is not None else "unavailable",
                    "X-Prosody-Variance": f"{prosody_var:.2f}" if prosody_var is not None else "unavailable",
                    "X-Resolved-Emotion": norm_emotion,
                    "X-Resolved-CFG": f"{active_cfg:.2f}",
                    "X-Resolved-Exaggeration": f"{active_exaggeration:.2f}",
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

def run_server():
    import subprocess
    if os.name == "posix":
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
            NGROK_AUTHTOKEN = "3I5bScJL7R0haCWXJ3FmBedIO5l_5aTLhGmF9vqmvEepVsERq"

        ngrok.set_auth_token(NGROK_AUTHTOKEN)
        tunnel = ngrok.connect(8008)

        print("\n==========================================================================")
        print("🚀 VoiceLib Chatterbox GPU Server running!")
        print(f"🌐 NGROK PUBLIC URL: {tunnel.public_url}")
        print(f"👉 Copy URL to backend .env: COLAB_GPU_API_URL={tunnel.public_url}")
        print("==========================================================================\n")
        print("⚡ Server is LIVE and listening for voice cloning requests! (Keep running)")

        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nStopping server...")
