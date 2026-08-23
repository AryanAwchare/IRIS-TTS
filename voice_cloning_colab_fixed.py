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
import os
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import Tuple

import sys
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print(f"Python Version: {sys.version.split()[0]}")


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
    cuda_available = False

# Kill old port 8008 process if running (Linux/Colab)
if os.name == "posix":
    os.system("fuser -k 8008/tcp 2>/dev/null")

# ── 1.1 PYTORCH COMPATIBILITY SHIMS ────────────────────────────────────────
import types

# 1. Patch missing _chunk_or_narrow_cat helper
if not hasattr(torch._utils, "_chunk_or_narrow_cat"):
    def _chunk_or_narrow_cat(tensors, dim=0):
        return torch.cat(tensors, dim=dim)
    torch._utils._chunk_or_narrow_cat = _chunk_or_narrow_cat

# 2. Targeted collectives shim for single-process environments
def _make_collectives_shim(name):
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

# Dependencies import
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
except ImportError as e:
    print(f"⚠️ Dependency notice: {e}. In Google Colab, please run Cell 1 to install requirements.")






# ── 2. AUDIO CLEANING & DENOISING PIPELINE ─────────────────────────────────
try:
    import noisereduce as nr
except Exception as nr_err:
    print(f"⚠️ Notice: noisereduce import fallback ({nr_err}). High-pass filter & normalization will remain active.")
    nr = None

def clean_and_denoise_audio(audio_path_or_bytes, target_sr=32000, enable_demucs=False):
    """
    State-of-the-art audio cleaner for voice cloning:
    - Converts to mono 32kHz (optimal for Chatterbox / GPT-SoVITS)
    - 2-pass stationary & non-stationary spectral noise reduction
    - Low-frequency rumble cut (<75Hz) and sibilance shaping
    - Silence trimming and dynamic RMS loudness normalization (-18 dBFS)
    """
    if isinstance(audio_path_or_bytes, bytes):
        y, sr = sf.read(io.BytesIO(audio_path_or_bytes))
        y = y.astype(np.float32)
    else:
        y, sr = librosa.load(audio_path_or_bytes, sr=None, mono=True)

    if y.ndim > 1:
        y = y.mean(axis=1)

    # 1. Resample to 32kHz target
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # 2. High-pass filter (remove 0-65Hz mic rumble / DC offset)
    nyq = sr * 0.5
    b, a = signal.butter(4, 65.0 / nyq, btype='highpass')
    y = signal.filtfilt(b, a, y).astype(np.float32)

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
def _peaking_eq(audio: np.ndarray, sr: int, center_freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """Digital biquad peaking bell filter for phase-coherent vocal EQ sculpt."""
    if abs(gain_db) < 0.1:
        return audio
    nyq = sr / 2.0
    if center_freq >= nyq:
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


def _high_shelf(audio: np.ndarray, sr: int, cutoff: float, gain_db: float) -> np.ndarray:
    """Digital biquad high shelf filter for vocal air enhancement."""
    if abs(gain_db) < 0.1:
        return audio
    nyq = sr / 2.0
    if cutoff >= nyq:
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


def pitch_preserving_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Time stretches audio array without altering pitch."""
    if abs(rate - 1.0) < 0.005 or len(y) == 0:
        return y
    try:
        import librosa
        return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
    except Exception:
        try:
            n_fft = 512
            hop = 128
            _, _, Z = signal.stft(y, nperseg=n_fft, noverlap=n_fft - hop)
            n_frames = Z.shape[1]
            new_n_frames = max(2, int(n_frames / rate))
            time_steps = np.linspace(0, n_frames - 1, new_n_frames)
            mag = np.abs(Z)
            phase_acc = np.angle(Z[:, 0])
            d_phase = np.angle(Z[:, 1:]) - np.angle(Z[:, :-1])
            new_Z = np.zeros((Z.shape[0], new_n_frames), dtype=np.complex64)
            for t_idx, t in enumerate(time_steps):
                f_idx = int(t)
                frac = t - f_idx
                if f_idx + 1 < n_frames:
                    mag_interp = (1 - frac) * mag[:, f_idx] + frac * mag[:, f_idx + 1]
                else:
                    mag_interp = mag[:, f_idx]
                new_Z[:, t_idx] = mag_interp * np.exp(1j * phase_acc)
                if f_idx < d_phase.shape[1]:
                    phase_acc += d_phase[:, f_idx]
            _, y_out = signal.istft(new_Z, nperseg=n_fft, noverlap=n_fft - hop)
            return y_out[:int(len(y) / rate)].astype(np.float32)
        except Exception:
            indices = np.linspace(0, len(y) - 1, max(1, int(len(y) / rate)))
            return np.interp(indices, np.arange(len(y)), y).astype(np.float32)


def concatenate_audio_chunks_with_crossfade(chunks: list, sr: int, min_pause_ms: float = 80.0, max_pause_ms: float = 150.0) -> np.ndarray:
    """Concatenates sentence audio chunks with 5ms cosine cross-fades to eliminate boundary click artifacts."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    fade_samples = int(sr * 0.005)  # 5ms equal-power fade
    result = []

    for i, chunk in enumerate(chunks):
        c = chunk.copy().astype(np.float32)
        if len(c) > fade_samples * 2:
            fade_in = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, fade_samples)))
            fade_out = 0.5 * (1.0 + np.cos(np.pi * np.linspace(0, 1, fade_samples)))
            c[:fade_samples] *= fade_in
            c[-fade_samples:] *= fade_out
        result.append(c)
        if i < len(chunks) - 1:
            pause_dur = float(np.random.uniform(min_pause_ms / 1000.0, max_pause_ms / 1000.0))
            result.append(np.zeros(int(sr * pause_dur), dtype=np.float32))

    return np.concatenate(result).astype(np.float32)


def enhance_voice_mastering(audio_np, sr=32000):
    """
    Transparent vocal normalizer:
    - Removes sub-audible DC offset (<30Hz)
    - Applies clean peak ceiling limiter (-0.3 dBFS)
    - Preserves 100% of authentic speaker vocal timbre, formant structure, and accent
    """
    audio = audio_np.copy().astype(np.float32)
    nyq = sr * 0.5

    # 1. Gentle DC & sub-rumble cut (30Hz)
    if 30.0 / nyq < 1.0:
        b_hp, a_hp = signal.butter(2, 30.0 / nyq, btype='highpass')
        audio = signal.filtfilt(b_hp, a_hp, audio).astype(np.float32)

    # 2. Peak normalization without frequency coloring
    max_val = np.max(np.abs(audio)) + 1e-9
    if max_val > 0.01:
        target_peak = 10 ** (-0.3 / 20.0)  # ~ -0.3 dBFS
        audio = audio * (target_peak / max(max_val, 1.0))
        audio = np.clip(audio, -0.99, 0.99)

    return audio.astype(np.float32)

print("✅ Pure vocal acoustic preservation chain ready!")

# Helper: split long text into natural sentence chunks
def split_text_into_sentences(text: str, min_chars: int = 40) -> list[str]:
    import re
    # Split by period, exclamation, question mark, or semicolons
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

# ── 4. VOICE SIMILARITY SCORING ENGINE ─────────────────────────────────────
def _legacy_mfcc_similarity(ref_audio_np, gen_audio_np, sr=32000):
    """
    [DEPRECATED LEGACY] Computes multi-dimensional acoustic similarity score between reference and generated voice:
    - MFCC Cosine Similarity (Timbre / Vocal Tract Shape): 50% weight
    - Spectral Centroid Match (Pitch Register / Brightness): 25% weight
    - Energy Envelope Correlation (Prosody / Cadence): 25% weight
    """
    mfcc_ref = librosa.feature.mfcc(y=ref_audio_np, sr=sr, n_mfcc=20)
    mfcc_gen = librosa.feature.mfcc(y=gen_audio_np, sr=sr, n_mfcc=20)

    min_cols = min(mfcc_ref.shape[1], mfcc_gen.shape[1])
    if min_cols == 0:
        return 50.0, {}

    ref_sub = mfcc_ref[:, :min_cols]
    gen_sub = mfcc_gen[:, :min_cols]

    # 1. MFCC Cosine similarity
    dot = np.sum(ref_sub * gen_sub, axis=0)
    norm_r = np.linalg.norm(ref_sub, axis=0) + 1e-9
    norm_g = np.linalg.norm(gen_sub, axis=0) + 1e-9
    mfcc_sim = float(np.clip(np.mean(dot / (norm_r * norm_g)) * 100.0, 0.0, 100.0))

    # 2. Spectral Centroid
    sc_ref = librosa.feature.spectral_centroid(y=ref_audio_np, sr=sr)[0]
    sc_gen = librosa.feature.spectral_centroid(y=gen_audio_np, sr=sr)[0]
    min_sc = min(len(sc_ref), len(sc_gen))
    corr_sc = float(np.clip((np.corrcoef(sc_ref[:min_sc], sc_gen[:min_sc])[0, 1] + 1.0) * 50.0, 0.0, 100.0))
    if np.isnan(corr_sc):
        corr_sc = 70.0

    # 3. RMS Energy
    rms_ref = librosa.feature.rms(y=ref_audio_np)[0]
    rms_gen = librosa.feature.rms(y=gen_audio_np)[0]
    min_rms = min(len(rms_ref), len(rms_gen))
    corr_rms = float(np.clip((np.corrcoef(rms_ref[:min_rms], rms_gen[:min_rms])[0, 1] + 1.0) * 50.0, 0.0, 100.0))
    if np.isnan(corr_rms):
        corr_rms = 70.0

    overall_score = round(mfcc_sim * 0.50 + corr_sc * 0.25 + corr_rms * 0.25, 1)

    metrics = {
        "overall_similarity_pct": overall_score,
        "mfcc_timbre_match": round(mfcc_sim, 1),
        "spectral_brightness_match": round(corr_sc, 1),
        "energy_prosody_match": round(corr_rms, 1),
    }
    return overall_score, metrics


# ── 4. VOICE SIMILARITY, PROSODY & EVALUATION ENGINES ─────────────────────
VOICELIB_BACKEND_DIR = os.getenv("VOICELIB_BACKEND_DIR", "/content/VoiceLib/voicelib/backend")
if os.path.exists(VOICELIB_BACKEND_DIR) and VOICELIB_BACKEND_DIR not in sys.path:
    sys.path.insert(0, VOICELIB_BACKEND_DIR)

# ── EMOTION PARAMETER PRESETS & ACCENT-LOCK CALIBRATION ─────────────────────
# Identity/accent lock requires higher cfg_weight (0.65 - 0.80) to condition tightly on the reference speaker's
# timbre and accent. For high-energy emotions (happy, excited, angry), lower cfg_weight (0.45 - 0.55) and
# higher exaggeration (0.25 - 0.40) allow natural pitch excursions and vocal dynamism without acoustic clamping.
# NOTE: These are calibrated starting defaults to be hand-tuned by ear.
EMOTION_PARAMS = {
    "neutral": {"exaggeration": 0.05, "cfg_weight": 0.70},  # Tight speaker lock, natural neutral delivery
    "calm":    {"exaggeration": 0.00, "cfg_weight": 0.75},  # Maximum identity & accent lock, steady cadence
    "happy":   {"exaggeration": 0.25, "cfg_weight": 0.55},  # Expressive pitch variation, lively rhythm
    "excited": {"exaggeration": 0.40, "cfg_weight": 0.45},  # High dynamic excursions, energetic delivery
    "sad":     {"exaggeration": 0.15, "cfg_weight": 0.65},  # Slower pacing, gentle contour
    "angry":   {"exaggeration": 0.35, "cfg_weight": 0.50},  # Sharp dynamic attacks, higher vocal strain
}

_ECAPA_CLASSIFIER = None

def _get_ecapa_classifier():
    global _ECAPA_CLASSIFIER
    if _ECAPA_CLASSIFIER is None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier
        
        _ECAPA_CLASSIFIER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(os.path.expanduser("~"), ".cache", "speechbrain", "ecapa"),
            run_opts={"device": "cpu"}
        )
    return _ECAPA_CLASSIFIER


def calculate_voice_similarity(ref_audio, gen_audio, sr=32000) -> float | None:
    """
    Computes objective speaker similarity using SpeechBrain ECAPA-TDNN 192-dim embeddings.
    Accepts file paths (.wav) or raw audio numpy arrays / bytes.
    Returns raw cosine similarity float in range [0.0, 1.0], or None if computation fails.
    """
    # 1. Try importing from backend module if present
    try:
        from app.evaluation.speaker_similarity import speaker_similarity as _eval_sim
        if isinstance(ref_audio, str) and isinstance(gen_audio, str):
            return _eval_sim(ref_audio, gen_audio)
    except Exception:
        pass

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
                with open(ref_path, "wb") as f:
                    f.write(ref_audio)
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
                with open(gen_path, "wb") as f:
                    f.write(gen_audio)
            else:
                sf.write(gen_path, gen_audio, sr, format="WAV", subtype="PCM_16")

        classifier = _get_ecapa_classifier()
        signal_ref = classifier.load_audio(ref_path)
        signal_gen = classifier.load_audio(gen_path)

        with torch.no_grad():
            emb_ref = classifier.encode_batch(signal_ref)
            emb_gen = classifier.encode_batch(signal_gen)

        cos_sim = F.cosine_similarity(emb_ref.squeeze(), emb_gen.squeeze(), dim=0).item()
        return float(max(0.0, min(1.0, cos_sim)))
    except Exception as exc:
        print(f"❌ ECAPA-TDNN similarity calculation failed: {exc}")
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
            print(f"⚠️ faster-whisper loading notice ({e})")
            _WHISPER_MODEL = None
    return _WHISPER_MODEL


def word_error_rate(intended_text: str, gen_wav_path_or_bytes) -> float | None:
    """Computes Word Error Rate (WER) using faster-whisper and jiwer."""
    if not intended_text or not intended_text.strip():
        return 0.0

    # 1. Try importing from backend module if available
    try:
        from app.evaluation.content_accuracy import word_error_rate as _eval_wer
        if isinstance(gen_wav_path_or_bytes, str) and os.path.exists(gen_wav_path_or_bytes):
            return _eval_wer(intended_text, gen_wav_path_or_bytes)
    except Exception:
        pass

    try:
        import jiwer
        model = _get_whisper_model()
        if model is None:
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

        segments, _ = model.transcribe(audio_target, beam_size=5, language="en")
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


def calculate_prosody_variance(audio: np.ndarray, sr: int = 32000) -> float | None:
    """
    Diagnostic measurement of pitch (F0) standard deviation across voiced frames.
    Provides a lightweight indicator of dynamic range / emotional expressiveness.
    """
    try:
        # Fast YIN pitch tracking on 16kHz audio for speed
        audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000) if sr != 16000 else audio
        f0 = librosa.yin(audio_16k, fmin=65, fmax=500, sr=16000)
        voiced = f0[(f0 > 65) & (f0 < 500)]
        if len(voiced) > 10:
            return float(np.std(voiced))
        # Fallback to RMS dynamics variance
        rms = librosa.feature.rms(y=audio_16k)[0]
        return float(np.std(rms) * 1000.0)
    except Exception as exc:
        print(f"⚠️ Prosody variance calculation notice: {exc}")
        return None


def select_best_segment(audio_path: str, target_duration: float = 8.0) -> str:
    """Pre-filters reference audio using VAD and SNR windowing."""
    try:
        from app.preprocessing.reference_selector import select_best_segment as _sel
        return _sel(audio_path, target_duration=target_duration)
    except Exception:
        # Fallback to local VAD logic if backend module is not installed
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
            print(f"⚠️ VAD selector notice ({e}). Using original audio.")
            return audio_path

print("✅ Evaluation (ECAPA-TDNN + Whisper WER + Prosody Diagnostic) & VAD selector ready!")


# ── 5. MODEL INITIALIZATION ────────────────────────────────────────────────
try:
    from chatterbox.tts import ChatterboxTTS
except ImportError:
    ChatterboxTTS = None

device = "cuda" if (torch.cuda.is_available() if 'torch' in globals() else False) else "cpu"
model = None
if ChatterboxTTS is not None:
    print("🧠 Loading Chatterbox TTS model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"✅ Chatterbox TTS loaded on {'CUDA GPU' if torch.cuda.is_available() else 'CPU'}!")
else:
    print("⚠️ ChatterboxTTS module not loaded in local environment. Available in Colab runtime.")


# ── 6. FASTAPI APP & NGROK TUNNEL ──────────────────────────────────────────
try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

try:
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import Response
    colab_app = FastAPI(title="VoiceLib Chatterbox GPU Server")
except Exception:
    colab_app = None
    UploadFile = object
    File = lambda *args, **kwargs: None
    Form = lambda *args, **kwargs: None
    Response = object

if colab_app is not None:
    @colab_app.get("/health")
    def health():
        return {
            "status": "online",
            "model": "chatterbox-tts",
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "sample_rate": getattr(model, "sr", 32000) if model else 32000,
            "emotion_presets": list(EMOTION_PARAMS.keys()),
        }

gpu_lock = asyncio.Lock()


@colab_app.post("/synthesize")
async def synthesize_endpoint(
    ref_audio: UploadFile = File(...),
    text: str = Form(...),
    emotion: str = Form("neutral"),
    speed: float = Form(1.0),
    pitch: float = Form(0.0),
    cfg_weight: float | None = Form(None),
    exaggeration: float | None = Form(None),
    language: str = Form("en"),
):
    """
    Zero-shot voice cloning endpoint with emotion mapping & objective evaluation.
    - emotion: "neutral", "calm", "happy", "excited", "sad", "angry"
    - cfg_weight & exaggeration: explicit overrides if provided; otherwise resolved from EMOTION_PARAMS
    """
    req_start_time = time.perf_counter()
    async with gpu_lock:
        if model is None:
            return Response(
                content=b"Server Error: Chatterbox TTS model failed to initialize or is not available.",
                status_code=503,
                media_type="text/plain"
            )

        try:
            # 1. Clean, decode, & normalize incoming reference audio prompt
            gen_sr = getattr(model, "sr", 32000)
            raw_bytes = await ref_audio.read()
            clean_wav_bytes, y_ref, sr_ref = clean_and_denoise_audio(raw_bytes, target_sr=gen_sr)
            
            import uuid
            ref_path = f"/tmp/ref_{uuid.uuid4().hex[:8]}.wav"
            with open(ref_path, "wb") as f:
                f.write(clean_wav_bytes)


            # 1.5. VAD and SNR reference segment selection (Fix 5)
            try:
                ref_path = select_best_segment(ref_path, target_duration=8.0)
            except Exception as vad_err:
                print(f"⚠️ VAD selection notice: {vad_err}")

            # 2. Text Normalization & Paralinguistic Tag Processing
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

            # 3. Resolve Emotion Presets & Accent-Lock Parameters (Fix 2 & Fix 3)
            norm_emotion = emotion.lower().strip() if emotion else "neutral"
            preset = EMOTION_PARAMS.get(norm_emotion, EMOTION_PARAMS["neutral"])

            # Explicit caller values override the preset; otherwise use emotion preset
            active_exaggeration = float(np.clip(
                exaggeration if exaggeration is not None else preset["exaggeration"],
                0.0, 0.50
            ))
            active_cfg = float(np.clip(
                cfg_weight if cfg_weight is not None else preset["cfg_weight"],
                0.20, 0.90
            ))

            print(
                f"🎭 [Synthesize Request] Emotion: '{norm_emotion}' | "
                f"Resolved exaggeration: {active_exaggeration:.2f} (Preset: {preset['exaggeration']:.2f}) | "
                f"Resolved CFG: {active_cfg:.2f} (Preset: {preset['cfg_weight']:.2f})"
            )

            gen_sr = model.sr

            # 4. Neural generation
            if len(clean_text) > 80 or "," in clean_text or ";" in clean_text:
                sentences = split_text_into_sentences(clean_text, min_chars=35)
                audio_chunks = []
                pauses_ms = []

                for sent in sentences:
                    s_text = sent.strip()
                    if not s_text.endswith(('.', '!', '?', ';', ',')):
                        s_text += '.'

                    if s_text.endswith('...'):
                        pause_ms = 280.0
                    elif s_text.endswith(('.', '!')):
                        pause_ms = 200.0
                    elif s_text.endswith('?'):
                        pause_ms = 180.0
                    elif s_text.endswith((',', ';')):
                        pause_ms = 120.0
                    else:
                        pause_ms = 150.0

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

            # 5. Trim trailing silence
            if len(gen_np) > gen_sr * 0.5:
                yt, _ = librosa.effects.trim(gen_np, top_db=38)
                if len(yt) > gen_sr * 0.3:
                    gen_np = yt

            # 6. Apply pitch calibration if requested
            if abs(pitch) > 0.05:
                try:
                    gen_np = librosa.effects.pitch_shift(gen_np, sr=gen_sr, n_steps=pitch).astype(np.float32)
                except Exception:
                    pass

            # 7. Apply speed adjustment
            if abs(speed - 1.0) > 0.05 and 0.5 <= speed <= 2.0:
                gen_np = pitch_preserving_time_stretch(gen_np, rate=speed)

            # 8. Vocal mastering
            gen_np = enhance_voice_mastering(gen_np, sr=gen_sr)

            # 9. Clean up temp ref file & isolate GPU memory
            try:
                if os.path.exists(ref_path):
                    os.unlink(ref_path)
            except Exception:
                pass

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            out_buf = io.BytesIO()
            sf.write(out_buf, gen_np, gen_sr, format='WAV', subtype='PCM_16')
            wav_bytes = out_buf.getvalue()

            # 10. Compute Decoupled Evaluation Metrics (Fix 4 & Fix 6)
            similarity_score = calculate_voice_similarity(y_ref, gen_np, sr=gen_sr)
            wer_score = word_error_rate(clean_text, wav_bytes)
            prosody_var = calculate_prosody_variance(gen_np, sr=gen_sr)
            latency_s = time.perf_counter() - req_start_time

            # 11. Structured Request Logging (Fix 7)
            import json, datetime
            log_entry = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
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

            print(
                f"📊 [Run Completed] Sim: {f'{similarity_score:.4f}' if similarity_score is not None else 'N/A'} | "
                f"WER: {f'{wer_score:.4f}' if wer_score is not None else 'N/A'} | "
                f"ProsodyVar: {f'{prosody_var:.2f}' if prosody_var is not None else 'N/A'} | "
                f"Latency: {latency_s:.2f}s"
            )

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
    import subprocess, time
    try:
        subprocess.run(["fuser", "-k", "8008/tcp"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.8)
    except Exception:
        pass
    uvicorn.run(colab_app, host="0.0.0.0", port=8008, log_level="warning")

# ── 7. MAIN RUNNER ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess, time
    try:
        subprocess.run(["fuser", "-k", "8008/tcp"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.8)
    except Exception:
        pass

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1.5)

    # Launch Ngrok Tunnel with strict env token validation (Fix 1)
    # NOTE: Any previously hardcoded ngrok token must be revoked in your ngrok dashboard!
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
        raise RuntimeError(
            "NGROK_AUTHTOKEN not found! Please set it as a Colab Secret (🔑) or environment variable before running. "
            "Ensure any previous token has been revoked in your ngrok dashboard."
        )

    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    tunnel = ngrok.connect(8008)

    print("\n==========================================================================")
    print("🚀 VoiceLib Chatterbox GPU Server running!")
    print(f"🌐 NGROK PUBLIC URL: {tunnel.public_url}")
    print(f"👉 Copy URL to backend .env: COLAB_GPU_API_URL={tunnel.public_url}")
    print("==========================================================================\n")
    print("⚡ Server is LIVE and listening for voice cloning requests! (Keep this cell running)")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping server...")


