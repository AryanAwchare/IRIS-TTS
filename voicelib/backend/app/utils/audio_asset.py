"""
audio_asset.py — First-class audio abstraction, canonicalization, and chunking.

Provides:
  - AudioAsset: strongly typed audio container (path, sample_rate, channels, duration, samples, loudness)
  - Canonical WAV conversion (standard 16-bit / 32-bit float PCM WAV at 44.1kHz or 32kHz)
  - Dynamic energy-guided slicing (8–20s windows cut at silence/RMS dips to avoid cutting sustained vowels)
  - Equal-power crossfade stitching (preserves constant acoustic energy without volume dips)
  - Loudness / RMS energy envelope analysis
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf


@dataclass
class AudioChunk:
    """A sliced segment of an AudioAsset with timing metadata and overlap margin."""
    chunk_index: int
    samples: np.ndarray
    sample_rate: int
    start_sec: float
    end_sec: float
    is_preview: bool = False
    overlap_samples: int = 0


@dataclass
class AudioAsset:
    """
    First-class audio representation passed across all separation, analysis, SVC, and mixing stages.
    Eliminates ad-hoc format and sample-rate mismatches across modules.
    """
    samples: np.ndarray             # float32 array, shape (num_samples,) or (channels, num_samples)
    sample_rate: int = 44100
    channels: int = 1
    duration: float = 0.0
    path: Optional[Path] = None
    format: str = "WAV"
    loudness_lufs: Optional[float] = None

    def __post_init__(self):
        if self.samples is not None:
            self.samples = np.asarray(self.samples, dtype=np.float32)
            if self.samples.ndim == 1:
                self.channels = 1
                num_samples = len(self.samples)
            elif self.samples.ndim == 2:
                # Standardize to (channels, num_samples)
                if self.samples.shape[0] > self.samples.shape[1] and self.samples.shape[1] <= 2:
                    self.samples = self.samples.T
                self.channels = self.samples.shape[0]
                num_samples = self.samples.shape[1]
            else:
                raise ValueError(f"Unsupported audio array dimension: {self.samples.ndim}")

            if self.sample_rate > 0:
                self.duration = round(float(num_samples) / float(self.sample_rate), 4)

    @property
    def num_samples(self) -> int:
        return self.samples.shape[-1] if self.samples is not None else 0

    def to_mono(self) -> AudioAsset:
        """Downmix to mono if multi-channel."""
        if self.channels == 1:
            return self
        mono_samples = np.mean(self.samples, axis=0, dtype=np.float32)
        return AudioAsset(
            samples=mono_samples,
            sample_rate=self.sample_rate,
            channels=1,
            path=self.path,
            format=self.format,
        )

    def resample(self, target_sr: int) -> AudioAsset:
        """Resample audio to target sample rate using scipy or librosa."""
        if target_sr == self.sample_rate:
            return self

        try:
            import librosa
            if self.channels == 1:
                resampled = librosa.resample(self.samples, orig_sr=self.sample_rate, target_sr=target_sr)
            else:
                resampled = np.stack([
                    librosa.resample(self.samples[ch], orig_sr=self.sample_rate, target_sr=target_sr)
                    for ch in range(self.channels)
                ])
        except Exception:
            from scipy import signal
            num_target_samples = int(round(self.num_samples * (float(target_sr) / float(self.sample_rate))))
            resampled = signal.resample(self.samples, num_target_samples, axis=-1).astype(np.float32)

        return AudioAsset(
            samples=resampled,
            sample_rate=target_sr,
            channels=self.channels,
            path=self.path,
            format=self.format,
        )

    def normalize(self, target_peak_db: float = -1.0) -> AudioAsset:
        """Peak-normalize audio to target dBFS."""
        max_peak = float(np.max(np.abs(self.samples)))
        if max_peak < 1e-6:
            return self
        target_amp = 10.0 ** (target_peak_db / 20.0)
        scale = target_amp / max_peak
        normed = np.clip(self.samples * scale, -1.0, 1.0)
        return AudioAsset(
            samples=normed,
            sample_rate=self.sample_rate,
            channels=self.channels,
            path=self.path,
            format=self.format,
        )

    def save(self, target_path: Union[str, Path], subtype: str = "PCM_16") -> Path:
        """Save to disk as canonical WAV file."""
        p = Path(target_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Soundfile expects shape (num_samples, channels)
        write_samples = self.samples.T if self.channels > 1 else self.samples
        sf.write(str(p), write_samples, self.sample_rate, subtype=subtype)
        return p

    def to_bytes(self, subtype: str = "PCM_16") -> bytes:
        """Serialize audio array into WAV byte stream."""
        buf = io.BytesIO()
        write_samples = self.samples.T if self.channels > 1 else self.samples
        sf.write(buf, write_samples, self.sample_rate, format="WAV", subtype=subtype)
        return buf.getvalue()


def load_audio_asset(
    source: Union[str, Path, bytes, io.BytesIO],
    target_sr: Optional[int] = None,
    mono: bool = False,
) -> AudioAsset:
    """
    Decodes audio from file path, bytes buffer, or stream into canonical AudioAsset.
    Automatically standardizes bit depth, channel configuration, and sample rate.
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)

    try:
        data, sr = sf.read(source, dtype="float32")
    except Exception as sf_err:
        # Fallback to librosa if soundfile failed (e.g. mp3 / m4a without libsndfile mp3 support)
        try:
            import librosa
            data, sr = librosa.load(source, sr=target_sr, mono=mono)
        except Exception as l_err:
            raise RuntimeError(f"Failed to decode audio source with soundfile ({sf_err}) and librosa ({l_err})")

    # Standardize shape to (channels, num_samples) or (num_samples,)
    if data.ndim == 2:
        data = data.T
        channels = data.shape[0]
    else:
        channels = 1

    path_obj = Path(source) if isinstance(source, (str, Path)) else None
    asset = AudioAsset(
        samples=data,
        sample_rate=sr,
        channels=channels,
        path=path_obj,
    )

    if mono and asset.channels > 1:
        asset = asset.to_mono()

    if target_sr and asset.sample_rate != target_sr:
        asset = asset.resample(target_sr)

    return asset


def slice_into_windows(
    asset: AudioAsset,
    min_window_sec: float = 8.0,
    max_window_sec: float = 20.0,
    overlap_sec: float = 0.5,
) -> List[AudioChunk]:
    """
    Dynamically slices an AudioAsset into 8-20s windows.
    Instead of hard cutting, it detects energy dips (RMS < -35dB or lowest local energy)
    within the [min_window_sec, max_window_sec] range so that sustained singing vowels
    are preserved intact without mid-note truncation.
    """
    mono_asset = asset.to_mono()
    samples = mono_asset.samples
    sr = asset.sample_rate
    total_len = len(samples)

    min_samples = int(min_window_sec * sr)
    max_samples = int(max_window_sec * sr)
    overlap_samples = int(overlap_sec * sr)

    if total_len <= max_samples:
        # Audio is already short enough to process as a single chunk
        return [
            AudioChunk(
                chunk_index=0,
                samples=asset.samples,
                sample_rate=sr,
                start_sec=0.0,
                end_sec=asset.duration,
                overlap_samples=0,
            )
        ]

    # Calculate frame-level RMS energy envelope (20ms frames, 10ms hop)
    frame_len = int(0.02 * sr)
    hop_len = int(0.01 * sr)
    num_frames = max(1, (total_len - frame_len) // hop_len)
    
    # Fast vectorized RMS energy calculation
    energy = np.zeros(num_frames, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_len
        frame = samples[start : start + frame_len]
        energy[i] = np.sqrt(np.mean(frame ** 2) + 1e-9)

    chunks: List[AudioChunk] = []
    curr_start = 0
    chunk_idx = 0

    while curr_start < total_len:
        target_end = curr_start + max_samples
        if target_end >= total_len:
            # Last slice reaches end of track
            chunk_samples = asset.samples[:, curr_start:] if asset.channels > 1 else asset.samples[curr_start:]
            chunks.append(
                AudioChunk(
                    chunk_index=chunk_idx,
                    samples=chunk_samples,
                    sample_rate=sr,
                    start_sec=round(curr_start / sr, 3),
                    end_sec=asset.duration,
                    overlap_samples=overlap_samples if chunk_idx > 0 else 0,
                )
            )
            break

        # Search for minimum energy point in the window [curr_start + min_samples, curr_start + max_samples]
        search_start_frame = max(0, (curr_start + min_samples) // hop_len)
        search_end_frame = min(num_frames - 1, target_end // hop_len)

        if search_end_frame > search_start_frame:
            window_energy = energy[search_start_frame:search_end_frame]
            min_local_frame = search_start_frame + int(np.argmin(window_energy))
            best_cut_sample = min_local_frame * hop_len
        else:
            best_cut_sample = target_end

        # Add overlap margin to the slice for smooth crossfading
        slice_end = min(total_len, best_cut_sample + overlap_samples)
        chunk_samples = asset.samples[:, curr_start:slice_end] if asset.channels > 1 else asset.samples[curr_start:slice_end]

        chunks.append(
            AudioChunk(
                chunk_index=chunk_idx,
                samples=chunk_samples,
                sample_rate=sr,
                start_sec=round(curr_start / sr, 3),
                end_sec=round(slice_end / sr, 3),
                overlap_samples=overlap_samples,
            )
        )

        curr_start = best_cut_sample
        chunk_idx += 1

    return chunks


def equal_power_crossfade_stitch(
    chunk_arrays: List[np.ndarray],
    overlap_samples: int,
) -> np.ndarray:
    """
    Seamlessly stitches converted audio chunks back into a continuous track
    using Equal-Power (Cosine / Sine) crossfading:
        w_in(t)  = sin(t * pi / 2)
        w_out(t) = cos(t * pi / 2)
    This guarantees that w_in^2 + w_out^2 = 1.0, preserving constant acoustic energy
    and eliminating the -3dB dip of linear crossfading.
    """
    if not chunk_arrays:
        return np.array([], dtype=np.float32)
    if len(chunk_arrays) == 1 or overlap_samples <= 0:
        return np.concatenate(chunk_arrays, axis=-1)

    is_multichannel = chunk_arrays[0].ndim == 2
    channels = chunk_arrays[0].shape[0] if is_multichannel else 1

    # Precalculate equal-power fade curves
    t = np.linspace(0.0, math.pi / 2.0, overlap_samples, endpoint=False, dtype=np.float32)
    fade_in = np.sin(t)
    fade_out = np.cos(t)

    # Estimate total length
    total_len = sum(c.shape[-1] for c in chunk_arrays) - (len(chunk_arrays) - 1) * overlap_samples
    if is_multichannel:
        output = np.zeros((channels, total_len), dtype=np.float32)
    else:
        output = np.zeros(total_len, dtype=np.float32)

    current_pos = 0
    for idx, chunk in enumerate(chunk_arrays):
        chunk_len = chunk.shape[-1]

        if idx == 0:
            if is_multichannel:
                output[:, :chunk_len] = chunk
            else:
                output[:chunk_len] = chunk
            current_pos = chunk_len
        else:
            # Overlap region
            crossfade_start = current_pos - overlap_samples
            actual_overlap = min(overlap_samples, chunk_len)

            if is_multichannel:
                for ch in range(channels):
                    existing = output[ch, crossfade_start : crossfade_start + actual_overlap]
                    incoming = chunk[ch, :actual_overlap]
                    f_in = fade_in[:actual_overlap]
                    f_out = fade_out[:actual_overlap]
                    output[ch, crossfade_start : crossfade_start + actual_overlap] = (
                        existing * f_out + incoming * f_in
                    )
                    # Non-overlapped portion
                    if chunk_len > actual_overlap:
                        output[ch, crossfade_start + actual_overlap : crossfade_start + chunk_len] = chunk[ch, actual_overlap:]
            else:
                existing = output[crossfade_start : crossfade_start + actual_overlap]
                incoming = chunk[:actual_overlap]
                f_in = fade_in[:actual_overlap]
                f_out = fade_out[:actual_overlap]
                output[crossfade_start : crossfade_start + actual_overlap] = (
                    existing * f_out + incoming * f_in
                )
                if chunk_len > actual_overlap:
                    output[crossfade_start + actual_overlap : crossfade_start + chunk_len] = chunk[actual_overlap:]

            current_pos = crossfade_start + chunk_len

    return output[:, :current_pos] if is_multichannel else output[:current_pos]
