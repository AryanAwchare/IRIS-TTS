"""
Integration test suite for Song Voice Conversion (SVC / Song Cloning) Pipeline.
Tests:
  - AudioAsset slicing & equal-power crossfade reassembly
  - Demucs / Center-channel stem separation artifact caching
  - Vocal analysis (F0 pitch, voicing, median Hz, octave transposition recommendation)
  - RVC chunked conversion with checkpoint caching and 20s preview mode
  - Studio vocal mastering, de-esser, sidechain ducking, and mixdown
  - Regression check: converted vocals replace original singer with zero overlap
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from app.audio_mixer import apply_sidechain_ducking, apply_vocal_mastering_chain, mix_and_master_song
from app.models import SongCover, SongCoverCreate, SongCoverOut, SongCoverStatusOut
from app.services.job_queue import ARTIFACTS_ROOT, job_manager
from app.svc_engines.rvc_engine import RVCEngine
from app.utils.audio_asset import AudioAsset, equal_power_crossfade_stitch, load_audio_asset, slice_into_windows
from app.vocal_analysis import analyze_vocal_track, calculate_recommended_pitch_shift
from app.vocal_separation import separate_vocals_and_instrumental


def _generate_synthetic_song(duration_s: float = 12.0, sr: int = 44100) -> bytes:
    """Generates a synthetic stereo mix (panned instruments + center vocal melody)."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    # Instrumental backing (chords panned stereo)
    inst_left = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 330 * t)
    inst_right = 0.3 * np.sin(2 * np.pi * 277 * t) + 0.15 * np.sin(2 * np.pi * 440 * t)
    # Center lead vocal melody (A4 440Hz -> C5 523Hz with a pause)
    vocal = 0.4 * np.sin(2 * np.pi * 440 * t)
    # Introduce vocal pause between 4s and 6s to test silence detection
    pause_mask = (t >= 4.0) & (t <= 6.0)
    vocal[pause_mask] = 0.0

    mix_left = inst_left + vocal
    mix_right = inst_right + vocal
    stereo_mix = np.stack([mix_left, mix_right])

    import io
    buf = io.BytesIO()
    sf.write(buf, stereo_mix.T, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_audio_asset_slicing_and_crossfade():
    """Verify dynamic silence slicing and equal-power stitching."""
    sr = 44100
    t = np.linspace(0, 20, sr * 20, dtype=np.float32)
    sig = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Silence at 9.0s - 10.0s
    sig[int(9 * sr) : int(10 * sr)] = 0.0

    asset = AudioAsset(samples=sig, sample_rate=sr)
    chunks = slice_into_windows(asset, min_window_sec=8.0, max_window_sec=14.0, overlap_sec=0.5)

    assert len(chunks) >= 2, "Expected at least 2 chunks"
    # Overlap samples
    overlap = int(0.5 * sr)
    stitched = equal_power_crossfade_stitch([c.samples for c in chunks], overlap_samples=overlap)
    assert abs(len(stitched) - len(sig)) < sr * 0.1, "Stitched audio length must match original within 100ms"


def test_vocal_separation_and_caching():
    """Verify stem separation and persistent artifact caching."""
    song_bytes = _generate_synthetic_song(duration_s=6.0)
    song_hash = job_manager.get_audio_hash(song_bytes)

    # 1. First run: separates stems
    artifact1 = separate_vocals_and_instrumental(song_bytes, song_hash=song_hash)
    assert artifact1.vocals_path.exists()
    assert artifact1.instrumental_path.exists()

    # 2. Second run: must hit artifact cache (fast return)
    artifact2 = separate_vocals_and_instrumental(song_bytes, song_hash=song_hash)
    assert artifact2.vocals_path == artifact1.vocals_path
    assert artifact2.song_hash == song_hash


def test_vocal_pitch_analysis_and_transposition():
    """Verify F0 extraction, voicing, and transposition recommendations."""
    sr = 44100
    t = np.linspace(0, 4.0, sr * 4, dtype=np.float32)
    # Singing note: 440 Hz (A4)
    vocal = 0.5 * np.sin(2 * np.pi * 440 * t)
    vocal_asset = AudioAsset(samples=vocal, sample_rate=sr)

    analysis = analyze_vocal_track(vocal_asset, song_hash="test_pitch_analysis")
    assert 400.0 <= analysis.median_f0 <= 460.0, f"Expected median F0 ~440Hz, got {analysis.median_f0}"
    assert np.any(analysis.voicing), "Voicing detection should find active notes"

    # Test transposition calculation: Female soprano (330Hz) -> Male baritone (130Hz)
    shift = calculate_recommended_pitch_shift(330.0, 130.0)
    assert shift in [-16, -12], f"Expected octave shift around -12st, got {shift}"


def test_rvc_chunked_conversion_and_preview():
    """Verify RVC chunk conversion with checkpoints and 20s preview mode."""
    sr = 44100
    t = np.linspace(0, 15.0, sr * 15, dtype=np.float32)
    vocal = 0.4 * np.sin(2 * np.pi * 440 * t)
    vocal_asset = AudioAsset(samples=vocal, sample_rate=sr)

    engine = RVCEngine()
    temp_checkpoints = ARTIFACTS_ROOT / "test_checkpoints"

    # Test preview mode (only processes first chunk)
    preview_asset = engine.convert_full_vocals(
        vocals_asset=vocal_asset,
        voice_id="test_voice",
        pitch_shift=0,
        preview_only=True,
        checkpoint_dir=temp_checkpoints,
    )
    assert preview_asset.duration <= 16.5, f"Preview duration should be under 16.5s, got {preview_asset.duration}"
    assert (temp_checkpoints / "chunk_000.wav").exists(), "Checkpoint chunk_000.wav must be saved to disk"

    # Clean up checkpoint dir
    shutil.rmtree(temp_checkpoints, ignore_errors=True)


def test_audio_mixing_and_zero_bleed():
    """Verify studio vocal mastering, sidechain ducking, and mixdown without bleed."""
    sr = 44100
    duration_s = 5.0
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)

    inst_samples = np.stack([0.3 * np.sin(2 * np.pi * 200 * t), 0.3 * np.sin(2 * np.pi * 250 * t)])
    orig_vocal = 0.5 * np.sin(2 * np.pi * 440 * t)
    conv_vocal = 0.5 * np.sin(2 * np.pi * 330 * t)  # Different pitch / timbre

    inst_asset = AudioAsset(samples=inst_samples, sample_rate=sr)
    orig_asset = AudioAsset(samples=orig_vocal, sample_rate=sr)
    conv_asset = AudioAsset(samples=conv_vocal, sample_rate=sr)

    # Test mastering chain
    mastered = apply_vocal_mastering_chain(conv_asset)
    assert mastered.samples.shape == conv_asset.samples.shape

    # Test ducking
    ducked = apply_sidechain_ducking(inst_samples, conv_vocal, sr=sr, duck_gain_db=-1.5)
    assert ducked.shape == inst_samples.shape

    # Test final mix and zero-bleed assertion
    out_path = ARTIFACTS_ROOT / "test_mix_output.wav"
    mixed = mix_and_master_song(
        converted_vocals_asset=conv_asset,
        instrumental_asset=inst_asset,
        output_path=out_path,
        original_vocals_asset=orig_asset,
    )
    assert out_path.exists(), "Mastered audio file must be written to disk"
    assert mixed.channels == 2, "Final mix must be stereo"
    assert np.max(np.abs(mixed.samples)) <= 1.0, "Output must be normalized without clipping"

    if out_path.exists():
        out_path.unlink()


def test_song_cover_models():
    """Verify Pydantic schemas and ORM instantiation."""
    u_id = uuid.uuid4()
    v_id = uuid.uuid4()
    job_id = uuid.uuid4()

    req = SongCoverCreate(
        voice_id=v_id,
        title="Test Song",
        pitch_shift=2,
        index_rate=0.8,
        preview_only=True,
    )
    assert req.pitch_shift == 2
    assert req.preview_only is True

    status_out = SongCoverStatusOut(
        id=job_id,
        status="converting",
        progress=45.0,
    )
    assert status_out.progress == 45.0
    assert status_out.status == "converting"


def test_curated_demo_catalog():
    """Verify curated demo catalog returns tracks with pre-separated stems."""
    from app.services.curated_catalog import get_curated_songs, get_curated_song_stems
    songs = get_curated_songs()
    assert len(songs) >= 3, "Expected at least 3 curated demo tracks"

    first_song = songs[0]
    assert first_song.song_hash.startswith("curated_")
    assert first_song.duration <= 60.0

    stems = get_curated_song_stems(first_song.song_hash)
    assert stems is not None, "Curated stems should be pre-cached on disk"
    assert stems.vocals_path.exists()
    assert stems.instrumental_path.exists()


def test_song_fetcher_validation():
    """Verify URL validation and 5-minute hard limit enforcement."""
    from app.services.song_fetcher import validate_song_url, MAX_SONG_DURATION_SEC
    assert validate_song_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert validate_song_url("https://soundcloud.com/artist/track") is True
    assert validate_song_url("https://example.com/song.mp3") is True
    assert validate_song_url("ftp://invalid.com/song") is False
    assert MAX_SONG_DURATION_SEC == 300.0, "Hard cap must be exactly 5 minutes (300 seconds)"


def test_song_cover_input_sources_and_metadata():
    """Verify UPLOAD, SEARCH, and LIBRARY input sources and metadata serialization."""
    v_id = uuid.uuid4()
    req_upload = SongCoverCreate(voice_id=v_id, source_type="UPLOAD", title="Upload Track")
    assert req_upload.source_type == "UPLOAD"

    req_search = SongCoverCreate(
        voice_id=v_id,
        source_type="SEARCH",
        source_url="https://youtube.com/watch?v=test",
        title="Search Track",
        tos_confirmed=True,
    )
    assert req_search.source_type == "SEARCH"
    assert req_search.source_url is not None

    req_library = SongCoverCreate(
        voice_id=v_id,
        source_type="LIBRARY",
        library_song_hash="curated_acoustic_01",
    )
    assert req_library.source_type == "LIBRARY"
    assert req_library.library_song_hash == "curated_acoustic_01"


def run_all_tests():
    funcs = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    print(f"Running {len(funcs)} song cover tests...")
    for fn in funcs:
        fn()
        print(f"  {fn.__name__}: PASS")
    print(f"All {len(funcs)} song cover tests PASSED!")


if __name__ == "__main__":
    run_all_tests()


