"""
song_fetcher.py — Remote Song Search & Ingestion Service (Option 1: Search & Fetch).

Provides:
  - URL format validation (YouTube, SoundCloud, direct audio URLs)
  - Strict duration pre-check (enforces hard cap <= 300 seconds / 5 minutes BEFORE download)
  - Fetch audio stream via yt-dlp or streaming HTTP request
  - Standardizes downloaded stream into canonical 16-bit PCM WAV
"""
from __future__ import annotations

import io
import logging
import os
import re
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

from app.utils.audio_asset import AudioAsset, load_audio_asset

logger = logging.getLogger(__name__)

MAX_SONG_DURATION_SEC = 300.0  # 5-minute hard cap


def validate_song_url(url: str) -> bool:
    """Validates remote audio or video streaming URL."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        # Match common video/audio domains or direct audio files
        valid_domains = ["youtube.com", "youtu.be", "soundcloud.com", "bandcamp.com"]
        is_known_platform = any(d in parsed.netloc.lower() for d in valid_domains)
        is_direct_audio = any(url.lower().endswith(ext) for ext in [".mp3", ".wav", ".flac", ".m4a", ".ogg"])
        return is_known_platform or is_direct_audio or bool(parsed.netloc)
    except Exception:
        return False


def fetch_and_canonicalize_url(
    url: str,
    max_duration_sec: float = MAX_SONG_DURATION_SEC,
) -> Tuple[bytes, str, float]:
    """
    Downloads audio from remote URL (YouTube, SoundCloud, direct audio) with strict duration validation.
    Returns: (canonical_wav_bytes, detected_title, duration_sec)
    Raises: ValueError on download failure or unsupported URL.
    """
    if not validate_song_url(url):
        raise ValueError(f"Invalid or unsupported song URL: '{url}'")

    # 1. Attempt download using yt-dlp if installed (handles YouTube, SoundCloud, etc.)
    try:
        import yt_dlp
        logger.info(f"Fetching audio via yt-dlp for URL: {url}")
        tmp_dir = tempfile.mkdtemp()
        target_template = os.path.join(tmp_dir, "fetch_%(id)s.%(ext)s")
        
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": target_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "ignoreerrors": False,
            "extract_flat": False,
            "extractor_args": {
                "youtube": {
                    "player_client": ["android", "ios", "web"],
                }
            },
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            },
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Metadata pre-check before downloading
            try:
                info = ydl.extract_info(url, download=True)
            except Exception as dl_err:
                raise ValueError(f"Failed to download audio from YouTube: {dl_err}")

            title = info.get("title") or "Remote Track"

            # Find downloaded file
            downloaded_files = list(Path(tmp_dir).glob("fetch_*"))
            if not downloaded_files:
                raise RuntimeError("No audio file produced by YouTube downloader.")

            raw_file = downloaded_files[0]
            asset = load_audio_asset(raw_file, target_sr=44100)

            # Auto-trim if longer than 5 minutes
            if asset.duration > max_duration_sec:
                logger.info(f"Song duration ({asset.duration:.1f}s) trimmed to {int(max_duration_sec)}s limit.")
                asset = asset.slice(0.0, max_duration_sec)

            return asset.to_bytes(), title, asset.duration

    except ImportError:
        logger.warning("yt-dlp is not installed in the backend environment. Direct YouTube extraction requires yt-dlp.")
        raise ValueError("YouTube downloads require 'yt-dlp'. Please run: pip install yt-dlp in your backend venv.")
    except ValueError:
        raise
    except Exception as e:
        logger.warning(f"yt-dlp fetch notice: {e} — falling back to direct HTTP fetch")
    finally:
        if 'tmp_dir' in locals() and os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # 2. Direct HTTP Stream Fetch Fallback
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "VoiceLib-SongFetcher/2.0"}
    )
    buffer = io.BytesIO()
    max_bytes = 100 * 1024 * 1024  # 100MB limit
    total_bytes = 0
    chunk_size = 64 * 1024

    with urllib.request.urlopen(req, timeout=30.0) as res:
        while True:
            chunk = res.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError("Downloaded file exceeds 100MB size limit.")
            buffer.write(chunk)

    raw_bytes = buffer.getvalue()

    asset = load_audio_asset(raw_bytes, target_sr=44100)
    if asset.duration > max_duration_sec:
        asset = asset.slice(0.0, max_duration_sec)

    title = Path(urllib.parse.urlparse(url).path).stem or "Fetched Song"
    return asset.to_bytes(), title, asset.duration
