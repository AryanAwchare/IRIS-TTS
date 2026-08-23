"""
S3-compatible object storage wrapper with automatic local disk fallback.

Works with:
    - AWS S3          (leave STORAGE_ENDPOINT_URL unset)
    - Cloudflare R2   (set STORAGE_ENDPOINT_URL to R2 endpoint)
    - MinIO           (set STORAGE_ENDPOINT_URL to http://minio:9000)
    - Local Disk      (automatic fallback when S3/MinIO is unreachable)
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.config import get_settings

logger = logging.getLogger(__name__)

LOCAL_STORAGE_DIR = Path("./local_storage_data")


def _get_client():
    s = get_settings()
    kwargs: dict = dict(
        aws_access_key_id=s.storage_access_key,
        aws_secret_access_key=s.storage_secret_key,
        region_name=s.storage_region,
    )
    if s.storage_endpoint_url:
        kwargs["endpoint_url"] = s.storage_endpoint_url
    return boto3.client("s3", **kwargs)


_local_fallback_active = False


def _is_local_mode() -> bool:
    s = get_settings()
    if _local_fallback_active or (s.storage_access_key in ("local", "minioadmin") and not s.storage_endpoint_url):
        return True
    return False


def ensure_bucket_exists() -> None:
    """Create S3 bucket or local storage directory if missing."""
    global _local_fallback_active
    s = get_settings()
    LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if _is_local_mode():
        _local_fallback_active = True
        logger.info("Using local disk storage fallback.")
        return

    try:
        client = _get_client()
        client.head_bucket(Bucket=s.storage_bucket_name)
        _local_fallback_active = False
        logger.info(f"Storage bucket '{s.storage_bucket_name}' ready.")
    except Exception as e:
        _local_fallback_active = True
        logger.warning(f"S3 connection check failed ({e}). Enabling local disk storage fallback.")


def upload_bytes(data: bytes, content_type: str, prefix: str = "uploads") -> str:
    global _local_fallback_active
    s = get_settings()
    key = f"{prefix}/{uuid.uuid4()}"

    if _is_local_mode():
        target_path = LOCAL_STORAGE_DIR / key
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
        logger.info(f"Saved {len(data):,} bytes to local disk fallback: {key}")
        return key

    try:
        _get_client().put_object(
            Bucket=s.storage_bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
        logger.info(f"Uploaded {len(data):,} bytes to S3 → {key}")
    except Exception as e:
        _local_fallback_active = True
        logger.warning(f"S3 upload failed ({e}). Saving to local disk fallback: {key}")
        target_path = LOCAL_STORAGE_DIR / key
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(data)
    return key


def download_bytes(key: str) -> bytes:
    global _local_fallback_active
    s = get_settings()

    if (LOCAL_STORAGE_DIR / key).exists() or _is_local_mode():
        target_path = LOCAL_STORAGE_DIR / key
        if target_path.exists():
            return target_path.read_bytes()

    try:
        response = _get_client().get_object(Bucket=s.storage_bucket_name, Key=key)
        data: bytes = response["Body"].read()
        logger.info(f"Downloaded {len(data):,} bytes from S3 ← {key}")
        return data
    except Exception as e:
        _local_fallback_active = True
        logger.warning(f"S3 download failed ({e}). Reading from local disk fallback: {key}")
        target_path = LOCAL_STORAGE_DIR / key
        if target_path.exists():
            return target_path.read_bytes()
        raise FileNotFoundError(f"Object {key} not found in S3 or local disk storage.")


def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    s = get_settings()
    api_base = (s.vite_api_base_url or "http://localhost:8000").rstrip("/")
    local_url = f"{api_base}/storage_files/{key}"

    if s.storage_public_base_url:
        return f"{s.storage_public_base_url.rstrip('/')}/{key}"

    if (LOCAL_STORAGE_DIR / key).exists() or _is_local_mode():
        return local_url

    try:
        url: str = _get_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": s.storage_bucket_name, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
    except Exception:
        return local_url


def delete_object(key: str) -> None:
    s = get_settings()
    target_path = LOCAL_STORAGE_DIR / key
    if target_path.exists():
        target_path.unlink()
        logger.info(f"Deleted local file: {key}")

    if not _is_local_mode():
        try:
            _get_client().delete_object(Bucket=s.storage_bucket_name, Key=key)
            logger.info(f"Deleted S3 object: {key}")
        except Exception:
            pass
