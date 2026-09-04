"""
Database engine, session factory, and base declarative model.
Uses SQLAlchemy 2.0 async pattern with asyncpg driver.
"""
from __future__ import annotations

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def _create_engine():
    settings = get_settings()
    url = settings.database_url
    if "sqlite" in url:
        return create_async_engine(url, echo=False)
    return create_async_engine(
        url,
        echo=False,
        # Supabase transaction-mode pooler (port 6543) closes idle connections
        # aggressively. Keep the pool small and recycle every 5 min.
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300,
        pool_timeout=30,
        pool_reset_on_return="rollback",
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
            "server_settings": {
                "application_name": "voicelib",
            },
        },
    )


engine = _create_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a DB session and ensures it is closed."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """
    Create all tables defined by ORM models.
    For development only — auto-adds missing columns for existing databases.
    """
    from app import models  # noqa: F401 — ensure models are registered
    from sqlalchemy import text

    async with engine.begin() as conn:
        settings = get_settings()
        is_sqlite = "sqlite" in settings.database_url.lower()

        if is_sqlite:
            # Enable Write-Ahead Logging & busy timeout for concurrent safety
            try:
                await conn.execute(text("PRAGMA journal_mode=WAL;"))
                await conn.execute(text("PRAGMA busy_timeout=5000;"))
            except Exception:
                pass

        await conn.run_sync(Base.metadata.create_all)

        if is_sqlite:
            # SQLite does NOT support 'ADD COLUMN IF NOT EXISTS' syntax
            # Inspect existing columns dynamically using PRAGMA table_info
            table_columns = {
                "voices": [
                    ("speech_capable", "BOOLEAN DEFAULT 1"),
                    ("singing_capable", "BOOLEAN DEFAULT 0"),
                    ("singing_identity", "JSON"),
                    ("opt_weights", "JSON"),
                    ("pronunciation_dict", "JSON"),
                ],
                "generations": [
                    ("engine", "VARCHAR(64) DEFAULT 'gpt-sovits-v3'"),
                    ("emotion", "VARCHAR(64) DEFAULT 'auto'"),
                    ("speed", "FLOAT DEFAULT 1.0"),
                    ("eval_status", "VARCHAR(32) DEFAULT 'pending'"),
                    ("speaker_similarity", "FLOAT"),
                    ("word_error_rate", "FLOAT"),
                    ("prosody_f0_std", "FLOAT"),
                    ("composite_grade", "VARCHAR(4)"),
                    ("composite_score", "FLOAT"),
                    ("eval_error", "TEXT"),
                    ("evaluated_at", "TIMESTAMP"),
                ],
                "song_covers": [
                    ("source_type", "VARCHAR(32) DEFAULT 'UPLOAD'"),
                    ("source_url", "VARCHAR(1024)"),
                    ("song_hash", "VARCHAR(64)"),
                    ("metadata_json", "JSON"),
                ],
            }
            for table_name, cols in table_columns.items():
                try:
                    res = await conn.execute(text(f"PRAGMA table_info({table_name});"))
                    existing_cols = {row[1] for row in res.fetchall()}
                    for col_name, col_def in cols:
                        if col_name not in existing_cols:
                            await conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def};"))
                except Exception:
                    pass
        else:
            # PostgreSQL migrations with IF NOT EXISTS
            migrations = [
                "ALTER TABLE voices ADD COLUMN IF NOT EXISTS speech_capable BOOLEAN DEFAULT TRUE;",
                "ALTER TABLE voices ADD COLUMN IF NOT EXISTS singing_capable BOOLEAN DEFAULT FALSE;",
                "ALTER TABLE voices ADD COLUMN IF NOT EXISTS singing_identity JSON;",
                "ALTER TABLE voices ADD COLUMN IF NOT EXISTS opt_weights JSON;",
                "ALTER TABLE voices ADD COLUMN IF NOT EXISTS pronunciation_dict JSON;",
                "ALTER TABLE generations ADD COLUMN IF NOT EXISTS engine VARCHAR(64) DEFAULT 'gpt-sovits-v3';",
                "ALTER TABLE generations ADD COLUMN IF NOT EXISTS emotion VARCHAR(64) DEFAULT 'auto';",
                "ALTER TABLE generations ADD COLUMN IF NOT EXISTS speed FLOAT DEFAULT 1.0;",
                "ALTER TABLE generations ADD COLUMN IF NOT EXISTS eval_status VARCHAR(32) DEFAULT 'pending';",
                "ALTER TABLE generations ADD COLUMN IF NOT EXISTS speaker_similarity FLOAT;",
                "ALTER TABLE generations ADD COLUMN IF NOT EXISTS word_error_rate FLOAT;",
                "ALTER TABLE generations ADD COLUMN IF NOT EXISTS prosody_f0_std FLOAT;",
                "ALTER TABLE generations ADD COLUMN IF NOT EXISTS composite_grade VARCHAR(4);",
                "ALTER TABLE generations ADD COLUMN IF NOT EXISTS composite_score FLOAT;",
                "ALTER TABLE generations ADD COLUMN IF NOT EXISTS eval_error TEXT;",
                "ALTER TABLE generations ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMP WITH TIME ZONE;",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS source_type VARCHAR(32) DEFAULT 'UPLOAD';",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS source_url VARCHAR(1024);",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS song_hash VARCHAR(64);",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS metadata_json JSON;",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS is_preview BOOLEAN DEFAULT FALSE;",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS preview_s3_key VARCHAR(512);",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS vocals_s3_key VARCHAR(512);",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS instrumental_s3_key VARCHAR(512);",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS converted_vocals_s3_key VARCHAR(512);",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS final_mix_s3_key VARCHAR(512);",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS pitch_shift INTEGER DEFAULT 0;",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS index_rate FLOAT DEFAULT 0.75;",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS protect_voiceless FLOAT DEFAULT 0.33;",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS progress FLOAT DEFAULT 0.0;",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS error_message TEXT;",
                "ALTER TABLE song_covers ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP WITH TIME ZONE;",
            ]
            for stmt in migrations:
                try:
                    await conn.execute(text(stmt))
                except Exception:
                    pass
