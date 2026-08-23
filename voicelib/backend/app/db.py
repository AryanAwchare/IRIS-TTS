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
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
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
        await conn.run_sync(Base.metadata.create_all)
        # Ensure schema completeness on pre-existing databases (PostgreSQL / SQLite)
        migrations = [
            "ALTER TABLE voices ADD COLUMN IF NOT EXISTS opt_weights JSON;",
            "ALTER TABLE voices ADD COLUMN IF NOT EXISTS pronunciation_dict JSON;",
            "ALTER TABLE generations ADD COLUMN IF NOT EXISTS engine VARCHAR(64) DEFAULT 'gpt-sovits-v3';",
            "ALTER TABLE generations ADD COLUMN IF NOT EXISTS emotion VARCHAR(64) DEFAULT 'neutral';",
            "ALTER TABLE generations ADD COLUMN IF NOT EXISTS speed FLOAT DEFAULT 1.0;",
        ]
        for stmt in migrations:
            try:
                await conn.execute(text(stmt))
            except Exception:
                pass

