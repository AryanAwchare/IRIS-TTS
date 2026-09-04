"""
Auth router — user registration and login.

FIX: Login endpoint uses constant-time password verification even for
     non-existent email addresses (timing defense against user enumeration).
"""
from __future__ import annotations

import threading
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import create_access_token, get_current_user, hash_password, verify_password
from app.db import get_db
from app.models import Token, User, UserCreate, UserOut

router = APIRouter()

# Timing defense: always run bcrypt even for non-existent emails.
# This prevents response-time enumeration of valid email addresses.
# The dummy hash is computed once at module load and reused.
_DUMMY_HASH: str = ""
_DUMMY_HASH_LOCK = threading.Lock()


def _get_dummy_hash() -> str:
    """Lazily compute and cache the dummy bcrypt hash for timing defense."""
    global _DUMMY_HASH
    if _DUMMY_HASH:
        return _DUMMY_HASH
    with _DUMMY_HASH_LOCK:
        if not _DUMMY_HASH:
            _DUMMY_HASH = hash_password("voicelib-timing-defense-placeholder-xK7mN2")
    return _DUMMY_HASH


@router.post(
    "/register",
    response_model=Token,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(
    payload: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Token:
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user = User(email=payload.email, hashed_password=hash_password(payload.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token(str(user.id))
    return Token(access_token=token, user=UserOut.model_validate(user))


@router.post(
    "/login",
    response_model=Token,
    summary="Log in and receive a JWT access token",
)
async def login(
    payload: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Token:
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    # FIX: Always run bcrypt regardless of whether the email exists.
    # Using a pre-computed dummy hash for non-existent users ensures the
    # response time is constant, preventing user enumeration via timing.
    hash_to_check = user.hashed_password if user else _get_dummy_hash()
    password_valid = verify_password(payload.password, hash_to_check)

    if not user or not password_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    token = create_access_token(str(user.id))
    return Token(access_token=token, user=UserOut.model_validate(user))


@router.get(
    "/me",
    response_model=UserOut,
    summary="Return the currently authenticated user",
)
async def me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserOut:
    return UserOut.model_validate(current_user)
