"""Encrypts question_responses at rest so raw DB access doesn't expose answers.

Uses a single server-held Fernet key (ENCRYPTION_KEY). This protects against
someone reading the database directly; it is NOT per-user access control —
authorization for who can see decrypted answers through the API is enforced
separately in app/routers/internal/reports.py.
"""

import json

from cryptography.fernet import Fernet

from app.config import settings

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        if not settings.encryption_key:
            raise RuntimeError("ENCRYPTION_KEY is not set")
        _fernet = Fernet(settings.encryption_key.encode())
    return _fernet


def encrypt_json(data: dict) -> str:
    payload = json.dumps(data).encode()
    return _get_fernet().encrypt(payload).decode()


def decrypt_json(token: str) -> dict:
    payload = _get_fernet().decrypt(token.encode())
    return json.loads(payload)
