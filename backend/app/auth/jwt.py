from datetime import datetime, timedelta

from jose import jwt, JWTError

from app.config import settings


def create_access_token(
    subject: str,
    role: str,
    hospital_id: str | None,
    email: str | None = None,
    impersonated_by: str | None = None,
) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": subject,
        "role": role,
        "hospital_id": hospital_id,
        "email": email,
        "impersonated_by": impersonated_by,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise ValueError("invalid or expired token") from exc
