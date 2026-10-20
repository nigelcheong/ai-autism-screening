from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.auth.api_key import hash_api_key
from app.auth.jwt import decode_access_token
from app.models.db import get_db
from app.models.entities import ApiKey

bearer_scheme = HTTPBearer(auto_error=False)


class TenantContext:
    """Resolved identity for the current request, from either a JWT or an API key."""

    def __init__(self, hospital_id: str | None, role: str, subject: str, email: str | None = None):
        self.hospital_id = hospital_id
        self.role = role
        self.subject = subject
        self.email = email


def get_current_staff(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> TenantContext:
    if credentials is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    return TenantContext(
        hospital_id=payload.get("hospital_id"),
        role=payload["role"],
        subject=payload["sub"],
        email=payload.get("email"),
    )


def require_role(*allowed_roles: str):
    def _check(ctx: TenantContext = Depends(get_current_staff)) -> TenantContext:
        if ctx.role not in allowed_roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "insufficient role")
        return ctx

    return _check


def resolve_tenant_from_api_key(api_key: str, db: Session) -> TenantContext:
    key_hash = hash_api_key(api_key)
    record = (
        db.query(ApiKey)
        .filter(ApiKey.key_hash == key_hash, ApiKey.active.is_(True))
        .first()
    )
    if record is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
    return TenantContext(hospital_id=str(record.hospital_id), role="external", subject=str(record.id))
