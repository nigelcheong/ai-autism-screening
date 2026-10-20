import secrets

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.api_key import hash_api_key, mask_api_key
from app.auth.dependencies import require_role, TenantContext
from app.core.time import utc_iso
from app.models.db import get_db
from app.models.entities import ApiKey

router = APIRouter(prefix="/internal/hospital/api-key", tags=["hospital-api-key"])


class ApiKeyStatus(BaseModel):
    active: bool
    masked_key: str | None
    created_at: str | None


class ApiKeyIssued(BaseModel):
    api_key: str  # only ever returned once, right after creation


def _current_key(db: Session, hospital_id: str) -> ApiKey | None:
    return (
        db.query(ApiKey)
        .filter(ApiKey.hospital_id == hospital_id, ApiKey.active.is_(True))
        .order_by(ApiKey.created_at.desc())
        .first()
    )


@router.get("", response_model=ApiKeyStatus)
def get_api_key_status(
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("hospital_admin")),
):
    key = _current_key(db, ctx.hospital_id)
    if key is None:
        return ApiKeyStatus(active=False, masked_key=None, created_at=None)
    return ApiKeyStatus(
        active=key.active,
        masked_key=mask_api_key(key.last_four),
        created_at=utc_iso(key.created_at),
    )


@router.post("/regenerate", response_model=ApiKeyIssued)
def regenerate_api_key(
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("hospital_admin")),
):
    # deactivate any existing key(s) before issuing a new one
    db.query(ApiKey).filter(ApiKey.hospital_id == ctx.hospital_id, ApiKey.active.is_(True)).update(
        {"active": False}
    )
    raw_key = secrets.token_urlsafe(32)
    record = ApiKey(
        hospital_id=ctx.hospital_id,
        key_hash=hash_api_key(raw_key),
        last_four=raw_key[-4:],
        active=True,
    )
    db.add(record)
    db.commit()
    return ApiKeyIssued(api_key=raw_key)
