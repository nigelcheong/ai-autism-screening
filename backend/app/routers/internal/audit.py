from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import not_, or_
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_staff, require_role, TenantContext
from app.core.audit import log_action
from app.core.filters import apply_date_range
from app.core.time import utc_iso
from app.models.db import get_db
from app.models.entities import AuditLog

router = APIRouter(prefix="/internal/admin/audit-log", tags=["audit-log"])

# AuditMiddleware blanket-logs every authenticated API call as "GET /path",
# "POST /path", etc. -- useful for debugging but noise for a human reading
# "what did this user do", so it's excluded by default (include_api_calls).
_API_NOISE_PREFIXES = ("GET ", "POST ", "PATCH ", "PUT ", "DELETE ")


class AuditLogOut(BaseModel):
    id: UUID
    actor_email: str
    actor_role: str
    action: str
    target: str | None
    created_at: str

    class Config:
        from_attributes = True


@router.get("", response_model=list[AuditLogOut])
def list_audit_log(
    search: str | None = None,
    actor_email: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    include_api_calls: bool = False,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    query = db.query(AuditLog)
    if not include_api_calls:
        query = query.filter(
            not_(or_(*[AuditLog.action.like(f"{prefix}%") for prefix in _API_NOISE_PREFIXES]))
        )
    if actor_email:
        query = query.filter(AuditLog.actor_email == actor_email)
    if search:
        query = query.filter(
            or_(
                AuditLog.actor_email.ilike(f"%{search}%"),
                AuditLog.action.ilike(f"%{search}%"),
                AuditLog.target.ilike(f"%{search}%"),
            )
        )
    query = apply_date_range(query, AuditLog.created_at, date_from, date_to)

    rows = query.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit).all()
    return [
        AuditLogOut(
            id=r.id,
            actor_email=r.actor_email,
            actor_role=r.actor_role,
            action=r.action,
            target=r.target,
            created_at=utc_iso(r.created_at),
        )
        for r in rows
    ]


@router.get("/actors", response_model=list[str])
def list_audit_actors(
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    rows = db.query(AuditLog.actor_email).distinct().order_by(AuditLog.actor_email).all()
    return [r[0] for r in rows]


# --- Any authenticated user: report a client-side page view, so the audit
# trail also covers navigation (not just API calls). ---

page_view_router = APIRouter(prefix="/internal/audit", tags=["audit-log"])


class PageViewIn(BaseModel):
    path: str


@page_view_router.post("/page-view", status_code=204)
def log_page_view(
    payload: PageViewIn,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    log_action(db, ctx, "page_view", target=payload.path)
