import re

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.auth.jwt import decode_access_token
from app.models.db import SessionLocal
from app.models.entities import AuditLog

# Paths that already log a clearer, hand-written event (see routers/internal/auth.py)
# or that are too noisy / not meaningful to audit.
_SKIP_EXACT = {
    ("POST", "/internal/auth/login"),
    ("POST", "/internal/auth/logout"),
    ("POST", "/internal/audit/page-view"),
}
_SKIP_PREFIXES = ("/docs", "/openapi.json", "/redoc", "/health")

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _normalize_path(path: str) -> str:
    return _UUID_RE.sub("{id}", path)


class AuditMiddleware(BaseHTTPMiddleware):
    """Best-effort blanket audit trail: logs every authenticated /internal/*
    request (method + path + outcome) so the audit log covers page-driven
    actions (viewing a report, listing patients, etc.), not just the
    hand-instrumented admin mutations. Never blocks or fails the request."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        try:
            path = request.url.path
            method = request.method
            if method == "OPTIONS" or path.startswith(_SKIP_PREFIXES):
                return response
            if (method, path) in _SKIP_EXACT:
                return response
            if not path.startswith("/internal/"):
                return response

            auth_header = request.headers.get("authorization", "")
            if not auth_header.lower().startswith("bearer "):
                return response
            token = auth_header.split(" ", 1)[1]
            try:
                payload = decode_access_token(token)
            except ValueError:
                return response

            db = SessionLocal()
            try:
                db.add(
                    AuditLog(
                        actor_email=payload.get("email") or payload.get("sub", "unknown"),
                        actor_role=payload.get("role", "unknown"),
                        action=f"{method} {_normalize_path(path)}",
                        target=f"{path}?{request.url.query}" if request.url.query else path,
                        hospital_id=payload.get("hospital_id"),
                    )
                )
                db.commit()
            finally:
                db.close()
        except Exception:
            # Auditing must never break the actual request.
            pass

        return response
