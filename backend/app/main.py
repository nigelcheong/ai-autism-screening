from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.routers.external.diagnose import limiter as external_limiter
from app.routers.external import diagnose as external_diagnose, patients as external_patients
from app.routers.internal import auth as internal_auth, patients as internal_patients
from app.routers.internal import diagnose as internal_diagnose, admin as internal_admin
from app.routers.internal import dashboard as internal_dashboard
from app.routers.internal import hospital_api_key as internal_hospital_api_key
from app.routers.internal import questions as internal_questions
from app.routers.internal import staff as internal_staff
from app.routers.internal import cases as internal_cases
from app.routers.internal import reports as internal_reports
from app.routers.internal import audit as internal_audit
from app.routers.internal import platform_stats as internal_platform_stats
from app.core.audit_middleware import AuditMiddleware

app = FastAPI(
    title="Autism Screening API",
    description=(
        "Multi-tenant API for hospital-facing autism screening (questionnaire and image based). "
        "`/internal/*` routes use JWT staff auth; `/v1/external/*` routes use a per-hospital API key."
    ),
    version="0.1.0",
)

app.state.limiter = external_limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AuditMiddleware)

app.include_router(internal_auth.router)
app.include_router(internal_patients.router)
app.include_router(internal_diagnose.router)
app.include_router(internal_admin.router)
app.include_router(internal_dashboard.router)
app.include_router(internal_hospital_api_key.router)
app.include_router(internal_questions.admin_router)
app.include_router(internal_questions.live_router)
app.include_router(internal_staff.router)
app.include_router(internal_cases.router)
app.include_router(internal_reports.router)
app.include_router(internal_audit.router)
app.include_router(internal_audit.page_view_router)
app.include_router(internal_platform_stats.router)
app.include_router(external_diagnose.router)
app.include_router(external_patients.router)


@app.get("/health")
def health():
    return {"status": "ok"}
