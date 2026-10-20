from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import require_role, TenantContext
from app.core.filters import apply_date_range
from app.core.time import utc_iso
from app.models.db import get_db
from app.models.entities import DiagnosticSession, Patient

router = APIRouter(prefix="/internal/doctor", tags=["doctor-cases"])


class CaseOut(BaseModel):
    id: UUID
    patient_id: UUID
    patient_name: str
    session_type: str
    risk_score: float | None
    risk_category: str | None
    created_at: str


@router.get("/cases", response_model=list[CaseOut])
def list_my_cases(
    search: str | None = None,
    risk_category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 25,
    offset: int = 0,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("doctor")),
):
    query = (
        db.query(DiagnosticSession, Patient.name)
        .join(Patient, Patient.id == DiagnosticSession.patient_id)
        .filter(
            DiagnosticSession.hospital_id == ctx.hospital_id,
            DiagnosticSession.assigned_doctor_id == ctx.subject,
        )
    )
    if search:
        query = query.filter(Patient.name.ilike(f"%{search}%"))
    if risk_category:
        query = query.filter(DiagnosticSession.risk_category == risk_category)
    query = apply_date_range(query, DiagnosticSession.created_at, date_from, date_to)

    rows = query.order_by(DiagnosticSession.created_at.desc()).offset(offset).limit(limit).all()
    return [
        CaseOut(
            id=session.id,
            patient_id=session.patient_id,
            patient_name=patient_name,
            session_type=session.session_type,
            risk_score=session.risk_score,
            risk_category=session.risk_category,
            created_at=utc_iso(session.created_at),
        )
        for session, patient_name in rows
    ]
