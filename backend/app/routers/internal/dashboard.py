from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_staff, TenantContext
from app.models.db import get_db
from app.models.entities import DiagnosticSession, Patient

router = APIRouter(prefix="/internal/dashboard", tags=["dashboard"])


class DashboardStats(BaseModel):
    total_scanned: int
    total_patients: int
    autism_positive: int


@router.get("/stats", response_model=DashboardStats)
def get_stats(
    db: Session = Depends(get_db), ctx: TenantContext = Depends(get_current_staff)
):
    session_filters = [DiagnosticSession.hospital_id == ctx.hospital_id]
    # Doctors only see stats for cases assigned to them, not the whole hospital.
    if ctx.role == "doctor":
        session_filters.append(DiagnosticSession.assigned_doctor_id == ctx.subject)

    total_scanned = db.query(func.count(DiagnosticSession.id)).filter(*session_filters).scalar()

    if ctx.role == "doctor":
        total_patients = (
            db.query(func.count(func.distinct(DiagnosticSession.patient_id)))
            .filter(*session_filters)
            .scalar()
        )
    else:
        total_patients = (
            db.query(func.count(Patient.id))
            .filter(Patient.hospital_id == ctx.hospital_id)
            .scalar()
        )

    # "Positive" = a session the model flagged high likelihood; moderate/low are not counted.
    autism_positive = (
        db.query(func.count(DiagnosticSession.id))
        .filter(*session_filters, DiagnosticSession.risk_category == "high")
        .scalar()
    )
    return DashboardStats(
        total_scanned=total_scanned or 0,
        total_patients=total_patients or 0,
        autism_positive=autism_positive or 0,
    )
