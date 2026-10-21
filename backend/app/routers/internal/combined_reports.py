from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_staff, require_role, TenantContext
from app.core.audit import log_action
from app.core.time import utc_iso
from app.models.db import get_db
from app.models.entities import CombinedReport, DiagnosticSession, Hospital, Patient, StaffUser
from app.routers.internal.reports import build_session_report, SessionReport

router = APIRouter(prefix="/internal/combined-reports", tags=["combined-reports"])


class CombinedReportIn(BaseModel):
    patient_id: str
    session_ids: list[str]
    comments: str = ""

    @field_validator("session_ids")
    @classmethod
    def exactly_two(cls, v: list[str]) -> list[str]:
        if len(set(v)) != 2:
            raise ValueError("combined reports take exactly two distinct sessions")
        return v


class CombinedReportCommentsIn(BaseModel):
    comments: str


class CombinedReportSummary(BaseModel):
    id: UUID
    patient_id: UUID
    patient_name: str
    session_ids: list[UUID]
    comments: str
    created_by_email: str | None
    created_at: str
    updated_at: str


class CombinedReportOut(CombinedReportSummary):
    hospital_name: str
    reports: list[SessionReport]


def _get_sessions_for(db: Session, ctx: TenantContext, patient_id: str, session_ids: list[str]):
    sessions = (
        db.query(DiagnosticSession)
        .filter(DiagnosticSession.id.in_(session_ids), DiagnosticSession.patient_id == patient_id)
        .all()
    )
    if len(sessions) != 2:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "one or both sessions not found for this patient")
    for s in sessions:
        if ctx.role != "platform_admin" and str(s.hospital_id) != ctx.hospital_id:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "not your hospital's session")
    # Stable order: always the earlier-created session first, regardless of
    # the order the two ids were selected in.
    return sorted(sessions, key=lambda s: s.created_at)


def _to_summary(db: Session, report: CombinedReport) -> dict:
    patient = db.query(Patient).filter(Patient.id == report.patient_id).first()
    creator = db.query(StaffUser).filter(StaffUser.id == report.created_by).first()
    return {
        "id": report.id,
        "patient_id": report.patient_id,
        "patient_name": patient.name if patient else "Unknown",
        "session_ids": [report.session_a_id, report.session_b_id],
        "comments": report.comments,
        "created_by_email": creator.email if creator else None,
        "created_at": utc_iso(report.created_at),
        "updated_at": utc_iso(report.updated_at),
    }


@router.get("", response_model=list[CombinedReportSummary])
def list_combined_reports(
    patient_id: str,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    query = db.query(CombinedReport).filter(CombinedReport.patient_id == patient_id)
    if ctx.role != "platform_admin":
        query = query.filter(CombinedReport.hospital_id == ctx.hospital_id)
    reports = query.order_by(CombinedReport.created_at.desc()).all()
    return [CombinedReportSummary(**_to_summary(db, r)) for r in reports]


@router.post("", response_model=CombinedReportOut)
def create_combined_report(
    payload: CombinedReportIn,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("staff", "hospital_admin", "doctor")),
):
    patient = (
        db.query(Patient)
        .filter(Patient.id == payload.patient_id, Patient.hospital_id == ctx.hospital_id)
        .first()
    )
    if patient is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "patient not found")

    sessions = _get_sessions_for(db, ctx, payload.patient_id, payload.session_ids)

    report = CombinedReport(
        hospital_id=ctx.hospital_id,
        patient_id=payload.patient_id,
        session_a_id=sessions[0].id,
        session_b_id=sessions[1].id,
        comments=payload.comments,
        created_by=ctx.subject,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    log_action(
        db, ctx, "create_combined_report", target=patient.name, hospital_id=ctx.hospital_id
    )

    hospital = db.query(Hospital).filter(Hospital.id == report.hospital_id).first()
    return CombinedReportOut(
        **_to_summary(db, report),
        hospital_name=hospital.name if hospital else "N/A",
        reports=[build_session_report(db, s) for s in sessions],
    )


@router.get("/{report_id}", response_model=CombinedReportOut)
def get_combined_report(
    report_id: str,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    report = db.query(CombinedReport).filter(CombinedReport.id == report_id).first()
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "combined report not found")
    if ctx.role != "platform_admin" and str(report.hospital_id) != ctx.hospital_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your hospital's report")

    sessions = (
        db.query(DiagnosticSession)
        .filter(DiagnosticSession.id.in_([report.session_a_id, report.session_b_id]))
        .all()
    )
    sessions_by_id = {s.id: s for s in sessions}
    ordered = [sessions_by_id[report.session_a_id], sessions_by_id[report.session_b_id]]

    hospital = db.query(Hospital).filter(Hospital.id == report.hospital_id).first()
    return CombinedReportOut(
        **_to_summary(db, report),
        hospital_name=hospital.name if hospital else "N/A",
        reports=[build_session_report(db, s) for s in ordered],
    )


@router.patch("/{report_id}", response_model=CombinedReportSummary)
def update_combined_report_comments(
    report_id: str,
    payload: CombinedReportCommentsIn,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("staff", "hospital_admin", "doctor")),
):
    report = db.query(CombinedReport).filter(CombinedReport.id == report_id).first()
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "combined report not found")
    if str(report.hospital_id) != ctx.hospital_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your hospital's report")

    report.comments = payload.comments
    report.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(report)
    return CombinedReportSummary(**_to_summary(db, report))
