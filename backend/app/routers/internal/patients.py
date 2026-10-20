from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_staff, TenantContext
from app.core.time import utc_iso
from app.models.db import get_db
from app.models.entities import Patient, DiagnosticSession

router = APIRouter(prefix="/internal/patients", tags=["patients"])


class PatientIn(BaseModel):
    name: str
    dob: datetime | None = None
    external_patient_id: str | None = None
    gender: str | None = None
    autism: bool | None = None
    jaundice: bool | None = None
    used_app_before: bool | None = None
    country_of_res: str | None = None
    relation: str | None = None


class PatientOut(PatientIn):
    id: UUID
    latest_risk_category: str | None = None
    latest_screening_at: str | None = None

    class Config:
        from_attributes = True


@router.get("", response_model=list[PatientOut])
def list_patients(
    search: str | None = None,
    risk_category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    query = db.query(Patient).filter(Patient.hospital_id == ctx.hospital_id)
    if ctx.role == "doctor":
        # Doctors see patients they have an assigned case for, plus any
        # patient they added themselves (even before a session exists).
        assigned_patient_ids = (
            db.query(DiagnosticSession.patient_id)
            .filter(
                DiagnosticSession.hospital_id == ctx.hospital_id,
                DiagnosticSession.assigned_doctor_id == ctx.subject,
            )
            .distinct()
        )
        from sqlalchemy import or_

        query = query.filter(
            or_(Patient.id.in_(assigned_patient_ids), Patient.created_by == ctx.subject)
        )
    if search:
        query = query.filter(Patient.name.ilike(f"%{search}%"))

    patients = query.order_by(Patient.name).all()

    # latest session per patient, for the risk badge shown in the list
    patient_ids = [p.id for p in patients]
    latest_by_patient: dict = {}
    if patient_ids:
        sessions = (
            db.query(DiagnosticSession)
            .filter(DiagnosticSession.patient_id.in_(patient_ids))
            .order_by(DiagnosticSession.created_at.desc())
            .all()
        )
        for s in sessions:
            latest_by_patient.setdefault(s.patient_id, s)

    date_from_dt = datetime.fromisoformat(date_from) if date_from else None
    date_to_dt = datetime.fromisoformat(date_to) + timedelta(days=1) if date_to else None

    results = []
    for p in patients:
        latest = latest_by_patient.get(p.id)
        if risk_category and (latest is None or latest.risk_category != risk_category):
            continue
        if (date_from_dt or date_to_dt) and (
            latest is None
            or (date_from_dt and latest.created_at < date_from_dt)
            or (date_to_dt and latest.created_at >= date_to_dt)
        ):
            continue
        results.append(
            PatientOut(
                id=p.id,
                name=p.name,
                dob=p.dob,
                external_patient_id=p.external_patient_id,
                gender=p.gender,
                autism=p.autism,
                jaundice=p.jaundice,
                used_app_before=p.used_app_before,
                country_of_res=p.country_of_res,
                relation=p.relation,
                latest_risk_category=latest.risk_category if latest else None,
                latest_screening_at=utc_iso(latest.created_at) if latest else None,
            )
        )

    return results[offset : offset + limit]


@router.post("", response_model=PatientOut)
def create_patient(
    payload: PatientIn,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    patient = Patient(hospital_id=ctx.hospital_id, created_by=ctx.subject, **payload.dict())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


class PatientPatch(BaseModel):
    name: str | None = None
    dob: datetime | None = None
    external_patient_id: str | None = None
    gender: str | None = None
    autism: bool | None = None
    jaundice: bool | None = None
    used_app_before: bool | None = None
    country_of_res: str | None = None
    relation: str | None = None


@router.patch("/{patient_id}", response_model=PatientOut)
def update_patient(
    patient_id: str,
    payload: PatientPatch,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    patient = (
        db.query(Patient)
        .filter(Patient.id == patient_id, Patient.hospital_id == ctx.hospital_id)
        .first()
    )
    if patient is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "patient not found")
    for field in (
        "name",
        "dob",
        "external_patient_id",
        "gender",
        "autism",
        "jaundice",
        "used_app_before",
        "country_of_res",
        "relation",
    ):
        value = getattr(payload, field)
        if value is not None:
            setattr(patient, field, value)
    db.commit()
    db.refresh(patient)
    return PatientOut(
        id=patient.id,
        name=patient.name,
        dob=patient.dob,
        external_patient_id=patient.external_patient_id,
        gender=patient.gender,
        autism=patient.autism,
        jaundice=patient.jaundice,
        used_app_before=patient.used_app_before,
        country_of_res=patient.country_of_res,
        relation=patient.relation,
    )


class SessionSummary(BaseModel):
    id: UUID
    session_type: str
    risk_score: float | None
    risk_category: str | None
    model_version: str | None
    assigned_doctor_id: UUID | None
    created_at: str

    class Config:
        from_attributes = True


@router.get("/{patient_id}/sessions", response_model=list[SessionSummary])
def list_patient_sessions(
    patient_id: str,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    patient = (
        db.query(Patient)
        .filter(Patient.id == patient_id, Patient.hospital_id == ctx.hospital_id)
        .first()
    )
    if patient is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "patient not found")

    query = db.query(DiagnosticSession).filter(DiagnosticSession.patient_id == patient_id)
    if ctx.role == "doctor":
        query = query.filter(DiagnosticSession.assigned_doctor_id == ctx.subject)

    sessions = query.order_by(DiagnosticSession.created_at.desc()).all()
    return [
        SessionSummary(
            id=s.id,
            session_type=s.session_type,
            risk_score=s.risk_score,
            risk_category=s.risk_category,
            model_version=s.model_version,
            assigned_doctor_id=s.assigned_doctor_id,
            created_at=utc_iso(s.created_at),
        )
        for s in sessions
    ]
