from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from app.auth.dependencies import resolve_tenant_from_api_key
from app.models.db import get_db
from app.models.entities import Patient
from app.routers.internal.patients import PatientIn, PatientOut

router = APIRouter(prefix="/v1/external/patients", tags=["external-patients"])


@router.get("", response_model=list[PatientOut])
def list_patients(x_api_key: str = Header(...), db: Session = Depends(get_db)):
    ctx = resolve_tenant_from_api_key(x_api_key, db)
    return db.query(Patient).filter(Patient.hospital_id == ctx.hospital_id).all()


@router.post("", response_model=PatientOut)
def create_patient(
    payload: PatientIn, x_api_key: str = Header(...), db: Session = Depends(get_db)
):
    ctx = resolve_tenant_from_api_key(x_api_key, db)
    patient = Patient(hospital_id=ctx.hospital_id, **payload.dict())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient
