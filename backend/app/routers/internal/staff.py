from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_staff, TenantContext
from app.models.db import get_db
from app.models.entities import StaffUser

router = APIRouter(prefix="/internal/staff", tags=["staff"])


class DoctorOut(BaseModel):
    id: UUID
    email: str

    class Config:
        from_attributes = True


@router.get("/doctors", response_model=list[DoctorOut])
def list_doctors(
    db: Session = Depends(get_db), ctx: TenantContext = Depends(get_current_staff)
):
    return (
        db.query(StaffUser)
        .filter(StaffUser.hospital_id == ctx.hospital_id, StaffUser.role == "doctor")
        .all()
    )
