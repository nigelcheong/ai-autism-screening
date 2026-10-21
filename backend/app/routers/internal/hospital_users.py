import re
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from passlib.context import CryptContext
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.auth.dependencies import require_role, TenantContext
from app.core.audit import log_action
from app.models.db import get_db
from app.models.entities import StaffUser

router = APIRouter(prefix="/internal/hospital/users", tags=["hospital-users"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# A hospital_admin can add their own hospital's day-to-day staff, but not
# another hospital_admin -- granting that peer-level role is a platform_admin
# action only (/internal/admin/hospitals/{id}/users), so one hospital_admin
# can't quietly promote a colleague without platform oversight.
SELF_SERVICE_ROLES = ("doctor", "staff")


class HospitalUserIn(BaseModel):
    email: str
    password: str
    role: str  # "doctor" | "staff"

    @field_validator("email")
    @classmethod
    def valid_email(cls, v: str) -> str:
        if not EMAIL_RE.match(v):
            raise ValueError("not a valid email address")
        return v


class HospitalUserOut(BaseModel):
    id: UUID
    email: str
    role: str

    class Config:
        from_attributes = True


class HospitalUserPatch(BaseModel):
    email: str | None = None
    role: str | None = None
    password: str | None = None

    @field_validator("email")
    @classmethod
    def valid_email(cls, v: str | None) -> str | None:
        if v is not None and not EMAIL_RE.match(v):
            raise ValueError("not a valid email address")
        return v


def _get_own_hospital_user(db: Session, hospital_id: str, user_id: str) -> StaffUser:
    user = (
        db.query(StaffUser)
        .filter(StaffUser.id == user_id, StaffUser.hospital_id == hospital_id)
        .first()
    )
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    return user


@router.get("", response_model=list[HospitalUserOut])
def list_my_hospital_users(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("hospital_admin")),
):
    return (
        db.query(StaffUser)
        .filter(StaffUser.hospital_id == ctx.hospital_id)
        .order_by(StaffUser.email)
        .offset(offset)
        .limit(limit)
        .all()
    )


@router.post("", response_model=HospitalUserOut)
def create_my_hospital_user(
    payload: HospitalUserIn,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("hospital_admin")),
):
    if payload.role not in SELF_SERVICE_ROLES:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, f"role must be one of {SELF_SERVICE_ROLES}"
        )
    if db.query(StaffUser).filter(StaffUser.email == payload.email).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "a user with this email already exists")

    user = StaffUser(
        hospital_id=ctx.hospital_id,
        email=payload.email,
        hashed_password=pwd_context.hash(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log_action(
        db, ctx, "create_hospital_user", target=f"{user.email} ({user.role})", hospital_id=ctx.hospital_id
    )
    return user


@router.patch("/{user_id}", response_model=HospitalUserOut)
def update_my_hospital_user(
    user_id: str,
    payload: HospitalUserPatch,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("hospital_admin")),
):
    user = _get_own_hospital_user(db, ctx.hospital_id, user_id)
    if payload.email is not None and payload.email != user.email:
        if db.query(StaffUser).filter(StaffUser.email == payload.email).first():
            raise HTTPException(status.HTTP_409_CONFLICT, "a user with this email already exists")
        old_email = user.email
        user.email = payload.email
        log_action(
            db, ctx, "change_user_email", target=f"{old_email} -> {payload.email}", hospital_id=ctx.hospital_id
        )
    if payload.role is not None:
        if payload.role not in SELF_SERVICE_ROLES:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY, f"role must be one of {SELF_SERVICE_ROLES}"
            )
        user.role = payload.role
        log_action(
            db, ctx, "change_user_role", target=f"{user.email} -> {payload.role}", hospital_id=ctx.hospital_id
        )
    if payload.password is not None:
        user.hashed_password = pwd_context.hash(payload.password)
        log_action(db, ctx, "reset_user_password", target=user.email, hospital_id=ctx.hospital_id)
    db.commit()
    db.refresh(user)
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_my_hospital_user(
    user_id: str,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("hospital_admin")),
):
    user = _get_own_hospital_user(db, ctx.hospital_id, user_id)
    if str(user.id) == ctx.subject:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "cannot delete your own account")
    email = user.email
    db.delete(user)
    db.commit()
    log_action(db, ctx, "delete_user", target=email, hospital_id=ctx.hospital_id)
