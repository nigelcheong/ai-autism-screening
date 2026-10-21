import re
import secrets
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from passlib.context import CryptContext
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.auth.dependencies import require_role, TenantContext
from app.auth.api_key import hash_api_key, mask_api_key
from app.auth.jwt import create_access_token
from app.core.audit import log_action
from app.core.time import utc_iso
from app.models.db import get_db
from app.models.entities import Hospital, ApiKey, StaffUser

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter(prefix="/internal/admin/hospitals", tags=["admin"])

PLANS = ("free", "basic", "pro", "enterprise")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class HospitalIn(BaseModel):
    name: str
    plan: str = "free"
    contact_name: str | None = None
    contact_designation: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None

    @field_validator("contact_email")
    @classmethod
    def valid_contact_email(cls, v: str | None) -> str | None:
        if v and not EMAIL_RE.match(v):
            raise ValueError("not a valid email address")
        return v


class HospitalPatch(BaseModel):
    name: str | None = None
    plan: str | None = None
    active: bool | None = None
    contact_name: str | None = None
    contact_designation: str | None = None
    contact_email: str | None = None
    contact_phone: str | None = None

    @field_validator("contact_email")
    @classmethod
    def valid_contact_email(cls, v: str | None) -> str | None:
        if v and not EMAIL_RE.match(v):
            raise ValueError("not a valid email address")
        return v


class HospitalOut(HospitalIn):
    id: UUID
    active: bool

    class Config:
        from_attributes = True


@router.get("", response_model=list[HospitalOut])
def list_hospitals(
    search: str | None = None,
    limit: int = 25,
    offset: int = 0,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    query = db.query(Hospital)
    if search:
        query = query.filter(Hospital.name.ilike(f"%{search}%"))
    return query.order_by(Hospital.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/{hospital_id}", response_model=HospitalOut)
def get_hospital(
    hospital_id: str,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    hospital = db.query(Hospital).filter(Hospital.id == hospital_id).first()
    if hospital is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "hospital not found")
    return hospital


@router.post("", response_model=HospitalOut)
def create_hospital(
    payload: HospitalIn,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    if payload.plan not in PLANS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"plan must be one of {PLANS}")
    hospital = Hospital(**payload.dict())
    db.add(hospital)
    db.commit()
    db.refresh(hospital)
    log_action(db, ctx, "create_hospital", target=hospital.name, hospital_id=str(hospital.id))
    return hospital


@router.patch("/{hospital_id}", response_model=HospitalOut)
def update_hospital(
    hospital_id: str,
    payload: HospitalPatch,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    hospital = db.query(Hospital).filter(Hospital.id == hospital_id).first()
    if hospital is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "hospital not found")
    if payload.name is not None and payload.name.strip() and payload.name != hospital.name:
        old_name = hospital.name
        hospital.name = payload.name.strip()
        log_action(db, ctx, "rename_hospital", target=f"{old_name} -> {hospital.name}", hospital_id=hospital_id)
    if payload.plan is not None:
        if payload.plan not in PLANS:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"plan must be one of {PLANS}")
        hospital.plan = payload.plan
        log_action(
            db, ctx, "update_hospital_plan", target=f"{hospital.name} -> {payload.plan}", hospital_id=hospital_id
        )
    if payload.contact_name is not None:
        hospital.contact_name = payload.contact_name
    if payload.contact_designation is not None:
        hospital.contact_designation = payload.contact_designation
    if payload.contact_email is not None:
        hospital.contact_email = payload.contact_email
    if payload.contact_phone is not None:
        hospital.contact_phone = payload.contact_phone
    if payload.active is not None:
        hospital.active = payload.active
        log_action(
            db,
            ctx,
            "activate_hospital" if payload.active else "deactivate_hospital",
            target=hospital.name,
            hospital_id=hospital_id,
        )
    db.commit()
    db.refresh(hospital)
    return hospital


@router.get("/{hospital_id}/api-key")
def get_hospital_api_key_status(
    hospital_id: str,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    key = (
        db.query(ApiKey)
        .filter(ApiKey.hospital_id == hospital_id, ApiKey.active.is_(True))
        .order_by(ApiKey.created_at.desc())
        .first()
    )
    if key is None:
        return {"active": False, "masked_key": None, "created_at": None}
    return {
        "active": key.active,
        "masked_key": mask_api_key(key.last_four),
        "created_at": utc_iso(key.created_at),
    }


@router.post("/{hospital_id}/api-key")
def issue_api_key(
    hospital_id: str,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    db.query(ApiKey).filter(ApiKey.hospital_id == hospital_id, ApiKey.active.is_(True)).update(
        {"active": False}
    )
    raw_key = secrets.token_urlsafe(32)
    record = ApiKey(
        hospital_id=hospital_id,
        key_hash=hash_api_key(raw_key),
        last_four=raw_key[-4:],
        active=True,
    )
    db.add(record)
    db.commit()
    log_action(db, ctx, "issue_api_key", hospital_id=hospital_id)
    # raw_key is only ever returned once, at creation time.
    return {"api_key": raw_key}


class HospitalUserIn(BaseModel):
    email: str
    password: str
    role: str  # "hospital_admin" | "doctor" | "staff"

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


@router.get("/{hospital_id}/users", response_model=list[HospitalUserOut])
def list_hospital_users(
    hospital_id: str,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    return (
        db.query(StaffUser)
        .filter(StaffUser.hospital_id == hospital_id)
        .offset(offset)
        .limit(limit)
        .all()
    )


@router.post("/{hospital_id}/users", response_model=HospitalUserOut)
def create_hospital_user(
    hospital_id: str,
    payload: HospitalUserIn,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    if payload.role not in ("hospital_admin", "doctor", "staff"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid role")
    if db.query(StaffUser).filter(StaffUser.email == payload.email).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "a user with this email already exists")

    user = StaffUser(
        hospital_id=hospital_id,
        email=payload.email,
        hashed_password=pwd_context.hash(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    log_action(db, ctx, "create_hospital_user", target=f"{user.email} ({user.role})", hospital_id=hospital_id)
    return user


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


def _get_hospital_user(db: Session, hospital_id: str, user_id: str) -> StaffUser:
    user = (
        db.query(StaffUser)
        .filter(StaffUser.id == user_id, StaffUser.hospital_id == hospital_id)
        .first()
    )
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    return user


@router.patch("/{hospital_id}/users/{user_id}", response_model=HospitalUserOut)
def update_hospital_user(
    hospital_id: str,
    user_id: str,
    payload: HospitalUserPatch,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    user = _get_hospital_user(db, hospital_id, user_id)
    if payload.email is not None and payload.email != user.email:
        if db.query(StaffUser).filter(StaffUser.email == payload.email).first():
            raise HTTPException(status.HTTP_409_CONFLICT, "a user with this email already exists")
        old_email = user.email
        user.email = payload.email
        log_action(db, ctx, "change_user_email", target=f"{old_email} -> {payload.email}", hospital_id=hospital_id)
    if payload.role is not None:
        if payload.role not in ("hospital_admin", "doctor", "staff"):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid role")
        user.role = payload.role
        log_action(db, ctx, "change_user_role", target=f"{user.email} -> {payload.role}", hospital_id=hospital_id)
    if payload.password is not None:
        user.hashed_password = pwd_context.hash(payload.password)
        log_action(db, ctx, "reset_user_password", target=user.email, hospital_id=hospital_id)
    db.commit()
    db.refresh(user)
    return user


@router.delete("/{hospital_id}/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_hospital_user(
    hospital_id: str,
    user_id: str,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    user = _get_hospital_user(db, hospital_id, user_id)
    email = user.email
    db.delete(user)
    db.commit()
    log_action(db, ctx, "delete_user", target=email, hospital_id=hospital_id)


class ImpersonateOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/{hospital_id}/users/{user_id}/impersonate", response_model=ImpersonateOut)
def impersonate_hospital_user(
    hospital_id: str,
    user_id: str,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    user = _get_hospital_user(db, hospital_id, user_id)
    token = create_access_token(
        subject=str(user.id),
        role=user.role,
        hospital_id=str(user.hospital_id),
        email=user.email,
        impersonated_by=ctx.subject,
    )
    log_action(db, ctx, "impersonate_user", target=user.email, hospital_id=hospital_id)
    return ImpersonateOut(access_token=token)
