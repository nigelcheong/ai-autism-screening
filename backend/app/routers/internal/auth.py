from fastapi import APIRouter, Depends, Header, HTTPException, status
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_staff, TenantContext
from app.auth.jwt import create_access_token, decode_access_token
from app.core.audit import log_event, log_action
from app.models.db import get_db
from app.models.entities import StaffUser, PlatformAdmin

router = APIRouter(prefix="/internal/auth", tags=["auth"])
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    admin = db.query(PlatformAdmin).filter(PlatformAdmin.email == payload.email).first()
    if admin and pwd_context.verify(payload.password, admin.hashed_password):
        token = create_access_token(
            subject=str(admin.id), role="platform_admin", hospital_id=None, email=admin.email
        )
        log_event(db, admin.email, "platform_admin", "login")
        return LoginResponse(access_token=token)

    staff = db.query(StaffUser).filter(StaffUser.email == payload.email).first()
    if staff and pwd_context.verify(payload.password, staff.hashed_password):
        token = create_access_token(
            subject=str(staff.id),
            role=staff.role,
            hospital_id=str(staff.hospital_id),
            email=staff.email,
        )
        log_event(db, staff.email, staff.role, "login", hospital_id=str(staff.hospital_id))
        return LoginResponse(access_token=token)

    log_event(db, payload.email, "unknown", "login_failed")
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")


@router.post("/refresh", response_model=LoginResponse)
def refresh(authorization: str = Header(...)):
    """Reissue the caller's own token with a fresh expiry.

    Used by the "stay logged in" option on the frontend's session-expiry
    warning. Requires the current token to still be valid (not yet
    expired) -- there's no separate longer-lived refresh token, so once a
    token actually expires the user must log in again.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
    try:
        payload = decode_access_token(authorization.removeprefix("Bearer "))
    except ValueError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token")

    new_token = create_access_token(
        subject=payload["sub"],
        role=payload["role"],
        hospital_id=payload.get("hospital_id"),
        email=payload.get("email"),
        impersonated_by=payload.get("impersonated_by"),
    )
    return LoginResponse(access_token=new_token)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    # JWTs are stateless (no server-side session to invalidate) — this
    # endpoint exists purely to record the logout event in the audit log.
    log_action(db, ctx, "logout")


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
def change_password(
    payload: ChangePasswordRequest,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    model = PlatformAdmin if ctx.role == "platform_admin" else StaffUser
    account = db.query(model).filter(model.id == ctx.subject).first()
    if account is None or not pwd_context.verify(payload.current_password, account.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "current password is incorrect")
    if len(payload.new_password) < 8:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "new password must be at least 8 characters")
    account.hashed_password = pwd_context.hash(payload.new_password)
    db.commit()
