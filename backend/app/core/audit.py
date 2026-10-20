from sqlalchemy.orm import Session

from app.auth.dependencies import TenantContext
from app.models.entities import AuditLog


def log_event(
    db: Session,
    actor_email: str,
    actor_role: str,
    action: str,
    target: str | None = None,
    hospital_id: str | None = None,
) -> None:
    db.add(
        AuditLog(
            actor_email=actor_email,
            actor_role=actor_role,
            action=action,
            target=target,
            hospital_id=hospital_id,
        )
    )
    db.commit()


def log_action(
    db: Session,
    ctx: TenantContext,
    action: str,
    target: str | None = None,
    hospital_id: str | None = None,
) -> None:
    log_event(
        db,
        actor_email=ctx.email or ctx.subject,
        actor_role=ctx.role,
        action=action,
        target=target,
        hospital_id=hospital_id or ctx.hospital_id,
    )
