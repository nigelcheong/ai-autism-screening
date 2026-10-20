from collections import Counter
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import require_role, TenantContext
from app.models.db import get_db
from app.models.entities import Hospital, DiagnosticSession

router = APIRouter(prefix="/internal/admin/platform-stats", tags=["platform-stats"])

DAYS = 14


class DayCount(BaseModel):
    date: str
    questionnaire: int
    image: int


class PlatformStats(BaseModel):
    total_hospitals: int
    active_hospitals: int
    total_sessions: int
    questionnaire_sessions: int
    image_sessions: int
    high_risk_sessions: int
    sessions_by_day: list[DayCount]


@router.get("", response_model=PlatformStats)
def get_platform_stats(
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    hospitals = db.query(Hospital).all()
    sessions = db.query(DiagnosticSession).all()

    since = datetime.utcnow() - timedelta(days=DAYS - 1)
    since_date = since.date()
    counts: dict[tuple, int] = Counter()
    for s in sessions:
        d = s.created_at.date()
        if d >= since_date:
            counts[(d, s.session_type)] += 1

    sessions_by_day = []
    for i in range(DAYS):
        d = (since + timedelta(days=i)).date()
        sessions_by_day.append(
            DayCount(
                date=d.isoformat(),
                questionnaire=counts.get((d, "questionnaire"), 0),
                image=counts.get((d, "image"), 0),
            )
        )

    return PlatformStats(
        total_hospitals=len(hospitals),
        active_hospitals=sum(1 for h in hospitals if h.active),
        total_sessions=len(sessions),
        questionnaire_sessions=sum(1 for s in sessions if s.session_type == "questionnaire"),
        image_sessions=sum(1 for s in sessions if s.session_type == "image"),
        high_risk_sessions=sum(1 for s in sessions if s.risk_category == "high"),
        sessions_by_day=sessions_by_day,
    )
