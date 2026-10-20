import json
import re
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy.orm import Session

from app.auth.dependencies import require_role, get_current_staff, TenantContext
from app.models.db import get_db
from app.models.entities import Question

router = APIRouter(tags=["question-bank"])

INSTRUMENTS = ("aq10", "qchat10")
KEY_PREFIXES = {"aq10": "aq", "qchat10": "qc"}


def _key_sort(key: str) -> int:
    match = re.search(r"(\d+)$", key)
    return int(match.group(1)) if match else 0


class QuestionOut(BaseModel):
    id: UUID
    instrument: str
    key: str
    text: str
    answer_scale: str
    options: list[str]
    scoring_indices: list[int]
    version: int
    active: bool
    sort_order: int

    @classmethod
    def from_row(cls, q: Question) -> "QuestionOut":
        return cls(
            id=q.id,
            instrument=q.instrument,
            key=q.key,
            text=q.text,
            answer_scale=q.answer_scale,
            options=json.loads(q.options_json),
            scoring_indices=json.loads(q.scoring_indices_json),
            version=q.version,
            active=q.active,
            sort_order=q.sort_order,
        )


class ActiveQuestionOut(BaseModel):
    id: UUID
    key: str
    text: str
    options: list[str]


class QuestionCreate(BaseModel):
    instrument: str
    text: str
    options: list[str]
    scoring_indices: list[int]
    answer_scale: str = ""

    @field_validator("instrument")
    @classmethod
    def valid_instrument(cls, v: str) -> str:
        if v not in INSTRUMENTS:
            raise ValueError(f"instrument must be one of {INSTRUMENTS}")
        return v

    @field_validator("options")
    @classmethod
    def at_least_two_options(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("a question needs at least 2 options")
        return v


class QuestionPatch(BaseModel):
    text: str | None = None
    options: list[str] | None = None
    scoring_indices: list[int] | None = None
    active: bool | None = None


class ReorderIn(BaseModel):
    keys: list[str]  # desired display order, e.g. ["aq3", "aq1", "aq2", ...]


# --- Admin: full question bank management (all versions, active + retired) ---

admin_router = APIRouter(prefix="/internal/admin/questions", tags=["question-bank"])


@admin_router.get("", response_model=list[QuestionOut])
def list_questions(
    instrument: str | None = None,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    query = db.query(Question)
    if instrument:
        query = query.filter(Question.instrument == instrument)
    rows = query.all()
    rows = sorted(rows, key=lambda q: (q.instrument, q.sort_order, _key_sort(q.key), q.version))
    return [QuestionOut.from_row(q) for q in rows]


@admin_router.post("", response_model=QuestionOut)
def create_question(
    payload: QuestionCreate,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    for idx in payload.scoring_indices:
        if idx < 0 or idx >= len(payload.options):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "scoring_indices out of range")

    existing = (
        db.query(Question.key, Question.sort_order)
        .filter(Question.instrument == payload.instrument)
        .all()
    )
    next_n = max([_key_sort(k) for k, _ in existing], default=0) + 1
    next_order = max([o for _, o in existing], default=-1) + 1
    question = Question(
        instrument=payload.instrument,
        key=f"{KEY_PREFIXES[payload.instrument]}{next_n}",
        text=payload.text,
        answer_scale=payload.answer_scale or f"{len(payload.options)}-point scale",
        options_json=json.dumps(payload.options),
        scoring_indices_json=json.dumps(payload.scoring_indices),
        version=1,
        active=True,
        sort_order=next_order,
    )
    db.add(question)
    db.commit()
    db.refresh(question)
    return QuestionOut.from_row(question)


@admin_router.patch("/reorder", status_code=status.HTTP_204_NO_CONTENT)
def reorder_questions(
    payload: ReorderIn,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    for order, key in enumerate(payload.keys):
        db.query(Question).filter(Question.key == key).update({"sort_order": order})
    db.commit()


@admin_router.patch("/{question_id}", response_model=QuestionOut)
def update_question(
    question_id: str,
    payload: QuestionPatch,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("platform_admin")),
):
    question = db.query(Question).filter(Question.id == question_id).first()
    if question is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "question not found")

    new_options = payload.options if payload.options is not None else json.loads(question.options_json)
    new_scoring = (
        payload.scoring_indices if payload.scoring_indices is not None else json.loads(question.scoring_indices_json)
    )
    for idx in new_scoring:
        if idx < 0 or idx >= len(new_options):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "scoring_indices out of range")

    content_changed = (
        (payload.text is not None and payload.text != question.text)
        or (payload.options is not None and payload.options != json.loads(question.options_json))
        or (
            payload.scoring_indices is not None
            and payload.scoring_indices != json.loads(question.scoring_indices_json)
        )
    )

    if content_changed:
        # New wording/options/scoring -> new version, so past sessions keep
        # referencing exactly what they were answered against (core/diagnosis.py).
        question.active = False
        new_version = Question(
            instrument=question.instrument,
            key=question.key,
            text=payload.text if payload.text is not None else question.text,
            answer_scale=question.answer_scale,
            options_json=json.dumps(new_options),
            scoring_indices_json=json.dumps(new_scoring),
            version=question.version + 1,
            active=payload.active if payload.active is not None else True,
            sort_order=question.sort_order,
        )
        db.add(new_version)
        db.commit()
        db.refresh(new_version)
        return QuestionOut.from_row(new_version)

    if payload.active is not None:
        question.active = payload.active
        db.commit()
        db.refresh(question)
    return QuestionOut.from_row(question)


# --- Any authenticated staff: the current live question set, for the screening UI ---

live_router = APIRouter(prefix="/internal/questions", tags=["question-bank"])


@live_router.get("/active", response_model=list[ActiveQuestionOut])
def list_active_questions(
    instrument: str,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    rows = db.query(Question).filter(Question.active.is_(True), Question.instrument == instrument).all()
    rows = sorted(rows, key=lambda q: (q.sort_order, _key_sort(q.key)))
    return [
        ActiveQuestionOut(id=q.id, key=q.key, text=q.text, options=json.loads(q.options_json))
        for q in rows
    ]
