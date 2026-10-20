from fastapi import APIRouter, Depends, Header, Request
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.auth.dependencies import resolve_tenant_from_api_key
from app.core.diagnosis import run_questionnaire_diagnosis, run_image_diagnosis, to_diagnosis_result
from app.models.db import get_db
from app.schemas.diagnose import QuestionnaireSubmission, ImageSubmission, DiagnosisResult

router = APIRouter(prefix="/v1/external/diagnose", tags=["external-diagnose"])
limiter = Limiter(key_func=get_remote_address)


@router.post("/questionnaire", response_model=DiagnosisResult)
@limiter.limit("30/minute")
def diagnose_questionnaire(
    request: Request,
    submission: QuestionnaireSubmission,
    x_api_key: str = Header(...),
    db: Session = Depends(get_db),
):
    ctx = resolve_tenant_from_api_key(x_api_key, db)
    session = run_questionnaire_diagnosis(db, ctx.hospital_id, submission)
    return to_diagnosis_result(session)


@router.post("/image", response_model=DiagnosisResult)
@limiter.limit("30/minute")
def diagnose_image(
    request: Request,
    submission: ImageSubmission,
    x_api_key: str = Header(...),
    db: Session = Depends(get_db),
):
    ctx = resolve_tenant_from_api_key(x_api_key, db)
    session = run_image_diagnosis(db, ctx.hospital_id, submission)
    return to_diagnosis_result(session)
