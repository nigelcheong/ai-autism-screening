from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_staff, TenantContext
from app.core.diagnosis import run_questionnaire_diagnosis, run_image_diagnosis, to_diagnosis_result
from app.core.uploads import save_uploaded_image
from app.models.db import get_db
from app.schemas.diagnose import QuestionnaireSubmission, ImageSubmission, DiagnosisResult

router = APIRouter(prefix="/internal/diagnose", tags=["diagnose"])


@router.post("/images/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    ctx: TenantContext = Depends(get_current_staff),
):
    stored_names = [await save_uploaded_image(f) for f in files]
    return {"image_urls": stored_names}


@router.post("/questionnaire", response_model=DiagnosisResult)
def diagnose_questionnaire(
    submission: QuestionnaireSubmission,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    session = run_questionnaire_diagnosis(db, ctx.hospital_id, submission)
    return to_diagnosis_result(session)


@router.post("/image", response_model=DiagnosisResult)
def diagnose_image(
    submission: ImageSubmission,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    session = run_image_diagnosis(db, ctx.hospital_id, submission)
    return to_diagnosis_result(session)
