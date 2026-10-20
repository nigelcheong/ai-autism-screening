import json
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.age import (
    IMAGE_SCREENING_MAX_MONTHS,
    IMAGE_SCREENING_MIN_MONTHS,
    age_in_months,
    is_eligible_for_image_screening,
)
from app.core.encryption import encrypt_json
from app.core.time import utc_iso
from app.ml.inference import run_questionnaire_inference, run_image_inference
from app.models.entities import DiagnosticSession, SessionImage, Question
from app.schemas.diagnose import QuestionnaireSubmission, ImageSubmission, DiagnosisResult


def to_diagnosis_result(session: DiagnosticSession) -> DiagnosisResult:
    return DiagnosisResult(
        id=session.id,
        session_type=session.session_type,
        risk_score=session.risk_score,
        risk_category=session.risk_category,
        model_version=session.model_version,
        created_at=utc_iso(session.created_at),
    )


def _snapshot_question_meta(db: Session, instrument: str, keys: list[str]) -> dict[str, dict]:
    """Capture the exact wording, options, and scoring live at answer time, so
    a later edit to the question bank doesn't change how a past session's
    report reads or scores."""
    meta: dict[str, dict] = {}
    for key in keys:
        question = (
            db.query(Question)
            .filter(Question.key == key, Question.instrument == instrument, Question.active.is_(True))
            .first()
        )
        if question:
            meta[key] = {
                "text": question.text,
                "options": json.loads(question.options_json),
                "scoring_indices": json.loads(question.scoring_indices_json),
            }
    return meta


def run_questionnaire_diagnosis(
    db: Session, hospital_id: str, submission: QuestionnaireSubmission
) -> DiagnosticSession:
    question_meta = _snapshot_question_meta(db, submission.instrument, list(submission.answers.keys()))
    result = run_questionnaire_inference(submission.answers, question_meta, submission.instrument)

    session = DiagnosticSession(
        hospital_id=hospital_id,
        patient_id=submission.patient_id,
        session_type="questionnaire",
        question_responses_encrypted=encrypt_json(
            {
                "instrument": submission.instrument,
                "answers": submission.answers,
                "question_meta": question_meta,
                "personal_info": submission.personal_info.dict(),
                # Snapshotted alongside the answers themselves, like
                # question_meta -- so a later default-model swap can't
                # retroactively change why a past report says what it says.
                "model_contributions": result.get("feature_contributions"),
            }
        ),
        model_version=result["model_version"],
        risk_score=result["risk_score"],
        risk_category=result["risk_category"],
        consent_given=submission.consent_given,
        assigned_doctor_id=submission.assigned_doctor_id,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def run_image_diagnosis(
    db: Session, hospital_id: str, submission: ImageSubmission
) -> DiagnosticSession:
    dob = submission.personal_info.dob if submission.personal_info else None
    if dob:
        months = age_in_months(datetime.fromisoformat(dob).date())
        if not is_eligible_for_image_screening(months):
            raise HTTPException(
                400,
                f"Image screening is validated for children {IMAGE_SCREENING_MIN_MONTHS} months "
                f"to {IMAGE_SCREENING_MAX_MONTHS // 12} years old; this patient is {months} months "
                "old. Use the questionnaire screening instead.",
            )

    result = run_image_inference(submission.image_urls)

    session = DiagnosticSession(
        hospital_id=hospital_id,
        patient_id=submission.patient_id,
        session_type="image",
        question_responses_encrypted=(
            encrypt_json({"personal_info": submission.personal_info.dict()})
            if submission.personal_info
            else None
        ),
        model_version=result["model_version"],
        risk_score=result["risk_score"],
        risk_category=result["risk_category"],
        consent_given=submission.consent_given,
        assigned_doctor_id=submission.assigned_doctor_id,
    )
    db.add(session)
    db.flush()

    per_image_scores = result.get("per_image_scores", [])
    per_image_heatmaps = result.get("per_image_heatmaps", [])
    per_image_shap_regions = result.get("per_image_shap_regions", [])
    per_image_shap_diagrams = result.get("per_image_shap_diagrams", [])
    for i, url in enumerate(submission.image_urls):
        score = per_image_scores[i] if i < len(per_image_scores) else None
        heatmap = per_image_heatmaps[i] if i < len(per_image_heatmaps) else None
        regions = per_image_shap_regions[i] if i < len(per_image_shap_regions) else None
        diagram = per_image_shap_diagrams[i] if i < len(per_image_shap_diagrams) else None
        db.add(
            SessionImage(
                session_id=session.id,
                image_url=url,
                risk_score=score,
                heatmap_url=heatmap,
                shap_regions_json=json.dumps(regions) if regions else None,
                shap_diagram_url=diagram,
            )
        )

    db.commit()
    db.refresh(session)
    return session
