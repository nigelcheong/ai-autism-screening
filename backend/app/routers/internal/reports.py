import base64
import json
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_staff, require_role, TenantContext
from app.core.encryption import decrypt_json
from app.core.risk_bands import BAND_LEGEND, describe_probability, short_label
from app.core.time import utc_iso
from app.core.uploads import heatmap_path, resolve_image_path, shap_diagram_path
from app.models.db import get_db
from app.models.entities import DiagnosticSession, Hospital, Patient, SessionImage, StaffUser

router = APIRouter(prefix="/internal/sessions", tags=["reports"])


class FeatureContribution(BaseModel):
    label: str
    value: str
    contributed_to_risk: bool
    image_data_url: str | None = None
    heatmap_data_url: str | None = None
    # A larger, standalone image (not a small thumbnail like the two above):
    # the photo enlarged with a 3x3 grid overlay and arrow callouts showing
    # each region's SHAP contribution directly on the image.
    shap_diagram_data_url: str | None = None


class SummaryStat(BaseModel):
    label: str
    value: str


class ExplanationItem(BaseModel):
    label: str
    direction: str  # "increased" | "decreased"
    magnitude: float  # 0-100, relative to this session's largest contribution


class SessionReport(BaseModel):
    id: UUID
    patient_name: str
    patient_dob: str | None
    patient_age_label: str | None
    patient_external_id: str | None
    hospital_name: str
    doctor_email: str | None
    session_type: str
    risk_score: float | None
    risk_category: str | None
    model_version: str | None
    model_description: str
    created_at: str
    generated_at: str
    summary_stats: list[SummaryStat]
    features: list[FeatureContribution]
    explanation: list[ExplanationItem]
    explanation_note: str
    note: str

    class Config:
        protected_namespaces = ()


INSTRUMENT_CUTOFFS = {
    "aq10": "6 of 10 or more (AQ-10 standard referral cutoff)",
    "qchat10": "3 of 10 or more (Q-CHAT-10 standard referral cutoff)",
}
INSTRUMENT_NAMES = {"aq10": "AQ-10", "qchat10": "Q-CHAT-10"}


def _model_description(model_version: str | None) -> str:
    """Friendly, report-ready name for a model_version string."""
    if not model_version:
        return "Not available"
    if model_version.startswith("stub"):
        return "Rule-based reference scoring (no trained model)"
    if model_version.startswith("aq10-logreg-"):
        n = model_version.removeprefix("aq10-logreg-")
        return f"AQ-10 Logistic Regression model (trained on {n} cases)"
    if model_version.startswith("qchat10-logreg-"):
        n = model_version.removeprefix("qchat10-logreg-")
        return f"Q-CHAT-10 Logistic Regression model (trained on {n} cases)"
    if model_version == "cnn_corrected_fold4_model":
        return (
            "cnn_corrected_fold4_model — DenseNet-201 convolutional neural network "
            "(fine-tuned on facial images, cross-validation fold 4)"
        )
    return model_version


def _age_label(dob: datetime | None) -> str | None:
    if dob is None:
        return None
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    months = (today.year - dob.year) * 12 + (today.month - dob.month)
    if today.day < dob.day:
        months -= 1
    months = max(months, 0)
    years, rem_months = divmod(months, 12)
    if years == 0:
        return f"{rem_months} month{'s' if rem_months != 1 else ''}"
    if rem_months == 0:
        return f"{years} year{'s' if years != 1 else ''}"
    return f"{years}y {rem_months}m"


def _image_data_url(path, resize: int | None = None) -> str:
    """Read an image file as a base64 data: URL, so the frontend can render
    it inline without a separate authenticated fetch per thumbnail."""
    if resize:
        import io

        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB").resize((resize, resize))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    else:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@router.get("/{session_id}/report", response_model=SessionReport)
def get_session_report(
    session_id: str,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(get_current_staff),
):
    session = db.query(DiagnosticSession).filter(DiagnosticSession.id == session_id).first()
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    if ctx.role != "platform_admin" and str(session.hospital_id) != ctx.hospital_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "not your hospital's session")

    patient = db.query(Patient).filter(Patient.id == session.patient_id).first()

    features: list[FeatureContribution] = []
    summary_stats: list[SummaryStat] = []
    explanation: list[ExplanationItem] = []
    explanation_note = ""
    note = ""

    if session.session_type == "questionnaire" and session.question_responses_encrypted:
        data = decrypt_json(session.question_responses_encrypted)
        answers: dict = data.get("answers", {})
        instrument = data.get("instrument")
        # question_meta is a per-session snapshot (text + options + scoring)
        # taken at submission time, so a later edit to the question bank
        # doesn't rewrite how old reports read or score. Older sessions
        # predating this snapshot used a binary agree/disagree AQ-10 only,
        # handled by the legacy branch below.
        question_meta: dict = data.get("question_meta", {})

        if question_meta:
            raw_score = 0
            for key, selected in answers.items():
                meta = question_meta.get(key)
                if not meta:
                    continue
                options = meta["options"]
                label_value = options[selected] if 0 <= selected < len(options) else str(selected)
                indicative = selected in meta["scoring_indices"]
                raw_score += int(indicative)
                features.append(
                    FeatureContribution(
                        label=meta["text"],
                        value=label_value,
                        contributed_to_risk=indicative,
                    )
                )
            instrument_label = INSTRUMENT_NAMES.get(instrument, instrument or "")
            summary_stats = [
                SummaryStat(label="Instrument used", value=instrument_label),
                SummaryStat(label="Items scored as indicative", value=f"{raw_score} of {len(question_meta)}"),
                SummaryStat(
                    label="Clinical referral cutoff",
                    value=INSTRUMENT_CUTOFFS.get(instrument, "—"),
                ),
                SummaryStat(
                    label="Model-predicted probability",
                    value=(
                        describe_probability(session.risk_score)
                        if session.risk_score is not None
                        else "—"
                    ),
                ),
                SummaryStat(label="Likelihood bands", value=BAND_LEGEND),
            ]
            note = (
                f"Scored using the {instrument_label} instrument. A checked item is a response "
                "that counts toward the risk score under that instrument's standard scoring rules. "
                "Question wording and options reflect what was actually shown at the time of this "
                "scan, even if the question bank has since been edited. The likelihood category "
                "shown above the chart is based on the instrument's own clinical cutoff score "
                "(items scored as indicative vs. the referral cutoff) -- the model-predicted "
                "probability is a separate, independent statistical estimate, and the two may not "
                "always agree. Neither is a diagnosis."
            )

            model_contributions: dict | None = data.get("model_contributions")
            if model_contributions:
                max_abs = max((abs(v) for v in model_contributions.values()), default=0) or 1.0
                ranked = sorted(model_contributions.items(), key=lambda kv: -abs(kv[1]))[:6]
                for key, contrib in ranked:
                    meta = question_meta.get(key)
                    if not meta:
                        continue
                    explanation.append(
                        ExplanationItem(
                            label=meta["text"],
                            direction="increased" if contrib > 0 else "decreased",
                            magnitude=round(abs(contrib) / max_abs * 100, 1),
                        )
                    )
                explanation_note = (
                    "Ranked by how much each answer pushed the trained model's estimate up or "
                    "down, using the model's own learned weight for that item. This is an exact "
                    "decomposition of its score, not an approximation."
                )
            else:
                explanation_note = (
                    "A per-answer explanation isn't available for this session (it predates "
                    "this feature, or ran on the rule-based stub)."
                )
        else:
            # Legacy sessions: binary agree/disagree AQ-10, no snapshot.
            from app.core.aq10 import AQ10_TEXT_BY_KEY

            question_texts: dict = data.get("question_texts", {})
            for key, value in answers.items():
                text = question_texts.get(key) or AQ10_TEXT_BY_KEY.get(key, key)
                agreed = bool(value)
                features.append(
                    FeatureContribution(
                        label=text,
                        value="Agree" if agreed else "Disagree",
                        contributed_to_risk=agreed,
                    )
                )
            note = (
                "Contribution shown reflects the legacy screening-stub scoring "
                "(each 'Agree' answer contributes to the risk score)."
            )

    elif session.session_type == "image":
        images = db.query(SessionImage).filter(SessionImage.session_id == session.id).all()
        scored = [img for img in images if img.risk_score is not None]
        any_shap = False
        for i, img in enumerate(images, start=1):
            heatmap_data_url = None
            if img.heatmap_url:
                hpath = heatmap_path(img.heatmap_url)
                if hpath:
                    heatmap_data_url = _image_data_url(hpath)

            shap_diagram_data_url = None
            if img.shap_diagram_url:
                dpath = shap_diagram_path(img.shap_diagram_url)
                if dpath:
                    shap_diagram_data_url = _image_data_url(dpath)

            image_data_url = None
            try:
                image_data_url = _image_data_url(resolve_image_path(img.image_url), resize=224)
            except HTTPException:
                pass  # original file missing/moved -- still show what we have

            if img.risk_score is not None:
                pct = img.risk_score * 100
                features.append(
                    FeatureContribution(
                        label=f"Image {i}",
                        value=f"{pct:.0f}% predicted likelihood",
                        contributed_to_risk=img.risk_score >= 0.5,
                        image_data_url=image_data_url,
                        heatmap_data_url=heatmap_data_url,
                        shap_diagram_data_url=shap_diagram_data_url,
                    )
                )
            else:
                features.append(
                    FeatureContribution(label=f"Image {i}", value="Analyzed", contributed_to_risk=True)
                )

            if img.shap_regions_json:
                any_shap = True
                prefix = f"Image {i}: " if len(images) > 1 else ""
                for region in json.loads(img.shap_regions_json):
                    explanation.append(
                        ExplanationItem(
                            label=f"{prefix}{region['label']}",
                            direction=region["direction"],
                            magnitude=region["magnitude"],
                        )
                    )

        # Ranked across all images together, top 6 overall (matches the
        # questionnaire's explanation length).
        explanation = sorted(explanation, key=lambda item: -item.magnitude)[:6]

        summary_stats = [SummaryStat(label="Images analyzed", value=str(len(images)))]
        if scored:
            avg = sum(img.risk_score for img in scored) / len(scored)
            top = max(scored, key=lambda img: img.risk_score)
            top_idx = images.index(top) + 1
            summary_stats += [
                SummaryStat(label="Average predicted probability", value=describe_probability(avg)),
                SummaryStat(
                    label="Highest-confidence image",
                    value=f"Image {top_idx} -- {short_label(top.risk_score)}",
                ),
                SummaryStat(label="Likelihood bands", value=BAND_LEGEND),
            ]
        note = (
            "Risk score is the model's average predicted likelihood across all submitted images, "
            "shown per-image above. The heat map on each image is Grad-CAM (pools gradients over "
            "the model's last convolutional layer) -- a visual attention map, not a clinical "
            "facial-feature finding. A checkmark next to an image below means that image's own "
            "predicted likelihood was 50% or higher; this is only which images individually leaned "
            "positive, not the overall likelihood band shown above (which uses the average)."
        )
        if any_shap:
            explanation_note = (
                "SHAP (shap.GradientExplainer) values, pooled into a 3x3 grid of image regions "
                "and ranked by how much each region pushed the prediction up or down -- also "
                "shown drawn directly on the enlarged photo below, with each region's push "
                "labeled in place. A second, independently-computed explanation alongside the "
                "Grad-CAM heat map above -- agreement between methods is a stronger signal than "
                "any one alone, but none of them is a diagnostic measurement."
            )
        else:
            explanation_note = (
                "A SHAP explanation isn't available for this session (it predates this feature, "
                "ran on the rule-based stub, or SHAP computation failed for this image)."
            )

    doctor = (
        db.query(StaffUser).filter(StaffUser.id == session.assigned_doctor_id).first()
        if session.assigned_doctor_id
        else None
    )
    hospital = db.query(Hospital).filter(Hospital.id == session.hospital_id).first()

    return SessionReport(
        id=session.id,
        patient_name=patient.name if patient else "Unknown",
        patient_dob=patient.dob.date().isoformat() if patient and patient.dob else None,
        patient_age_label=_age_label(patient.dob) if patient else None,
        patient_external_id=patient.external_patient_id if patient else None,
        hospital_name=hospital.name if hospital else "—",
        doctor_email=doctor.email if doctor else None,
        session_type=session.session_type,
        risk_score=session.risk_score,
        risk_category=session.risk_category,
        model_version=session.model_version,
        model_description=_model_description(session.model_version),
        created_at=utc_iso(session.created_at),
        generated_at=datetime.now(timezone.utc).isoformat(),
        summary_stats=summary_stats,
        features=features,
        explanation=explanation,
        explanation_note=explanation_note,
        note=note,
    )


class AssignDoctorIn(BaseModel):
    doctor_id: str | None  # null clears the assignment


class SessionOut(BaseModel):
    id: UUID
    assigned_doctor_id: UUID | None

    class Config:
        from_attributes = True


@router.patch("/{session_id}/assign-doctor", response_model=SessionOut)
def assign_doctor(
    session_id: str,
    payload: AssignDoctorIn,
    db: Session = Depends(get_db),
    ctx: TenantContext = Depends(require_role("staff", "hospital_admin", "doctor")),
):
    session = (
        db.query(DiagnosticSession)
        .filter(DiagnosticSession.id == session_id, DiagnosticSession.hospital_id == ctx.hospital_id)
        .first()
    )
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
    session.assigned_doctor_id = payload.doctor_id
    db.commit()
    db.refresh(session)
    return session
