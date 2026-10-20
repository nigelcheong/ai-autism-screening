from uuid import UUID

from pydantic import BaseModel, field_validator


class PersonalInfo(BaseModel):
    age: int
    dob: str | None = None  # ISO date; used to pick the screening instrument by age
    gender: str  # "male" | "female" | "others"
    autism: bool  # family history of autism
    jaundice: bool
    used_app_before: bool
    country_of_res: str
    relation: str  # "Self" | "Parent" | "Relative" | "Health care professional" | "Others"


def _require_consent(v: bool) -> bool:
    if not v:
        raise ValueError("consent_given must be true — patient/guardian consent is required to run a screening")
    return v


class QuestionnaireSubmission(BaseModel):
    patient_id: str
    instrument: str  # "aq10" | "qchat10"
    answers: dict[str, int]  # {"aq1": 2, "qc3": 0, ...} -> selected option index
    personal_info: PersonalInfo
    consent_given: bool = False
    assigned_doctor_id: str | None = None

    @field_validator("consent_given")
    @classmethod
    def consent_required(cls, v: bool) -> bool:
        return _require_consent(v)


class ImageSubmission(BaseModel):
    patient_id: str
    image_urls: list[str]
    personal_info: PersonalInfo | None = None
    consent_given: bool = False
    assigned_doctor_id: str | None = None

    @field_validator("consent_given")
    @classmethod
    def consent_required(cls, v: bool) -> bool:
        return _require_consent(v)


class DiagnosisResult(BaseModel):
    id: UUID
    session_type: str
    risk_score: float
    risk_category: str
    model_version: str
    created_at: str

    class Config:
        from_attributes = True
        protected_namespaces = ()
