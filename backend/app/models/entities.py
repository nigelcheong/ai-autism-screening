import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Boolean, DateTime, ForeignKey, Text, Float, Integer, Enum
)
from sqlalchemy.orm import relationship

from app.models.db import Base, GUID


def uuid_pk():
    return Column(GUID(), primary_key=True, default=uuid.uuid4)


class PlatformAdmin(Base):
    __tablename__ = "platform_admins"
    id = uuid_pk()
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)


class Hospital(Base):
    __tablename__ = "hospitals"
    id = uuid_pk()
    name = Column(String, nullable=False)
    plan = Column(String, default="free")
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Primary point of contact at the hospital, for the platform admin to reach.
    contact_name = Column(String, nullable=True)
    contact_designation = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)

    staff_users = relationship("StaffUser", back_populates="hospital")
    patients = relationship("Patient", back_populates="hospital")


class StaffUser(Base):
    __tablename__ = "staff_users"
    id = uuid_pk()
    hospital_id = Column(GUID(), ForeignKey("hospitals.id"), nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum("hospital_admin", "doctor", "staff", name="staff_role"), nullable=False)

    hospital = relationship("Hospital", back_populates="staff_users")


class ApiKey(Base):
    __tablename__ = "api_keys"
    id = uuid_pk()
    hospital_id = Column(GUID(), ForeignKey("hospitals.id"), nullable=False)
    key_hash = Column(String, nullable=False)
    last_four = Column(String, nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Patient(Base):
    __tablename__ = "patients"
    id = uuid_pk()
    hospital_id = Column(GUID(), ForeignKey("hospitals.id"), nullable=False)
    external_patient_id = Column(String, nullable=True)
    name = Column(String, nullable=False)
    dob = Column(DateTime, nullable=True)
    # Demographic/history fields asked on the first screening's Personal
    # Information step. Persisted here (like dob) so returning for another
    # screening doesn't ask the same questions again -- only what's still
    # missing, or what the user explicitly chooses to edit.
    gender = Column(String, nullable=True)
    autism = Column(Boolean, nullable=True)  # family history of autism
    jaundice = Column(Boolean, nullable=True)
    used_app_before = Column(Boolean, nullable=True)
    country_of_res = Column(String, nullable=True)
    relation = Column(String, nullable=True)  # who typically answers for this patient
    # Who added this patient record. A doctor's patient list is filtered to
    # patients they have a session with -- without this, a patient a doctor
    # creates on the fly (with no session yet) would be invisible to them
    # until someone assigns them a session.
    created_by = Column(GUID(), ForeignKey("staff_users.id"), nullable=True)

    hospital = relationship("Hospital", back_populates="patients")
    sessions = relationship("DiagnosticSession", back_populates="patient")


class Question(Base):
    __tablename__ = "questions"
    id = uuid_pk()
    # Which screening instrument this question belongs to: "aq10" (ages 4-11,
    # 4-point agree/disagree scale) or "qchat10" (ages 18-48 months, each
    # question has its own 5-point frequency scale). The age-based flow picks
    # the instrument; see app/core/age.py.
    instrument = Column(String, nullable=False, default="aq10")
    # Logical slot this question occupies within its instrument ("aq1".."aq10",
    # "qc1".."qc10", ...). Answers are keyed by this, so editing a question
    # inserts a new row with the same key and a higher version, rather than
    # mutating text in place — old sessions keep the exact wording and options
    # they were answered against (see core/diagnosis.py).
    key = Column(String, nullable=False)
    text = Column(String, nullable=False)
    answer_scale = Column(String, nullable=False)  # human-readable label, e.g. "4-point agree/disagree"
    # JSON-encoded list[str] of the answer options shown for this question, in
    # order (e.g. ["Always","Usually","Sometimes","Rarely","Never"]).
    options_json = Column(Text, nullable=False, default="[]")
    # JSON-encoded list[int]: which option indices (0-based, into options_json)
    # count as a "concerning" / scorable response for this question.
    scoring_indices_json = Column(Text, nullable=False, default="[]")
    version = Column(Integer, default=1)
    active = Column(Boolean, default=True)
    # Display order in the screening UI, independent of `key` (which must stay
    # stable since it's how historical answers are addressed).
    sort_order = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class DiagnosticSession(Base):
    __tablename__ = "diagnostic_sessions"
    id = uuid_pk()
    hospital_id = Column(GUID(), ForeignKey("hospitals.id"), nullable=False)
    patient_id = Column(GUID(), ForeignKey("patients.id"), nullable=False)
    session_type = Column(Enum("questionnaire", "image", name="session_type"), nullable=False)
    assigned_doctor_id = Column(GUID(), ForeignKey("staff_users.id"), nullable=True)
    # Fernet-encrypted JSON blob (answers + personal_info) — see app/core/encryption.py.
    # Never store or return this raw; always go through encrypt_json/decrypt_json.
    question_responses_encrypted = Column(Text, nullable=True)
    model_version = Column(String, nullable=True)
    risk_score = Column(Float, nullable=True)
    risk_category = Column(Enum("low", "moderate", "high", name="risk_category"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    consent_given = Column(Boolean, default=False)

    patient = relationship("Patient", back_populates="sessions")
    images = relationship("SessionImage", back_populates="session")


class SessionImage(Base):
    __tablename__ = "session_images"
    id = uuid_pk()
    session_id = Column(GUID(), ForeignKey("diagnostic_sessions.id"), nullable=False)
    image_url = Column(String, nullable=False)
    # This image's own model-predicted probability (0-1) of the positive
    # class, so the report can show a per-image breakdown, not just the
    # session-level average. Null for stub-scored sessions.
    risk_score = Column(Float, nullable=True)
    # Filename (under uploads/images/heatmaps/) of this image's Grad-CAM
    # attention overlay -- which pixels most drove the model's prediction
    # for this specific photo. Null for stub-scored sessions.
    heatmap_url = Column(String, nullable=True)
    # JSON-encoded list of {label, direction, magnitude} -- this image's
    # SHAP values (shap.GradientExplainer), pooled into a 3x3 grid of
    # regions and ranked, the same shape as the questionnaire's per-answer
    # explanation. A chart, not a second heat map: Grad-CAM above is the
    # visual explanation; this is the numeric one. Null for stub-scored
    # sessions.
    shap_regions_json = Column(Text, nullable=True)
    # Filename (under uploads/images/shap_diagrams/) of this image's SHAP
    # grid diagram -- the same region values as shap_regions_json, drawn
    # directly on an enlarged copy of the photo with a 3x3 grid overlay and
    # arrow callouts, instead of (or alongside) the ranked-list chart.
    # Null for stub-scored sessions.
    shap_diagram_url = Column(String, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("DiagnosticSession", back_populates="images")


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = uuid_pk()
    actor_email = Column(String, nullable=False)
    actor_role = Column(String, nullable=False)
    action = Column(String, nullable=False)  # e.g. "impersonate_user", "update_hospital_plan"
    target = Column(String, nullable=True)  # human-readable target, e.g. an email or hospital name
    hospital_id = Column(GUID(), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
