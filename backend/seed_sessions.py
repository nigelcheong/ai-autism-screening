"""Seed a few sample diagnostic sessions so the dashboard tiles aren't empty. Local dev only."""

from app.models.db import SessionLocal
from app.models.entities import Hospital, Patient, DiagnosticSession

if __name__ == "__main__":
    db = SessionLocal()
    try:
        hospital = db.query(Hospital).filter(Hospital.name == "Demo Hospital").first()
        patient = db.query(Patient).filter(Patient.hospital_id == hospital.id).first()

        existing = db.query(DiagnosticSession).filter(
            DiagnosticSession.hospital_id == hospital.id
        ).count()
        if existing:
            print(f"{existing} sessions already exist, skipping")
        else:
            samples = [
                ("questionnaire", 0.2, "low"),
                ("questionnaire", 0.5, "moderate"),
                ("questionnaire", 0.8, "high"),
                ("image", 0.75, "high"),
                ("questionnaire", 0.1, "low"),
            ]
            for session_type, score, category in samples:
                db.add(
                    DiagnosticSession(
                        hospital_id=hospital.id,
                        patient_id=patient.id,
                        session_type=session_type,
                        risk_score=score,
                        risk_category=category,
                        model_version="stub-0.0",
                        consent_given=True,
                    )
                )
            db.commit()
            print(f"seeded {len(samples)} sample sessions")
    finally:
        db.close()
