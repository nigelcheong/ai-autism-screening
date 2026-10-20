"""Seed a demo hospital + one login per role + a patient, for local testing.

Run: python seed_demo.py
All passwords: password123
"""

from passlib.context import CryptContext

from app.models.db import SessionLocal
from app.models.entities import Hospital, StaffUser, Patient, PlatformAdmin

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

DEMO_USERS = [
    ("staff@demo.hospital", "staff"),
    ("doctor@demo.hospital", "doctor"),
    ("admin@demo.hospital", "hospital_admin"),
]


def get_or_create_staff(db, hospital_id, email, role):
    staff = db.query(StaffUser).filter(StaffUser.email == email).first()
    if not staff:
        staff = StaffUser(
            hospital_id=hospital_id,
            email=email,
            hashed_password=pwd_context.hash("password123"),
            role=role,
        )
        db.add(staff)
    return staff


if __name__ == "__main__":
    db = SessionLocal()
    try:
        hospital = db.query(Hospital).filter(Hospital.name == "Demo Hospital").first()
        if not hospital:
            hospital = Hospital(name="Demo Hospital", plan="free", active=True)
            db.add(hospital)
            db.commit()
            db.refresh(hospital)

        for email, role in DEMO_USERS:
            get_or_create_staff(db, hospital.id, email, role)

        platform_admin = (
            db.query(PlatformAdmin).filter(PlatformAdmin.email == "platform@autism.dev").first()
        )
        if not platform_admin:
            db.add(
                PlatformAdmin(
                    email="platform@autism.dev",
                    hashed_password=pwd_context.hash("password123"),
                )
            )

        patient = db.query(Patient).filter(Patient.name == "Test Patient").first()
        if not patient:
            patient = Patient(hospital_id=hospital.id, name="Test Patient")
            db.add(patient)

        db.commit()
        print(f"hospital_id={hospital.id}\n")
        print("Demo logins (all password: password123)")
        print("-" * 45)
        for email, role in DEMO_USERS:
            print(f"  {role:<16} {email}")
        print(f"  {'platform_admin':<16} platform@autism.dev")
    finally:
        db.close()
