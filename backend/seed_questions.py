"""Seed the `questions` table with both age-based screening instruments.

Run once against a fresh DB: `python seed_questions.py`
Source: AQ-10 / Q-CHAT-10 (Allison, Auyeung & Baron-Cohen, 2012), as provided
in frontend/QuestionsData/.
"""

import json

from app.core.instruments import AQ10_ITEMS, QCHAT10_ITEMS
from app.models.db import SessionLocal
from app.models.entities import Question


def _seed_instrument(db, instrument: str, key_prefix: str, items, answer_scale_label: str):
    for i, (text, options, scoring_indices) in enumerate(items):
        db.add(
            Question(
                instrument=instrument,
                key=f"{key_prefix}{i + 1}",
                text=text,
                answer_scale=answer_scale_label,
                options_json=json.dumps(options),
                scoring_indices_json=json.dumps(scoring_indices),
                version=1,
                active=True,
                sort_order=i,
            )
        )


def seed():
    db = SessionLocal()
    try:
        existing = db.query(Question).count()
        if existing:
            print(f"questions table already has {existing} rows, skipping seed")
            return
        _seed_instrument(db, "aq10", "aq", AQ10_ITEMS, "4-point agree/disagree")
        _seed_instrument(db, "qchat10", "qc", QCHAT10_ITEMS, "5-point frequency")
        db.commit()
        print(f"seeded {len(AQ10_ITEMS)} AQ-10 items and {len(QCHAT10_ITEMS)} Q-CHAT-10 items")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
