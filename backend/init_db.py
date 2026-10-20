"""Create all tables for local development (sqlite). Production (Supabase/Postgres)
schema + RLS policies should be applied via SQL migrations instead, not this script."""

from app.models.db import Base, engine
from app.models import entities  # noqa: F401 (registers models on Base.metadata)

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("tables created")
