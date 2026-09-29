"""Idempotent deployment setup: create missing tables and application roles."""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from database.models import Base, Role
from database.setup_admin import ROLES


def initialize_schema(engine):
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(text("SELECT pg_advisory_xact_lock(746291830)"))
        Base.metadata.create_all(connection)
        with Session(bind=connection) as session:
            for role_id, name in ROLES:
                if session.get(Role, role_id) is None:
                    session.add(Role(role_id=role_id, role_name=name))
            session.flush()


if __name__ == "__main__":
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise SystemExit("DATABASE_URL is required for database setup.")
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            url = "postgresql+psycopg2://" + url[len(prefix):]
            break
    engine = create_engine(url)
    try:
        initialize_schema(engine)
        print("Database tables and roles initialized; existing data preserved.")
    finally:
        engine.dispose()
