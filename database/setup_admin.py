"""Create missing tables/roles and an admin without replacing existing users.

Run: python -m database.setup_admin
Requires DATABASE_URL; prompts privately for the new administrator password.
"""
import getpass
import os
import uuid

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from database.models import Base, Role, User

ROLES = [
    ("admin", "Administrator"),
    ("regional_manager", "Regional Manager"),
    ("manager", "Store Manager"),
    ("analyst", "Data Analyst"),
]


def initialize(engine, username, email, password):
    if os.getenv("DINEIQ_SEED_DEMO") == "1":
        raise ValueError("Disable DINEIQ_SEED_DEMO before secure admin setup.")
    if not username.strip() or "@" not in email or len(password) < 16:
        raise ValueError("Provide a username, email and password of at least 16 characters.")
    # Import after validation; demo seeding must never be enabled for this command.
    from database.connection import hash_password
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            from sqlalchemy import text
            connection.execute(text("SELECT pg_advisory_xact_lock(746291830)"))
        Base.metadata.create_all(connection)
        with Session(bind=connection) as session:
            existing = session.scalars(select(User).where(
                (User.username == username) | (User.email == email)
            )).first()
            if existing:
                raise ValueError("Username or email already exists; no account was changed.")
            for role_id, name in ROLES:
                if session.get(Role, role_id) is None:
                    session.add(Role(role_id=role_id, role_name=name))
            session.flush()
            session.add(User(
                user_id="USER-" + uuid.uuid4().hex,
                username=username, email=email,
                hashed_password=hash_password(password),
                full_name="Administrator", role_id="admin", is_active=True,
            ))
            session.flush()


def main():
    os.environ["DINEIQ_SEED_DEMO"] = "0"
    url = os.getenv("DATABASE_URL", "")
    if not url:
        raise SystemExit("Set DATABASE_URL to your actual database connection string first.")
    if url.startswith("postgres://"):
        url = "postgresql+psycopg2://" + url[len("postgres://"):]
    username = input("Admin username: ").strip()
    email = input("Admin email: ").strip()
    password = getpass.getpass("Admin password (16+ characters): ")
    if password != getpass.getpass("Confirm password: "):
        raise SystemExit("Passwords did not match; no changes made.")
    engine = create_engine(url)
    try:
        initialize(engine, username, email, password)
        print("Tables, roles and administrator created. Password was not logged.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
