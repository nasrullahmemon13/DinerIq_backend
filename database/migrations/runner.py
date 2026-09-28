"""
DineIQ Analytics - Database Migration Runner
Applies DDL schema migrations to SQLite and PostgreSQL engines.
Supports SRS Requirements (lxi) through (lxv).
"""

import os
import sqlite3
from sqlalchemy import text
from database.connection import get_engine, CURRENT_DIR

MIGRATIONS_DIR = os.path.dirname(os.path.abspath(__file__))

def run_migrations():
    """Reads SQL files in database/migrations and applies them to the configured DB engine."""
    engine = get_engine()
    sql_files = sorted([f for f in os.listdir(MIGRATIONS_DIR) if f.endswith(".sql")])

    print(f"Running database migrations from {MIGRATIONS_DIR}...")
    with engine.connect() as conn:
        for sql_file in sql_files:
            file_path = os.path.join(MIGRATIONS_DIR, sql_file)
            print(f"Applying migration: {sql_file}")
            with open(file_path, "r", encoding="utf-8") as f:
                sql_content = f.read()

            statements = [s.strip() for s in sql_content.split(";") if s.strip()]
            for stmt in statements:
                try:
                    conn.execute(text(stmt))
                    conn.commit()
                except Exception as e:
                    if "already exists" in str(e).lower():
                        continue
                    else:
                        print(f"Warning during migration statement execution: {e}")
            print(f"[OK] Applied {sql_file}")

if __name__ == "__main__":
    run_migrations()
