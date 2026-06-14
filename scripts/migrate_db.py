"""
Database initialization script.
Creates all SQLite tables from the SQLAlchemy models.

Run once on fresh installations:
    python scripts/migrate_db.py

Safe to run multiple times — uses CREATE TABLE IF NOT EXISTS semantics.
"""

import os
import sys

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import eventlet
eventlet.monkey_patch()

from dotenv import load_dotenv
load_dotenv()

from app import create_app
from app.extensions import db
from app.models import Patient, RecordingSession, ECGRecord, Prediction  # noqa


def init_db(config_name: str = "development") -> None:
    app = create_app(config_name)
    with app.app_context():
        db.create_all()
        db_path = app.config.get("SQLALCHEMY_DATABASE_URI", "?")
        print(f"Database initialized: {db_path}")
        print("Tables created:")
        for table in db.engine.table_names():
            print(f"  - {table}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="development", choices=["development", "production"])
    args = parser.parse_args()
    init_db(args.env)
