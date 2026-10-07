"""Global pytest configuration and fixtures for singlangbydeshu."""

import os
import sqlite3
import tempfile

import pytest

# Configure isolated database path BEFORE any project modules are imported
_temp_dir = tempfile.TemporaryDirectory()
_test_db_path = os.path.join(_temp_dir.name, "test_session_isolated.db")
os.environ["SINGLANG_DB_PATH"] = _test_db_path

import database as db  # Import after environment variable is configured


@pytest.fixture(autouse=True)
def clean_test_database():
    """Ensure every test has a freshly initialized, isolated database."""
    db.init_db()
    yield
    # Clean tables after each test
    try:
        with db.get_connection() as conn:
            conn.execute("DELETE FROM translations")
            conn.execute("DELETE FROM practice_history")
            conn.commit()
    except (sqlite3.Error, OSError):
        pass
