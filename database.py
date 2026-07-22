import sqlite3
import time
import os

DB_PATH = 'database.db'


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite database tables for translation history and practice stats."""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # Translation History Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gesture_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                hand_mode TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Practice / Quiz History Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS practice_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_sign TEXT NOT NULL,
                matched_sign TEXT NOT NULL,
                score INTEGER NOT NULL,
                streak INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()


def record_translation(gesture_name, confidence, hand_mode='Single-Hand'):
    """Records a single recognized sign translation into the database."""
    if not gesture_name or gesture_name == 'Detecting...':
        return
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO translations (gesture_name, confidence, hand_mode)
            VALUES (?, ?, ?)
        ''', (gesture_name, float(confidence), hand_mode))
        conn.commit()


def get_translation_history(limit=50):
    """Retrieves recent translation history from the database."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, gesture_name, confidence, hand_mode, timestamp
            FROM translations
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def record_practice_session(target_sign, matched_sign, score, streak):
    """Records a completed quiz game match or session."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO practice_history (target_sign, matched_sign, score, streak)
            VALUES (?, ?, ?, ?)
        ''', (target_sign, matched_sign, int(score), int(streak)))
        conn.commit()


def get_practice_history(limit=50):
    """Retrieves quiz practice history."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, target_sign, matched_sign, score, streak, timestamp
            FROM practice_history
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def clear_history():
    """Clears all translation logs."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM translations')
        conn.commit()


# Automatically initialize DB schema on import
init_db()
