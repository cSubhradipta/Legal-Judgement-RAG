import json
import sqlite3
from pathlib import Path

DATABASE = Path(__file__).resolve().parent / "data" / "chat_history.sqlite3"


def setup():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE, timeout=30) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                conversation_id TEXT PRIMARY KEY,
                messages TEXT NOT NULL,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


def load_history(conversation_id):
    setup()
    with sqlite3.connect(DATABASE, timeout=30) as connection:
        row = connection.execute(
            "SELECT messages FROM chat_history WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
    if row is None:
        return []
    try:
        messages = json.loads(row[0])
        return messages if isinstance(messages, list) else []
    except (ValueError, TypeError):
        return []


def save_history(conversation_id, messages):
    setup()
    content = json.dumps(messages, ensure_ascii=False)
    with sqlite3.connect(DATABASE, timeout=30) as connection:
        connection.execute(
            """
            INSERT INTO chat_history (conversation_id, messages, last_updated)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(conversation_id) DO UPDATE SET
                messages = excluded.messages,
                last_updated = CURRENT_TIMESTAMP
            """,
            (conversation_id, content),
        )
