"""
SQLite-based persistence for conversation state.
Uses LangGraph's SqliteSaver for checkpointing.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import message_to_dict, messages_from_dict
from langgraph.checkpoint.sqlite import SqliteSaver

from src.utils.config import get_config


class SQLiteCheckpointer:
    """
    Custom SQLite persistence layer for Streamlit session state.
    Stores thread_id, messages, and booking state in SQLite.
    Survives page refreshes and browser restarts.
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_config().get_checkpoint_db_path()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the custom checkpoints table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT PRIMARY KEY,
                    messages TEXT,
                    booking_date TEXT,
                    booking_time TEXT,
                    booking_email TEXT,
                    booking_service TEXT,
                    current_agent TEXT,
                    last_action TEXT,
                    suggested_slots TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def save(self, thread_id: str, messages: list, state_dict: dict) -> None:
        """Save conversation state to SQLite."""
        messages_json = json.dumps([message_to_dict(m) for m in messages])
        booking = state_dict.get("booking_details", {})
        suggested = state_dict.get("suggested_slots", [])

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO checkpoints (thread_id, messages, booking_date, booking_time, 
                                         booking_email, booking_service, current_agent, 
                                         last_action, suggested_slots, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    messages = excluded.messages,
                    booking_date = excluded.booking_date,
                    booking_time = excluded.booking_time,
                    booking_email = excluded.booking_email,
                    booking_service = excluded.booking_service,
                    current_agent = excluded.current_agent,
                    last_action = excluded.last_action,
                    suggested_slots = excluded.suggested_slots,
                    updated_at = excluded.updated_at
                """,
                (
                    thread_id,
                    messages_json,
                    booking.get("date"),
                    booking.get("time"),
                    booking.get("email"),
                    booking.get("service"),
                    state_dict.get("current_agent", "triage"),
                    state_dict.get("last_action", ""),
                    json.dumps(suggested),
                    sqlite3.datetime.datetime.now().isoformat()
                )
            )
            conn.commit()

    def load(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Load conversation state from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM checkpoints WHERE thread_id = ?", (thread_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            messages = json.loads(row["messages"]) if row["messages"] else []
            suggested = json.loads(row["suggested_slots"]) if row["suggested_slots"] else []

            return {
                "messages": messages_from_dict(messages) if messages else [],
                "booking_details": {
                    "date": row["booking_date"],
                    "time": row["booking_time"],
                    "email": row["booking_email"],
                    "service": row["booking_service"]
                },
                "current_agent": row["current_agent"],
                "last_action": row["last_action"],
                "suggested_slots": suggested
            }

    def delete(self, thread_id: str) -> None:
        """Delete a checkpoint by thread_id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.commit()


def get_checkpointer():
    """
    Create and return a LangGraph SqliteSaver instance.
    Uses a separate database file to avoid schema conflicts with
    the custom SQLiteCheckpointer table.
    """
    # Use a separate file for LangGraph's internal checkpointing
    db_path = Path("data/langgraph_checkpoints.sqlite")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)