"""
SQLite database setup for appointments and LangGraph checkpoints.
Auto-creates tables on first run.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from src.utils.config import get_config


class DatabaseManager:
    """Manages SQLite connections and schema for appointment storage."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or get_config().get_db_path()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Appointments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS appointments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    email TEXT NOT NULL,
                    service TEXT,
                    status TEXT DEFAULT 'confirmed',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, time)
                )
            """)

            # Mock availability slots table (pre-seeded)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS availability_slots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    is_available INTEGER DEFAULT 1,
                    UNIQUE(date, time)
                )
            """)

            conn.commit()

        self._seed_slots()

    def _seed_slots(self) -> None:
        """Pre-populate availability slots for the next 14 days."""
        from datetime import timedelta
        import random

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if already seeded
            cursor.execute("SELECT COUNT(*) FROM availability_slots")
            if cursor.fetchone()[0] > 0:
                return

            today = datetime.now().date()
            times = ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]

            for day_offset in range(14):
                date_str = (today + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                for time_str in times:
                    # Randomly make some slots unavailable (30% chance)
                    is_available = 0 if random.random() < 0.3 else 1
                    cursor.execute(
                        "INSERT OR IGNORE INTO availability_slots (date, time, is_available) VALUES (?, ?, ?)",
                        (date_str, time_str, is_available)
                    )

            conn.commit()

    def check_availability(self, date: str) -> Dict[str, Any]:
        """Check available slots for a given date."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT time, is_available FROM availability_slots WHERE date = ? ORDER BY time",
                (date,)
            )
            rows = cursor.fetchall()

            if not rows:
                return {
                    "date": date,
                    "available": False,
                    "slots": [],
                    "message": f"No slots found for {date}. Please try another date."
                }

            available_slots = [
                {"time": row["time"], "available": bool(row["is_available"])}
                for row in rows
            ]

            open_slots = [s for s in available_slots if s["available"]]

            return {
                "date": date,
                "available": len(open_slots) > 0,
                "slots": available_slots,
                "open_slots": [s["time"] for s in open_slots],
                "message": f"{len(open_slots)} slots available on {date}." if open_slots else f"All slots booked on {date}."
            }

    def reserve_slot(self, date: str, time: str, email: str, service: str = "General") -> Dict[str, Any]:
        """Reserve an appointment slot."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if slot is still available
            cursor.execute(
                "SELECT is_available FROM availability_slots WHERE date = ? AND time = ?",
                (date, time)
            )
            row = cursor.fetchone()

            if not row:
                return {
                    "success": False,
                    "message": f"Slot {date} {time} does not exist.",
                    "alternative_dates": self._get_alternative_slots(date)
                }

            if row[0] == 0:
                return {
                    "success": False,
                    "message": f"Slot {date} at {time} is already booked.",
                    "alternative_dates": self._get_alternative_slots(date)
                }

            # Mark slot as unavailable
            cursor.execute(
                "UPDATE availability_slots SET is_available = 0 WHERE date = ? AND time = ?",
                (date, time)
            )

            # Insert appointment
            cursor.execute(
                "INSERT INTO appointments (date, time, email, service) VALUES (?, ?, ?, ?)",
                (date, time, email, service)
            )

            conn.commit()

            return {
                "success": True,
                "appointment_id": cursor.lastrowid,
                "date": date,
                "time": time,
                "email": email,
                "service": service,
                "message": f"Appointment confirmed for {date} at {time}."
            }

    def _get_alternative_slots(self, exclude_date: str, limit: int = 3) -> List[Dict[str, str]]:
        """Find alternative available slots strictly AFTER the requested date."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Only suggest dates AFTER the requested date.
            # If user asked for July 13, we show July 14 onwards — never today or earlier.
            cursor.execute(
                """
                SELECT date, time FROM availability_slots 
                WHERE is_available = 1 AND date > ? 
                ORDER BY date, time 
                LIMIT ?
                """,
                (exclude_date, limit)
            )

            return [
                {"date": row["date"], "time": row["time"]}
                for row in cursor.fetchall()
            ]

    def get_appointments(self, email: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve appointments, optionally filtered by email."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if email:
                cursor.execute(
                    "SELECT * FROM appointments WHERE email = ? ORDER BY date, time",
                    (email,)
                )
            else:
                cursor.execute("SELECT * FROM appointments ORDER BY date, time")

            return [dict(row) for row in cursor.fetchall()]

    # ═══════════════════════════════════════════════════════════
    # FIX: Get most recently created appointments for sidebar
    # Orders by created_at DESC so newest bookings appear first
    # ═══════════════════════════════════════════════════════════
    def get_recent_appointments(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most recently created appointments (for sidebar)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # ═══════════════════════════════════════════════════════════
            # FIX: Order by id DESC as fallback when created_at is NULL
            # or inconsistent. id is AUTOINCREMENT so higher = newer.
            # ═══════════════════════════════════════════════════════════
            cursor.execute(
                """
                SELECT * FROM appointments 
                ORDER BY 
                    CASE WHEN created_at IS NULL THEN 0 ELSE 1 END DESC,
                    created_at DESC,
                    id DESC 
                LIMIT ?
                """,
                (limit,)
            )

            return [dict(row) for row in cursor.fetchall()]


# Singleton instance
_db_manager: DatabaseManager | None = None


def get_db() -> DatabaseManager:
    """Get or create the singleton DatabaseManager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager