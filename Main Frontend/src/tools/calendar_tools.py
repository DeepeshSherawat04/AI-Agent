"""
Mock Calendar Tools:
- check_availability(date)
- reserve_slot(date, time, email)
- send_booking_notification(email, details)

These are LangChain tool-wrapped functions for LLM agent use.
"""

from typing import Dict, Any, Optional
from langchain.tools import tool

from src.utils.database import get_db
from src.utils.webhook import send_booking_notification as _send_notification


@tool
def check_availability(date: str) -> Dict[str, Any]:
    """
    Check available appointment slots for a given date.

    Args:
        date: Date in YYYY-MM-DD format (e.g., "2026-07-15")

    Returns:
        Dict with available flag, slot list, and message.
    """
    db = get_db()
    result = db.check_availability(date)
    return result


@tool
def reserve_slot(date: str, time: str, email: str, service: str) -> Dict[str, Any]:
    """
    Reserve an appointment slot in the database.

    Args:
        date: Appointment date in YYYY-MM-DD format
        time: Appointment time in HH:MM format (24-hour)
        email: User email for confirmation
        service: Type of service (e.g., Meeting, Consultation)

    Returns:
        Dict with success status, appointment details, and alternative slots if failed.
    """
    db = get_db()
    result = db.reserve_slot(date, time, email, service)
    return result


@tool
def send_booking_notification(email: str, details: str) -> Dict[str, Any]:
    """
    Send a mock booking confirmation notification via webhook.

    Args:
        email: Recipient email address
        details: JSON string of booking details (date, time, service, appointment_id)

    Returns:
        Dict with success status and delivery info.
    """
    import json
    try:
        details_dict = json.loads(details)
    except json.JSONDecodeError:
        details_dict = {"raw": details}

    return _send_notification(email, details_dict)


@tool
def get_alternative_slots(date: str, limit: int = 3) -> Dict[str, Any]:
    """
    Get alternative available slots when the requested date is fully booked.

    Args:
        date: The originally requested date (to exclude)
        limit: Maximum number of alternatives to return (default 3)

    Returns:
        Dict with list of alternative date/time slots.
    """
    db = get_db()
    alternatives = db._get_alternative_slots(date, limit)

    return {
        "original_date": date,
        "alternatives": alternatives,
        "count": len(alternatives),
        "message": f"Found {len(alternatives)} alternative slots." if alternatives else "No alternatives available."
    }


# Tool registry for easy access
CALENDAR_TOOLS = [
    check_availability,
    reserve_slot,
    send_booking_notification,
    get_alternative_slots,
]