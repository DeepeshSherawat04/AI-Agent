"""
Webhook client for mock notifications (Webhook.site / Pipedream).
Sends POST requests to simulate email/WhatsApp confirmations.
"""

import json
import requests
from typing import Dict, Any, Optional
from src.utils.config import get_config


def send_booking_notification(
    email: str,
    details: Dict[str, Any],
    webhook_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Send a mock booking confirmation notification via webhook.

    Args:
        email: Recipient email address
        details: Booking details dict (date, time, service, appointment_id)
        webhook_url: Override URL (uses config default if None)

    Returns:
        Dict with success status and response info
    """
    url = webhook_url or get_config().webhook_url

    if not url:
        return {
            "success": False,
            "message": "No webhook URL configured. Set WEBHOOK_URL in .env",
            "mock_mode": True
        }

    # BUG FIX: Remove fallback to "General Consultation" since service is now
    # explicitly collected from the user before this function is called.
    service = details.get("service") or "Not specified"

    payload = {
        "event": "appointment_booked",
        "recipient": email,
        "appointment": {
            "date": details.get("date"),
            "time": details.get("time"),
            "service": service,
            "appointment_id": details.get("appointment_id"),
            "status": "confirmed"
        },
        "notification": {
            "channel": "email",
            "subject": f"Appointment Confirmed — {details.get('date')} at {details.get('time')}",
            "body": (
                f"Hello,\n\n"
                f"Your appointment has been confirmed.\n\n"
                f"📅 Date: {details.get('date')}\n"
                f"⏰ Time: {details.get('time')}\n"
                f"📋 Service: {service}\n"
                f"🔢 Booking ID: {details.get('appointment_id')}\n\n"
                f"Thank you for choosing TrulyIAS!"
            )
        },
        "timestamp": details.get("timestamp", ""),
        "source": "TrulyIAS Scheduling Assistant"
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()

        return {
            "success": True,
            "status_code": response.status_code,
            "message": "Notification sent successfully via webhook.",
            "webhook_response": response.text[:200] if response.text else "OK",
            "mock_mode": False
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "message": f"Webhook failed: {str(e)}. Notification logged locally.",
            "mock_mode": True,
            "error": str(e)
        }